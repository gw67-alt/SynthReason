import random
import math
import sys
from collections import defaultdict, Counter
from tqdm import tqdm
import numpy as np
import pickle

# NEW: Import the Hugging Face datasets library
try:
    from datasets import load_dataset, get_dataset_split_names
    HF_DATASETS_AVAILABLE = True
except ImportError:
    HF_DATASETS_AVAILABLE = False
    print("Hugging Face 'datasets' library not found. Please install it to use Hugging Face datasets: pip install datasets")

# NEW: Attempt to import PyOpenCL
try:
    import pyopencl as cl
    import pyopencl.array as cl_array
    PYOPENCL_AVAILABLE = True
    print("PyOpenCL found.")
except ImportError:
    PYOPENCL_AVAILABLE = False
    print("PyOpenCL not found. OpenCL acceleration will not be available. Please install it: pip install pyopencl")


class RetrocausalSymbolicMarkov:
    """
    Markov chain text generator.
    Original description mentioned retrocausal knowledge updates and symbolic distributions.
    This version simplifies the training process, removing K and S matrices,
    to better suit parallelization, focusing retrocausal/symbolic aspects on generation if retained.
    """

    # Static configuration variables
    SENTENCE_END_CHARS = set('.!?')

    def __init__(self, n=2, alpha=0.3, cascade_threshold=0.7, extra_dim_depth=3): # alpha is no longer used for K
        if not isinstance(n, int) or n < 1:
            raise ValueError("n-gram size 'n' must be a positive integer.")

        self.n = n
        # self.alpha = alpha # No longer used for K updates during training
        self.cascade_threshold = cascade_threshold # Used in generation
        self.extra_dim_depth = extra_dim_depth # Used in generation

        self.m = {} # Transitions: {context_tuple: {next_word: adjusted_frequency}}
        self.s = [] # Sentence starting n-grams
        self.all_words = set() # Will store actual words

        # K and S matrices removed for simpler, more parallelizable training
        # self.K = {}
        # self.S = {}
        
        self.cascade_memory = [] # Primarily for generation-time effects now
        self.epochs = 0 # Number of n-grams processed

        # Extra-dimensional integration structures (for generation)
        self.D = {}
        self.integration_history = [] # Might be populated during generation if that logic is kept
        self.fold_tensor = {}
        self.retrocausal_loops = [] # Populated during generation

        # Cache for expensive calculations
        self.similarity_cache = {}
        self.permutation_cache = {}
        self.dimensional_cache = {}

        # For OpenCL
        self.word_to_id = {}
        self.id_to_word = {}
        self.next_word_id = 0 # Counter for new words

        self.cl_ctx = None
        self.cl_queue = None
        self.cl_prg = None
        if PYOPENCL_AVAILABLE:
            try:
                # Try to get a GPU context first, then CPU
                self.cl_ctx = None
                platforms = cl.get_platforms()
                for platform in platforms:
                    try:
                        gpu_devices = platform.get_devices(device_type=cl.device_type.GPU)
                        if gpu_devices:
                            self.cl_ctx = cl.Context(devices=[gpu_devices[0]])
                            print(f"OpenCL context created on GPU: {gpu_devices[0].name}")
                            break
                    except cl.RuntimeError:
                        continue # Try next platform or device type
                if not self.cl_ctx: # If no GPU or GPU context failed, try CPU
                    for platform in platforms:
                        try:
                            cpu_devices = platform.get_devices(device_type=cl.device_type.CPU)
                            if cpu_devices:
                                self.cl_ctx = cl.Context(devices=[cpu_devices[0]])
                                print(f"OpenCL context created on CPU: {cpu_devices[0].name}")
                                break
                        except cl.RuntimeError:
                            continue
                if not self.cl_ctx: # If still no context, raise an error or use interactive
                     self.cl_ctx = cl.create_some_context(interactive=False) # Fallback
                     print(f"OpenCL context created on: {self.cl_ctx.devices[0].name} (fallback)")

                self.cl_queue = cl.CommandQueue(self.cl_ctx)
            except Exception as e:
                print(f"Could not initialize PyOpenCL context: {e}. OpenCL features disabled.")
                self.cl_ctx = None



    def _get_id(self, word, create=True):
        if word not in self.word_to_id:
            if create:
                self.word_to_id[word] = self.next_word_id
                self.id_to_word[self.next_word_id] = word
                self.next_word_id += 1
            else:
                return -1 
        return self.word_to_id[word]

    def _compile_opencl_kernels(self):
        if not self.cl_ctx or self.cl_prg:
            return

        kernel_source = f"""
        __kernel void find_ngrams(__global const int* word_ids,
                                  int num_total_words,
                                  int n_gram_size,
                                  __global int* out_context_ids,
                                  __global int* out_next_word_ids,
                                  __global int* out_indices,
                                  __global int* result_count
                                  ) {{
            int gid = get_global_id(0);

            if (gid < num_total_words - n_gram_size) {{
                int current_result_idx = atomic_inc(result_count);
                out_indices[current_result_idx] += gid;

                for (int i = 0; i < n_gram_size; ++i) {{
                    out_context_ids[current_result_idx * n_gram_size + i] = word_ids[gid + i];
                }}
                out_next_word_ids[current_result_idx] = word_ids[gid + n_gram_size];
            }}
        }}
        """
        try:
            self.cl_prg = cl.Program(self.cl_ctx, kernel_source).build()
            print("OpenCL kernels compiled successfully.")
        except Exception as e:
            print(f"Error compiling OpenCL kernels: {e}")
            self.cl_prg = None
            global PYOPENCL_AVAILABLE
            PYOPENCL_AVAILABLE = False


    def train_opencl(self, t):
        if not PYOPENCL_AVAILABLE or not self.cl_ctx or self.n <= 0:
            print("OpenCL not available, context not initialized, or n invalid. Falling back to standard training.")
            return self.train(t)

        if not self.cl_prg:
            self._compile_opencl_kernels()
            if not self.cl_prg:
                print("OpenCL kernels failed to compile. Falling back to standard training.")
                return self.train(t)

        if not isinstance(t, str) or not t:
            raise TypeError("Training data must be a non-empty string.")

        words = t.split()
        num_words = len(words)

        if num_words <= self.n:
            print(f"Warning: Training data has only {num_words} words, <= n-gram size {self.n}. Skipping OpenCL training.")
            return

        print(f"Training model with OpenCL on {num_words} words (n={self.n}). K/S matrices removed from training.")

        self.all_words.update(words)
        word_ids_list = [self._get_id(word, create=True) for word in words]
        word_ids_np = np.array(word_ids_list, dtype=np.int32)

        overall_word_freqs = Counter(words)
        total_word_count = float(num_words)

        num_possible_ngrams = num_words - self.n
        if num_possible_ngrams <= 0:
            print("Not enough words to form any n-grams. Skipping.")
            return

        mf = cl.mem_flags
        word_ids_g = cl.Buffer(self.cl_ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=word_ids_np)
        out_context_ids_g = cl.Buffer(self.cl_ctx, mf.WRITE_ONLY, size=num_possible_ngrams * self.n * np.dtype(np.int32).itemsize)
        out_next_word_ids_g = cl.Buffer(self.cl_ctx, mf.WRITE_ONLY, size=num_possible_ngrams * np.dtype(np.int32).itemsize)
        out_indices_g = cl.Buffer(self.cl_ctx, mf.WRITE_ONLY, size=num_possible_ngrams * np.dtype(np.int32).itemsize)
        result_count_g = cl.Buffer(self.cl_ctx, mf.READ_WRITE | mf.COPY_HOST_PTR, hostbuf=np.zeros(1, dtype=np.int32)) # READ_WRITE for atomic_inc

        kernel_args = (
            word_ids_g, np.int32(num_words), np.int32(self.n),
            out_context_ids_g, out_next_word_ids_g, out_indices_g, result_count_g
        )
        try:
            self.cl_prg.find_ngrams(self.cl_queue, (num_possible_ngrams,), None, *kernel_args).wait()
        except cl.Error as e: 
            print(f"OpenCL kernel execution error: {e}. Falling back to standard training.")
            return self.train(t)

        final_result_count_np = np.empty(1, dtype=np.int32)
        cl.enqueue_copy(self.cl_queue, final_result_count_np, result_count_g).wait()
        actual_results = final_result_count_np[0]

        contexts_flat_np = np.empty(actual_results * self.n, dtype=np.int32)
        next_words_np = np.empty(actual_results, dtype=np.int32)
        indices_np = np.empty(actual_results, dtype=np.int32)

        if actual_results > 0:
            cl.enqueue_copy(self.cl_queue, contexts_flat_np, out_context_ids_g) # Removed size, is_blocking=True by default for direct array copy
            cl.enqueue_copy(self.cl_queue, next_words_np, out_next_word_ids_g)
            cl.enqueue_copy(self.cl_queue, indices_np, out_indices_g)
        
        contexts_np = contexts_flat_np.reshape(actual_results, self.n) if actual_results > 0 else np.array([])


        temp_m_word_based = defaultdict(lambda: defaultdict(float))
        temp_s_word_based = set()
        
        # Simplified CPU post-processing. K, S, and complex symbolic adjustments removed from here.
        for i in tqdm(range(actual_results), desc="CPU Post-processing (Simplified)"):
            context_ids = tuple(contexts_np[i])
            next_word_id_val = next_words_np[i]
            original_idx = indices_np[i]

            current_ngram_words = tuple(self.id_to_word[cid] for cid in context_ids)
            next_word_actual_words = self.id_to_word[next_word_id_val]

            if original_idx == 0 or \
               (original_idx > 0 and words[original_idx-1] and \
                any(c in self.SENTENCE_END_CHARS for c in words[original_idx-1][-1:])):
                temp_s_word_based.add(current_ngram_words)

            base_increment = 1.0
            norm_freq_next_word = overall_word_freqs.get(next_word_actual_words, 0) / total_word_count
            global_freq_factor = 1.0 + math.log1p(norm_freq_next_word * 100) * 0.1 # Reduced influence
            
            len_next_word = len(next_word_actual_words)
            length_factor = 1.0 + math.log1p(len_next_word) * 0.02 # Reduced influence

            # Simplified increment: No l_inner_product from training, no K-factor
            symbolic_enhancement_factor = global_freq_factor * length_factor
            symbolic_enhancement_factor = max(0.5, min(symbolic_enhancement_factor, 2.0)) # Bound
            
            adjusted_increment_val = base_increment * symbolic_enhancement_factor
            noise_val = random.uniform(-0.01, 0.01) * adjusted_increment_val # Smaller noise
            final_increment_val = max(0.01, adjusted_increment_val + noise_val) # Ensure decent minimum

            temp_m_word_based[current_ngram_words][next_word_actual_words] += final_increment_val
        
        self.epochs += num_possible_ngrams

        self.m = {gram: dict(counts) for gram, counts in temp_m_word_based.items() if counts}
        self.s = list(filter(lambda sg: sg in self.m, temp_s_word_based))

        if not self.s and self.m:
            num_starts_to_sample = min(max(1, int(len(self.m) * 0.05)), 50)
            if num_starts_to_sample > len(self.m): num_starts_to_sample = len(self.m)
            if num_starts_to_sample > 0:
                self.s = random.sample(list(self.m.keys()), k=num_starts_to_sample)
        
        # self.cascade_memory is not populated from training anymore
        self.cascade_memory = [] 

        print(f"OpenCL-assisted simplified training complete. Model has {len(self.m)} contexts, {len(self.s)} valid starts.")
        
        word_ids_g.release()
        out_context_ids_g.release()
        out_next_word_ids_g.release()
        out_indices_g.release()
        result_count_g.release()

    # update_knowledge is no longer called by train_opencl
    # def update_knowledge(self, K_prev, alpha, S_n, x, x_true):
    #     I_x_true = 1.0 if x == x_true else 0.0
    #     A_gate = S_n * I_x_true
    #     error_term = A_gate - K_prev
    #     K_n = K_prev + alpha * S_n * error_term
    #     return max(0.0, min(1.0, K_n))


    def _symbolic_probability(self, context_tuple, options_dict):
        # Simplified: removed K-factor and cascade_mem_boost from training-derived memory
        if not isinstance(context_tuple, tuple):
            try:
                context_tuple = tuple(str(c) for c in context_tuple) if hasattr(context_tuple, '__iter__') and not isinstance(context_tuple, str) else tuple([str(context_tuple)])
            except TypeError:
                context_tuple = tuple()
        
        candidate_words_list = list(options_dict.keys())
        base_scores_list = list(options_dict.values())

        if not candidate_words_list:
            return [], []

        enhanced_scores = []
        for i, word_cand in enumerate(candidate_words_list):
            current_base_score = base_scores_list[i]
            # K-factor removed
            # cascade_mem_boost from training-derived self.cascade_memory removed
            # If generate() populates self.cascade_memory, that's a different mechanism
            # For now, direct score without these factors.
            final_score_for_word = current_base_score 
            enhanced_scores.append(max(0.0001, final_score_for_word))
        
        final_refined_scores = []
        if not context_tuple: # No context, use enhanced_scores directly
            final_refined_scores = enhanced_scores
        else:
            for i, word_cand in enumerate(candidate_words_list):
                score_after_k_cascade = enhanced_scores[i] # Name is now a bit misleading, it's just score
                context_sim_factor_accum = 1.0
                # L-semi-inner product for context similarity can be kept for generation if _safe_l_semi_inner_product is kept
                for context_word_idx, word_from_context in enumerate(context_tuple[:max(1, self.n //2)]):
                    l_sim = self._safe_l_semi_inner_product(str(word_from_context), word_cand)
                    context_sim_factor_accum *= (1.0 + l_sim * 0.15) 
                context_sim_factor_accum = min(context_sim_factor_accum, 3.0) # Cap total similarity boost

                score_after_context_sim = score_after_k_cascade * context_sim_factor_accum
                final_refined_scores.append(max(0.00001, score_after_context_sim)) # Ensure positive

        if not final_refined_scores: return [], []
        scored_candidates = sorted(zip(final_refined_scores, candidate_words_list), reverse=True)
        num_final_candidates = min(len(scored_candidates), 75) 
        top_candidates_words = [word for score, word in scored_candidates[:num_final_candidates]]
        top_candidates_scores = [score for score, word in scored_candidates[:num_final_candidates]]

        if not top_candidates_words:
            return [],[] 
        return top_candidates_words, top_candidates_scores


    # ... (Keep compute_extra_dimension, calculate_fold_strength, detect_retrocausal_integration,
    #      recompute_with_extra_dimension, calculate_cascade_coherence, retrocausal_information_kernel,
    #      _calculate_distinct_permutations_log, _safe_l_semi_inner_product, _compute_l_semi_inner_product
    #      These are complex and define the generation behavior. If _safe_l_semi_inner_product is too slow,
    #      it might also need simplification or removal, which would further change the model.
    #      For now, they are kept, meaning generation can still be complex.
    #      Note: calculate_fold_strength and calculate_cascade_coherence use self.cascade_memory.
    #      This list will now primarily be populated during the generate() call itself.
    #      The original train() method is also kept as an alternative or for comparison.) ...

    def compute_extra_dimension(self, context, time_index, future_states):
        """
        Compute extra-dimensional vector when retrocausal integration occurs.
        Uses hyperbolic geometry H³ ⊗ S¹ for temporal folding.

        Args:
            context: Current context state
            time_index: Temporal position
            future_states: Anticipated future configurations

        Returns:
            numpy.ndarray: Extra-dimensional vector in H³⊗S¹ space
        """
        cache_key = (str(context)[:20], time_index, len(future_states))
        if cache_key in self.dimensional_cache:
            return self.dimensional_cache[cache_key]

        extra_dim = np.zeros(self.extra_dim_depth * 2)  # Real + Imaginary components

        try:
            for i in range(self.extra_dim_depth):
                tau = time_index + 1
                hyperbolic_factor = math.tanh(tau / 10.0)

                context_hash = hash(str(context)) % 1000000
                context_angle = (context_hash / 1000000.0) * 2 * math.pi

                future_influence = 0.0
                if future_states and len(future_states) > i:
                    try:
                        fs_val_str = str(future_states[i])[:10] 
                        if fs_val_str.replace('.','',1).replace('-','',1).isdigit():
                            future_val = float(fs_val_str)
                        else:
                            future_val = hash(str(future_states[i])) % 1000
                        future_influence = future_val / 1000.0
                    except (ValueError, OverflowError, TypeError):
                        future_influence = 0.5
                
                r = hyperbolic_factor * (1.0 + future_influence)
                r = min(max(r, -700), 700) 
                theta = context_angle + i * math.pi / self.extra_dim_depth

                extra_dim[i*2] = math.cosh(r) * math.cos(theta)
                extra_dim[i*2 + 1] = math.sinh(r) * math.sin(theta)

            for i in range(self.extra_dim_depth):
                temporal_phase = (time_index / 10.0) % (2 * math.pi)
                # calculate_fold_strength uses self.cascade_memory, which is now mostly generation-time
                fold_strength = self.calculate_fold_strength(context, time_index) 
                fold_factor_arg = -fold_strength * temporal_phase / (2 * math.pi)
                fold_factor = math.exp(max(fold_factor_arg, -700))

                extra_dim[i*2] *= fold_factor
                extra_dim[i*2 + 1] *= math.cos(temporal_phase + i * math.pi / 3)
            
            norm = np.linalg.norm(extra_dim)
            if norm > 1e-9 and norm < 1e6 : 
                extra_dim = extra_dim / norm
            elif norm >= 1e6: 
                extra_dim = (extra_dim / norm) * 1e5 
            elif norm <= 1e-9 and norm !=0 : 
                extra_dim = np.random.normal(0, 0.01, self.extra_dim_depth * 2)
            else: 
                extra_dim = np.random.normal(0, 0.1, self.extra_dim_depth * 2)

        except (OverflowError, ValueError, ZeroDivisionError) as e:
            extra_dim = np.random.normal(0, 0.1, self.extra_dim_depth * 2)

        if len(self.dimensional_cache) < 1000:
            self.dimensional_cache[cache_key] = extra_dim
        return extra_dim

    def calculate_fold_strength(self, context, time_index):
        # This function now relies on self.cascade_memory being populated during generation
        fold_strength = 0.0
        try: 
            for i, past_context in enumerate(self.cascade_memory): # self.cascade_memory should be managed by generate()
                if past_context == context:
                    loop_length = len(self.cascade_memory) - i
                    fold_strength += 1.0 / (1.0 + loop_length)
        except TypeError: 
            pass

        if len(self.cascade_memory) > 2:
            # calculate_cascade_coherence also uses self.cascade_memory
            recent_coherence = self.calculate_cascade_coherence(self.cascade_memory[-3:])
            fold_strength += recent_coherence * 0.5
        return min(2.0, fold_strength)

    def detect_retrocausal_integration(self, current_context, generation_history):
        # generation_history is passed in, typically from the generate() method's local history
        if len(generation_history) < 3:
            return False
        context_str = str(current_context)
        for i, past_state in enumerate(generation_history[-10:]):
            past_str = str(past_state)
            similarity = self._safe_l_semi_inner_product(context_str, past_str) # Relies on this function
            if similarity > 0.8:
                # calculate_cascade_coherence here would use the short sequence passed.
                loop_coherence = self.calculate_cascade_coherence([past_state, current_context])
                if loop_coherence > self.cascade_threshold:
                    if len(self.retrocausal_loops) < 100: # self.retrocausal_loops is for generation
                        self.retrocausal_loops.append({
                            'start_index': len(generation_history) - 10 + i,
                            'end_index': len(generation_history),
                            'coherence': loop_coherence,
                            'context_similarity': similarity
                        })
                    return True
        return False

    def recompute_with_extra_dimension(self, context, candidate_words, candidate_weights, time_index):
        # This is a generation-time function
        future_states = []
        sorted_candidates = sorted(zip(candidate_weights, candidate_words), reverse=True)
        for weight, word in sorted_candidates[:5]: 
            future_states.append(word)

        extra_dim_vector = self.compute_extra_dimension(context, time_index, future_states)
        current_context_tuple = tuple(context) if isinstance(context, list) else context
        if not isinstance(current_context_tuple, tuple):
            current_context_tuple = tuple(str(current_context_tuple))

        if len(self.D) > 2000: 
            keys_to_remove = list(self.D.keys())[:len(self.D)-2000]
            for key_to_remove in keys_to_remove:
                del self.D[key_to_remove]
        self.D[(time_index, current_context_tuple)] = extra_dim_vector

        enhanced_weights = []
        for i, word in enumerate(candidate_words):
            base_weight = candidate_weights[i] if i < len(candidate_weights) else 1.0
            word_hash = hash(word) % 1000000 
            word_vector_components = []
            for j in range(self.extra_dim_depth * 2):
                angle = (word_hash / 1000000.0) * 2 * math.pi + (j * math.pi / (self.extra_dim_depth * 2))
                word_vector_components.append(math.cos(angle))
            word_vector = np.array(word_vector_components)
            try:
                norm_extra_dim = np.linalg.norm(extra_dim_vector)
                norm_word_vec = np.linalg.norm(word_vector)
                if norm_extra_dim > 1e-9 and norm_word_vec > 1e-9: 
                    dimensional_projection = np.dot(extra_dim_vector, word_vector) / (norm_extra_dim * norm_word_vec)
                else:
                    dimensional_projection = 0.0
                dimensional_boost = 1.0 + abs(dimensional_projection) * 0.5
            except (ValueError, OverflowError, ZeroDivisionError):
                dimensional_boost = 1.0
            
            fold_strength = self.calculate_fold_strength(current_context_tuple, time_index) # Uses generation's cascade_memory
            fold_correction_arg = -fold_strength * 0.1
            fold_correction = math.exp(max(fold_correction_arg, -700))
            
            symmetry_factor = 1.0
            if self.retrocausal_loops: # Populated by detect_retrocausal_integration during generation
                recent_loop = self.retrocausal_loops[-1]
                symmetry_factor = 1.0 + recent_loop.get('coherence', 0.0) * 0.3
            
            enhanced_weight = base_weight * dimensional_boost * fold_correction * symmetry_factor
            enhanced_weights.append(max(0.0001, enhanced_weight))
        return candidate_words, enhanced_weights

    def calculate_cascade_coherence(self, context_sequence):
        # This function now relies on context_sequence passed in (e.g., from generate or detect_retrocausal_integration)
        # or self.cascade_memory if called from calculate_fold_strength (where self.cascade_memory is generation's history)
        if len(context_sequence) < 2:
            return 0.0
        coherence = 1.0
        max_iterations = min(3, len(context_sequence) - 1)
        for i in range(max_iterations):
            try:
                c1 = context_sequence[i]
                c2 = context_sequence[i + 1]
                context1_str = ' '.join(map(str, c1)) if isinstance(c1, (list, tuple)) else str(c1)
                context2_str = ' '.join(map(str, c2)) if isinstance(c2, (list, tuple)) else str(c2)
                overlap = self._safe_l_semi_inner_product(context1_str, context2_str) # Relies on this
                decay = math.exp(-0.1 * i)
                coherence *= decay * (overlap ** 2)
                coherence = max(0.0, min(1.0, coherence))
                if coherence < 1e-9:
                    coherence = 0.0
                    break
            except Exception as e:
                coherence *= 0.5 
                coherence = max(0.0, min(1.0, coherence))
                break
        return coherence

    def retrocausal_information_kernel(self, t0, future_states, omega=1.0):
        # Generation-time utility
        information = 0.0j 
        max_states_to_sum = max(10, min(len(future_states), 20)) # Ensure not to exceed list length, cap at 20
        for tau_idx in range(max_states_to_sum):
            if tau_idx >= len(future_states): break # Safety break
            current_future_state = future_states[tau_idx]
            try:
                state_val_str = str(current_future_state)
                try:
                    state_val = float(state_val_str[:10].replace(',','')) 
                except ValueError:
                    state_val = float(hash(state_val_str) % 1000 - 500) / 500.0
                phase_arg = -1j * omega * tau_idx 
                information += state_val * np.exp(phase_arg)
            except (ValueError, OverflowError, TypeError) as e:
                continue 
        abs_info = abs(information)
        return min(abs_info, 1e3)

    @staticmethod
    def _calculate_distinct_permutations_log(word):
        # General utility
        if not word or len(word) == 0:
            return 0.0
        word = str(word)[:30] if len(word) > 30 else str(word)
        n = len(word)
        if n == 0: return 0.0
        counts = Counter(word)
        try:
            log_n_factorial = math.lgamma(n + 1)
            log_denominator_sum = sum(math.lgamma(count + 1) for count in counts.values())
            log_permutations = log_n_factorial - log_denominator_sum
            return max(0.0, min(20.0, log_permutations))
        except (ValueError, OverflowError):
            return 0.0

    def _safe_l_semi_inner_product(self, word1, word2):
        # General utility, kept for generation-time complexities
        if word1 is None or word2 is None: return 0.0
        try:
            s_word1 = str(word1)
            s_word2 = str(word2)
        except Exception:
            return 0.0
        if not s_word1 or not s_word2:
            return 0.0
        cache_key = (s_word1[:20], s_word2[:20])
        if cache_key in self.similarity_cache:
            return self.similarity_cache[cache_key]
        s_word1_limited = s_word1[:50]
        s_word2_limited = s_word2[:50]
        result = self._compute_l_semi_inner_product(s_word1_limited, s_word2_limited) # Calls the next method
        if len(self.similarity_cache) < 2000:
            self.similarity_cache[cache_key] = result
        return result

    def _compute_l_semi_inner_product(self, word1, word2):
        # General utility, underlying computation for the above.
        # Note: Uses self.cascade_memory for a boost. This memory is now mostly for generation.
        if not word1 or not word2: return 0.0
        char_product = 0.0
        w1_len = len(word1)
        w2_len = len(word2)
        if w1_len == 0 or w2_len == 0: return 0.0
        for i, c1 in enumerate(word1):
            pos_weight_1 = 1.0 - (abs(i - (w1_len -1) / 2.0) / (w1_len / 2.0 + 1e-9))
            pos_weight_1 = max(0, pos_weight_1)
            for j, c2 in enumerate(word2):
                if c1 == c2:
                    pos_weight_2 = 1.0 - (abs(j - (w2_len-1) / 2.0) / (w2_len / 2.0 + 1e-9))
                    pos_weight_2 = max(0, pos_weight_2)
                    char_product += pos_weight_1 * pos_weight_2
        char_product_normalized = char_product / ( (w1_len * w2_len)**0.5 + 1e-9)
        char_product_normalized = min(char_product_normalized, 1.0)
        length_ratio = min(w1_len, w2_len) / (max(w1_len, w2_len) + 1e-9)
        vowels = set('aeiouAEIOU')
        w1_alpha = "".join(filter(str.isalpha, word1))
        w2_alpha = "".join(filter(str.isalpha, word2))
        w1_pattern = ''.join('V' if c in vowels else 'C' for c in w1_alpha)
        w2_pattern = ''.join('V' if c in vowels else 'C' for c in w2_alpha)
        phonetic_factor = 0.0
        if w1_pattern and w2_pattern:
            pattern_len_max = min(3, min(len(w1_pattern), len(w2_pattern)))
            if pattern_len_max > 0:
                match_start = sum(1 for k in range(pattern_len_max) if w1_pattern[k] == w2_pattern[k])
                match_end = sum(1 for k in range(pattern_len_max) if w1_pattern[-(k+1)] == w2_pattern[-(k+1)])
                phonetic_factor = (match_start + match_end) / (2.0 * pattern_len_max)
        prefix_len = 0
        for i in range(min(w1_len, w2_len, 5)):
            if word1[i] == word2[i]: prefix_len += 1
            else: break
        suffix_len = 0
        for i in range(1, min(w1_len, w2_len, 5) + 1):
            if word1[-i] == word2[-i]: suffix_len += 1
            else: break
        prefix_suffix_factor = (prefix_len + suffix_len) / (float(min(min(w1_len,w2_len),5) * 2) + 1e-9)
        prefix_suffix_factor = min(prefix_suffix_factor, 1.0)
        perm_key1 = word1[:20] 
        perm_key2 = word2[:20]
        log_perm1_val, log_perm2_val = 0.0, 0.0
        if perm_key1 in self.permutation_cache: log_perm1_val = self.permutation_cache[perm_key1]
        elif len(self.permutation_cache) < 2000: self.permutation_cache[perm_key1] = log_perm1_val = self._calculate_distinct_permutations_log(perm_key1)
        else: log_perm1_val = self._calculate_distinct_permutations_log(perm_key1) # Compute if cache full
        if perm_key2 in self.permutation_cache: log_perm2_val = self.permutation_cache[perm_key2]
        elif len(self.permutation_cache) < 2000: self.permutation_cache[perm_key2] = log_perm2_val = self._calculate_distinct_permutations_log(perm_key2)
        else: log_perm2_val = self._calculate_distinct_permutations_log(perm_key2) # Compute if cache full
        diff_log_perm = abs(log_perm1_val - log_perm2_val)
        denominator_perm = max(log_perm1_val, log_perm2_val, 1.0)
        normalized_diff_perm = diff_log_perm / denominator_perm if denominator_perm > 1e-9 else 0.0
        permutation_similarity_factor = 1.0 - min(normalized_diff_perm, 1.0)
        
        cascade_boost = 1.0
        if self.cascade_memory: # self.cascade_memory is now populated by generate()
            recent_coherence_proxy = min(0.5, len(self.cascade_memory) * 0.05) 
            cascade_boost = 1.0 + recent_coherence_proxy * 0.2 # Small boost

        l_inner_product = (
            0.30 * char_product_normalized +
            0.15 * length_ratio +
            0.20 * phonetic_factor +
            0.20 * prefix_suffix_factor +
            0.15 * permutation_similarity_factor
        )
        l_inner_product *= cascade_boost
        return max(0.0, min(2.0, l_inner_product))

    def train(self, t):
        # This is the original CPU-intensive training method.
        # It's kept for comparison or if OpenCL is not desired/available.
        # It uses the K and S matrix logic if they were present.
        # Since K and S are removed from __init__, this train method would need them re-added
        # or it would also operate in a simplified manner.
        # For this exercise, we assume if train() is called, it implies a simplified model
        # without K and S as they are no longer attributes.
        # Let's make it also use the simplified counting from train_opencl's CPU part for consistency.

        if not isinstance(t, str) or not t:
            raise TypeError("Training data must be a non-empty string.")
        words = t.split()
        num_words = len(words)
        if num_words <= self.n:
            print(f"Warning: Training data (CPU) has only {num_words} words, <= n-gram size {self.n}. Skipping.")
            return

        print(f"Training model (CPU simplified) on {num_words} words with n={self.n}...")
        overall_word_freqs = Counter(words)
        total_word_count = float(num_words)
        temp_m = defaultdict(lambda: defaultdict(float))
        temp_s = set()
        self.all_words.update(words)
        
        # current_run_epoch_offset = self.epochs # Not used if K/S are gone
        # self.epochs += (num_words - self.n) # Still track epochs

        for i in tqdm(range(num_words - self.n), desc="Simplified CPU Training"):
            current_ngram = tuple(words[i:i+self.n])
            next_word_actual = words[i+self.n]
            
            # Simplified increment logic (mirrors the simplified train_opencl CPU part)
            base_increment = 1.0
            norm_freq_next_word = overall_word_freqs.get(next_word_actual, 0) / total_word_count
            global_freq_factor = 1.0 + math.log1p(norm_freq_next_word * 100) * 0.1 
            len_next_word = len(next_word_actual)
            length_factor = 1.0 + math.log1p(len_next_word) * 0.02
            symbolic_enhancement_factor = global_freq_factor * length_factor
            symbolic_enhancement_factor = max(0.5, min(symbolic_enhancement_factor, 2.0))
            adjusted_increment_val = base_increment * symbolic_enhancement_factor
            noise_val = random.uniform(-0.01, 0.01) * adjusted_increment_val
            final_increment_val = max(0.01, adjusted_increment_val + noise_val)
            
            if i == 0 or (i > 0 and words[i-1] and any(c in self.SENTENCE_END_CHARS for c in words[i-1][-1:])):
                temp_s.add(current_ngram)
            temp_m[current_ngram][next_word_actual] += final_increment_val
        
        self.epochs += (num_words - self.n) # Correctly update epochs

        self.m = {gram: dict(counts) for gram, counts in temp_m.items() if counts}
        self.s = list(filter(lambda sg: sg in self.m, temp_s))
        if not self.s and self.m:
            num_starts_to_sample = min(max(1, int(len(self.m) * 0.05)), 50)
            if num_starts_to_sample > len(self.m):
                num_starts_to_sample = len(self.m)
            if num_starts_to_sample > 0:
                self.s = random.sample(list(self.m.keys()), k=num_starts_to_sample)
        
        self.cascade_memory = [] # Reset for generation
        print(f"Simplified CPU training complete. Model has {len(self.m)} contexts, {len(self.s)} valid starts.")


    def generate(self, length=50, start_context_str=None):
        if not self.m:
            return "Model not trained."
        
        # `self.cascade_memory` will be populated here during generation for generation-time effects
        self.cascade_memory = [] # Reset for this generation run
        self.retrocausal_loops = [] # Reset for this generation run

        current_gram_tuple = None
        generated_words_list = []
        if start_context_str:
            sc_words = start_context_str.split()
            if len(sc_words) >= self.n:
                current_gram_tuple = tuple(sc_words[-self.n:])
                if current_gram_tuple not in self.m:
                    current_gram_tuple = None 
                else:
                    generated_words_list = list(sc_words)
                    self.cascade_memory.extend(list(current_gram_tuple)) # Initialize cascade_memory
        
        if current_gram_tuple is None:
            if not self.s:
                if self.m:
                    sample_k = min(1, max(1,int(len(self.m)*0.1))) # Ensure at least 1 if m exists
                    if sample_k > len(self.m): sample_k = len(self.m)
                    if sample_k > 0:
                        try: 
                            self.s = random.sample(list(self.m.keys()), k=sample_k)
                        except ValueError: 
                            return "Cannot start generation: error selecting random start."
                if not self.s: return "Cannot start generation: no valid start points in the model."
            try: 
                current_gram_tuple = random.choice(self.s)
            except IndexError: 
                return "Cannot start generation: no sentence starts available."
            generated_words_list = list(current_gram_tuple)
            self.cascade_memory.extend(list(current_gram_tuple)) # Initialize cascade_memory

        
        local_generation_history_ngrams = [current_gram_tuple] # For detect_retrocausal_integration
        MAX_GEN_HIST_LEN = 15
        
        num_words_to_generate = length - len(generated_words_list)

        for _ in range(num_words_to_generate):
            if len(generated_words_list) >= length: break
            if current_gram_tuple not in self.m or not self.m[current_gram_tuple]:
                if not self.s: break
                try: current_gram_tuple = random.choice(self.s)
                except IndexError: break
                if current_gram_tuple not in self.m or not self.m[current_gram_tuple]: break
                # Potentially add new current_gram_tuple to generated_words_list and cascade_memory if restarting
                # For now, just reset context
                local_generation_history_ngrams = [current_gram_tuple] # Reset history on jump
                self.cascade_memory.append(current_gram_tuple) # Add new context to cascade
                if len(self.cascade_memory) > 20 : self.cascade_memory.pop(0) # Keep cascade_memory bounded


            options_for_current_gram = self.m[current_gram_tuple]
            candidate_words, candidate_raw_scores = self._symbolic_probability(current_gram_tuple, options_for_current_gram)
            if not candidate_words:
                if not self.s: break
                try: current_gram_tuple = random.choice(self.s)
                except IndexError: break
                local_generation_history_ngrams = [current_gram_tuple]
                self.cascade_memory.append(current_gram_tuple)
                if len(self.cascade_memory) > 20 : self.cascade_memory.pop(0)
                continue
            
            time_idx_generation = len(generated_words_list) # "Time" in generation sequence
            # detect_retrocausal_integration uses local_generation_history_ngrams
            if self.detect_retrocausal_integration(current_gram_tuple, local_generation_history_ngrams):
                _, final_scores_for_choice = self.recompute_with_extra_dimension(
                    current_gram_tuple, candidate_words, candidate_raw_scores, time_idx_generation
                )
            else:
                final_scores_for_choice = candidate_raw_scores
            
            sum_final_scores = sum(final_scores_for_choice)
            if sum_final_scores > 1e-9:
                probabilities_for_choice = [s / sum_final_scores for s in final_scores_for_choice]
            else: 
                if candidate_words:
                    prob = 1.0 / len(candidate_words)
                    probabilities_for_choice = [prob] * len(candidate_words)
                else: 
                    break 
            
            if not candidate_words : break 
            try:
                next_chosen_word = random.choices(candidate_words, weights=probabilities_for_choice, k=1)[0]
            except ValueError as e: 
                if candidate_words: next_chosen_word = random.choice(candidate_words) 
                else: break 
            
            generated_words_list.append(next_chosen_word)
            current_gram_tuple = tuple(generated_words_list[-self.n:])

            # Update local generation history for detect_retrocausal_integration
            local_generation_history_ngrams.append(current_gram_tuple)
            if len(local_generation_history_ngrams) > MAX_GEN_HIST_LEN:
                local_generation_history_ngrams.pop(0)
            
            # Update self.cascade_memory for use by fold_strength, coherence, L-semi-inner-product boost
            self.cascade_memory.append(current_gram_tuple) # Appending the new context
            if len(self.cascade_memory) > 20: # Keep it bounded (e.g., last 20 contexts)
                 self.cascade_memory.pop(0)

        return " ".join(generated_words_list)


    def save_model(self, filepath):
        cl_ctx_backup, cl_queue_backup, cl_prg_backup = self.cl_ctx, self.cl_queue, self.cl_prg
        self.cl_ctx, self.cl_queue, self.cl_prg = None, None, None
        try:
            with open(filepath, 'wb') as f:
                pickle.dump(self, f)
            print(f"Model saved successfully to {filepath}")
        except Exception as e:
            print(f"Error saving model to {filepath}: {e}")
        finally:
            self.cl_ctx, self.cl_queue, self.cl_prg = cl_ctx_backup, cl_queue_backup, cl_prg_backup


    @staticmethod
    def load_model(filepath):
        try:
            with open(filepath, 'rb') as f:
                model = pickle.load(f)
            print(f"Model loaded successfully from {filepath}")
            if PYOPENCL_AVAILABLE and (not hasattr(model, 'cl_ctx') or model.cl_ctx is None):
                try:
                    # Try to get a GPU context first, then CPU
                    model.cl_ctx = None
                    platforms = cl.get_platforms()
                    for platform in platforms:
                        try:
                            gpu_devices = platform.get_devices(device_type=cl.device_type.GPU)
                            if gpu_devices:
                                model.cl_ctx = cl.Context(devices=[gpu_devices[0]])
                                break
                        except cl.RuntimeError: continue
                    if not model.cl_ctx: # If no GPU or GPU context failed, try CPU
                        for platform in platforms:
                            try:
                                cpu_devices = platform.get_devices(device_type=cl.device_type.CPU)
                                if cpu_devices:
                                    model.cl_ctx = cl.Context(devices=[cpu_devices[0]])
                                    break
                            except cl.RuntimeError: continue
                    if not model.cl_ctx:
                        model.cl_ctx = cl.create_some_context(interactive=False)

                    model.cl_queue = cl.CommandQueue(model.cl_ctx)
                    model.cl_prg = None 
                    print(f"OpenCL context re-initialized on load for: {model.cl_ctx.devices[0].name}")
                except Exception as e:
                    print(f"Could not re-initialize PyOpenCL context on load: {e}. OpenCL features may be disabled.")
                    model.cl_ctx = None
            return model
        except FileNotFoundError:
            print(f"Error: Model file not found at {filepath}")
            return None
        except Exception as e:
            print(f"Error loading model from {filepath}: {e}")
            return None

if __name__ == "__main__":
    print("Initializing Simplified Symbolic Markov Model...")
    model = None 

    load_choice = input("Do you want to (L)oad a saved model or (T)rain a new one? [L/T]: ").strip().upper()

    if load_choice == 'L':
        model_filepath = input("Enter the filepath of the saved model: ").strip()
        model = RetrocausalSymbolicMarkov.load_model(model_filepath)
        if model is None:
            print("Failed to load model. Will proceed to train a new one if training data is provided.")
    
    if model is None: 
        print("Proceeding to train a new model.")
        model = RetrocausalSymbolicMarkov(n=2, cascade_threshold=0.75, extra_dim_depth=2) # Alpha removed
        training_text = None
        default_training_text = (
            "The quick brown fox jumps over the lazy dog. "
            "A lazy dog sleeps all day. The quick cat also jumps. "
            "Foxes are quick and dogs can be lazy. Cats are nimble. "
            "What does the fox say? The dog barks. The cat meows. "
            "Symbolic AI and machine learning can create interesting text. "
            "This model attempts to blend Markov chains with advanced ideas. "
            "Knowledge is updated based on future states. This is a test. " # This sentence is now less true for training
            "Let's see how well it generates coherent sentences. "
            "The brown fox is very quick. The dog remains lazy. "
            "Jumping is fun for the fox and the cat. "
            "AI is evolving. Markov models predict the next state. "
            "This sentence provides more data for training. This is the end of the training data."
        )

        data_source_choice = input("Load training data from (1) File, (2) Hugging Face, or (3) Default text? [1/2/3]: ").strip()

        if data_source_choice == '1':
            filename = input("Enter filename for training data: ")
            try:
                with open(filename, 'r', encoding='utf-8') as file:
                    raw_text = file.read().lower()
                    training_text = ' '.join(raw_text.split()) 
                print(f"Loaded training data from file: {filename}")
            except FileNotFoundError:
                print(f"Error: File '{filename}' not found. Using default short training data.")
                training_text = default_training_text
            except Exception as e:
                print(f"An error occurred while reading the file: {e}. Using default short training data.")
                training_text = default_training_text
        elif data_source_choice == '2':
            if not HF_DATASETS_AVAILABLE:
                print("Hugging Face 'datasets' library is not available. Falling back to default text.")
                training_text = default_training_text
            else:
                try:
                    hf_dataset_name = input("Enter Hugging Face dataset name (e.g., 'wikitext'): ").strip()
                    hf_config_name = input("Enter dataset configuration (e.g., 'wikitext-2-raw-v1', blank for default): ").strip() or None
                    
                    available_splits = get_dataset_split_names(hf_dataset_name, config_name=hf_config_name)
                    print(f"Available splits for {hf_dataset_name} (config: {hf_config_name}): {available_splits}")
                    hf_split = input(f"Enter dataset split (e.g., 'train'): ").strip()
                    if hf_split not in available_splits:
                        print(f"Split '{hf_split}' not valid. Using 'train' or first available if 'train' not present.")
                        hf_split = 'train' if 'train' in available_splits else available_splits[0]
                        
                    hf_text_column = input("Enter the name of the column containing text (e.g., 'text'): ").strip()
                    hf_num_rows_str = input("Enter max number of rows to use (e.g., 1000, blank for all): ").strip()
                    hf_num_rows = int(hf_num_rows_str) if hf_num_rows_str else None

                    print(f"Loading HF Dataset: {hf_dataset_name}, Config: {hf_config_name}, Split: {hf_split}, Column: {hf_text_column}")
                    dataset_args = {"path": hf_dataset_name, "split": hf_split}
                    if hf_config_name: dataset_args["name"] = hf_config_name
                    
                    ds = load_dataset(**dataset_args)
                    if hf_text_column not in ds.features:
                         raise ValueError(f"Text column '{hf_text_column}' not found. Available: {list(ds.features)}")

                    all_hf_texts = []
                    iterator = ds.select(range(hf_num_rows)) if hf_num_rows else ds
                    for ex in tqdm(iterator, desc="Processing HF dataset"):
                        text_content = ex.get(hf_text_column)
                        if text_content and isinstance(text_content, str):
                            all_hf_texts.append(text_content.lower())
                    
                    training_text = " ".join(all_hf_texts)
                    training_text = ' '.join(training_text.split()) 
                    print(f"Loaded {len(training_text.split())} words from Hugging Face.")
                    if not training_text: raise ValueError("No text from HF")
                except Exception as e:
                    print(f"HF loading error: {e}. Using default.")
                    training_text = default_training_text
        else:
            print("Using default short training data.")
            training_text = default_training_text

        if training_text:
            print("\nTraining the model...")
            use_opencl_train = input("Use OpenCL for training if available? (Y/N): ").strip().upper()
            if use_opencl_train == 'Y' and PYOPENCL_AVAILABLE and model.cl_ctx: # Check model.cl_ctx too
                model.train_opencl(training_text)
            else:
                if use_opencl_train == 'Y' and (not PYOPENCL_AVAILABLE or not model.cl_ctx):
                    print("OpenCL not available or context failed, falling back to standard CPU training.")
                model.train(training_text) # Calls the simplified CPU train
        else:
            print("No training data loaded. Model cannot be trained and was not loaded.")
    
    if model: 
        save_choice = input("\nDo you want to save the current model? [Y/N]: ").strip().upper()
        if save_choice == 'Y':
            save_filepath = input("Enter filepath to save the model (e.g., retro_model_simplified.pkl): ").strip()
            if save_filepath:
                model.save_model(save_filepath)
            else:
                print("No filepath provided. Model not saved.")
                
    if model and model.m: 
        print("\nInteractive Generation Mode (type 'quit' or 'exit' to stop):")
        while True:
            seed = input("USER: ")
            if seed.lower() in ['quit', 'exit']:
                break
            if not seed.strip():
                print("AI: (No input provided, try seeding with some text or type 'quit')")
                continue
            gen_length = 250 
            generated_text = model.generate(length=gen_length, start_context_str=seed)
            print(f"AI: {generated_text}")
    elif model:
        print("Model is loaded/initialized but appears to have no training data/transitions. Cannot start generation.")
    else:
        print("Model could not be initialized or loaded. Exiting.")

    print("\nModel demonstration finished.")
