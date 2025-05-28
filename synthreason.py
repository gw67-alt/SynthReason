import random
import math
import sys
from collections import defaultdict, Counter
from tqdm import tqdm
import numpy as np
import pickle

# Import the Hugging Face datasets library (optional)
try:
    from datasets import load_dataset, get_dataset_split_names
    HF_DATASETS_AVAILABLE = True
except ImportError:
    HF_DATASETS_AVAILABLE = False
    print("Hugging Face 'datasets' library not found. Optional: pip install datasets")


class RetrocausalSymbolicMarkov:
    """
    Markov chain text generator.
    This version uses CPU-based training with symbolic factor adjustments.
    """

    # Static configuration variables
    SENTENCE_END_CHARS = set('.!?')

    def __init__(self, n=2, alpha=0.3, cascade_threshold=0.7, extra_dim_depth=3): # alpha is effectively unused
        if not isinstance(n, int) or n < 1:
            raise ValueError("n-gram size 'n' must be a positive integer.")

        self.n = n
        self.cascade_threshold = cascade_threshold
        self.extra_dim_depth = extra_dim_depth

        self.m = {} # Transitions: {context_tuple: {next_word: adjusted_frequency}}
        self.s = [] # Sentence starting n-grams
        self.all_words = set() # Stores all unique words encountered

        self.cascade_memory = [] # For generation-time effects
        self.epochs = 0 # Number of n-gram instances processed

        # Extra-dimensional integration structures (for generation)
        self.D = {}
        self.integration_history = []
        self.fold_tensor = {}
        self.retrocausal_loops = []

        # Cache for expensive calculations
        self.similarity_cache = {}
        self.permutation_cache = {}
        self.dimensional_cache = {}

    def _symbolic_probability(self, context_tuple, options_dict):
        if not isinstance(context_tuple, tuple):
            try:
                context_tuple = tuple(str(c) for c in context_tuple) if hasattr(context_tuple, '__iter__') and not isinstance(context_tuple, str) else tuple([str(context_tuple)])
            except TypeError:
                context_tuple = tuple()
        
        candidate_words_list = list(options_dict.keys())
        base_scores_list = list(options_dict.values())

        if not candidate_words_list: return [], []

        enhanced_scores = []
        for i, word_cand in enumerate(candidate_words_list):
            current_base_score = base_scores_list[i]
            final_score_for_word = current_base_score 
            enhanced_scores.append(max(0.0001, final_score_for_word))
        
        final_refined_scores = []
        if not context_tuple: 
            final_refined_scores = enhanced_scores
        else:
            for i, word_cand in enumerate(candidate_words_list):
                score_after_k_cascade = enhanced_scores[i]
                context_sim_factor_accum = 1.0
                num_context_words_for_sim = max(1, self.n // 2) 
                for context_word_from_tuple in context_tuple[:num_context_words_for_sim]:
                    l_sim = self._safe_l_semi_inner_product(str(context_word_from_tuple), word_cand)
                    context_sim_factor_accum *= (1.0 + l_sim * 0.15) 
                context_sim_factor_accum = min(context_sim_factor_accum, 3.0)

                score_after_context_sim = score_after_k_cascade * context_sim_factor_accum
                final_refined_scores.append(max(0.00001, score_after_context_sim)) 

        if not final_refined_scores: return [], []
        scored_candidates = sorted(zip(final_refined_scores, candidate_words_list), reverse=True)
        num_final_candidates = min(len(scored_candidates), 75) 
        top_candidates_words = [word for score, word in scored_candidates[:num_final_candidates]]
        top_candidates_scores = [score for score, word in scored_candidates[:num_final_candidates]]

        if not top_candidates_words: return [],[] 
        return top_candidates_words, top_candidates_scores

    def compute_extra_dimension(self, context, time_index, future_states):
        cache_key = (str(context)[:20], time_index, len(future_states))
        if cache_key in self.dimensional_cache:
            return self.dimensional_cache[cache_key]

        extra_dim = np.zeros(self.extra_dim_depth * 2)
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
                    except: future_influence = 0.5 # Default on error
                r = hyperbolic_factor * (1.0 + future_influence)
                r = min(max(r, -700), 700) 
                theta = context_angle + i * math.pi / self.extra_dim_depth
                extra_dim[i*2] = math.cosh(r) * math.cos(theta)
                extra_dim[i*2 + 1] = math.sinh(r) * math.sin(theta)
            for i in range(self.extra_dim_depth): # Second loop for temporal folding
                temporal_phase = (time_index / 10.0) % (2 * math.pi)
                fold_strength = self.calculate_fold_strength(context, time_index) 
                fold_factor_arg = -fold_strength * temporal_phase / (2 * math.pi)
                fold_factor = math.exp(max(fold_factor_arg, -700)) # Prevent underflow
                extra_dim[i*2] *= fold_factor
                extra_dim[i*2 + 1] *= math.cos(temporal_phase + i * math.pi / 3) # Phase shift for S1 part
            
            norm = np.linalg.norm(extra_dim)
            if norm > 1e-9 and norm < 1e6 : extra_dim = extra_dim / norm
            elif norm >= 1e6: extra_dim = (extra_dim / norm) * 1e5 
            elif norm <= 1e-9 and norm !=0 : extra_dim = np.random.normal(0, 0.01, self.extra_dim_depth * 2)
            else: extra_dim = np.random.normal(0, 0.1, self.extra_dim_depth * 2) # If norm is 0
        except (OverflowError, ValueError, ZeroDivisionError): # Catch math errors
            extra_dim = np.random.normal(0, 0.1, self.extra_dim_depth * 2)

        if len(self.dimensional_cache) < 1000: # Cache limit
            self.dimensional_cache[cache_key] = extra_dim
        return extra_dim

    def calculate_fold_strength(self, context, time_index):
        fold_strength = 0.0
        try: 
            for i, past_context in enumerate(self.cascade_memory):
                if past_context == context:
                    loop_length = len(self.cascade_memory) - i
                    fold_strength += 1.0 / (1.0 + loop_length)
        except TypeError: pass # If context or past_context is not comparable
        if len(self.cascade_memory) > 2:
            recent_coherence = self.calculate_cascade_coherence(self.cascade_memory[-3:])
            fold_strength += recent_coherence * 0.5
        return min(2.0, fold_strength) # Cap fold_strength

    def detect_retrocausal_integration(self, current_context, generation_history):
        if len(generation_history) < 3: return False
        context_str = str(current_context)
        for i, past_state in enumerate(generation_history[-10:]): # Check recent history
            past_str = str(past_state)
            similarity = self._safe_l_semi_inner_product(context_str, past_str)
            if similarity > 0.8: # High similarity threshold
                loop_coherence = self.calculate_cascade_coherence([past_state, current_context])
                if loop_coherence > self.cascade_threshold:
                    if len(self.retrocausal_loops) < 100: # Limit number of stored loops
                        self.retrocausal_loops.append({
                            'start_index': len(generation_history) - min(10, len(generation_history)) + i, # Adjust start_index
                            'end_index': len(generation_history),
                            'coherence': loop_coherence,
                            'context_similarity': similarity
                        })
                    return True
        return False

    def recompute_with_extra_dimension(self, context, candidate_words, candidate_weights, time_index):
        future_states = []
        weights_to_sort = [candidate_weights[i] if i < len(candidate_weights) else 0.0 for i in range(len(candidate_words))]
        sorted_candidates = sorted(zip(weights_to_sort, candidate_words), reverse=True)

        for weight, word in sorted_candidates[:5]: # Top 5 candidates for future states
            future_states.append(word)

        extra_dim_vector = self.compute_extra_dimension(context, time_index, future_states)
        
        current_context_tuple = tuple(context) if isinstance(context, list) else context
        if not isinstance(current_context_tuple, tuple):
            current_context_tuple = tuple([str(current_context_tuple)]) # Ensure tuple for key

        if len(self.D) > 2000: # Cache eviction for self.D
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
                else: dimensional_projection = 0.0
                dimensional_boost = 1.0 + abs(dimensional_projection) * 0.5
            except: dimensional_boost = 1.0 # Default on error
            
            fold_strength = self.calculate_fold_strength(current_context_tuple, time_index)
            fold_correction_arg = -fold_strength * 0.1
            fold_correction = math.exp(max(fold_correction_arg, -700)) # Prevent underflow
            
            symmetry_factor = 1.0
            if self.retrocausal_loops:
                recent_loop = self.retrocausal_loops[-1]
                symmetry_factor = 1.0 + recent_loop.get('coherence', 0.0) * 0.3
            
            enhanced_weight = base_weight * dimensional_boost * fold_correction * symmetry_factor
            enhanced_weights.append(max(0.0001, enhanced_weight)) # Ensure positive weight
        return candidate_words, enhanced_weights

    def calculate_cascade_coherence(self, context_sequence):
        if len(context_sequence) < 2: return 0.0
        coherence = 1.0
        max_iterations = min(3, len(context_sequence) - 1) # Check at most 2 transitions
        for i in range(max_iterations):
            try:
                c1, c2 = context_sequence[i], context_sequence[i + 1]
                context1_str = ' '.join(map(str, c1)) if isinstance(c1, (list, tuple)) else str(c1)
                context2_str = ' '.join(map(str, c2)) if isinstance(c2, (list, tuple)) else str(c2)
                overlap = self._safe_l_semi_inner_product(context1_str, context2_str)
                decay = math.exp(-0.1 * i) # Decay for older transitions
                coherence *= decay * (overlap ** 2)
                coherence = max(0.0, min(1.0, coherence)) # Bound coherence
                if coherence < 1e-9: coherence = 0.0; break # Stop if coherence drops too low
            except: coherence *= 0.5; coherence = max(0.0, min(1.0, coherence)); break
        return coherence

    def retrocausal_information_kernel(self, t0, future_states, omega=1.0):
        information = 0.0j 
        max_states_to_sum = min(len(future_states), 20) # Adjusted to not exceed list, cap at 20
        for tau_idx in range(max_states_to_sum):
            # No need for `if tau_idx >= len(future_states): break` due to loop range
            current_future_state = future_states[tau_idx]
            try:
                state_val_str = str(current_future_state)
                try: state_val = float(state_val_str[:10].replace(',','')) 
                except ValueError: state_val = float(hash(state_val_str) % 1000 - 500) / 500.0 # Fallback
                phase_arg = -1j * omega * tau_idx 
                information += state_val * np.exp(phase_arg)
            except: continue 
        return min(abs(information), 1e3) # Cap max info

    @staticmethod
    def _calculate_distinct_permutations_log(word):
        if not word : return 0.0 # Simplified check
        word_str = str(word)[:30] # Limit length for performance
        n = len(word_str)
        if n == 0: return 0.0
        counts = Counter(word_str)
        try:
            log_n_factorial = math.lgamma(n + 1)
            log_denominator_sum = sum(math.lgamma(count + 1) for count in counts.values())
            return max(0.0, min(20.0, log_n_factorial - log_denominator_sum)) # Cap result
        except: return 0.0 # Catch any math errors

    def _safe_l_semi_inner_product(self, word1, word2):
        if word1 is None or word2 is None: return 0.0
        try: s_word1, s_word2 = str(word1), str(word2)
        except: return 0.0
        if not s_word1 or not s_word2: return 0.0
        
        cache_key = (s_word1[:20], s_word2[:20]) # Use slices for cache key consistency
        if cache_key in self.similarity_cache: return self.similarity_cache[cache_key]
        
        result = self._compute_l_semi_inner_product(s_word1[:50], s_word2[:50]) # Limit string length for computation
        if len(self.similarity_cache) < 2000: self.similarity_cache[cache_key] = result
        return result

    def _compute_l_semi_inner_product(self, word1, word2):
        if not word1 or not word2: return 0.0
        char_product, w1_len, w2_len = 0.0, len(word1), len(word2)
        if w1_len == 0 or w2_len == 0: return 0.0

        for i, c1 in enumerate(word1):
            pw1 = max(0, 1.0 - (abs(i - (w1_len -1) / 2.0) / (w1_len / 2.0 + 1e-9)))
            for j, c2 in enumerate(word2):
                if c1 == c2:
                    pw2 = max(0, 1.0 - (abs(j - (w2_len-1) / 2.0) / (w2_len / 2.0 + 1e-9)))
                    char_product += pw1 * pw2
        
        char_prod_norm = min(char_product / ((w1_len * w2_len)**0.5 + 1e-9), 1.0) if (w1_len * w2_len > 0) else 0.0
        len_ratio = min(w1_len, w2_len) / (max(w1_len, w2_len) + 1e-9) if max(w1_len, w2_len) > 0 else 0.0
        
        vowels = set('aeiouAEIOU')
        w1a, w2a = "".join(filter(str.isalpha, word1)), "".join(filter(str.isalpha, word2))
        p1, p2 = (''.join('V' if c in vowels else 'C' for c in w1a), 
                  ''.join('V' if c in vowels else 'C' for c in w2a))
        phon_factor = 0.0
        if p1 and p2:
            p_len = min(3, len(p1), len(p2))
            if p_len > 0:
                m_start = sum(1 for k in range(p_len) if p1[k] == p2[k])
                m_end = sum(1 for k in range(p_len) if p1[-(k+1)] == p2[-(k+1)])
                phon_factor = (m_start + m_end) / (2.0 * p_len)
        
        pre_len, suf_len = 0,0
        min_comp_len = min(w1_len, w2_len, 5)
        for i in range(min_comp_len):
            if word1[i] == word2[i]: pre_len += 1
            else: break
        for i in range(1, min_comp_len + 1):
            if word1[-i] == word2[-i]: suf_len += 1
            else: break
        pre_suf_factor = min((pre_len + suf_len) / (float(min_comp_len * 2) + 1e-9), 1.0) if min_comp_len > 0 else 0.0
        
        pk1, pk2 = word1[:20], word2[:20]
        lp1 = self.permutation_cache.get(pk1)
        if lp1 is None: 
            lp1 = self._calculate_distinct_permutations_log(pk1)
            if len(self.permutation_cache) < 2000: self.permutation_cache[pk1] = lp1
        lp2 = self.permutation_cache.get(pk2)
        if lp2 is None:
            lp2 = self._calculate_distinct_permutations_log(pk2)
            if len(self.permutation_cache) < 2000: self.permutation_cache[pk2] = lp2
        
        denom_perm = max(lp1, lp2, 1.0) # Ensure denom_perm is at least 1.0
        perm_sim_factor = 1.0 - min(abs(lp1 - lp2) / (denom_perm + 1e-9), 1.0) # Add epsilon to denom
        
        cascade_boost = 1.0 + min(0.5, len(self.cascade_memory) * 0.05) * 0.2 if self.cascade_memory else 1.0
        l_inner = (0.30*char_prod_norm + 0.15*len_ratio + 0.20*phon_factor + 0.20*pre_suf_factor + 0.15*perm_sim_factor)
        return max(0.0, min(2.0, l_inner * cascade_boost)) # Cap final result

    def train(self, t):
        if not isinstance(t, str) or not t:
            raise TypeError("Training data must be a non-empty string.")
        
        words = t.split()
        num_words = len(words)

        if num_words <= self.n:
            print(f"Warning: Training data (CPU) has only {num_words} words, not enough for context length {self.n} and a next word. Skipping.")
            return

        print(f"Training model (CPU simplified) on {num_words} words with n-gram context length={self.n}.")
        print("Symbolic factors will be calculated based on incrementally processed data prefixes.")
        
        temp_m = defaultdict(lambda: defaultdict(float))
        temp_s = set()
        self.all_words.update(words) # Accumulate all words seen
        
        num_ngram_instances = num_words//2
        
        for i in tqdm(range(num_ngram_instances), desc="Simplified CPU Training (Incremental Prefix Factors)"):
            current_prefix_words = words[0 : i + self.n + 1]
            current_total_word_count = float(len(current_prefix_words))
            if current_total_word_count == 0: continue # Should not happen if num_words > self.n

            current_overall_word_freqs = Counter(current_prefix_words)
            current_ngram_context = tuple(words[i : i + self.n])
            next_word_actual = words[i + self.n]
            
            base_increment = 1.0
            norm_freq_next_word = current_overall_word_freqs.get(next_word_actual, 0) / current_total_word_count
            global_freq_factor = 1.0 + math.log1p(norm_freq_next_word * 100) * 0.1 
            len_next_word = len(next_word_actual)
            length_factor = 1.0 + math.log1p(len_next_word) * 0.02
            symbolic_enhancement_factor = global_freq_factor * length_factor
            symbolic_enhancement_factor = max(0.5, min(symbolic_enhancement_factor, 2.0))
            adjusted_increment_val = base_increment * symbolic_enhancement_factor
            noise_val = random.uniform(-0.01, 0.01) * adjusted_increment_val
            final_increment_val = max(0.01, adjusted_increment_val + noise_val)
            
            is_sentence_start = (i == 0) or \
                                (words[i-1] and any(c in self.SENTENCE_END_CHARS for c in words[i-1][-1:]))
            if is_sentence_start:
                temp_s.add(current_ngram_context)
            temp_m[current_ngram_context][next_word_actual] += final_increment_val
            
        self.epochs += num_ngram_instances

        # Merge with existing self.m and self.s for cumulative training
        for gram, next_word_counts_for_gram in temp_m.items():
            if gram not in self.m:
                self.m[gram] = defaultdict(float)
            for next_word, increment_value in next_word_counts_for_gram.items():
                self.m[gram][next_word] *= increment_value*i
        
        current_starts_as_set = set(self.s)
        current_starts_as_set.update(temp_s)
        self.s = list(filter(lambda sg: sg in self.m, current_starts_as_set)) # Filter to ensure starts are in m
                
        if not self.s and self.m:
            num_starts_to_sample = min(max(1, int(len(self.m) * 0.05)), 50)
            if num_starts_to_sample > len(self.m): num_starts_to_sample = len(self.m)
            if num_starts_to_sample > 0:
                self.s = random.sample(list(self.m.keys()), k=num_starts_to_sample)
                print(f"  No explicit sentence starts identified; sampled {len(self.s)} contexts as potential starts.")
                
        self.cascade_memory = []
        print(f"Simplified CPU training complete. Model has {len(self.m)} contexts (n-gram length {self.n}), {len(self.s)} valid starts.")


    def generate(self, length=50, start_context_str=None):
        if not self.m: return "Model not trained."
        self.cascade_memory, self.retrocausal_loops = [], []
        current_gram_tuple, generated_words_list = None, []

        if start_context_str:
            sc_words = start_context_str.split()
            if len(sc_words) >= self.n:
                current_gram_tuple = tuple(sc_words[-self.n:])
                if current_gram_tuple not in self.m: current_gram_tuple = None
                else: 
                    generated_words_list = list(sc_words)
                    self.cascade_memory = [current_gram_tuple] # Initialize with the starting tuple
        
        if current_gram_tuple is None:
            if not self.s: 
                if self.m:
                    sample_k = max(1, min(int(len(self.m)*0.1), len(self.m), 50)) # Cap sample_k
                    if sample_k > 0: self.s = random.sample(list(self.m.keys()), k=sample_k)
                if not self.s: return "Cannot start: no valid start points in model."
            try: current_gram_tuple = random.choice(self.s)
            except IndexError: return "Cannot start: no sentence starts available."
            generated_words_list = list(current_gram_tuple)
            self.cascade_memory = [current_gram_tuple] # Initialize with the starting tuple

        local_gen_hist_ngrams = [current_gram_tuple]
        MAX_GEN_HIST_LEN = 15
        
        num_to_gen = length - len(generated_words_list)
        for gen_idx in range(num_to_gen):
            if len(generated_words_list) >= length: break
            
            if current_gram_tuple not in self.m or not self.m[current_gram_tuple]:
                if not self.s: break
                try: current_gram_tuple = random.choice(self.s)
                except IndexError: break
                if current_gram_tuple not in self.m or not self.m[current_gram_tuple]: break
                local_gen_hist_ngrams = [current_gram_tuple] 
                # Consider adding a portion of new context to generated_words_list if jumping
                # generated_words_list.extend(list(current_gram_tuple)) # Could lead to repetition if not handled carefully
                self.cascade_memory.append(current_gram_tuple)
                if len(self.cascade_memory) > 20: self.cascade_memory.pop(0)


            options = self.m[current_gram_tuple]
            cand_words, cand_scores = self._symbolic_probability(current_gram_tuple, options)
            if not cand_words:
                if not self.s: break
                try: current_gram_tuple = random.choice(self.s)
                except IndexError: break
                local_gen_hist_ngrams = [current_gram_tuple]
                self.cascade_memory.append(current_gram_tuple)
                if len(self.cascade_memory) > 20: self.cascade_memory.pop(0)
                continue
            
            time_idx_gen = len(generated_words_list)
            
            if self.detect_retrocausal_integration(current_gram_tuple, local_gen_hist_ngrams):
                _, final_scores = self.recompute_with_extra_dimension(
                    current_gram_tuple, cand_words, cand_scores, time_idx_gen 
                )
            else:
                final_scores = cand_scores
            
            sum_scores = sum(final_scores)
            if sum_scores > 1e-9: probs = [s / sum_scores for s in final_scores]
            else: probs = [1.0 / len(cand_words)] * len(cand_words) if cand_words else []
            
            if not cand_words or not probs : break 
            try:
                next_word = random.choices(cand_words, weights=probs, k=1)[0]
            except ValueError: # Handles issues like empty cand_words or mismatched lengths if probs is wrong
                next_word = random.choice(cand_words) if cand_words else None
            
            if next_word is None: break # Stop if no word could be chosen
            
            generated_words_list.append(next_word)
            current_gram_tuple = tuple(generated_words_list[-self.n:])
            
            local_gen_hist_ngrams.append(current_gram_tuple)
            if len(local_gen_hist_ngrams) > MAX_GEN_HIST_LEN: local_gen_hist_ngrams.pop(0)
            
            self.cascade_memory.append(current_gram_tuple)
            if len(self.cascade_memory) > 20: self.cascade_memory.pop(0)
            
        return " ".join(generated_words_list)

    def save_model(self, filepath):
        # No OpenCL attributes to worry about anymore
        try:
            with open(filepath, 'wb') as f:
                pickle.dump(self, f)
            print(f"Model saved successfully to {filepath}")
        except Exception as e:
            print(f"Error saving model to {filepath}: {e}")

    @staticmethod
    def load_model(filepath):
        try:
            with open(filepath, 'rb') as f:
                model = pickle.load(f)
            print(f"Model loaded successfully from {filepath}")
            # No OpenCL re-initialization needed
            return model
        except FileNotFoundError:
            print(f"Error: Model file not found at {filepath}")
            return None
        except Exception as e:
            print(f"Error loading model from {filepath}: {e}")
            return None

if __name__ == "__main__":
    print("Initializing Simplified Symbolic Markov Model (CPU Only)...")
    model = None 

    load_choice = input("Do you want to (L)oad a saved model or (T)rain a new one? [L/T]: ").strip().upper()

    if load_choice == 'L':
        model_filepath = input("Enter the filepath of the saved model: ").strip()
        model = RetrocausalSymbolicMarkov.load_model(model_filepath)
        if model is None:
            print("Failed to load model. Will proceed to train a new one if training data is provided.")
    
    if model is None: 
        print("Proceeding to train a new model.")
        try:
            n_val_str = input("Enter n-gram context size (e.g., 2 for bigram context): ").strip()
            n_val = int(n_val_str) if n_val_str.isdigit() else 2
            if n_val < 1: n_val = 2 # Default to 2 if invalid
            
            depth_val_str = input("Enter extra dimension depth for generation (e.g., 3): ").strip()
            depth_val = int(depth_val_str) if depth_val_str.isdigit() else 3
            if depth_val < 1: depth_val = 3

            model = RetrocausalSymbolicMarkov(n=n_val, cascade_threshold=0.75, extra_dim_depth=depth_val)
        except ValueError: # Catch if int conversion fails for some reason
            print("Invalid input for n or depth. Using defaults (n=2, depth=3).")
            model = RetrocausalSymbolicMarkov(n=2, cascade_threshold=0.75, extra_dim_depth=3)

        training_text = None
        default_training_text = (
            "The quick brown fox jumps over the lazy dog. "
            "A lazy dog sleeps all day. The quick cat also jumps. "
            "Foxes are quick and dogs can be lazy. Cats are nimble. "
            "What does the fox say? The dog barks. The cat meows. "
            "Symbolic AI and machine learning can create interesting text. "
            "This model attempts to blend Markov chains with advanced ideas. "
            "Knowledge is updated based on future states. This is a test. " 
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
                    # Limit text size for very large files to prevent memory issues during split/join
                    max_words_from_file = 200000
                    training_text = ' '.join(raw_text.split()[:max_words_from_file]) 
                print(f"Loaded training data from file: {filename} (up to {max_words_from_file} words).")
            except FileNotFoundError:
                print(f"Error: File '{filename}' not found. Using default short training data.")
                training_text = default_training_text
            except Exception as e:
                print(f"An error occurred while reading the file: {e}. Using default short training data.")
                training_text = default_training_text
        elif data_source_choice == '2' and HF_DATASETS_AVAILABLE:
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
                hf_num_rows_str = input("Enter max number of rows to use (e.g., 10000, blank for all from split): ").strip()
                hf_num_rows = int(hf_num_rows_str) if hf_num_rows_str.isdigit() else None

                print(f"Loading HF Dataset: {hf_dataset_name}, Config: {hf_config_name}, Split: {hf_split}, Column: {hf_text_column}")
                dataset_args = {"path": hf_dataset_name, "split": hf_split}
                if hf_config_name: dataset_args["name"] = hf_config_name
                
                ds_full = load_dataset(**dataset_args)
                
                # Select rows before iterating if hf_num_rows is specified
                ds_to_process = ds_full.select(range(hf_num_rows)) if hf_num_rows and hf_num_rows < len(ds_full) else ds_full

                if hf_text_column not in ds_to_process.features:
                        raise ValueError(f"Text column '{hf_text_column}' not found. Available: {list(ds_to_process.features)}")

                all_hf_texts = []
                for ex in tqdm(ds_to_process, desc="Processing HF dataset rows"):
                    text_content = ex.get(hf_text_column)
                    if text_content and isinstance(text_content, str):
                        all_hf_texts.append(text_content.lower())
                
                training_text = " ".join(all_hf_texts)
                training_text = ' '.join(training_text.split()) # Normalize whitespace
                print(f"Loaded {len(training_text.split())} words from Hugging Face dataset.")
                if not training_text: raise ValueError("No text extracted from Hugging Face dataset.")
            except Exception as e:
                print(f"Hugging Face dataset loading error: {e}. Using default short training data.")
                training_text = default_training_text
        elif data_source_choice == '2' and not HF_DATASETS_AVAILABLE:
             print("Hugging Face 'datasets' library is not available. Falling back to default text.")
             training_text = default_training_text
        else:
            print("Using default short training data.")
            training_text = default_training_text

        if training_text:
            print("\nTraining the model (CPU only)...")
            model.train(training_text) # Always calls the CPU train method now
        else:
            print("No training data loaded. Model cannot be trained and was not loaded.")
    
    if model: 
        save_choice = input("\nDo you want to save the current model? [Y/N]: ").strip().upper()
        if save_choice == 'Y':
            save_filepath = input("Enter filepath to save the model (e.g., retro_model_cpu.pkl): ").strip()
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
            
            gen_length_str = input("Enter generation length (e.g., 100, default 50): ").strip()
            try:
                gen_length = int(gen_length_str) if gen_length_str.isdigit() else 50
                if gen_length <=0: gen_length = 50
            except ValueError:
                print("Invalid length, using default 50.")
                gen_length = 50

            generated_text = model.generate(length=gen_length, start_context_str=seed)
            print(f"AI: {generated_text}")
    elif model:
        print("Model is loaded/initialized but appears to have no training data/transitions. Cannot start generation.")
    else:
        print("Model could not be initialized or loaded. Exiting.")

    print("\nModel demonstration finished.")
