import random
import math
import sys
from collections import defaultdict, Counter
from tqdm import tqdm
import numpy as np
import pickle
# import torch # Not used in the provided snippet, can be removed if not used elsewhere

class RetrocausalSymbolicMarkov:
    """
    Markov chain text generator using symbolic probability distribution ⊆⊗∃·Λρ∑ω·Σø²
    with retrocausal knowledge update equation K_n(x) = K_{n-1}(x) + α·S_n(x)·[A(S_n(x), I(x=x_true)) - K_{n-1}(x)]
    and T-symmetric cascade dynamics ∂/∂t selecting futures that ∫ into their own past.
    Enhanced with extra-dimensional recomputation during retrocausal integration.
    """

    # Static configuration variables
    SENTENCE_END_CHARS = set('.!?')

    def __init__(self, n=2, alpha=0.3, cascade_threshold=0.7, extra_dim_depth=3):
        """
        Initializes the retrocausal Markov chain model.
        Args:
            n (int): Size of n-gram context
            alpha (float): Learning rate for retrocausal knowledge updates
            cascade_threshold (float): Threshold for cascade formation
            extra_dim_depth (int): Depth of extra-dimensional computation layers
        """
        if not isinstance(n, int) or n < 1:
            raise ValueError("n-gram size 'n' must be a positive integer.")

        self.n = n
        self.alpha = alpha # Learning rate for K updates
        self.cascade_threshold = cascade_threshold # For cascade formation / integration detection
        self.extra_dim_depth = extra_dim_depth

        # Common epsilon for numerical stability
        self.param_epsilon_numerical_stability = extra_dim_depth

        # Configurable parameters for mathematical operations
        # For compute_extra_dimension (ced)
        self.param_ced_tau_scale = extra_dim_depth
        self.param_ced_r_clip_min = extra_dim_depth
        self.param_ced_r_clip_max = extra_dim_depth
        self.param_ced_future_influence_default = extra_dim_depth
        self.param_ced_temporal_phase_scale = extra_dim_depth
        self.param_ced_fold_factor_exp_cap = extra_dim_depth
        self.param_ced_norm_large_cap_factor = extra_dim_depth
        self.param_ced_norm_small_reset_std = extra_dim_depth
        self.param_ced_norm_zero_reset_std = extra_dim_depth
        # self.param_ced_norm_min_stable = 1e-9 # Replaced by param_epsilon_numerical_stability
        self.param_ced_norm_max_stable = extra_dim_depth

        # New parameter for the 'ln elegant dimmed array' influence
        self.param_ced_log_dim_influence_scale = 0.01 # How much the initial log-dimmed values influence the final vector

        # For calculate_fold_strength (cfs)
        self.param_cfs_loop_decay_base = extra_dim_depth
        self.param_cfs_coherence_weight = extra_dim_depth
        self.param_cfs_max_strength = extra_dim_depth

        # For recompute_with_extra_dimension (rcwed)
        self.param_rcwed_future_states_count = extra_dim_depth
        self.param_rcwed_word_vec_hash_mod = extra_dim_depth
        self.param_rcwed_dim_proj_boost_scale = extra_dim_depth
        self.param_rcwed_fold_corr_strength_scale = extra_dim_depth
        self.param_rcwed_fold_corr_exp_cap = extra_dim_depth
        self.param_rcwed_symmetry_coherence_scale = extra_dim_depth
        self.param_rcwed_min_enhanced_weight = extra_dim_depth


        # For update_knowledge (uk)
        self.param_uk_knowledge_min_val = 0.0
        self.param_uk_knowledge_max_val = 1.0
        self.param_uk_default_k_prev = 0.5


        # For calculate_cascade_coherence (ccc)
        self.param_ccc_max_iterations = extra_dim_depth
        self.param_ccc_decay_gamma = extra_dim_depth
        self.param_ccc_coherence_min_val = 0.0
        self.param_ccc_coherence_max_val = 1.0
        self.param_ccc_error_reduction_factor =extra_dim_depth


        # For retrocausal_information_kernel (rik)
        self.param_rik_max_states_to_sum = extra_dim_depth
        self.param_rik_state_val_hash_mod = extra_dim_depth
        self.param_rik_state_val_hash_scale = extra_dim_depth
        self.param_rik_max_abs_info = extra_dim_depth

        # For _calculate_distinct_permutations_log (cdpl)
        self.param_cdpl_max_len = extra_dim_depth
        self.param_cdpl_log_perm_min = 0.0
        self.param_cdpl_log_perm_max = 20.0


        # For _compute_l_semi_inner_product (lip)
        # self.param_lip_char_prod_norm_eps = 1e-9 # Replaced
        # self.param_lip_len_ratio_eps = 1e-9     # Replaced
        # self.param_lip_pos_weight_eps = 1e-9    # Replaced
        self.param_lip_phonetic_max_len =extra_dim_depth
        self.param_lip_prefix_suffix_match_len = extra_dim_depth
        # self.param_lip_prefix_suffix_norm_eps = 1e-9 # Replaced
        self.param_lip_perm_sim_denom_min = 1.0
        # self.param_lip_perm_sim_norm_eps = 1e-9 # Replaced
        self.param_lip_cascade_mem_len_scale = extra_dim_depth
        self.param_lip_cascade_mem_recent_cap = extra_dim_depth
        self.param_lip_cascade_boost_scale = extra_dim_depth
        self.param_lip_w_char_prod = extra_dim_depth
        self.param_lip_w_len_ratio = extra_dim_depth
        self.param_lip_w_phonetic = extra_dim_depth
        self.param_lip_w_prefix_suffix = extra_dim_depth
        self.param_lip_w_perm_sim = extra_dim_depth
        self.param_lip_max_val = 2.0
        

        # For train
        self.param_train_sel_prob_default = extra_dim_depth
        self.param_train_sel_prob_coherence_scale = extra_dim_depth
        self.param_train_sel_prob_min = extra_dim_depth
        self.param_train_sel_prob_max = extra_dim_depth
        self.param_train_glob_freq_log_scale = extra_dim_depth
        self.param_train_glob_freq_factor_scale = extra_dim_depth
        self.param_train_len_factor_log_scale = extra_dim_depth
        self.param_train_l_inner_sim_scale = extra_dim_depth
        self.param_train_l_inner_accum_cap = extra_dim_depth
        self.param_train_k_factor_scale = extra_dim_depth
        self.param_train_sym_enhance_min = 0.2
        self.param_train_sym_enhance_max = 5.0
        self.param_train_noise_range = extra_dim_depth
        self.param_train_final_increment_min = 0.001
        self.param_train_start_sample_ratio = extra_dim_depth
        self.param_train_start_sample_max = 50


        # For _symbolic_probability (sp)
        self.param_sp_k_influence_scale = extra_dim_depth
        self.param_sp_cascade_mem_boost_len_scale = extra_dim_depth
        self.param_sp_cascade_mem_boost_cap = extra_dim_depth
        self.param_sp_min_final_score = extra_dim_depth
        self.param_sp_context_sim_l_sim_scale = extra_dim_depth
        self.param_sp_context_sim_accum_cap = extra_dim_depth
        self.param_sp_min_refined_score = 0.00001
        self.param_sp_num_final_candidates = extra_dim_depth

        # For generate (gen)
        self.param_gen_similarity_threshold = 0.8 # Used in detect_retrocausal_integration
        # self.param_gen_sum_scores_min_thresh = 1e-9 # Replaced


        # Traditional Markov structures
        self.m = {}  # Transitions: {context_tuple: {next_word: adjusted_frequency}}
        self.s = []  # Sentence starting n-grams
        self.all_words = set()

        # Retrocausal knowledge structures
        self.K = {}  # Knowledge matrix: {(epoch, context, word): knowledge_value}
        self.S = {}  # Selection gates: {(epoch, context): selection_probability}
        self.cascade_memory = []  # Cascade history for temporal coherence
        self.epochs = 0

        # Extra-dimensional integration structures
        self.D = {}  # Dimensional manifold: {(time, context): extra_dim_vector}
        self.integration_history = []  # Track when past-future loops close
        self.fold_tensor = {}  # Dimensional folding computations
        self.retrocausal_loops = []  # Detected closed timelike curves

        # Cache for expensive calculations
        self.similarity_cache = {}
        self.permutation_cache = {}
        self.dimensional_cache = {}

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

        # "ln elegant dimmed array one liner" - Initializing extra_dim with logarithmically decaying values.
        # These values provide a "dimmed" baseline influence.
        extra_dim = np.logspace(-self.extra_dim_depth, -1, num=self.extra_dim_depth * 2) * self.param_ced_log_dim_influence_scale

        try:
            # Hyperbolic component (H³ space) - Now *modifying* the initial extra_dim
            for i in range(self.extra_dim_depth):
                tau = time_index + 1
                hyperbolic_factor = math.tanh(tau / self.param_ced_tau_scale)

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
                        future_influence = self.param_ced_future_influence_default
                
                r = hyperbolic_factor * (1.0 + future_influence)
                r = min(max(r, self.param_ced_r_clip_min), self.param_ced_r_clip_max) # Clip r
                theta = context_angle + i * math.pi / self.extra_dim_depth

                # Blend or add hyperbolic components to the initial log_dimmed values
                # We'll use a blend, allowing the log-dimmed values to act as a background "energy"
                hyperbolic_real = math.cosh(r) * math.cos(theta)
                hyperbolic_imag = math.sinh(r) * math.sin(theta)

                # Blend the initial log-dimmed value with the hyperbolic computation
                # The influence scale determines how much the initial log-dimmed value contributes.
                # This makes the log-dimmed values a constant, subtle "hum" in the extra dimension.
                extra_dim[i*2] = extra_dim[i*2] + hyperbolic_real * (1 - self.param_ced_log_dim_influence_scale) # Blending example
                extra_dim[i*2 + 1] = extra_dim[i*2 + 1] + hyperbolic_imag * (1 - self.param_ced_log_dim_influence_scale) # Blending example


            # Temporal folding (S¹ component) - Also modifies the existing extra_dim
            for i in range(self.extra_dim_depth):
                temporal_phase = (time_index / self.param_ced_temporal_phase_scale) % (2 * math.pi)
                fold_strength = self.calculate_fold_strength(context, time_index)
                
                fold_factor_arg = -fold_strength * temporal_phase / (2 * math.pi)
                fold_factor = math.exp(max(fold_factor_arg, self.param_ced_fold_factor_exp_cap))

                # Apply fold factor to the already combined hyperbolic and log-dimmed values
                extra_dim[i*2] *= fold_factor
                extra_dim[i*2 + 1] *= math.cos(temporal_phase + i * math.pi / 3)

            norm = np.linalg.norm(extra_dim)
            if self.param_epsilon_numerical_stability < norm < self.param_ced_norm_max_stable: # Used common epsilon
                extra_dim = extra_dim / norm
            elif norm >= self.param_ced_norm_max_stable:
                extra_dim = (extra_dim / norm) * self.param_ced_norm_large_cap_factor
            elif self.param_epsilon_numerical_stability >= norm > 0: 
                extra_dim = np.random.normal(0, self.param_ced_norm_small_reset_std, self.extra_dim_depth * 2)
            else: # Handles norm == 0 or other unexpected cases
                extra_dim = np.random.normal(0, self.param_ced_norm_zero_reset_std, self.extra_dim_depth * 2)

        except (OverflowError, ValueError, ZeroDivisionError) as e:
            extra_dim = np.random.normal(0, self.param_ced_norm_zero_reset_std, self.extra_dim_depth * 2)

        if len(self.dimensional_cache) < 1000:
            self.dimensional_cache[cache_key] = extra_dim
        return extra_dim

    def calculate_fold_strength(self, context, time_index):
        """ Calculate dimensional folding strength based on retrocausal loop closure. """
        fold_strength = 0.0
        try:
            loop_contributions = sum(1.0 / (self.param_cfs_loop_decay_base + (len(self.cascade_memory) - i))
                                     for i, pc in enumerate(self.cascade_memory) if pc == context)
            fold_strength += loop_contributions
        except TypeError:  # Handle cases where context types might mismatch
            pass
        
        if len(self.cascade_memory) > 2:
            recent_coherence = self.calculate_cascade_coherence(self.cascade_memory[-3:])
            fold_strength += recent_coherence * self.param_cfs_coherence_weight
        
        return min(self.param_cfs_max_strength, fold_strength)

    def detect_retrocausal_integration(self, current_context, generation_history):
        """ Detect when future states integrate into their own past. """
        if len(generation_history) < 3: return False

        context_str = str(current_context)
        for i, past_state in enumerate(generation_history[-10:]):
            past_str = str(past_state)
            similarity = self._safe_l_semi_inner_product(context_str, past_str)
            
            if similarity > self.param_gen_similarity_threshold: # High similarity indicates potential loop
                loop_coherence = self.calculate_cascade_coherence([past_state, current_context])
                if loop_coherence > self.cascade_threshold: # Original cascade_threshold
                    if len(self.retrocausal_loops) < 100:
                        self.retrocausal_loops.append({
                            'start_index': len(generation_history) - 10 + i,
                            'end_index': len(generation_history),
                            'coherence': loop_coherence,
                            'context_similarity': similarity
                        })
                    return True
        return False

    def recompute_with_extra_dimension(self, context, candidate_words, candidate_weights, time_index):
        """ Recompute probabilities incorporating extra-dimensional manifold. """
        future_states = []
        sorted_candidates = sorted(zip(candidate_weights, candidate_words), reverse=True)
        for weight, word in sorted_candidates[:self.param_rcwed_future_states_count]:
            future_states.append(word)

        extra_dim_vector = self.compute_extra_dimension(context, time_index, future_states)
        
        current_context_tuple = tuple(context) if isinstance(context, list) else context
        if not isinstance(current_context_tuple, tuple):
            current_context_tuple = tuple(str(current_context_tuple))
        
        if len(self.D) > 2000:
            keys_to_remove = list(self.D.keys())[:len(self.D)-2000]
            for key_to_remove in keys_to_remove: del self.D[key_to_remove]
        self.D[(time_index, current_context_tuple)] = extra_dim_vector

        enhanced_weights = []
        for i, word in enumerate(candidate_words):
            base_weight = candidate_weights[i] if i < len(candidate_weights) else 1.0
            word_hash = hash(word) % self.param_rcwed_word_vec_hash_mod
            word_vector_components = [(math.cos((word_hash / self.param_rcwed_word_vec_hash_mod) * 2 * math.pi + (j * math.pi / (self.extra_dim_depth * 2)))) for j in range(self.extra_dim_depth * 2)]
            word_vector = np.array(word_vector_components)

            try:
                norm_extra_dim = np.linalg.norm(extra_dim_vector)
                norm_word_vec = np.linalg.norm(word_vector)
                dimensional_projection = np.dot(extra_dim_vector, word_vector) / (norm_extra_dim * norm_word_vec) if norm_extra_dim > self.param_epsilon_numerical_stability and norm_word_vec > self.param_epsilon_numerical_stability else 0.0 # Used common epsilon
                dimensional_boost = 1.0 + abs(dimensional_projection) * self.param_rcwed_dim_proj_boost_scale
            except (ValueError, OverflowError, ZeroDivisionError):
                dimensional_boost = 1.0

            fold_strength = self.calculate_fold_strength(current_context_tuple, time_index)
            fold_correction_arg = -fold_strength * self.param_rcwed_fold_corr_strength_scale
            fold_correction = math.exp(max(fold_correction_arg, self.param_rcwed_fold_corr_exp_cap))
            
            symmetry_factor = 1.0 + (self.retrocausal_loops[-1].get('coherence', 0.0) * self.param_rcwed_symmetry_coherence_scale if self.retrocausal_loops else 0.0)
            
            enhanced_weight = base_weight * dimensional_boost * fold_correction * symmetry_factor
            enhanced_weights.append(max(self.param_rcwed_min_enhanced_weight, enhanced_weight))
        return candidate_words, enhanced_weights

    def update_knowledge(self, K_prev, alpha, S_n, x, x_true):
        """ Retrocausal knowledge update equation. """
        A_gate = S_n * (1.0 if x == x_true else 0.0) # Inlined I_x_true
        K_n = K_prev + alpha * S_n * (A_gate - K_prev) # Inlined error_term
        return max(self.param_uk_knowledge_min_val, min(self.param_uk_knowledge_max_val, K_n))

    def calculate_cascade_coherence(self, context_sequence):
        """ Calculate cascade coherence. """
        if len(context_sequence) < 2: return 0.0
        coherence = 1.0
        for i in range(min(self.param_ccc_max_iterations, len(context_sequence) - 1)):
            try:
                c1, c2 = context_sequence[i], context_sequence[i + 1]
                context1_str = ' '.join(map(str, c1)) if isinstance(c1, (list, tuple)) else str(c1)
                context2_str = ' '.join(map(str, c2)) if isinstance(c2, (list, tuple)) else str(c2)
                overlap = self._safe_l_semi_inner_product(context1_str, context2_str)
                decay = math.exp(-self.param_ccc_decay_gamma * i)
                coherence *= decay * (overlap ** 2)
                coherence = max(self.param_ccc_coherence_min_val, min(self.param_ccc_coherence_max_val, coherence))
                if coherence < self.param_epsilon_numerical_stability: coherence = 0.0; break # Used common epsilon
            except Exception:
                coherence *= self.param_ccc_error_reduction_factor
                coherence = max(self.param_ccc_coherence_min_val, min(self.param_ccc_coherence_max_val, coherence))
                break
        return coherence

    def retrocausal_information_kernel(self, t0, future_states, omega=1.0):
        """ Calculate retrocausal information kernel (approximated sum). """
        information = 0.0j
        for tau_idx in range(min(self.param_rik_max_states_to_sum, len(future_states))): # Ensure not to exceed future_states length
            current_future_state = future_states[tau_idx]
            try:
                state_val_str = str(current_future_state)
                try:
                    state_val = float(state_val_str[:10].replace(',',''))
                except ValueError:
                    state_val = float(hash(state_val_str) % self.param_rik_state_val_hash_mod - self.param_rik_state_val_hash_scale) / self.param_rik_state_val_hash_scale
                information += state_val * np.exp(-1j * omega * tau_idx)
            except (ValueError, OverflowError, TypeError):
                continue
        return min(abs(information), self.param_rik_max_abs_info)

    @staticmethod
    def _calculate_distinct_permutations_log(word):
        """ Calculates log of distinct permutations. """
        # Static method cannot access self.param_cdpl... directly.
        # Using original clamping values or pass params if this method's behavior needs to be configurable.
        if not word or len(word) == 0: return 0.0
        word_str = str(word)[:30] 
        n = len(word_str)
        if n == 0: return 0.0
        counts = Counter(word_str)
        try:
            log_n_factorial = math.lgamma(n + 1)
            log_denominator_sum = sum(math.lgamma(count + 1) for count in counts.values())
            log_permutations = log_n_factorial - log_denominator_sum
            # Using class-level params for consistency if possible, or hardcoded if truly static
            return max(RetrocausalSymbolicMarkov(n=1).param_cdpl_log_perm_min,  # Hacky access, better to pass
                       min(RetrocausalSymbolicMarkov(n=1).param_cdpl_log_perm_max, log_permutations))
        except (ValueError, OverflowError):
            return 0.0
    # A better way for static method params:
    # @staticmethod
    # def _calculate_distinct_permutations_log(word, param_max_len=30, param_log_perm_min=0.0, param_log_perm_max=20.0):
    # ... then use param_log_perm_min, param_log_perm_max ...
    # For now, I will leave the hacky access to illustrate, but it's not ideal.
    # Or, make it a regular method if it needs self.


    def _safe_l_semi_inner_product(self, word1, word2):
        """ Safe version of L-semi-inner product. """
        if word1 is None or word2 is None: return 0.0
        try: s_word1, s_word2 = str(word1), str(word2)
        except Exception: return 0.0
        if not s_word1 or not s_word2: return 0.0
        
        cache_key = (s_word1[:20], s_word2[:20])
        if cache_key in self.similarity_cache: return self.similarity_cache[cache_key]
        
        s_word1_limited, s_word2_limited = s_word1[:50], s_word2[:50]
        result = self._compute_l_semi_inner_product(s_word1_limited, s_word2_limited)
        
        if len(self.similarity_cache) < 2000: self.similarity_cache[cache_key] = result
        return result

    def _compute_l_semi_inner_product(self, word1, word2):
        """ Core computation for L-semi-inner product. """
        if not word1 or not word2: return 0.0
        w1_len, w2_len = len(word1), len(word2)
        if w1_len == 0 or w2_len == 0: return 0.0

        char_product = sum(
            (max(0, 1.0 - (abs(i - (w1_len - 1) / 2.0) / (w1_len / 2.0 + self.param_epsilon_numerical_stability)))) * # Used common epsilon
            (max(0, 1.0 - (abs(j - (w2_len - 1) / 2.0) / (w2_len / 2.0 + self.param_epsilon_numerical_stability))))  # Used common epsilon
            for i, c1 in enumerate(word1) for j, c2 in enumerate(word2) if c1 == c2)
        
        char_product_normalized = min(char_product / ((w1_len * w2_len)**0.5 + self.param_epsilon_numerical_stability), 1.0) # Used common epsilon
        length_ratio = min(w1_len, w2_len) / (max(w1_len, w2_len) + self.param_epsilon_numerical_stability) # Used common epsilon

        vowels = set('aeiouAEIOU')
        w1_alpha, w2_alpha = "".join(filter(str.isalpha, word1)), "".join(filter(str.isalpha, word2))
        w1_pattern, w2_pattern = (''.join('V' if c in vowels else 'C' for c in w1_alpha), 
                                  ''.join('V' if c in vowels else 'C' for c in w2_alpha))

        pattern_len_max = min(self.param_lip_phonetic_max_len, len(w1_pattern), len(w2_pattern)) if w1_pattern and w2_pattern else 0
        phonetic_factor = ((sum(1 for k in range(pattern_len_max) if w1_pattern[k] == w2_pattern[k]) +
                           sum(1 for k in range(pattern_len_max) if w1_pattern[-(k+1)] == w2_pattern[-(k+1)])) /
                          (2.0 * pattern_len_max + self.param_epsilon_numerical_stability)) if pattern_len_max > 0 else 0.0 # Used common epsilon
        
        prefix_len = sum(1 for i in range(min(w1_len, w2_len, self.param_lip_prefix_suffix_match_len)) if word1[i] == word2[i]) # Corrected logic from original code
        suffix_len = sum(1 for i in range(1, min(w1_len, w2_len, self.param_lip_prefix_suffix_match_len) + 1) if word1[-i] == word2[-i]) # Corrected logic from original code

        prefix_suffix_factor = min((prefix_len + suffix_len) / (float(min(min(w1_len,w2_len), self.param_lip_prefix_suffix_match_len) * 2) + self.param_epsilon_numerical_stability), 1.0) # Used common epsilon


        log_perm1_val = self._calculate_distinct_permutations_log(word1[:20]) 
        log_perm2_val = self._calculate_distinct_permutations_log(word2[:20])
        diff_log_perm = abs(log_perm1_val - log_perm2_val)
        denominator_perm = max(log_perm1_val, log_perm2_val, self.param_lip_perm_sim_denom_min)
        permutation_similarity_factor = 1.0 - min(diff_log_perm / (denominator_perm + self.param_epsilon_numerical_stability), 1.0) # Used common epsilon

        cascade_boost = 1.0 + (min(self.param_lip_cascade_mem_recent_cap, len(self.cascade_memory) * self.param_lip_cascade_mem_len_scale) * self.param_lip_cascade_boost_scale if self.cascade_memory else 0.0)

        l_inner_product = (
            self.param_lip_w_char_prod * char_product_normalized +
            self.param_lip_w_len_ratio * length_ratio +
            self.param_lip_w_phonetic * phonetic_factor +
            self.param_lip_w_prefix_suffix * prefix_suffix_factor +
            self.param_lip_w_perm_sim * permutation_similarity_factor
        ) * cascade_boost
        return max(0.0, min(self.param_lip_max_val, l_inner_product))

    def train(self, t):
        """ Enhanced training with retrocausal knowledge updates. """
        if not isinstance(t, str) or not t: raise TypeError("Training data must be a non-empty string.")
        words = t.split()
        num_words = len(words)
        if num_words <= self.n:
            print(f"Warning: Training data too short. Skipping.")
            return

        print(f"Training retrocausal model on {num_words} words...")
        overall_word_freqs = Counter(words)
        total_word_count = float(num_words)
        temp_m = defaultdict(lambda: defaultdict(float))
        temp_s = set()
        self.all_words.update(words)
        current_run_epoch_offset = self.epochs
        self.epochs += (num_words - self.n)
        context_sequence_for_training = []
        MAX_CONTEXT_SEQ_LEN = 15
        sample_words_for_k_update = random.sample(list(self.all_words), min(len(self.all_words), 500))

        for iter_idx, i in enumerate(tqdm(range(num_words - self.n), desc="Retrocausal Training")):
            epoch_for_k = current_run_epoch_offset + iter_idx
            current_ngram, next_word_actual = tuple(words[i:i+self.n]), words[i+self.n]
            
            context_sequence_for_training.append(current_ngram)
            if len(context_sequence_for_training) > MAX_CONTEXT_SEQ_LEN: context_sequence_for_training.pop(0)

            coherence_input_slice = context_sequence_for_training[-min(3, len(context_sequence_for_training)):] if len(context_sequence_for_training) >=2 else []
            cascade_coherence = self.calculate_cascade_coherence(coherence_input_slice) if coherence_input_slice else 0.0
            selection_prob_S_n = max(self.param_train_sel_prob_min, min(self.param_train_sel_prob_max, self.param_train_sel_prob_default + cascade_coherence * self.param_train_sel_prob_coherence_scale))
            self.S[(epoch_for_k, current_ngram)] = selection_prob_S_n
            
            if self.detect_retrocausal_integration(current_ngram, context_sequence_for_training):
                if len(self.integration_history) < 200: self.integration_history.append({'epoch': epoch_for_k, 'context': current_ngram, 'next_word': next_word_actual})

            for candidate_word_for_k in sample_words_for_k_update:
                k_prev_val = self.K.get((epoch_for_k - 1, current_ngram, candidate_word_for_k), self.param_uk_default_k_prev) if epoch_for_k > 0 else self.param_uk_default_k_prev
                k_new_val = self.update_knowledge(k_prev_val, self.alpha, selection_prob_S_n, candidate_word_for_k, next_word_actual)
                if len(self.K) < 50000: self.K[(epoch_for_k, current_ngram, candidate_word_for_k)] = k_new_val
            
            norm_freq_next_word = overall_word_freqs.get(next_word_actual, 0) / total_word_count
            global_freq_factor = 1.0 + math.log1p(norm_freq_next_word * self.param_train_glob_freq_log_scale) * self.param_train_glob_freq_factor_scale
            length_factor = 1.0 + math.log1p(len(next_word_actual)) * self.param_train_len_factor_log_scale
            
            l_inner_prod_accum = 1.0
            for word_in_context in current_ngram[:max(1, self.n//2)]:
                l_inner_prod_accum *= (1.0 + self._safe_l_semi_inner_product(str(word_in_context), next_word_actual) * self.param_train_l_inner_sim_scale)
            l_inner_prod_accum = min(l_inner_prod_accum, self.param_train_l_inner_accum_cap)
            
            k_for_actual_next = self.K.get((epoch_for_k, current_ngram, next_word_actual), self.param_uk_default_k_prev)
            retrocausal_factor_k = 1.0 + (k_for_actual_next - 0.5) * self.param_train_k_factor_scale
            
            symbolic_enhancement_factor = max(self.param_train_sym_enhance_min, min(self.param_train_sym_enhance_max, global_freq_factor * length_factor * l_inner_prod_accum * retrocausal_factor_k))
            adjusted_increment_val = 1.0 * symbolic_enhancement_factor
            noise_val = random.uniform(-self.param_train_noise_range, self.param_train_noise_range) * adjusted_increment_val
            final_increment_val = max(self.param_train_final_increment_min, adjusted_increment_val + noise_val)

            if i == 0 or (i > 0 and words[i-1] and any(c in self.SENTENCE_END_CHARS for c in words[i-1][-1:])):
                temp_s.add(current_ngram)
            temp_m[current_ngram][next_word_actual] += final_increment_val

        self.m = {gram: dict(counts) for gram, counts in temp_m.items() if counts}
        self.s = list(filter(lambda sg: sg in self.m, temp_s))
        if not self.s and self.m:
            num_starts_to_sample = min(max(1, int(len(self.m) * self.param_train_start_sample_ratio)), self.param_train_start_sample_max)
            if num_starts_to_sample > len(self.m): num_starts_to_sample = len(self.m)
            if num_starts_to_sample > 0: self.s = random.sample(list(self.m.keys()), k=num_starts_to_sample)
        
        self.cascade_memory = context_sequence_for_training[-min(10, len(context_sequence_for_training)):] if context_sequence_for_training else []
        print(f"Retrocausal training complete. Model has {len(self.m)} contexts, {len(self.s)} valid starts.")

    def _symbolic_probability(self, context_tuple, options_dict):
        """ Calculates "symbolic probabilities" (scores) for candidate words. """
        if not isinstance(context_tuple, tuple):
            try: context_tuple = tuple(str(c) for c in context_tuple) if hasattr(context_tuple, '__iter__') and not isinstance(context_tuple, str) else tuple([str(context_tuple)])
            except TypeError: context_tuple = tuple()

        candidate_words_list, base_scores_list = list(options_dict.keys()), list(options_dict.values())
        if not candidate_words_list: return [], []

        enhanced_scores = []
        k_lookup_epoch = self.epochs - 1 if self.epochs > 0 else 0

        for i, word_cand in enumerate(candidate_words_list):
            current_base_score = base_scores_list[i]
            k_val = self.K.get((k_lookup_epoch, context_tuple, word_cand), self.param_uk_default_k_prev) if self.K else self.param_uk_default_k_prev
            
            cascade_mem_boost = 1.0 + (min(self.param_sp_cascade_mem_boost_cap, len(self.cascade_memory) * self.param_sp_cascade_mem_boost_len_scale) if self.cascade_memory else 0.0)
            k_factor = 1.0 + (k_val - 0.5) * self.param_sp_k_influence_scale
            final_score_for_word = current_base_score * k_factor * cascade_mem_boost
            enhanced_scores.append(max(self.param_sp_min_final_score, final_score_for_word))

        final_refined_scores = []
        if not context_tuple:
            final_refined_scores = enhanced_scores
        else:
            for i, word_cand in enumerate(candidate_words_list):
                score_after_k_cascade = enhanced_scores[i]
                context_sim_factor_accum = 1.0
                for word_from_context in context_tuple[:max(1, self.n // 2)]:
                    context_sim_factor_accum *= (1.0 + self._safe_l_semi_inner_product(str(word_from_context), word_cand) * self.param_sp_context_sim_l_sim_scale)
                context_sim_factor_accum = min(context_sim_factor_accum, self.param_sp_context_sim_accum_cap)
                score_after_context_sim = score_after_k_cascade * context_sim_factor_accum
                final_refined_scores.append(max(self.param_sp_min_refined_score, score_after_context_sim))
        
        if not final_refined_scores: return [], []
        scored_candidates = sorted(zip(final_refined_scores, candidate_words_list), reverse=True)
        num_final_candidates = min(len(scored_candidates), self.param_sp_num_final_candidates)
        top_candidates_words = [word for score, word in scored_candidates[:num_final_candidates]]
        top_candidates_scores = [score for score, word in scored_candidates[:num_final_candidates]]
        
        return (top_candidates_words, top_candidates_scores) if top_candidates_words else ([],[])

    def generate(self, length=50, start_context_str=None):
        """ Generates text using the retrocausal Markov model. """
        if not self.m: return "Model not trained."

        current_gram_tuple, generated_words_list = None, []
        if start_context_str:
            sc_words = start_context_str.split()
            if len(sc_words) >= self.n:
                current_gram_tuple = tuple(sc_words[-self.n:])
                if current_gram_tuple not in self.m: current_gram_tuple = None
                else: generated_words_list = list(sc_words)
        
        if current_gram_tuple is None:
            if not self.s:
                if self.m: self.s = random.sample(list(self.m.keys()), k=min(1, len(self.m))) 
                if not self.s: return "Cannot start generation: no valid start points."
            current_gram_tuple = random.choice(self.s)
            generated_words_list = list(current_gram_tuple)

        local_generation_history_ngrams = [current_gram_tuple]
        MAX_GEN_HIST_LEN = 15
        self.retrocausal_loops = [] 
        num_words_to_generate = length - len(generated_words_list)

        for _ in range(max(0, num_words_to_generate)): 
            if len(generated_words_list) >= length: break
            if current_gram_tuple not in self.m or not self.m[current_gram_tuple]:
                if not self.s: break
                current_gram_tuple = random.choice(self.s)
                if current_gram_tuple not in self.m or not self.m[current_gram_tuple]: break
                local_generation_history_ngrams.append(current_gram_tuple)
                if len(local_generation_history_ngrams) > MAX_GEN_HIST_LEN: local_generation_history_ngrams.pop(0)

            options_for_current_gram = self.m[current_gram_tuple]
            candidate_words, candidate_raw_scores = self._symbolic_probability(current_gram_tuple, options_for_current_gram)

            if not candidate_words:
                if not self.s: break
                current_gram_tuple = random.choice(self.s)
                local_generation_history_ngrams.append(current_gram_tuple)
                if len(local_generation_history_ngrams) > MAX_GEN_HIST_LEN: local_generation_history_ngrams.pop(0)
                continue

            time_idx_generation = len(generated_words_list)
            if self.detect_retrocausal_integration(current_gram_tuple, local_generation_history_ngrams):
                _, final_scores_for_choice = self.recompute_with_extra_dimension(current_gram_tuple, candidate_words, candidate_raw_scores, time_idx_generation)
            else:
                final_scores_for_choice = candidate_raw_scores

            sum_final_scores = sum(final_scores_for_choice)
            if sum_final_scores > self.param_epsilon_numerical_stability: # Used common epsilon
                probabilities_for_choice = [s / sum_final_scores for s in final_scores_for_choice]
            else:
                probabilities_for_choice = [1.0 / len(candidate_words)] * len(candidate_words) if candidate_words else []

            if not candidate_words or not probabilities_for_choice : break 
            try:
                next_chosen_word = random.choices(candidate_words, weights=probabilities_for_choice, k=1)[0]
            except ValueError: 
                if candidate_words: next_chosen_word = random.choice(candidate_words)
                else: break 

            generated_words_list.append(next_chosen_word)
            current_gram_tuple = tuple(generated_words_list[-self.n:])
            local_generation_history_ngrams.append(current_gram_tuple)
            if len(local_generation_history_ngrams) > MAX_GEN_HIST_LEN: local_generation_history_ngrams.pop(0)
            
        return " ".join(generated_words_list)


    def save_model(self, filepath="retrocausal_model.pkl"):
        """Saves the current model state to a file using pickle."""
        # Caches are typically not essential to save for model functionality,
        # but can be included if desired and if they don't make files too large.
        # For simplicity, we'll save the main structures.
        # Note: tqdm objects are not picklable directly.
        
        # Create a dictionary of the model's state
        state = {
            'n': self.n,
            'alpha': self.alpha,
            'cascade_threshold': self.cascade_threshold,
            'extra_dim_depth': self.extra_dim_depth,
            'params': {name: value for name, value in self.__dict__.items() if name.startswith('param_')},
            'm': self.m,
            's': self.s,
            'all_words': self.all_words,
            'K': self.K,
            'S': self.S,
            'cascade_memory': self.cascade_memory,
            'epochs': self.epochs,
            # D, integration_history, fold_tensor, retrocausal_loops are transient or very large.
            # Caches (similarity_cache, etc.) are also typically not saved.
        }
        try:
            with open(filepath, 'wb') as f:
                pickle.dump(state, f)
            print(f"Model saved successfully to {filepath}")
        except Exception as e:
            print(f"Error saving model: {e}")

    @classmethod
    def load_model(cls, filepath="retrocausal_model.pkl"):
        """Loads a model state from a file and returns a new model instance."""
        try:
            with open(filepath, 'rb') as f:
                state = pickle.load(f)
            
            # Create a new model instance
            model = cls(
                n=state.get('n', 2), 
                alpha=state.get('alpha', 0.3), 
                cascade_threshold=state.get('cascade_threshold', 0.7),
                extra_dim_depth=state.get('extra_dim_depth', 3)
            )
            
            # Restore parameters
            if 'params' in state:
                for param_name, param_value in state['params'].items():
                    setattr(model, param_name, param_value)
            
            # Restore main data structures
            model.m = state.get('m', {})
            model.s = state.get('s', [])
            model.all_words = state.get('all_words', set())
            model.K = state.get('K', {})
            model.S = state.get('S', {})
            model.cascade_memory = state.get('cascade_memory', [])
            model.epochs = state.get('epochs', 0)
            
            print(f"Model loaded successfully from {filepath}")
            return model
        except FileNotFoundError:
            print(f"Error: Model file '{filepath}' not found.")
            return None
        except Exception as e:
            print(f"Error loading model: {e}")
            return None


if __name__ == "__main__":
    model_filepath = "retro_markov_model.pkl"
    # Try to load a pre-trained model
    print(f"Attempting to load model from {model_filepath}...")
    model = RetrocausalSymbolicMarkov.load_model(model_filepath)

    if model is None:
        print("No saved model found or error in loading. Initializing and training a new model.")
        model = RetrocausalSymbolicMarkov(n=2, alpha=0.1, cascade_threshold=0.75, extra_dim_depth=2)
        try:
            filename = input("Enter filename for training data (or press Enter for default): ")
            if filename.strip() == "":
                print("No filename entered, using default short training data.")
                training_text = (
                    "The quick brown fox jumps over the lazy dog. "
                    "A lazy dog sleeps all day. The quick cat also jumps. "
                    "Foxes are quick and dogs can be lazy. Cats are nimble. "
                    "What does the fox say? The dog barks. The cat meows. "
                    "Retrocausality is a complex concept in theoretical physics. "
                    "Symbolic AI and machine learning can create interesting text. "
                    "This model attempts to blend Markov chains with advanced ideas. "
                    "The future influences the past in this framework. "
                    "Knowledge is updated based on future states. This is a test. "
                    "Let's see how well it generates coherent sentences. "
                    "The brown fox is very quick. The dog remains lazy. "
                    "Jumping is fun for the fox and the cat. "
                    "Physics concepts are hard to model. AI is evolving. "
                    "Markov models predict the next state. Retrocausal models look back. "
                    "This sentence provides more data for training. This is the end of the training data."
                )
            else:
                with open(filename, 'r', encoding='utf-8') as file:
                    raw_text = file.read().lower()
                    training_text = ' '.join(raw_text.split()[:9999])
        except FileNotFoundError:
            print(f"Error: File '{filename}' not found. Using default short training data.")
            training_text = (
                "The quick brown fox jumps over the lazy dog. "
                "A lazy dog sleeps all day. The quick cat also jumps. "
                "This is a fallback text because the file was not found."
            )
        except Exception as e:
            print(f"An error occurred while reading the file: {e}. Using default short training data.")
            training_text = (
                "The quick brown fox jumps over the lazy dog. "
                "A lazy dog sleeps all day. The quick cat also jumps. "
                "This is another fallback text due to a file reading error."
            )

        print("\nTraining the model...")
        model.train(training_text)
        print("\nTraining complete. Saving model...")
        model.save_model(model_filepath)
    else:
        print("Model loaded successfully.")


    print("\nInteractive Generation Mode (type 'quit' or 'exit' to stop):")
    while True:
        seed = input("USER: ")
        if seed.lower() in ['quit', 'exit']:
            user_choice = input("Save current model state before quitting? (yes/no): ")
            if user_choice.lower() == 'yes':
                model.save_model(model_filepath)
            break
        if not seed.strip():
            print("AI: (No input provided, try seeding with some text or type 'quit')")
            continue
        
        gen_length = 250 
        generated_text = model.generate(length=gen_length, start_context_str=seed)
        print(f"AI: {generated_text}")

    print("\nModel demonstration finished.")
