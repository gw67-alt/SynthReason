import random
import math
import sys
from collections import defaultdict, Counter
from tqdm import tqdm
import numpy as np
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
        self.alpha = alpha
        self.cascade_threshold = cascade_threshold
        self.extra_dim_depth = extra_dim_depth

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

        # Initialize extra-dimensional vector
        extra_dim = np.zeros(self.extra_dim_depth * 2)  # Real + Imaginary components

        try:
            # Hyperbolic component (H³ space)
            for i in range(self.extra_dim_depth):
                # Temporal curvature based on retrocausal strength
                tau = time_index + 1
                hyperbolic_factor = math.tanh(tau / 10.0)  # Prevent overflow

                # Context embedding in hyperbolic space
                context_hash = hash(str(context)) % 1000000
                context_angle = (context_hash / 1000000.0) * 2 * math.pi

                # Future state influence
                future_influence = 0.0
                if future_states and len(future_states) > i:
                    try:
                        # Attempt to convert future_states[i] to a floatable string or hash it
                        fs_val_str = str(future_states[i])[:10] # Limit length for safety
                        if fs_val_str.replace('.','',1).replace('-','',1).isdigit():
                            future_val = float(fs_val_str)
                        else:
                            future_val = hash(str(future_states[i])) % 1000
                        future_influence = future_val / 1000.0
                    except (ValueError, OverflowError, TypeError):
                        future_influence = 0.5


                # Hyperbolic embedding: cosh(r)cos(θ), cosh(r)sin(θ), sinh(r)
                r = hyperbolic_factor * (1.0 + future_influence)
                r = min(max(r, -700), 700) # Prevent overflow in cosh/sinh
                theta = context_angle + i * math.pi / self.extra_dim_depth

                extra_dim[i*2] = math.cosh(r) * math.cos(theta)  # Real part
                extra_dim[i*2 + 1] = math.sinh(r) * math.sin(theta)  # Imaginary part

            # Temporal folding (S¹ component)
            for i in range(self.extra_dim_depth):
                temporal_phase = (time_index / 10.0) % (2 * math.pi)
                fold_strength = self.calculate_fold_strength(context, time_index)

                # Apply folding transformation
                fold_factor_arg = -fold_strength * temporal_phase / (2 * math.pi)
                fold_factor = math.exp(max(fold_factor_arg, -700)) # Prevent underflow for exp

                extra_dim[i*2] *= fold_factor
                extra_dim[i*2 + 1] *= math.cos(temporal_phase + i * math.pi / 3)

            # Normalize to prevent overflow
            norm = np.linalg.norm(extra_dim)
            if norm > 1e-9 and norm < 1e6 : # Added check for very large norms before division
                extra_dim = extra_dim / norm
            elif norm >= 1e6: # If norm is excessively large, cap it
                 extra_dim = (extra_dim / norm) * 1e5 # Scale down to a large but manageable magnitude
            elif norm <= 1e-9 and norm !=0 : # Very small norm, might be unstable
                 extra_dim = np.random.normal(0, 0.01, self.extra_dim_depth * 2) # Reset to small random
            else: # Handles norm == 0 or other unexpected cases
                extra_dim = np.random.normal(0, 0.1, self.extra_dim_depth * 2)


        except (OverflowError, ValueError, ZeroDivisionError) as e:
            # Fallback to stable random vector
            # print(f"Warning: Exception in compute_extra_dimension: {e}")
            extra_dim = np.random.normal(0, 0.1, self.extra_dim_depth * 2)

        # Cache result
        if len(self.dimensional_cache) < 1000: # Limit cache size
            self.dimensional_cache[cache_key] = extra_dim

        return extra_dim

    def calculate_fold_strength(self, context, time_index):
        """
        Calculate dimensional folding strength based on retrocausal loop closure.
        """
        fold_strength = 0.0

        # Check for temporal loops
        try: # Add try-except for safety if context/cascade_memory items are not directly comparable
            for i, past_context in enumerate(self.cascade_memory):
                if past_context == context:
                    # Found a loop - calculate folding
                    loop_length = len(self.cascade_memory) - i
                    fold_strength += 1.0 / (1.0 + loop_length)
        except TypeError: # Handle cases where context types might mismatch
            pass


        # Add coherence-based folding
        if len(self.cascade_memory) > 2:
            recent_coherence = self.calculate_cascade_coherence(self.cascade_memory[-3:])
            fold_strength += recent_coherence * 0.5

        return min(2.0, fold_strength) # Cap fold strength

    def detect_retrocausal_integration(self, current_context, generation_history):
        """
        Detect when future states integrate into their own past.
        Returns True if integration detected, triggering extra-dimensional recomputation.
        """
        if len(generation_history) < 3:
            return False

        # Check for causal loops
        context_str = str(current_context) # Ensure string for L-semi-inner product
        for i, past_state in enumerate(generation_history[-10:]):  # Check last 10 states
            past_str = str(past_state) # Ensure string

            # Detect semantic similarity indicating temporal folding
            similarity = self._safe_l_semi_inner_product(context_str, past_str)
            if similarity > 0.8:  # High similarity indicates potential loop
                # Check if this creates a closed timelike curve
                # Ensure elements passed to calculate_cascade_coherence are suitable (e.g. iterables of strings)
                loop_coherence = self.calculate_cascade_coherence([past_state, current_context])
                if loop_coherence > self.cascade_threshold:
                    # Limit the number of stored retrocausal_loops to prevent memory issues
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
        """
        Recompute probabilities incorporating extra-dimensional manifold when
        retrocausal integration is detected.
        """
        # print(f"Retrocausal integration detected at t={time_index}") # Optional: can be verbose
        # print(f"Recomputing with extra-dimensional folding...")

        # Get anticipated future states (simplified prediction)
        future_states = []
        # Sort candidates by weight (probability) before picking top ones if not already sorted
        # Assuming candidate_weights are probabilities and sum to 1 (or are proportional scores)
        sorted_candidates = sorted(zip(candidate_weights, candidate_words), reverse=True)
        for weight, word in sorted_candidates[:5]: # Use top 5 candidates as future predictions
            future_states.append(word)


        # Compute extra-dimensional vector
        extra_dim_vector = self.compute_extra_dimension(context, time_index, future_states)

        # Store dimensional state
        # Ensure context is hashable for dictionary key
        current_context_tuple = tuple(context) if isinstance(context, list) else context
        if not isinstance(current_context_tuple, tuple): # Fallback if context isn't list or tuple
            current_context_tuple = tuple(str(current_context_tuple))

        # Limit the size of self.D to prevent memory exhaustion
        if len(self.D) > 2000: # Example limit
            # Simple FIFO eviction
            keys_to_remove = list(self.D.keys())[:len(self.D)-2000]
            for key_to_remove in keys_to_remove:
                del self.D[key_to_remove]
        self.D[(time_index, current_context_tuple)] = extra_dim_vector


        # Recompute weights using dimensional projection
        enhanced_weights = []
        for i, word in enumerate(candidate_words):
            base_weight = candidate_weights[i] if i < len(candidate_weights) else 1.0 # Should always be in range

            # Project word into extra-dimensional space
            word_hash = hash(word) % 1000000 # Using hash for simplicity
            # Simple word vector based on hash, ensure dimensionality matches extra_dim_vector
            word_vector_components = []
            for j in range(self.extra_dim_depth * 2):
                 angle = (word_hash / 1000000.0) * 2 * math.pi + (j * math.pi / (self.extra_dim_depth * 2))
                 word_vector_components.append(math.cos(angle)) # Example component generation
            word_vector = np.array(word_vector_components)


            # Calculate dimensional overlap (cosine similarity for stability)
            try:
                norm_extra_dim = np.linalg.norm(extra_dim_vector)
                norm_word_vec = np.linalg.norm(word_vector)

                if norm_extra_dim > 1e-9 and norm_word_vec > 1e-9: # Avoid division by zero
                    # Cosine similarity
                    dimensional_projection = np.dot(extra_dim_vector, word_vector) / (norm_extra_dim * norm_word_vec)
                else:
                    dimensional_projection = 0.0

                # Boost factor based on projection: (1 + |projection|) / 2 maps [-1,1] to [0,1], then scale
                # Or simpler: 1.0 + abs(projection) * scale
                dimensional_boost = 1.0 + abs(dimensional_projection) * 0.5 # Keep boost moderate
            except (ValueError, OverflowError, ZeroDivisionError):
                dimensional_boost = 1.0

            # Apply folding corrections
            fold_strength = self.calculate_fold_strength(current_context_tuple, time_index)
            fold_correction_arg = -fold_strength * 0.1
            fold_correction = math.exp(max(fold_correction_arg, -700)) # Gentle exponential damping


            # Temporal symmetry enhancement
            symmetry_factor = 1.0
            if self.retrocausal_loops: # Check if list is not empty
                recent_loop = self.retrocausal_loops[-1]
                symmetry_factor = 1.0 + recent_loop.get('coherence', 0.0) * 0.3 # Use .get for safety

            enhanced_weight = base_weight * dimensional_boost * fold_correction * symmetry_factor
            enhanced_weights.append(max(0.0001, enhanced_weight)) # Ensure positive weight, smaller min

        # print(f"Extra-dimensional vector norm: {np.linalg.norm(extra_dim_vector):.3f}") # Optional
        # print(f"Applied folding corrections with strength: {fold_strength:.3f}") # Optional

        return candidate_words, enhanced_weights

    def update_knowledge(self, K_prev, alpha, S_n, x, x_true):
        """
        Retrocausal knowledge update equation:
        K_n(x) = K_{n-1}(x) + α·S_n(x)·[A(S_n(x), I(x=x_true)) - K_{n-1}(x)]

        Args:
            K_prev (float): Prior knowledge K_{n-1}(x) about position x
            alpha (float): Learning rate α ∈ [0,1]
            S_n (float): Selection/query indicator S_n(x) (can be prob [0,1])
            x: Current position being evaluated
            x_true: True position

        Returns:
            float: Updated knowledge K_n(x)
        """
        # Indicator function I(x=x_true)
        I_x_true = 1.0 if x == x_true else 0.0

        # AND gate function A(S_n(x), I(x=x_true))
        # If S_n is a probability, this is a soft AND.
        A_gate = S_n * I_x_true

        # Knowledge update equation
        error_term = A_gate - K_prev
        K_n = K_prev + alpha * S_n * error_term

        return max(0.0, min(1.0, K_n))  # Clamp to [0,1]

    def calculate_cascade_coherence(self, context_sequence):
        """
        Calculate cascade coherence Π_cascade = ∏_n e^{-Γ_n t} ⟨Ψ_{n+1}|Ψ_n⟩²
        Fixed to prevent recursion issues.
        """
        if len(context_sequence) < 2:
            return 0.0

        coherence = 1.0
        # Limit iterations for performance and to focus on recent context
        max_iterations = min(3, len(context_sequence) - 1) # Reduced from 5 for speed

        for i in range(max_iterations):
            try:
                # Ensure contexts are strings for _safe_l_semi_inner_product
                # Handle cases where items in context_sequence might not be directly joinable
                # or might be single words (strings) themselves.
                c1 = context_sequence[i]
                c2 = context_sequence[i + 1]

                context1_str = ' '.join(map(str, c1)) if isinstance(c1, (list, tuple)) else str(c1)
                context2_str = ' '.join(map(str, c2)) if isinstance(c2, (list, tuple)) else str(c2)


                overlap = self._safe_l_semi_inner_product(context1_str, context2_str)

                # Decay factor e^{-Γ_n t}
                decay = math.exp(-0.1 * i)  # Γ_n = 0.1, t = i (time step from start of sequence slice)

                # Update coherence
                coherence *= decay * (overlap ** 2) # overlap is already [0,2], (overlap^2) could be up to 4.

                # Prevent coherence from becoming too small or too large if overlap is big
                coherence = max(0.0, min(1.0, coherence)) # Keep coherence bounded [0,1] after each step

                if coherence < 1e-9: # If effectively zero, no need to continue
                    coherence = 0.0
                    break

            except Exception as e:
                # print(f"Error in calculate_cascade_coherence: {e}") # Optional: log error
                coherence *= 0.5 # Reduce coherence on error and keep it bounded
                coherence = max(0.0, min(1.0, coherence))
                break
        return coherence # Already bounded


    def retrocausal_information_kernel(self, t0, future_states, omega=1.0):
        """
        Calculate retrocausal information kernel: ∫_{t₀}^∞ Ψ(τ) e^{-iω(τ-t₀)} dτ
        Approximated by a sum. Assumes future_states contains numeric or hashable values.
        """
        information = 0.0j # Initialize as complex number
        # Limit number of future states considered for performance
        max_states_to_sum = max(10, len(future_states)) # Reduced from 20

        for tau_idx in range(max_states_to_sum):
            # The 't0' parameter is not used in the loop if tau_idx directly represents (τ - t0)
            # Let's assume future_states[tau_idx] corresponds to Ψ(t_current + tau_idx * Δt)
            # and the phase is ω * (tau_idx * Δt). If Δt=1, then it's ω * tau_idx.
            # The original formula's (τ-t0) implies integration from t0.
            # Here, future_states is likely a list of *next possible words/states* from current point.
            # So tau_idx is the step into that future.

            current_future_state = future_states[tau_idx]
            try:
                # Attempt to get a numeric value for the state
                state_val_str = str(current_future_state)
                try:
                    state_val = float(state_val_str[:10].replace(',','')) # Handle potential commas, limit length
                except ValueError:
                    # If not directly float, hash it to get a pseudo-numeric value
                    state_val = float(hash(state_val_str) % 1000 - 500) / 500.0 # Map hash to [-1, 1] range

                phase_arg = -1j * omega * tau_idx # tau_idx acts as (τ - t0) or time step
                information += state_val * np.exp(phase_arg)
            except (ValueError, OverflowError, TypeError) as e:
                # print(f"Warning: Could not process future_state '{current_future_state}' in kernel: {e}")
                continue # Skip problematic state

        # Return magnitude, capped to prevent extreme values
        abs_info = abs(information)
        return min(abs_info, 1e3) # Cap the magnitude, reduced from 1e6


    @staticmethod
    def _calculate_distinct_permutations_log(word):
        """
        Calculates the natural logarithm of the number of distinct permutations of characters in a word.
        """
        if not word or len(word) == 0:
            return 0.0

        word = str(word)[:30] if len(word) > 30 else str(word) # Ensure string, limit length more strictly
        n = len(word)
        if n == 0: return 0.0

        counts = Counter(word)

        try:
            # Use math.lgamma for log(n!)
            log_n_factorial = math.lgamma(n + 1)
            log_denominator_sum = sum(math.lgamma(count + 1) for count in counts.values())
            log_permutations = log_n_factorial - log_denominator_sum
            # Clamp result to a reasonable range, e.g., related to information content.
            return max(0.0, min(20.0, log_permutations))  # Reduced upper clamp from 50
        except (ValueError, OverflowError):
            return 0.0 # Return neutral value on error

    def _safe_l_semi_inner_product(self, word1, word2):
        """
        Safe version of L-semi-inner product. Ensures inputs are strings.
        """
        if word1 is None or word2 is None: return 0.0 # Handle None inputs

        # Ensure inputs are strings and handle potential non-string types gracefully.
        try:
            s_word1 = str(word1)
            s_word2 = str(word2)
        except Exception: # Broad exception if str() conversion fails
            return 0.0


        if not s_word1 or not s_word2: # Check after conversion
            return 0.0

        # Check cache first
        cache_key = (s_word1[:20], s_word2[:20])  # Limit key size for cache efficiency
        if cache_key in self.similarity_cache:
            return self.similarity_cache[cache_key]

        # Limit string lengths for core computation
        s_word1_limited = s_word1[:50] # Reduced from 100
        s_word2_limited = s_word2[:50] # Reduced from 100

        result = self._compute_l_semi_inner_product(s_word1_limited, s_word2_limited)

        # Cache result (limit cache size)
        if len(self.similarity_cache) < 2000: # Cache more if memory allows, adjust based on usage
            self.similarity_cache[cache_key] = result
        # else: # Optional: Implement LRU or other eviction if cache grows too large
            # pass
        return result

    def _compute_l_semi_inner_product(self, word1, word2):
        """
        Core computation for L-semi-inner product. Assumes word1 and word2 are strings.
        """
        # Basic checks (should be guaranteed by caller but good for standalone use)
        if not word1 or not word2: return 0.0

        char_product = 0.0
        w1_len = len(word1)
        w2_len = len(word2)

        if w1_len == 0 or w2_len == 0: return 0.0

        # Character product with positional weighting
        for i, c1 in enumerate(word1):
            # Weight diminishes from center: 1 at center, 0 at ends.
            # Avoid division by zero if len is 1 (w1_len/2 can be 0.5, i can be 0).
            # Denominator should be non-zero. Max distance is len/2.
            pos_weight_1 = 1.0 - (abs(i - (w1_len -1) / 2.0) / (w1_len / 2.0 + 1e-9))
            pos_weight_1 = max(0, pos_weight_1) # Ensure non-negative
            for j, c2 in enumerate(word2):
                if c1 == c2:
                    pos_weight_2 = 1.0 - (abs(j - (w2_len-1) / 2.0) / (w2_len / 2.0 + 1e-9))
                    pos_weight_2 = max(0, pos_weight_2)
                    char_product += pos_weight_1 * pos_weight_2

        # Normalize char_product: divide by a measure of max possible score.
        # Max score is if all chars match and are central.
        # Simplification: normalize by product of lengths or min length.
        # A rough upper bound for char_product is min(w1_len, w2_len) if weights are often high.
        char_product_normalized = char_product / ( (w1_len * w2_len)**0.5 + 1e-9) # Geometric mean of lengths
        char_product_normalized = min(char_product_normalized, 1.0) # Clamp


        length_ratio = min(w1_len, w2_len) / (max(w1_len, w2_len) + 1e-9)

        # Phonetic patterns (Vowel/Consonant)
        vowels = set('aeiouAEIOU')
        # Keep only alphabetic characters for pattern matching
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


        # Prefix/suffix matching (exact char match)
        prefix_len = 0
        for i in range(min(w1_len, w2_len, 5)):  # Shorter limit for prefix/suffix
            if word1[i] == word2[i]: prefix_len += 1
            else: break
        suffix_len = 0
        for i in range(1, min(w1_len, w2_len, 5) + 1):
            if word1[-i] == word2[-i]: suffix_len += 1
            else: break
        # Normalize by max possible matched length (2 * limit)
        prefix_suffix_factor = (prefix_len + suffix_len) / (float(min(min(w1_len,w2_len),5) * 2) + 1e-9)
        prefix_suffix_factor = min(prefix_suffix_factor, 1.0) # Clamp


        # Permutation similarity (log of distinct permutations)
        perm_key1 = word1[:20] # Use original word slice for perm cache
        perm_key2 = word2[:20]

        # Retrieve from cache or compute, with cache size limit
        log_perm1_val, log_perm2_val = 0.0, 0.0
        if perm_key1 in self.permutation_cache: log_perm1_val = self.permutation_cache[perm_key1]
        elif len(self.permutation_cache) < 2000: self.permutation_cache[perm_key1] = log_perm1_val = self._calculate_distinct_permutations_log(perm_key1)
        else: log_perm1_val = self._calculate_distinct_permutations_log(perm_key1) # Compute if cache full

        if perm_key2 in self.permutation_cache: log_perm2_val = self.permutation_cache[perm_key2]
        elif len(self.permutation_cache) < 2000: self.permutation_cache[perm_key2] = log_perm2_val = self._calculate_distinct_permutations_log(perm_key2)
        else: log_perm2_val = self._calculate_distinct_permutations_log(perm_key2)

        diff_log_perm = abs(log_perm1_val - log_perm2_val)
        # Denominator for normalization: max log_perm or a constant if both are small
        denominator_perm = max(log_perm1_val, log_perm2_val, 1.0) # Use 1.0 as min denominator
        normalized_diff_perm = diff_log_perm / denominator_perm if denominator_perm > 1e-9 else 0.0
        permutation_similarity_factor = 1.0 - min(normalized_diff_perm, 1.0)


        # Retrocausal enhancement (simplified, from cascade memory length)
        cascade_boost = 1.0
        if self.cascade_memory: # Check if not empty
            recent_coherence_proxy = min(0.5, len(self.cascade_memory) * 0.05) # More gentle proxy
            cascade_boost = 1.0 + recent_coherence_proxy * 0.2


        # Combine factors with adjusted weights
        # Ensure weights sum to approx 1.0 before cascade_boost
        # Weights: char_prod:0.3, len_ratio:0.15, phonetic:0.2, prefix_suffix:0.2, perm_sim:0.15 (Sum = 1.0)
        l_inner_product = (
            0.30 * char_product_normalized +
            0.15 * length_ratio +
            0.20 * phonetic_factor +
            0.20 * prefix_suffix_factor +
            0.15 * permutation_similarity_factor
        )
        l_inner_product *= cascade_boost # Apply overall boost

        # Clamp final result to a predictable range, e.g., [0, 2.0]
        return max(0.0, min(2.0, l_inner_product))

    def train(self, t):
        """
        Enhanced training with retrocausal knowledge updates.
        """
        if not isinstance(t, str) or not t:
            raise TypeError("Training data must be a non-empty string.")

        words = t.split()
        num_words = len(words)

        if num_words <= self.n:
            # If training data is too short, either raise error or try to handle gracefully (e.g. by skipping training)
            print(f"Warning: Training data has only {num_words} words, less than or equal to n-gram size {self.n}. Skipping training for this data.")
            return


        print(f"Training retrocausal model on {num_words} words with n={self.n}, α={self.alpha}...")
        # print(f"Applying T-symmetric cascade dynamics and symbolic count adjustments...") # Optional verbosity
        # print(f"Extra-dimensional depth: {self.extra_dim_depth}")

        overall_word_freqs = Counter(words)
        total_word_count = float(num_words) # Should be num_words, not total_word_count from before

        temp_m = defaultdict(lambda: defaultdict(float))
        temp_s = set() # Store potential sentence starts
        self.all_words.update(words) # Add all words from current training text to the global set

        # Epoch management: self.epochs tracks total number of n-grams processed across all training calls.
        # current_run_epoch_offset is where this training run starts in the global epoch count.
        current_run_epoch_offset = self.epochs
        self.epochs += (num_words - self.n) # Increment global epoch count

        # Bounded context sequence for local coherence calculation and integration detection
        context_sequence_for_training = []
        MAX_CONTEXT_SEQ_LEN = 15 # Max length for this local sequence

        # Determine sample_words for K updates. Use most frequent ones if possible.
        if len(self.all_words) > 500:
            # Get 500 most common words from current training data + existing all_words
            # This is a simplification; true "most common" would involve global counts.
            # For now, use a random sample of all known words if too many.
            sample_words_for_k_update = random.sample(list(self.all_words), 500)
        else:
            sample_words_for_k_update = list(self.all_words)


        for iter_idx, i in enumerate(tqdm(range(num_words - self.n), desc="Retrocausal Training")): # Removed disable=True
            # epoch_for_k is the specific "time step" or "event index" for K matrix entries.
            epoch_for_k = current_run_epoch_offset + iter_idx

            current_ngram = tuple(words[i:i+self.n])
            next_word_actual = words[i+self.n]

            # Update and maintain bounded context_sequence_for_training
            context_sequence_for_training.append(current_ngram)
            if len(context_sequence_for_training) > MAX_CONTEXT_SEQ_LEN:
                context_sequence_for_training.pop(0)


            # Calculate selection probability S_n for K update
            selection_prob_S_n = 0.5 # Default
            if len(context_sequence_for_training) >= 2:
                # Use a short recent slice for coherence calc
                coherence_input_slice = context_sequence_for_training[-min(3, len(context_sequence_for_training)):]
                cascade_coherence = self.calculate_cascade_coherence(coherence_input_slice)
                # selection_prob should be higher if coherence is high.
                selection_prob_S_n = max(0.1, min(1.0, 0.5 + cascade_coherence * 0.5)) # Scale coherence effect
            self.S[(epoch_for_k, current_ngram)] = selection_prob_S_n # Store S_n


            # Retrocausal integration detection during training
            if self.detect_retrocausal_integration(current_ngram, context_sequence_for_training):
                # print(f"Integration detected at training step {iter_idx}, context: {current_ngram}") # Optional
                if len(self.integration_history) < 200: # Limit history size
                    self.integration_history.append({
                        'epoch': epoch_for_k, 'context': current_ngram, 'next_word': next_word_actual
                    })


            # Retrocausal knowledge (K) update for a sample of words
            # This is computationally intensive.
            for candidate_word_for_k in sample_words_for_k_update:
                # K_prev: knowledge from the previous step/state for this (ngram, candidate_word) pair.
                # Key for K_prev: (epoch_for_k - 1, current_ngram, candidate_word_for_k)
                # If epoch_for_k is 0 (first step of first training), no K_prev.
                k_prev_val = 0.5 # Default if no prior knowledge
                if epoch_for_k > 0 : # Only look for K_prev if not the very first global step
                    k_prev_val = self.K.get((epoch_for_k - 1, current_ngram, candidate_word_for_k), 0.5)

                k_new_val = self.update_knowledge(
                    k_prev_val, self.alpha, selection_prob_S_n,
                    candidate_word_for_k, # x (the word being evaluated)
                    next_word_actual      # x_true (the actual next word)
                )
                # Limit K matrix size to prevent memory overflow
                if len(self.K) < 50000: # Example limit, adjust based on memory
                     self.K[(epoch_for_k, current_ngram, candidate_word_for_k)] = k_new_val
                # else: could implement LRU or periodic pruning for K


            # Traditional Markov transition count update, enhanced with symbolic factors
            base_increment = 1.0
            # Global frequency factor for next_word_actual
            norm_freq_next_word = overall_word_freqs.get(next_word_actual, 0) / total_word_count
            global_freq_factor = 1.0 + math.log1p(norm_freq_next_word * 100) * 0.2 # Reduced influence


            len_next_word = len(next_word_actual)
            length_factor = 1.0 + math.log1p(len_next_word) * 0.05 # Reduced influence


            # L-semi-inner product factor between context words and next_word_actual
            l_inner_prod_accum = 1.0
            # Consider only a few words from context for performance
            for context_word_idx, word_in_context in enumerate(current_ngram[:max(1, self.n//2)]):
                similarity = self._safe_l_semi_inner_product(str(word_in_context), next_word_actual)
                l_inner_prod_accum *= (1.0 + similarity * 0.1) # Reduced influence
            l_inner_prod_accum = min(l_inner_prod_accum, 2.0) # Cap this factor


            # Retrocausal knowledge factor from K for the *actual* next word
            # Use the K value just computed (or default if K was pruned)
            k_for_actual_next = self.K.get((epoch_for_k, current_ngram, next_word_actual), 0.5)
            retrocausal_factor_k = 1.0 + (k_for_actual_next - 0.5) * 0.4 # Centered, reduced influence


            symbolic_enhancement_factor = (global_freq_factor * length_factor *
                                           l_inner_prod_accum * retrocausal_factor_k)
            symbolic_enhancement_factor = max(0.2, min(symbolic_enhancement_factor, 5.0)) # Bound


            adjusted_increment_val = base_increment * symbolic_enhancement_factor
            # Noise proportional to increment, but small
            noise_val = random.uniform(-0.02, 0.02) * adjusted_increment_val
            final_increment_val = max(0.001, adjusted_increment_val + noise_val)


            # Identify sentence starts
            # An n-gram is a sentence start if it's the first in text,
            # or if the word preceding it ends with a sentence-end character.
            if i == 0 or (i > 0 and words[i-1] and any(c in self.SENTENCE_END_CHARS for c in words[i-1][-1:])):
                temp_s.add(current_ngram)

            temp_m[current_ngram][next_word_actual] += final_increment_val


        # Update main model structures (m, s, cascade_memory)
        self.m = {gram: dict(counts) for gram, counts in temp_m.items() if counts} # Store only if has transitions
        # Filter self.s: ensure all start grams are actually in self.m (have transitions)
        self.s = list(filter(lambda sg: sg in self.m, temp_s))


        # Fallback if self.s is empty but self.m has data
        if not self.s and self.m:
            num_starts_to_sample = min(max(1, int(len(self.m) * 0.05)), 50) # Sample 5% or at least 1, up to 50
            # Ensure k is not greater than the population size
            if num_starts_to_sample > len(self.m): # check if sampling more than available
                num_starts_to_sample = len(self.m)
            if num_starts_to_sample > 0: # ensure there's something to sample
                self.s = random.sample(list(self.m.keys()), k=num_starts_to_sample)


        # Update cascade_memory with the tail of the sequence from this training run
        self.cascade_memory = context_sequence_for_training[-min(10, len(context_sequence_for_training)):] if context_sequence_for_training else []


        print(f"Retrocausal training complete. Model has {len(self.m)} contexts, {len(self.s)} valid starts.")
        # print(f"Knowledge matrix K contains {len(self.K)} entries.") # Optional
        # print(f"Integration events: {len(self.integration_history)}, Loops: {len(self.retrocausal_loops)}") # Optional

    def _symbolic_probability(self, context_tuple, options_dict):
        """
        Calculates "symbolic probabilities" (more like scores) for candidate words.
        Args:
            context_tuple (tuple): The current n-gram context.
            options_dict (dict): Dictionary of {next_word: base_frequency/score}.
        Returns:
            tuple: (list_of_candidate_words, list_of_corresponding_scores)
        """
        # Ensure context is a tuple (should be guaranteed by caller like generate)
        # If not, try to convert, but this function expects a tuple.
        if not isinstance(context_tuple, tuple):
            try:
                context_tuple = tuple(str(c) for c in context_tuple) if hasattr(context_tuple, '__iter__') and not isinstance(context_tuple, str) else tuple([str(context_tuple)])
            except TypeError:
                context_tuple = tuple() # Fallback to empty tuple if conversion fails

        candidate_words_list = list(options_dict.keys())
        base_scores_list = list(options_dict.values())

        if not candidate_words_list:
            return [], []

        enhanced_scores = []
        # Knowledge epoch for retrieval: use the "latest" available or a default.
        # self.epochs is total steps. K keys are (step_index, context, word).
        # A robust way would be to search for latest K entry for (context, word) or average.
        # Simpler: use a recent epoch index if K is large, or default.
        # This part remains heuristic without a clear time concept during generation matching training epochs.
        # Let's use a K value from a "recent representative epoch" if K is populated from training.
        # Fallback to 0.5 if no specific K found.
        # Assuming self.epochs is the total number of steps processed in training.
        # If K is keyed by (training_step_index, context, word), then for generation,
        # we might not have a direct matching "epoch".
        # For now, try to get K using a "typical" recent epoch from training, or default to 0.5.
        # This makes K's influence during generation somewhat averaged/defaulted if specific epoch alignment is missing.
        k_lookup_epoch = self.epochs -1 if self.epochs > 0 else 0 # Example "recent" epoch index

        for i, word_cand in enumerate(candidate_words_list):
            current_base_score = base_scores_list[i]
            retrocausal_k_weight = 0.5  # Default knowledge influence

            # Attempt to retrieve K. This is a simplification.
            # In a full system, K retrieval during generation might be more complex
            # (e.g. time-decayed, averaged over relevant past K entries for this context-word pair)
            if self.K:
                 # Try finding K for this context-word pair at the k_lookup_epoch.
                 # If context_tuple is not in K for that epoch, this will default.
                 k_val = self.K.get((k_lookup_epoch, context_tuple, word_cand))
                 if k_val is not None:
                     retrocausal_k_weight = k_val
                 # else: # If specific epoch key not found, could try other strategies or use default 0.5
                      # pass

            # Cascade boost from recent generation memory (if any)
            cascade_mem_boost = 1.0
            if self.cascade_memory: # If there's history from current generation
                cascade_mem_boost = 1.0 + min(0.2, len(self.cascade_memory) * 0.02) # Gentle boost


            # Combine: base_score * (K_influence) * (cascade_influence)
            # K influence: 1.0 is neutral if k_weight=0.5. Boost if >0.5, reduce if <0.5.
            k_factor = 1.0 + (retrocausal_k_weight - 0.5) * 0.4 # Scaled and centered

            final_score_for_word = current_base_score * k_factor * cascade_mem_boost
            enhanced_scores.append(max(0.0001, final_score_for_word))


        # Further refine scores based on L-semi-inner-product with context
        # This is the "tensor" like calculation part.
        final_refined_scores = []
        if not context_tuple: # No context, use enhanced_scores directly
            final_refined_scores = enhanced_scores
        else:
            for i, word_cand in enumerate(candidate_words_list):
                score_after_k_cascade = enhanced_scores[i]

                # Similarity factor from current context to this candidate word
                context_sim_factor_accum = 1.0
                # Use first few elements of context_tuple for this similarity calculation
                for context_word_idx, word_from_context in enumerate(context_tuple[:max(1, self.n //2)]): # Limit context use
                    l_sim = self._safe_l_semi_inner_product(str(word_from_context), word_cand)
                    # Overlap part was complex; L-sim itself should capture good textual similarity.
                    # Boost based on L-sim: 1.0 + L_sim * weight
                    context_sim_factor_accum *= (1.0 + l_sim * 0.15) # Modest weight for L-sim with context
                context_sim_factor_accum = min(context_sim_factor_accum, 3.0) # Cap total similarity boost

                score_after_context_sim = score_after_k_cascade * context_sim_factor_accum
                final_refined_scores.append(max(0.00001, score_after_context_sim)) # Ensure positive


        # Filter and select top candidates based on these refined scores (not yet probabilities)
        # This part replaces the mean_freq filtering logic with sorting by score

        if not final_refined_scores: return [], []

        # Pair scores with words, sort, and take top N (e.g., 50-100)
        scored_candidates = sorted(zip(final_refined_scores, candidate_words_list), reverse=True)

        # Limit number of candidates for final probability distribution
        # This also implicitly handles cases where many scores are tiny.
        num_final_candidates = min(len(scored_candidates), 75) # Take up to 75 best candidates

        top_candidates_words = [word for score, word in scored_candidates[:num_final_candidates]]
        top_candidates_scores = [score for score, word in scored_candidates[:num_final_candidates]]

        if not top_candidates_words:
            return [],[] # Should not happen if options_dict was non-empty

        # These are scores, not probabilities yet. Normalization happens in generate().
        return top_candidates_words, top_candidates_scores


    def generate(self, length=50, start_context_str=None):
        """
        Generates text using the retrocausal Markov model.
        start_context_str: Optional string to seed generation.
        """
        if not self.m:
            # print("Model not trained or no transitions available. Please train first.")
            return "Model not trained." # Simple message

        # Determine starting n-gram
        current_gram_tuple = None
        generated_words_list = []

        if start_context_str:
            sc_words = start_context_str.split()
            if len(sc_words) >= self.n:
                current_gram_tuple = tuple(sc_words[-self.n:])
                if current_gram_tuple not in self.m:
                    # print(f"Warning: Provided start context '{start_context_str}' (n-gram {current_gram_tuple}) not found in model. Trying a random start.")
                    current_gram_tuple = None # Fallback to random start
                else:
                    generated_words_list = list(sc_words) # Start generated text with the full seed if valid
            # else:
                # print(f"Warning: Provided start context '{start_context_str}' is shorter than n-gram size {self.n}. Trying a random start.")

        if current_gram_tuple is None: # If no valid start_context or it was too short/invalid
            if not self.s: # No sentence starts defined at all
                 # Fallback: pick any key from self.m if s is empty
                 if self.m:
                     # Ensure k is not greater than population if self.m is small
                     sample_k = min(1, len(self.m))
                     if sample_k > 0:
                        self.s = random.sample(list(self.m.keys()), k=sample_k)
                 if not self.s: return "Cannot start generation: no valid start points in the model."

            current_gram_tuple = random.choice(self.s)
            generated_words_list = list(current_gram_tuple)


        # Generation history for retrocausal integration detection (stores n-grams)
        # This is distinct from self.cascade_memory which is for training/L-sim boost
        local_generation_history_ngrams = [current_gram_tuple]
        MAX_GEN_HIST_LEN = 15 # Keep this local history bounded

        # Clear or manage generation-specific state like self.retrocausal_loops
        self.retrocausal_loops = [] # Reset for this generation run
        
        # Adjust loop iterations based on how many words are already in generated_words_list
        # The goal is for the *final* length of generated_words_list to be 'length'.
        # If generated_words_list starts with 'k' words, we need 'length - k' more words.
        
        num_words_to_generate = length - len(generated_words_list)

        for _ in range(num_words_to_generate): # Iterate for the number of additional words needed
            if len(generated_words_list) >= length: break # Safety break, though loop count should handle it

            if current_gram_tuple not in self.m or not self.m[current_gram_tuple]:
                # Dead end: try to restart from a random sentence start
                # print(f"Stuck at context {current_gram_tuple}. Attempting restart.")
                if not self.s: break # Cannot restart
                current_gram_tuple = random.choice(self.s)
                if current_gram_tuple not in self.m or not self.m[current_gram_tuple]: break # Still stuck, end.
                # When restarting, decide if the restart n-gram itself should be appended.
                # Current logic: the loop will pick the *next* word based on this new current_gram_tuple.
                # If you want the restart n-gram to be part of output immediately:
                # generated_words_list.extend(list(current_gram_tuple))
                # if len(generated_words_list) >= length: break
                # And update history accordingly. For now, just reset context and continue.
                local_generation_history_ngrams.append(current_gram_tuple) # Add new context to history
                if len(local_generation_history_ngrams) > MAX_GEN_HIST_LEN: local_generation_history_ngrams.pop(0)
                # The loop continues to pick a word based on this new current_gram_tuple.


            # Get candidate words and their raw scores from symbolic_probability
            options_for_current_gram = self.m[current_gram_tuple]
            candidate_words, candidate_raw_scores = self._symbolic_probability(current_gram_tuple, options_for_current_gram)

            if not candidate_words: # No candidates from symbolic_probability
                # print(f"No candidates from symbolic_probability for {current_gram_tuple}. Attempting restart.")
                if not self.s: break
                current_gram_tuple = random.choice(self.s)
                local_generation_history_ngrams.append(current_gram_tuple)
                if len(local_generation_history_ngrams) > MAX_GEN_HIST_LEN: local_generation_history_ngrams.pop(0)
                continue # Retry word choice with new context


            # Retrocausal integration check and recomputation
            time_idx_generation = len(generated_words_list) # "Time" in generation sequence
            if self.detect_retrocausal_integration(current_gram_tuple, local_generation_history_ngrams):
                # print(f"Integration detected in generation at t={time_idx_generation}") # Optional
                _, final_scores_for_choice = self.recompute_with_extra_dimension(
                    current_gram_tuple, candidate_words, candidate_raw_scores, time_idx_generation
                )
            else:
                final_scores_for_choice = candidate_raw_scores


            # Normalize final scores to probabilities for random.choices
            sum_final_scores = sum(final_scores_for_choice)
            if sum_final_scores > 1e-9:
                probabilities_for_choice = [s / sum_final_scores for s in final_scores_for_choice]
            else: # Fallback: if all scores are zero/tiny, use uniform probability over candidates
                if candidate_words: # Ensure there are candidates
                    prob = 1.0 / len(candidate_words)
                    probabilities_for_choice = [prob] * len(candidate_words)
                else: # Should not happen if checks above are effective
                    # print("Critical: No candidates and scores are zero. Ending generation.")
                    break


            # Choose next word based on probabilities
            if not candidate_words : break # Safety break if it somehow gets here with no candidates

            try:
                next_chosen_word = random.choices(candidate_words, weights=probabilities_for_choice, k=1)[0]
            except ValueError as e: # e.g. if weights don't sum to positive, or empty lists
                # print(f"Error in random.choices: {e}. Candidates: {len(candidate_words)}, Weights: {probabilities_for_choice[:5]}. Picking random.")
                if candidate_words: next_chosen_word = random.choice(candidate_words) # Fallback
                else: break # Cannot proceed

            generated_words_list.append(next_chosen_word)
            current_gram_tuple = tuple(generated_words_list[-self.n:]) # Update context

            # Update local generation history
            local_generation_history_ngrams.append(current_gram_tuple)
            if len(local_generation_history_ngrams) > MAX_GEN_HIST_LEN:
                local_generation_history_ngrams.pop(0)

            # Optional: Early stop if sentence end character and decent length
            # if any(c in self.SENTENCE_END_CHARS for c in next_chosen_word[-1:]) and len(generated_words_list) > length * 0.6:
            #     break

        return " ".join(generated_words_list)


if __name__ == "__main__":
    print("Initializing Retrocausal Symbolic Markov Model...")
    # Parameters can be tuned:
    # n: n-gram size (context window)
    # alpha: learning rate for K updates
    # cascade_threshold: for detecting retrocausal integration
    # extra_dim_depth: for extra-dimensional computation
    model = RetrocausalSymbolicMarkov(n=2, alpha=0.1, cascade_threshold=0.75, extra_dim_depth=2)

    # Simple sample training data
    try:
        filename = input("Enter filename (or press Enter for default training data): ")
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
                # Simple text cleaning: lowercasing and splitting by sentences (periods)
                # then joining sentences to ensure words are space-separated properly.
                # More sophisticated cleaning might be needed for real-world text.
                raw_text = file.read().lower()
                # Replace multiple spaces/newlines with single space
                raw_text = ' '.join(raw_text.split())
                # Simple sentence splitting and rejoining for now, might need better tokenization.
                # The original ' '.join(file.read().lower().split(".")) could lose words if a sentence didn't end with '.'
                # or create double spaces if user had "word. word".
                # A simple split by space is often safer for Markov chains if sentence structure isn't strictly used.
                training_text = raw_text # Using the cleaned raw_text

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

    print("\nInteractive Generation Mode (type 'quit' or 'exit' to stop):")
    while True:
        seed = input("USER: ")
        if seed.lower() in ['quit', 'exit']:
            break
        if not seed.strip(): # If user just presses enter, maybe provide a default prompt or random start
            print("AI: (No input provided, try seeding with some text or type 'quit')")
            continue

        # Determine generation length. Can be fixed or adaptive.
        # For very short seeds, might want longer generation.
        gen_length = 250 # Default length
      

        generated_text = model.generate(length=gen_length, start_context_str=seed)
        print(f"AI: {generated_text}")

    print("\nModel demonstration finished.")
