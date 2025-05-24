import random
import math
import sys
from collections import defaultdict, Counter
from tqdm import tqdm
import numpy as np
import torch

class RetrocausalSymbolicMarkov:
    """
    Markov chain text generator using symbolic probability distribution ⊆⊗∃·Λρ∑ω·Σø²
    with retrocausal knowledge update equation K_n(x) = K_{n-1}(x) + α·S_n(x)·[A(S_n(x), I(x=x_true)) - K_{n-1}(x)]
    and T-symmetric cascade dynamics ∂/∂t selecting futures that ∫ into their own past.
    """

    # Static configuration variables
    SENTENCE_END_CHARS = set('.!?')

    def __init__(self, n=2, alpha=0.3, cascade_threshold=0.7):
        """
        Initializes the retrocausal Markov chain model.
        Args:
            n (int): Size of n-gram context
            alpha (float): Learning rate for retrocausal knowledge updates
            cascade_threshold (float): Threshold for cascade formation
        """
        if not isinstance(n, int) or n < 1:
            raise ValueError("n-gram size 'n' must be a positive integer.")
        
        self.n = n
        self.alpha = alpha
        self.cascade_threshold = cascade_threshold
        
        # Traditional Markov structures
        self.m = {}  # Transitions: {context_tuple: {next_word: adjusted_frequency}}
        self.s = []  # Sentence starting n-grams
        self.all_words = set()
        
        # Retrocausal knowledge structures
        self.K = {}  # Knowledge matrix: {(epoch, context, word): knowledge_value}
        self.S = {}  # Selection gates: {(epoch, context): selection_probability}
        self.cascade_memory = []  # Cascade history for temporal coherence
        self.epochs = 0
        
        # Cache for expensive calculations
        self.similarity_cache = {}
        self.permutation_cache = {}

    def update_knowledge(self, K_prev, alpha, S_n, x, x_true):
        """
        Retrocausal knowledge update equation: 
        K_n(x) = K_{n-1}(x) + α·S_n(x)·[A(S_n(x), I(x=x_true)) - K_{n-1}(x)]
        
        Args:
            K_prev (float): Prior knowledge K_{n-1}(x) about position x
            alpha (float): Learning rate α ∈ [0,1]
            S_n (float): Selection/query indicator S_n(x) ∈ {0,1}
            x: Current position being evaluated
            x_true: True position
            
        Returns:
            float: Updated knowledge K_n(x)
        """
        # Indicator function I(x=x_true)
        I_x_true = 1.0 if x == x_true else 0.0
        
        # AND gate function A(S_n(x), I(x=x_true))
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
        max_iterations = min(5, len(context_sequence) - 1)  # Limit iterations
        
        for i in range(max_iterations):
            try:
                # Calculate overlap between consecutive contexts
                context1_str = ' '.join(str(w) for w in context_sequence[i]) if isinstance(context_sequence[i], (list, tuple)) else str(context_sequence[i])
                context2_str = ' '.join(str(w) for w in context_sequence[i + 1]) if isinstance(context_sequence[i + 1], (list, tuple)) else str(context_sequence[i + 1])
                
                overlap = self._safe_l_semi_inner_product(context1_str, context2_str)
                
                # Decay factor e^{-Γ_n t}
                decay = math.exp(-0.1 * i)  # Γ_n = 0.1
                
                # Update coherence
                coherence *= decay * (overlap ** 2)
                
                # Prevent coherence from becoming too small
                if coherence < 1e-10:
                    break
                    
            except Exception as e:
                # Fallback on calculation error
                coherence *= 0.5
                break
                
        return max(0.0, min(1.0, coherence))

    def retrocausal_information_kernel(self, t0, future_states, omega=1.0):
        """
        Calculate retrocausal information kernel: ∫_{t₀}^∞ Ψ(τ) e^{-iω(τ-t₀)} dτ
        """
        information = 0.0
        max_states = min(20, len(future_states))  # Limit to prevent overflow
        
        for tau in range(max_states):
            if tau >= t0 and tau < len(future_states):
                try:
                    phase = -1j * omega * (tau - t0)
                    state_val = float(future_states[tau]) if hasattr(future_states[tau], '__float__') else 1.0
                    information += state_val * np.exp(phase)
                except (ValueError, OverflowError):
                    continue
                    
        return abs(information) if abs(information) < 1e10 else 1.0

    @staticmethod
    def _calculate_distinct_permutations_log(word):
        """
        Calculates the natural logarithm of the number of distinct permutations of characters in a word.
        """
        if not word or len(word) == 0:
            return 0.0
        
        # Limit word length to prevent overflow
        word = word[:50] if len(word) > 50 else word
        n = len(word)

        counts = Counter(word)
        
        try:
            log_n_factorial = math.lgamma(n + 1)
            log_denominator_sum = sum(math.lgamma(count + 1) for count in counts.values())
            log_permutations = log_n_factorial - log_denominator_sum
            return max(0.0, min(50.0, log_permutations))  # Clamp result
        except (ValueError, OverflowError):
            return 0.0

    def _safe_l_semi_inner_product(self, word1, word2):
        """
        Safe version of L-semi-inner product that prevents recursion.
        """
        if not word1 or not word2:
            return 0.0
            
        # Check cache first
        cache_key = (word1[:20], word2[:20])  # Limit key size
        if cache_key in self.similarity_cache:
            return self.similarity_cache[cache_key]
        
        # Limit string lengths to prevent excessive computation
        word1 = str(word1)[:100]
        word2 = str(word2)[:100]
        
        result = self._compute_l_semi_inner_product(word1, word2)
        
        # Cache result (limit cache size)
        if len(self.similarity_cache) < 1000:
            self.similarity_cache[cache_key] = result
            
        return result

    def _compute_l_semi_inner_product(self, word1, word2):
        """
        Core computation for L-semi-inner product without recursion.
        """
        if not word1 or not word2:
            return 0.0
            
        # Traditional similarity components
        char_product = 0.0
        w1_len = len(word1)
        w2_len = len(word2)
        
        # Limit computation for very long strings
        if w1_len > 50 or w2_len > 50:
            word1 = word1[:50]
            word2 = word2[:50]
            w1_len = len(word1)
            w2_len = len(word2)
        
        for i, c1 in enumerate(word1):
            pos_weight_1 = 1.0 - (abs(i - w1_len/2) / (w1_len + 0.001))
            for j, c2 in enumerate(word2):
                if c1 == c2:
                    pos_weight_2 = 1.0 - (abs(j - w2_len/2) / (w2_len + 0.001))
                    char_product += pos_weight_1 * pos_weight_2
        
        length_ratio = min(w1_len, w2_len) / max(w1_len, w2_len) if max(w1_len, w2_len) > 0 else 0.0
        
        # Phonetic patterns
        vowels = set('aeiou')
        w1_pattern = ''.join('V' if c.lower() in vowels else 'C' for c in word1)
        w2_pattern = ''.join('V' if c.lower() in vowels else 'C' for c in word2)
        
        pattern_length = min(3, min(len(w1_pattern), len(w2_pattern)))
        pattern_match_start = 0
        pattern_match_end = 0
        
        if pattern_length > 0:
            pattern_match_start = sum(1 for i in range(pattern_length) 
                                      if i < len(w1_pattern) and i < len(w2_pattern) and w1_pattern[i] == w2_pattern[i])
            pattern_match_end = sum(1 for i in range(1, pattern_length + 1)
                                    if i <= len(w1_pattern) and i <= len(w2_pattern) 
                                    and w1_pattern[-i] == w2_pattern[-i])
            
        phonetic_factor = (pattern_match_start + pattern_match_end) / (2 * pattern_length) if pattern_length > 0 else 0.0
        
        # Prefix/suffix matching
        prefix_len = 0
        for i in range(min(w1_len, w2_len, 10)):  # Limit to first 10 chars
            if word1[i] == word2[i]:
                prefix_len += 1
            else:
                break
                
        suffix_len = 0
        for i in range(1, min(w1_len, w2_len, 10) + 1):  # Limit to last 10 chars
            if word1[-i] == word2[-i]:
                suffix_len += 1
            else:
                break
                
        prefix_suffix_factor = (prefix_len + suffix_len) / (w1_len + w2_len) if (w1_len + w2_len) > 0 else 0.0

        # Permutation similarity (cached)
        perm_key1 = word1[:20]
        perm_key2 = word2[:20]
        
        if perm_key1 not in self.permutation_cache:
            self.permutation_cache[perm_key1] = self._calculate_distinct_permutations_log(perm_key1)
        if perm_key2 not in self.permutation_cache:
            self.permutation_cache[perm_key2] = self._calculate_distinct_permutations_log(perm_key2)
            
        log_perm1 = self.permutation_cache[perm_key1]
        log_perm2 = self.permutation_cache[perm_key2]
        
        diff_log_perm = abs(log_perm1 - log_perm2)
        denominator_perm = max(log_perm1, log_perm2, 1.0)
        normalized_diff_perm = diff_log_perm / denominator_perm if denominator_perm > 0 else 0.0
        permutation_similarity_factor = 1.0 - min(normalized_diff_perm, 1.0)

        # Retrocausal enhancement (simplified to prevent recursion)
        cascade_boost = 1.0
        if len(self.cascade_memory) > 0:
            # Simple coherence estimate without recursion
            recent_coherence = min(0.5, len(self.cascade_memory) * 0.1)
            cascade_boost = 1.0 + recent_coherence * 0.2

        # Combine all factors
        l_inner_product = (
            0.30 * char_product + 
            0.15 * length_ratio + 
            0.15 * phonetic_factor + 
            0.15 * prefix_suffix_factor +
            0.10 * permutation_similarity_factor +
            0.15 * cascade_boost
        )
        
        return min(2.0, l_inner_product * 2.0)  # Clamp result

    def train(self, t):
        """
        Enhanced training with retrocausal knowledge updates.
        """
        if not isinstance(t, str) or not t:
            raise TypeError("Training data must be a non-empty string.")

        words = t.split()
        num_words = len(words)

        if num_words <= self.n:
            raise ValueError(f"Training data has only {num_words} words, need more than n-gram size {self.n}.")

        print(f"Training retrocausal model on {num_words} words with n={self.n}, α={self.alpha}...")
        print(f"Applying T-symmetric cascade dynamics and symbolic count adjustments...")

        overall_word_freqs = Counter(words)
        total_word_count = float(num_words)

        temp_m = defaultdict(lambda: defaultdict(float))
        temp_s = set()
        self.all_words = set(words)
        
        # Initialize retrocausal structures
        self.epochs = num_words - self.n
        context_sequence = []
        
        # Limit retrocausal processing to manageable subset
        sample_words = list(self.all_words)[:500]  # Limit to 500 most common words
        
        for epoch, i in enumerate(tqdm(range(num_words - self.n), desc="Retrocausal Training")):
            g = tuple(words[i:i+self.n])
            next_word = words[i+self.n]
            context_sequence.append(g)
            
            # Calculate selection probability based on cascade coherence
            if len(context_sequence) >= 2:
                cascade_coherence = self.calculate_cascade_coherence(context_sequence[-3:])  # Limit history
                selection_prob = max(0.1, min(1.0, cascade_coherence + 0.5))
            else:
                selection_prob = 1.0
            
            self.S[(epoch, g)] = selection_prob
            
            # Retrocausal knowledge update for sample of words (not all)
            for candidate_word in sample_words:
                if candidate_word in overall_word_freqs:
                    # Get previous knowledge (or initialize to 0.5)
                    K_prev = self.K.get((epoch-1, g, candidate_word), 0.5)
                    
                    # Update knowledge using retrocausal equation
                    K_new = self.update_knowledge(
                        K_prev, self.alpha, selection_prob, 
                        candidate_word, next_word
                    )
                    
                    self.K[(epoch, g, candidate_word)] = K_new
            
            # Traditional symbolic adjustment enhanced with retrocausal knowledge
            base_increment = 1.0
            global_freq_factor = 1.0 + math.log1p(overall_word_freqs[next_word] / total_word_count) * 0.5
            length_factor = 1.0 + math.log1p(len(next_word)) * 0.1

            # L-semi-inner product with retrocausal enhancement (limited)
            l_inner_prod_factor = 1.0
            for context_word in g:
                similarity = self._safe_l_semi_inner_product(context_word, next_word)
                l_inner_prod_factor *= (1.0 + similarity * 0.3)
            
            # Retrocausal knowledge factor
            retrocausal_knowledge = self.K.get((epoch, g, next_word), 0.5)
            retrocausal_factor = 1.0 + retrocausal_knowledge * 0.8
            
            # Combined symbolic factor
            symbolic_factor = (global_freq_factor * length_factor * 
                             l_inner_prod_factor * retrocausal_factor)
            
            adjusted_increment = base_increment * symbolic_factor
            noise = random.uniform(-0.05, 0.05)
            final_increment = max(0.01, adjusted_increment + noise)

            if i == 0 or any(c in self.SENTENCE_END_CHARS for c in words[i-1][-1:]):
                temp_s.add(g)

            temp_m[g][next_word] += final_increment

        # Store cascade memory for future use (limited size)
        self.cascade_memory = context_sequence[-10:]  # Keep last 10 contexts
        
        self.m = {gram: dict(next_words_counts) for gram, next_words_counts in temp_m.items()}
        self.s = list(filter(lambda start_gram: start_gram in self.m, temp_s))

        if not self.s and self.m:
            k_sample = min(len(self.m), 100)
            if k_sample > 0:
                self.s = random.sample(list(self.m.keys()), k=k_sample)

        print(f"Retrocausal training complete. Model has {len(self.m)} contexts, {len(self.s)} starts.")
        print(f"Knowledge matrix contains {len(self.K)} entries. Cascade coherence integrated.")

    def _symbolic_probability(self, context, options):
        """
        Enhanced symbolic probability with retrocausal influences.
        """
        if not isinstance(context, tuple):
            context = tuple(str(context)) if context else tuple()

        words_list = list(options.keys())
        freqs = list(options.values())

        if not words_list:
            return [], []

        # Apply retrocausal knowledge weighting (limited)
        enhanced_freqs = []
        for i, word in enumerate(words_list):
            base_freq = freqs[i]
            
            # Get retrocausal knowledge (simplified)
            retrocausal_weight = 0.5  # Default
            if self.epochs > 0:
                # Look for recent knowledge
                k_val = self.K.get((self.epochs-1, context, word), 0.5)
                retrocausal_weight = k_val
            
            # Apply cascade coherence boost (simplified)
            cascade_boost = 1.0
            if len(self.cascade_memory) > 0:
                # Simple boost based on memory length
                cascade_boost = 1.0 + min(0.3, len(self.cascade_memory) * 0.05)
            
            enhanced_freq = base_freq * (1.0 + retrocausal_weight * 0.5) * cascade_boost
            enhanced_freqs.append(enhanced_freq)

        # Traditional symbolic processing with enhanced frequencies
        mean_freq = sum(enhanced_freqs) / len(enhanced_freqs) if enhanced_freqs else 0
        subset_indices = [i for i, f in enumerate(enhanced_freqs) if f > mean_freq * 0.5]
        if not subset_indices:
            subset_indices = list(range(len(words_list)))

        # Limit subset size to prevent memory issues
        if len(subset_indices) > 100:
            subset_indices = subset_indices[:100]

        subsetWords = [words_list[i] for i in subset_indices]
        subsetFreqs = [enhanced_freqs[i] for i in subset_indices]

        if not subsetWords:
            return [], []

        # Enhanced tensor calculation with retrocausal factors (optimized)
        tensorValues = []
        for i, word in enumerate(subsetWords):
            value = subsetFreqs[i]
            for contextWord in context:
                l_similarity = self._safe_l_semi_inner_product(str(contextWord), word)
                overlap = len(set(str(contextWord)) & set(word))  # Changed | to & for intersection
                combined_similarity = (float(overlap) + 1.0) * (1.0 + l_similarity * 0.5)
                value *= combined_similarity
            tensorValues.append(max(0.001, value))  # Prevent zero values

        # Safe tensor operations
        try:
            tensorValues_torch = torch.tensor(tensorValues, dtype=torch.float32)
            sorted_tensor, sort_indices = torch.sort(tensorValues_torch)
            modified_sorted = sorted_tensor.clone()
            
            # Apply retrocausal modulation (simplified)
            for i in range(len(modified_sorted)):
                if i % 2 == 0:
                    modified_sorted[i] *= 0.8
                # Additional retrocausal boost
                if len(self.cascade_memory) > 2:
                    modified_sorted[i] *= 1.1

            inverse_indices = torch.argsort(sort_indices)
            unsorted_result = modified_sorted[inverse_indices]
            
            return subsetWords, unsorted_result.tolist()
            
        except Exception as e:
            # Fallback to simple frequency-based selection
            return subsetWords, subsetFreqs

    def gen(self, seed=None, count=100, window_size=20, word_filter=None):
        """
        Generate text with retrocausal influences.
        """
        if not self.m:
            raise ValueError("Model not trained. Call train() first.")

        if count < 1:
            raise ValueError("Word count must be positive.")

        current_context = None
        result_words = []
        generation_cascade = []  # Track generation cascade

        if not self.s:
            if self.m:
                k_sample = min(len(self.m), 100)
                if k_sample > 0:
                    self.s = random.sample(list(self.m.keys()), k=k_sample)
                if not self.s:
                    raise ValueError("No valid starting contexts available.")
            else:
                raise ValueError("Model has no transitions and no starting contexts.")

        # Initialize with seed or random start
        if seed:
            seed_words = seed.lower().split()
            if len(seed_words) >= self.n:
                potential_context = tuple(seed_words[-self.n:])
                if potential_context in self.m:
                    current_context = potential_context
                    result_words = seed_words
                else:
                    current_context = random.choice(self.s)
                    result_words = seed_words + list(current_context)
            else:
                current_context = random.choice(self.s)
                result_words = list(current_context)
        else:
            current_context = random.choice(self.s)
            result_words = list(current_context)

        retry_count = 0
        max_retries = 15

        print(f"Generating with retrocausal influences (α={self.alpha}, cascade_threshold={self.cascade_threshold})...")

        while len(result_words) < count:
            if current_context not in self.m or not self.m[current_context]:
                retry_count += 1
                if retry_count >= max_retries:
                    break
                
                possible_starts = [s for s in self.s if s in self.m and self.m[s]]
                if not possible_starts:
                    possible_starts = [c for c in self.m.keys() if self.m[c]]
                if not possible_starts:
                    break
                
                current_context = random.choice(possible_starts)
                continue

            # Track generation cascade (limited size)
            generation_cascade.append(current_context)
            if len(generation_cascade) > 10:
                generation_cascade = generation_cascade[-10:]
            
            # Calculate current cascade coherence (simplified)
            current_coherence = min(0.8, len(generation_cascade) * 0.1)
            
            # Get enhanced probabilities with retrocausal influences
            try:
                candidate_words, candidate_weights = self._symbolic_probability(
                    current_context, self.m[current_context]
                )
            except Exception as e:
                print(f"Probability calculation error: {e}")
                candidate_words = list(self.m[current_context].keys())
                candidate_weights = list(self.m[current_context].values())

            if not candidate_words or not candidate_weights or sum(candidate_weights) <= 0:
                retry_count += 1
                if retry_count >= max_retries:
                    break
                continue

            # Apply retrocausal selection with cascade coherence (simplified)
            if current_coherence > self.cascade_threshold:
                # Boost random selection of words
                for i in range(len(candidate_weights)):
                    if random.random() < 0.3:
                        candidate_weights[i] *= 1.2

            # Word filtering with retrocausal considerations
            window_words = result_words[-min(len(result_words), window_size):]
            next_word = None

            if word_filter is not None:
                available_words = candidate_words.copy()
                available_weights = candidate_weights.copy()
                
                for attempt in range(5):
                    if not available_words:
                        break
                    
                    current_total = sum(available_weights)
                    if current_total <= 0:
                        break
                    
                    normalized_weights = [w / current_total for w in available_weights]
                    try:
                        chosen_idx = random.choices(range(len(available_words)), 
                                                  weights=normalized_weights, k=1)[0]
                        candidate = available_words[chosen_idx]
                        
                        if word_filter(candidate, window_words):
                            next_word = candidate
                            break
                        
                        available_words.pop(chosen_idx)
                        available_weights.pop(chosen_idx)
                    except (ValueError, IndexError):
                        break
                
                if next_word is None and candidate_words:
                    next_word = random.choice(candidate_words)
            else:
                try:
                    if sum(candidate_weights) > 0:
                        next_word = random.choices(candidate_words, weights=candidate_weights, k=1)[0]
                    elif candidate_words:
                        next_word = random.choice(candidate_words)
                except (ValueError, IndexError):
                    if candidate_words:
                        next_word = random.choice(candidate_words)

            if next_word is None:
                retry_count += 1
                continue

            result_words.append(next_word)
            current_context = tuple(result_words[-self.n:])
            retry_count = 0

        # Update cascade memory with generation results (limited)
        self.cascade_memory.extend(generation_cascade[-5:])
        self.cascade_memory = self.cascade_memory[-10:]  # Keep manageable size

        return ' '.join(result_words[-count:])


# Example usage with retrocausal filter
def retrocausal_repetition_filter(word, window_words):
    """Enhanced filter that considers retrocausal coherence"""
    return word not in window_words

if __name__ == "__main__":
    CONFIG = {
        'input_filename': "test.txt",
        'ngram_size': 2,
        'words_to_generate': 250,
        'window_size': 100,
        'alpha': 10.7,
        'cascade_threshold': 10.3
    }

    print(f"--- Retrocausal Symbolic Markov Generator ---")
    print(f"T-Symmetric Dynamics: ∂/∂t selecting futures that ∫ into their own past")
    print(f"Knowledge Update: K_n(x) = K_{{n-1}}(x) + α·S_n(x)·[A(S_n(x), I(x=x_true)) - K_{{n-1}}(x)]")

    try:
        filename = input(f"Enter input filename (default: {CONFIG['input_filename']}): ")
        if not filename:
            filename = CONFIG['input_filename']

        with open(filename, 'r', encoding='utf-8') as file:
            txt = ' '.join(file.read().lower().split()[:9999])
            if not txt:
                print(f"Error: Input file '{filename}' is empty.")
                sys.exit(1)

    except FileNotFoundError:
        print(f"Error: Input file '{filename}' not found.")
        print("Using sample text for demonstration.")
        txt = ("the quantum field oscillates through time creating retrocausal pathways "
               "where future states influence past configurations through coherent cascade "
               "dynamics the laser cavity maintains temporal symmetry while knowledge "
               "updates propagate backwards selecting consistent futures that integrate "
               "into their own past through symbolic probability distributions")

    try:
        ngram_input = input(f"Enter n-gram size (default: {CONFIG['ngram_size']}): ")
        if ngram_input:
            CONFIG['ngram_size'] = int(ngram_input)
        
        alpha_input = input(f"Enter learning rate α (default: {CONFIG['alpha']}): ")
        if alpha_input:
            CONFIG['alpha'] = float(alpha_input)
            
        threshold_input = input(f"Enter cascade threshold (default: {CONFIG['cascade_threshold']}): ")
        if threshold_input:
            CONFIG['cascade_threshold'] = float(threshold_input)
            
    except ValueError:
        print("Invalid input. Using defaults.")

    try:
        model = RetrocausalSymbolicMarkov(
            n=CONFIG['ngram_size'],
            alpha=CONFIG['alpha'], 
            cascade_threshold=CONFIG['cascade_threshold']
        )
        model.train(txt)
    except Exception as e:
        print(f"Error during training: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

    if not model.m or not model.s:
        print("Model training failed. Cannot generate text.")
    else:
        print(f"\n--- Retrocausal Generation Ready ---")
        print(f"Parameters: n={model.n}, α={model.alpha}, threshold={model.cascade_threshold}")
        print(f"Knowledge entries: {len(model.K)}, Cascade memory: {len(model.cascade_memory)}")
        print("Enter seed text or press Enter for random start. Type 'exit' to quit.")

        while True:
            try:
                user_input = input("\nSEED: ")
                if user_input.lower() in ['quit', 'exit']:
                    break

                generated = model.gen(
                    seed=user_input or None,
                    count=CONFIG['words_to_generate'],
                    window_size=CONFIG['window_size'],
                    word_filter=retrocausal_repetition_filter
                )

                print("\n--- Retrocausally Generated Text ---")
                if user_input:
                    print(f"(Seed: '{user_input}')")
                print(generated)
                
                # Show cascade coherence info
                if len(model.cascade_memory) > 2:
                    recent_coherence = model.calculate_cascade_coherence(model.cascade_memory[-3:])
                    print(f"Recent cascade coherence: {recent_coherence:.3f}")
                
                print("-" * 60)

            except KeyboardInterrupt:
                print("\nExiting retrocausal generation.")
                break
            except Exception as e:
                print(f"Generation Error: {e}")
                import traceback
                traceback.print_exc()

    print("\n--- Retrocausal cascade complete ---")
