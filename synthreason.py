import random
import math
import sys
from collections import defaultdict, Counter
from tqdm import tqdm
import numpy as np
import torch

class SymbolicMarkov:
    """
    Markov chain text generator using symbolic probability distribution ⊆⊗∃·Λρ∑ω·Σø²
    and symbolic training count adjustments ∀λ±ε with L-semi-inner product.
    """

    # Static configuration variables
    SENTENCE_END_CHARS = set('.!?')

    def __init__(self, n=2):
        """
        Initializes the Markov chain model with n-gram size.
        Args:
            n (int): Size of n-gram context
        """
        if not isinstance(n, int) or n < 1:
            raise ValueError("n-gram size 'n' must be a positive integer.")
        self.n = n
        self.m = {}  # Transitions: {context_tuple: {next_word: adjusted_frequency}}
        self.s = []  # Sentence starting n-grams
        self.all_words = set() # Store all words seen during training

    @staticmethod
    def _calculate_distinct_permutations_log(word):
        """
        Calculates the natural logarithm of the number of distinct permutations of characters in a word.
        Formula: log(N! / (n1! * n2! * ... * nk!))
                 = log(N!) - (log(n1!) + log(n2!) + ...)
                 = lgamma(N+1) - sum(lgamma(ni+1) for ni in counts)
        Returns 0.0 for empty strings or if calculation errors occur.
        """
        if not word:
            return 0.0
        
        n = len(word)
        if n == 0:
             return 0.0

        counts = Counter(word)
        
        try:
            # math.lgamma(x) computes log(gamma(x)). log(k!) = lgamma(k+1).
            log_n_factorial = math.lgamma(n + 1)
            
            log_denominator_sum = 0.0
            for char_count in counts.values():
                log_denominator_sum += math.lgamma(char_count + 1)
            
            log_permutations = log_n_factorial - log_denominator_sum
            # Result should theoretically be non-negative.
            # Floating point inaccuracies might make it slightly negative for very simple cases (e.g. single char "a" should be log(1!) - log(1!) = 0)
            return max(0.0, log_permutations) 
        except (ValueError, OverflowError):
            # Fallback in case of unexpected math issues
            return 0.0

    def _l_semi_inner_product(self, word1, word2):
        """
        Calculates an L-semi-inner product between two words.
        This provides a measure of linguistic similarity.
        Now includes a factor based on the similarity of log-counts of distinct character permutations.
        """
        if not word1 or not word2:
            return 0.0
            
        # 1. Character overlap with position weighting
        char_product = 0.0
        w1_len = len(word1)
        w2_len = len(word2)
        
        for i, c1 in enumerate(word1):
            pos_weight_1 = 1.0 - (abs(i - w1_len/2) / (w1_len + 0.001))
            for j, c2 in enumerate(word2):
                if c1 == c2:
                    pos_weight_2 = 1.0 - (abs(j - w2_len/2) / (w2_len + 0.001))
                    char_product += pos_weight_1 * pos_weight_2
        
        # 2. Length similarity component (normalized)
        length_ratio = min(w1_len, w2_len) / max(w1_len, w2_len) if max(w1_len, w2_len) > 0 else 0.0
        
        # 3. Approximate phonetic similarity through vowel/consonant patterns
        vowels = set('aeiou') # Standard vowels for phonetic approximation
        w1_pattern = ''.join('V' if c in vowels else 'C' for c in word1.lower())
        w2_pattern = ''.join('V' if c in vowels else 'C' for c in word2.lower())
        
        pattern_length = min(3, min(len(w1_pattern), len(w2_pattern)))
        pattern_match_start = sum(1 for i in range(pattern_length) 
                                  if i < len(w1_pattern) and i < len(w2_pattern) and w1_pattern[i] == w2_pattern[i])
        pattern_match_end = sum(1 for i in range(1, pattern_length + 1)
                                if i <= len(w1_pattern) and i <= len(w2_pattern) 
                                and w1_pattern[-i] == w2_pattern[-i])
        phonetic_factor = (pattern_match_start + pattern_match_end) / (2 * pattern_length) if pattern_length > 0 else 0.0
        
        # 4. Special case for exact prefix or suffix matches
        prefix_len = 0
        for i in range(min(w1_len, w2_len)):
            if word1[i] == word2[i]:
                prefix_len += 1
            else:
                break
                
        suffix_len = 0
        for i in range(1, min(w1_len, w2_len) + 1):
            if word1[-i] == word2[-i]:
                suffix_len += 1
            else:
                break
        prefix_suffix_factor = (prefix_len + suffix_len) / (w1_len + w2_len) if (w1_len + w2_len) > 0 else 0.0

        # 5. NEW: Permutation-based similarity factor
        log_perm1 = self._calculate_distinct_permutations_log(word1)
        log_perm2 = self._calculate_distinct_permutations_log(word2)
        
        diff_log_perm = abs(log_perm1 - log_perm2)
        # Normalize the difference: 1 if identical, 0 if maximally different based on scale
        # Using max of logs (or 1.0 if logs are small) as denominator
        denominator_perm = max(log_perm1, log_perm2, 1.0) # Avoid division by zero and give scale
        normalized_diff_perm = diff_log_perm / denominator_perm if denominator_perm > 0 else 0.0
        permutation_similarity_factor = 1.0 - min(normalized_diff_perm, 1.0) # Ensure factor is [0,1]

        # Combine all factors with new weights (summing to 1.0)
        # Old weights: char=0.4, len=0.2, phon=0.2, pre_suf=0.2
        # New weights: perm=0.1, others scaled to sum to 0.9
        w_char = 0.36       # was 0.4
        w_len = 0.18        # was 0.2
        w_phon = 0.18       # was 0.2
        w_pre_suf = 0.18    # was 0.2
        w_perm = 0.10       # new factor

        l_inner_product = (
            w_char * char_product + 
            w_len * length_ratio + 
            w_phon * phonetic_factor + 
            w_pre_suf * prefix_suffix_factor +
            w_perm * permutation_similarity_factor # Added new factor
        )
        
        return l_inner_product * 2.0 # Amplify the effect (existing scaling)

    def train(self, t):
        if not isinstance(t, str) or not t:
            raise TypeError("Training data must be a non-empty string.")

        words = t.split()
        num_words = len(words)

        if num_words <= self.n: # Ensure enough words for at least one n-gram and next word
            raise ValueError(f"Training data has only {num_words} words, need more than n-gram size {self.n}.")

        print(f"Training on {num_words} words with n={self.n}...")
        print(f"Applying symbolic count adjustments ∀λ±ε with L-semi-inner product...")

        overall_word_freqs = Counter(words)
        total_word_count = float(num_words)

        temp_m = defaultdict(lambda: defaultdict(float))
        temp_s = set()
        self.all_words = set(words)
        
        # Loop range: Ensure i+self.n is a valid index for next_word
        # And if g2 and next_word2 are used, range should be num_words - (self.n + 3)
        # The original code used num_words - self.n - 3 which might be too short if g2 is not used.
        # Let's assume g2 and next_word2 are not essential for the core logic for now,
        # as they were not used in the symbolic adjustment calculation.
        # If they are to be used, the loop range and their definition needs care.
        # For now, let's use a range that ensures 'g' and 'next_word' are valid.
        # The loop should go up to the point where `i+self.n` is the last word.
        # So `i` goes up to `num_words - self.n - 1`.
        # range(num_words - self.n) would make `i` go up to `num_words - self.n - 1`.
        # words[i : i+self.n] is g
        # words[i+self.n] is next_word

        for i in tqdm(range(num_words - self.n), desc="Collecting Adjusted Frequencies (∀λ±ε+L)"):
            g = tuple(words[i:i+self.n])
            next_word = words[i+self.n]
            
            # --- Symbolic Adjustment Calculation ---
            base_increment = 1.0 # Changed from 2.0 to 1.0 as it's a base before multiplication

            global_freq_factor = 1.0 + math.log1p(overall_word_freqs[next_word] / total_word_count) * 0.5
            length_factor = 1.0 + math.log1p(len(next_word)) * 0.1

            # L-semi-inner product adjustment
            l_inner_prod_factor = 1.0
            # Calculate L-semi-inner product between each word in context and next word
            for context_word_from_g in g: # Iterate through each word in the actual context tuple 'g'
                # CORRECTED: Use _l_semi_inner_product directly for a scalar similarity score
                single_similarity_score = self._l_semi_inner_product(context_word_from_g, next_word)
                # CORRECTED: Use the calculated score in the factor
                l_inner_prod_factor *= (1.0 + single_similarity_score * 0.3) 
            
            # REMOVED problematic self.s update:
            # self.s = list(filter(lambda g_filter_param: l_inner_prod_factor in self.m, context_word))
            # This was incorrect. Sentence starts are collected below via temp_s.

            symbolic_factor = global_freq_factor * length_factor * l_inner_prod_factor
            adjusted_increment = base_increment * symbolic_factor
            noise = random.uniform(-0.05, 0.05)
            final_increment = max(0.01, adjusted_increment + noise)
            # --- End Symbolic Adjustment ---

            if i == 0 or any(c in self.SENTENCE_END_CHARS for c in words[i-1][-1:]):
                temp_s.add(g)

            temp_m[g][next_word] += final_increment

        self.m = {gram: dict(next_words_counts) for gram, next_words_counts in temp_m.items()}
        self.s = list(filter(lambda start_gram: start_gram in self.m, temp_s))

        if not self.s and self.m:
            print("Warning: No sentence starts detected based on punctuation. Using random contexts as starting points.")
            # Ensure sample size k is not greater than population size
            k_sample = min(len(self.m), 100)
            if k_sample > 0 : # only sample if there's something to sample from
                 self.s = random.sample(list(self.m.keys()), k=k_sample)

        if not self.m: # Simplified this condition, if m is empty, s will also be empty or useless
             print("Warning: Model training resulted in no transitions. Check input data and n-gram size.")
        
        print(f"Training complete. Model has {len(self.m)} contexts and {len(self.s)} sentence starts. Frequencies adjusted by ∀λ±ε with L-semi-inner product.")

    def _symbolic_probability(self, context, options):
        """
        Applies the symbolic probability distribution ⊆⊗∃·Λρ∑ω·Σø² to select the next word.
        Enhanced with L-semi-inner product influences.

        Args:
            context (tuple): The current context (n-gram). It must be a tuple of strings.
            options (dict): Possible next words with their *adjusted* frequencies.

        Returns:
            tuple: (words, weights) Lists of candidate words and their probabilities
        """
        # Ensure context is usable as an iterable of words (strings)
        if not isinstance(context, tuple) or not all(isinstance(w, str) for w in context):
            # This case should ideally not be hit if called correctly,
            # but as a safeguard if a dict was passed as context by mistake:
            if isinstance(context, dict): # Fallback/error recovery attempt for wrong context type
                # Try to make a pseudo-context, e.g., from its keys if they are strings
                # This is a patch for the specific incorrect call seen previously,
                # but the method fundamentally expects context to be an n-gram tuple.
                pseudo_context_words = [str(k) for k in context.keys()]
                if pseudo_context_words:
                    # Heuristic: take the first n words or all if less than n
                    context = tuple(pseudo_context_words[:max(1,self.n)]) 
                else: # or an empty context if keys weren't useful
                    context = tuple() 
            # If it's not a dict but still not a tuple of strings, an empty tuple is a safe default.
            # Or raise an error: raise TypeError("Context must be a tuple of strings")
            # For now, let's allow it to proceed with an empty context if conversion fails.
            elif not isinstance(context, tuple): # if it's not a tuple at all (e.g. a string)
                context = tuple(str(context)) # try to make it a tuple of one string


        words_list = list(options.keys()) # Renamed to avoid conflict with 'words' module
        freqs = list(options.values())

        if not words_list:
            return [], []

        mean_freq = sum(freqs) / len(freqs) if freqs else 0
        subset_indices = [i for i, f in enumerate(freqs) if f > mean_freq * 0.5]
        if not subset_indices:
            subset_indices = list(range(len(words_list)))

        subsetWords = [words_list[i] for i in subset_indices]*3
        subsetFreqs = [freqs[i] for i in subset_indices]*3

        if not subsetWords:
            return [], []

        tensorValues = []
        for i, word in enumerate(subsetWords):
            value = 1.0
            # This loop expects 'context' to be an iterable of strings (context words)
            for contextWord_from_ngram in context: # Renamed to avoid conflict
                l_similarity = self._l_semi_inner_product(str(contextWord_from_ngram), word) # Ensure contextWord is str
                overlap = len(set(str(contextWord_from_ngram)) | set(word)) 
                combined_similarity = (float(overlap) + 1.0) * (1.0 + l_similarity)
                value *= subsetFreqs[i] * combined_similarity
            tensorValues.append(value)

        tensorValues_torch = torch.tensor(tensorValues, dtype=torch.float32) # Explicit dtype

        sorted_tensor, sort_indices = torch.sort(tensorValues_torch)
        modified_sorted = sorted_tensor.clone()
        for i_val in range(len(modified_sorted)): # Renamed loop variable
            if i_val % 2 == 0:
                modified_sorted[i_val] /= 2

        inverse_indices = torch.argsort(sort_indices)
        unsorted_result = modified_sorted[inverse_indices]

        return subsetWords, unsorted_result.tolist()

    # ... (gen method and __main__ block remain the same as your provided code) ...
    # (Make sure to paste them back if you're replacing the whole file)

    def gen(self, seed=None, count=100, window_size=20, word_filter=None):
        """
        Generates text using the trained Markov model with symbolic probability distribution.

        Args:
            seed (str, optional): Starting sequence of words.
            count (int): Number of words to generate.
            window_size (int): Size of the moving window for word filtering.
            word_filter (callable, optional): A function that takes (word, window_words) and returns
                                            True if the word should be allowed, False otherwise.
        Returns:
            str: Generated text.
        """
        if not self.m:
            raise ValueError("Model not trained. Call train() first.")

        if count < 1:
            raise ValueError("Word count must be positive.")

        current_context = None # Use a different name than the 'context' parameter of _symbolic_probability
        result_words = [] # Use a different name than the 'words' variable inside _symbolic_probability

        if not self.s:
            if self.m:
                print("Warning: No sentence starts available, but model has contexts. Choosing a random context.")
                k_sample = min(len(self.m), 100) # Ensure k is not > population
                if k_sample > 0:
                    self.s = random.sample(list(self.m.keys()), k=k_sample)
                if not self.s: 
                    raise ValueError("No valid starting contexts available and recovery failed.")
            else:
                raise ValueError("Model has no transitions and no starting contexts.")
        
        if not self.s : # Final check if self.s is still empty
             raise ValueError("No starting contexts available for generation.")


        if seed:
            seed_words = seed.lower().split()
            if len(seed_words) >= self.n:
                potential_context_tuple = tuple(seed_words[-self.n:]) # Renamed
                if potential_context_tuple in self.m:
                    current_context = potential_context_tuple
                    result_words = seed_words
                else:
                    print(f"Warning: Seed context {potential_context_tuple} not found in model. Starting with a random context.")
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
        max_filter_attempts = 5

        while len(result_words) < count: 
            if current_context not in self.m or not self.m[current_context]:
                retry_count += 1
                if retry_count >= max_retries:
                    print(f"\nWarning: Max retries ({max_retries}) reached finding a valid next context. Stopping generation.")
                    break 

                possible_starts = [s_item for s_item in self.s if s_item in self.m and self.m[s_item]] 
                if not possible_starts:
                    possible_starts = [c_item for c_item in self.m.keys() if self.m[c_item]]

                if not possible_starts:
                    print("\nWarning: No valid contexts found to continue generation. Stopping.")
                    break 
                current_context = random.choice(possible_starts)
                print(f"Info: Resetting context to {current_context} (Retry {retry_count}/{max_retries})")
                continue 

            # Call to _symbolic_probability using current_context (an n-gram tuple)
            # and self.m[current_context] (a dict of next word options)
            candidate_words, candidate_weights = self._symbolic_probability(current_context, self.m[current_context]) # Renamed return vars

            if not candidate_words or not candidate_weights or len(candidate_words) != len(candidate_weights) or sum(candidate_weights) <= 0: # check for sum > 0
                retry_count += 1
                if retry_count >= max_retries:
                    print(f"\nWarning: Max retries ({max_retries}) reached finding valid next words. Stopping generation.")
                    break
                
                possible_starts = [s_item for s_item in self.s if s_item in self.m and self.m[s_item]]
                if not possible_starts:
                    possible_starts = [c_item for c_item in self.m.keys() if self.m[c_item]]
                if not possible_starts:
                    print("\nWarning: No valid contexts found to continue generation. Stopping.")
                    break
                current_context = random.choice(possible_starts)
                print(f"Info: Resetting context to {current_context} due to no valid next words (Retry {retry_count}/{max_retries})")
                continue 

            window_words_for_filter = result_words[-min(len(result_words), window_size):] # Renamed
            next_word_chosen = None # Renamed

            if word_filter is not None:
                available_words = candidate_words.copy()
                available_weights = candidate_weights.copy()
                found_valid_word = False
                filter_attempts = 0

                while available_words and filter_attempts < max_filter_attempts:
                    current_total_weight = sum(available_weights)
                    if current_total_weight <= 0: 
                        break 
                    normalized_weights = [w / current_total_weight for w in available_weights]
                    try:
                        chosen_idx = random.choices(
                            range(len(available_words)),
                            weights=normalized_weights,
                            k=1
                        )[0]
                    except ValueError as e:
                        print(f"Warning: Error during weighted choice (filter loop): {e}. Weights: {normalized_weights}")
                        break 
                    candidate_word_from_filter = available_words[chosen_idx] # Renamed

                    if word_filter(candidate_word_from_filter, window_words_for_filter):
                        next_word_chosen = candidate_word_from_filter
                        found_valid_word = True
                        break 
                    available_words.pop(chosen_idx)
                    available_weights.pop(chosen_idx)
                    filter_attempts += 1
                
                if not found_valid_word:
                    try:
                        # Fallback to original candidates if filter fails to find one
                        if sum(candidate_weights) > 0 : # Ensure weights are valid for random.choices
                             next_word_chosen = random.choices(candidate_words, weights=candidate_weights, k=1)[0]
                        elif candidate_words: # If weights are bad but words exist, choose uniformly
                             next_word_chosen = random.choice(candidate_words)
                        else: # No words to choose from
                             print("Error: Cannot select fallback word, no options available after filter.")
                             retry_count+=1
                             continue
                    except ValueError as e:
                        print(f"Warning: Error during weighted choice (fallback): {e}. Weights: {candidate_weights}. Choosing uniformly if possible.")
                        if candidate_words: 
                            next_word_chosen = random.choice(candidate_words)
                        else: 
                            print("Error: Cannot select fallback word, no options available.")
                            retry_count+=1
                            continue
            else:
                try:
                    if sum(candidate_weights) > 0: # Ensure weights are valid
                        next_word_chosen = random.choices(candidate_words, weights=candidate_weights, k=1)[0]
                    elif candidate_words: # Uniform choice if weights are invalid
                        next_word_chosen = random.choice(candidate_words)
                    else: # No words available
                        print("Error: Cannot select word, no options available (no filter).")
                        retry_count+=1
                        continue
                except ValueError as e:
                    print(f"Warning: Error during weighted choice (no filter): {e}. Weights: {candidate_weights}. Choosing uniformly if possible.")
                    if candidate_words:
                        next_word_chosen = random.choice(candidate_words)
                    else:
                        print("Error: Cannot select word, no options available.")
                        retry_count+=1
                        continue
            
            if next_word_chosen is None:
                print("Warning: Failed to select a next word. Resetting context.")
                retry_count += 1
                continue

            result_words.append(next_word_chosen)
            current_context = tuple(result_words[-self.n:])
            retry_count = 0 
        
        # Ensure 'count' words are returned if available, from the end of result_words
        # If seed made result_words longer than 'count' initially, take the last 'count' elements generated *after* the seed.
        # This part is tricky: if seed is long, and count is short.
        # The current logic `result[-count:]` works generally to give final 'count' words.
        # If initial `result_words` (from seed) is already >= count, it might just return part of the seed.
        # A better approach for "generate 'count' new words" would be to track words generated *after* seed.
        # However, the current implementation aims for total 'count' words.
        
        # The prompt implies result should be of 'count' length.
        # If seed made result_words already count or more, and no new words were added.
        # The final result might be shorter than count if generation stopped early.
        return ' '.join(result_words[-count:])


# Example usage
def no_repetition_filter(word, window_words):
    """Prevents a word from being repeated within the window"""
    return word not in window_words

if __name__ == "__main__":
    # Configuration
    CONFIG = {
        'input_filename': "test.txt", 
        'ngram_size': 2, 
        'words_to_generate': 250, # Reduced for quicker testing
        'window_size': 100      # Reduced for filter testing
    }

    print(f"--- Symbolic Markov Text Generator (Training: ∀λ±ε+L | Generation: ⊆⊗∃·Λρ∑ω·Σø²) ---")

    try:
        filename_to_use = input(f"Enter input filename (default: {CONFIG['input_filename']}): ")
        if not filename_to_use:
            filename_to_use = CONFIG['input_filename']
        CONFIG['input_filename'] = filename_to_use

        with open(CONFIG['input_filename'], 'r', encoding='utf-8') as file:
            txt = ' '.join(file.read().lower().split())
            if not txt:
                print(f"Error: Input file '{CONFIG['input_filename']}' is empty.")
                sys.exit(1)

    except FileNotFoundError:
        print(f"Error: Input file '{CONFIG['input_filename']}' not found.")
        print("Using sample text for demonstration.")
        txt = ("this is a simple sample text for the markov chain generator it demonstrates "
               "basic functionality and provides enough words for training with small ngrams "
               "repeating some words is necessary for the model to learn transitions this sample "
               "is quite short so results may be limited a larger corpus will yield better output "
               "the quick brown fox jumps over the lazy dog and the lazy dog slept") # Added more
    except Exception as e:
        print(f"Error reading file: {e}")
        sys.exit(1)

    try:
        ngram_input = input(f"Enter n-gram size (default: {CONFIG['ngram_size']}): ")
        if ngram_input:
            CONFIG['ngram_size'] = int(ngram_input)
            if CONFIG['ngram_size'] < 1:
                print(f"Invalid n-gram size: {CONFIG['ngram_size']}. Using default: 2")
                CONFIG['ngram_size'] = 2
    except ValueError:
        print(f"Invalid n-gram size input. Using default: {CONFIG['ngram_size']}")

    try:
        model = SymbolicMarkov(CONFIG['ngram_size'])
        model.train(txt)
    except ValueError as e:
        print(f"Error during initialization or training: {e}")
        sys.exit(1)
    except Exception as e: 
        print(f"An unexpected error occurred during training: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

    if not model.m or not model.s: # Check if model is usable
        print("\nModel training resulted in no usable data. Cannot generate text.")
    else:
        print(f"\n--- Ready to generate text ---")
        print(f"Using n={model.n}, generating {CONFIG['words_to_generate']} words.")
        print(f"Applying filter: no_repetition_filter (window: {CONFIG['window_size']})")
        print(f"Enter seed text (at least {model.n} words recommended) or press Enter for random start. Type 'exit' to quit.")

        while True:
            try:
                user_input = input("\nSEED: ")
                if user_input.lower() in ['quit', 'exit']:
                    break

                generated = model.gen(
                    seed=user_input or None,
                    count=CONFIG['words_to_generate'],
                    window_size=CONFIG['window_size'],
                    word_filter=no_repetition_filter
                )

                print("\n--- Generated Text ---")
                if user_input:
                    print(f"(Seed: '{user_input}')")
                print(generated)
                print("-" * 60)

            except KeyboardInterrupt:
                print("\nExiting.")
                break
            except ValueError as e: 
                print(f"Generation Error: {e}")
            except Exception as e:
                print(f"An unexpected error occurred during generation: {e}")
                import traceback
                traceback.print_exc()
    print("\n--- Generation Complete ---")
