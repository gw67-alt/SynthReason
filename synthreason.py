import random
import math
import sys
from collections import defaultdict, Counter
from tqdm import tqdm

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
        # Transitions will store floats due to symbolic adjustments
        self.m = {}  # Transitions: {context_tuple: {next_word: adjusted_frequency}}
        self.s = []  # Sentence starting n-grams
        self.all_words = set() # Store all words seen during training

    def _l_semi_inner_product(self, word1, word2):
        """
        Calculates an L-semi-inner product between two words.
        This provides a measure of linguistic similarity beyond simple character overlap.
        
        The L-semi-inner product considers:
        1. Character overlap (weighted)
        2. Length similarity 
        3. Phonetic properties approximation
        4. Position-based weighting
        
        Args:
            word1 (str): First word
            word2 (str): Second word
            
        Returns:
            float: L-semi-inner product value
        """
        if not word1 or not word2:
            return 0.0
            
        # 1. Character overlap with position weighting
        char_product = 0.0
        w1_len = len(word1)
        w2_len = len(word2)
        
        # Calculate normalized position weights
        for i, c1 in enumerate(word1):
            pos_weight_1 = 1.0 - (abs(i - w1_len/2) / (w1_len + 0.001))
            
            for j, c2 in enumerate(word2):
                if c1 == c2:
                    pos_weight_2 = 1.0 - (abs(j - w2_len/2) / (w2_len + 0.001))
                    # Weight matches by their positions (center characters matter more)
                    char_product += pos_weight_1 * pos_weight_2
        
        # 2. Length similarity component (normalized)
        length_ratio = min(w1_len, w2_len) / max(w1_len, w2_len) if max(w1_len, w2_len) > 0 else 0.0
        
        # 3. Approximate phonetic similarity through vowel/consonant patterns
        vowels = set('aeiou')
        w1_pattern = ''.join('V' if c in vowels else 'C' for c in word1.lower())
        w2_pattern = ''.join('V' if c in vowels else 'C' for c in word2.lower())
        
        # Compare vowel-consonant patterns at beginning and end of words
        pattern_length = min(3, min(len(w1_pattern), len(w2_pattern)))
        pattern_match_start = sum(1 for i in range(pattern_length) 
                              if i < len(w1_pattern) and i < len(w2_pattern) and w1_pattern[i] == w2_pattern[i])
        
        pattern_match_end = sum(1 for i in range(1, pattern_length+1)
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
        
        # Combine all factors with weights
        l_inner_product = (
            0.4 * char_product + 
            0.2 * length_ratio + 
            0.2 * phonetic_factor + 
            0.2 * prefix_suffix_factor
        )
        
        # Scale the result to a reasonable range
        return l_inner_product * 2.0  # Amplify the effect

    def train(self, t):
        """
        Trains the Markov model on the provided text, applying symbolic
        adjustments ∀λ±ε to the transition counts and incorporating
        L-semi-inner product calculations.

        Interpretation:
            ∀: Adjust increment based on global frequency of the next word.
            λ: Adjust increment based on the length of the next word.
            ±ε: Add small random noise to the increment.
            L: Apply L-semi-inner product between context and next word.

        Args:
            t (str): The training text.
        """
        if not isinstance(t, str) or not t:
            raise TypeError("Training data must be a non-empty string.")

        words = t.split()
        num_words = len(words)

        if num_words <= self.n:
            raise ValueError(f"Training data has only {num_words} words, need more than n-gram size {self.n}.")

        print(f"Training on {num_words} words with n={self.n}...")
        print(f"Applying symbolic count adjustments ∀λ±ε with L-semi-inner product...")

        # Calculate global word frequencies first (for ∀)
        print("Calculating global word frequencies (∀)...")
        overall_word_freqs = Counter(words)
        total_word_count = float(num_words) # Use float for division

        # Build frequency dictionary using defaultdict with float values
        temp_m = defaultdict(lambda: defaultdict(float))
        temp_s = set()  # Use set for faster lookup

        # Track all words seen during training
        self.all_words = set(words)

        # First pass: collect base frequencies with symbolic adjustments ∀λ±ε
        for i in tqdm(range(num_words - self.n-3), desc="Collecting Adjusted Frequencies (∀λ±ε+L)"):
            g = tuple(words[i:i+self.n])
            next_word = words[i+self.n]
            g2 = tuple(words[i+3:i+self.n+3])
            next_word2 = words[i+self.n+3]

            # --- Symbolic Adjustment Calculation ---
            base_increment = 1.0

            # ∀ (Universal/Global Influence): Boost based on overall word frequency
            # Normalize frequency (0 to 1) and use log1p for smoother scaling
            global_freq_factor = 1.0 + math.log1p(overall_word_freqs[next_word] / total_word_count) * 0.5 # Scaled influence
            # Note: log1p(x) = log(1+x), good for small x

            # λ (Lambda/Word Property): Boost based on word length
            # Use log1p again for smoother scaling, smaller factor
            length_factor = 1.0 + math.log1p(len(next_word)) * 0.1 # Scaled influence

            # L-semi-inner product adjustment
            l_inner_prod_factor = 1.0
            # Calculate L-semi-inner product between each word in context and next word
            for context_word in g:
                l_inner_prod = self._l_semi_inner_product(context_word, next_word)
                # Scale the effect - higher values for similar words
                l_inner_prod_factor *= (1.0 + l_inner_prod * 0.3)  # Moderate influence
                self.s = list(filter(lambda g: l_inner_prod_factor in self.m, context_word))
            # Combine ∀, λ, and L factors multiplicatively
            symbolic_factor = global_freq_factor * length_factor * l_inner_prod_factor

            # Apply the combined factor to the base increment
            adjusted_increment = base_increment * symbolic_factor

            # ±ε (Noise/Variability): Add small random noise
            noise = random.uniform(-0.05, 0.05) # Small epsilon range
            final_increment = adjusted_increment + noise

            # Ensure the increment is positive
            final_increment = max(0.01, final_increment) # Ensure at least a small positive value
            # --- End Symbolic Adjustment ---

            # Record sentence starts
            if i == 0 or any(c in self.SENTENCE_END_CHARS for c in words[i-1][-1:]):
                temp_s.add(g)


            # Increment frequency with the symbolically adjusted value
            temp_m[g][next_word] += final_increment

        # Convert to regular dictionaries for the final model
        self.m = {g: dict(next_words) for g, next_words in temp_m.items()}

        # Store valid sentence starts
        self.s = list(filter(lambda g: g in self.m, temp_s))

        # If no valid starts, use random contexts
        if not self.s and self.m:
            print("Warning: No sentence starts detected based on punctuation. Using random contexts as starting points.")
            self.s = random.sample(list(self.m.keys()), k=min(len(self.m), 100))

        if not self.s and not self.m:
             print("Warning: Model training resulted in no transitions. Check input data and n-gram size.")


        print(f"Training complete. Model has {len(self.m)} contexts and {len(self.s)} sentence starts. Frequencies adjusted by ∀λ±ε with L-semi-inner product.")

    def _symbolic_probability(self, context, options):
        """
        Applies the symbolic probability distribution ⊆⊗∃·Λρ∑ω·Σø² to select the next word.
        Enhanced with L-semi-inner product influences.

        Args:
            context (tuple): The current context (n-gram)
            options (dict): Possible next words with their *adjusted* frequencies

        Returns:
            tuple: (words, weights) Lists of candidate words and their probabilities
        """
        words = list(options.keys())
        # Frequencies are now floats due to training adjustments
        freqs = list(options.values())

        # Early return if no options
        if not words:
            return [], []

        # --- Start of Symbolic Probability Calculation (⊆⊗∃·Λρ∑ω·Σø²) ---

        # ⊆ (subset) - Filter options (keep frequencies > adjusted threshold or all)
        # Since frequencies are floats, compare against a small value or mean
        mean_freq = sum(freqs) / len(freqs) if freqs else 0
        # Keep words with frequency > slightly above mean or all if that leaves nothing
        subset_indices = [i for i, f in enumerate(freqs) if f > mean_freq * 0.5] # Keep if > 50% of mean freq

        # If nothing would remain, use all words
        if not subset_indices:
            subset_indices = list(range(len(words)))

        subsetWords = [words[i] for i in subset_indices]
        subsetFreqs = [freqs[i] for i in subset_indices]

        # If we still have nothing, return empty lists
        if not subsetWords:
            return [], []

        # ⊗ (tensor product) - Relationship between context and options
        tensorValues = []
        for word in subsetWords:
            tensorValue = 1.0 # Use float
            for contextWord in context:
                # Enhanced with L-semi-inner product
                l_similarity = self._l_semi_inner_product(contextWord, word)
                overlap = len(set(contextWord) & set(word))
                combined_similarity = (float(overlap) + 1.0) * (1.0 + l_similarity)
                tensorValue *= combined_similarity
            tensorValues.append(tensorValue)

        return subsetWords, tensorValues


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

        if not self.s:
            # Try to recover if training failed silently
            if self.m:
                 print("Warning: No sentence starts available, but model has contexts. Choosing a random context.")
                 self.s = random.sample(list(self.m.keys()), k=min(len(self.m), 100))
                 if not self.s: # Still no starts possible
                     raise ValueError("No valid starting contexts available and recovery failed.")
            else:
                raise ValueError("Model has no transitions and no starting contexts.")


        # Process seed if provided
        if seed:
            seed_words = seed.lower().split()
            if len(seed_words) >= self.n:
                # Check if the exact seed context exists
                potential_context = tuple(seed_words[-self.n:])
                if potential_context in self.m:
                    context = potential_context
                    result = seed_words
                else:
                    # Seed context not found, fall back to random start but keep seed words
                    print(f"Warning: Seed context {potential_context} not found in model. Starting with a random context.")
                    context = random.choice(self.s)
                    result = seed_words + list(context) # Append random context start
            else:
                # Seed too short, use random start
                context = random.choice(self.s)
                result = list(context)
        else:
            # No seed, use a random start context
            context = random.choice(self.s)
            result = list(context)

        # Generate text
        retry_count = 0
        max_retries = 15 # Increased retries slightly
        max_filter_attempts = 5

        while len(result) < count: # Check before loop body
            # Check context validity
            if context not in self.m or not self.m[context]:
                retry_count += 1
                if retry_count >= max_retries:
                    print(f"\nWarning: Max retries ({max_retries}) reached finding a valid next context. Stopping generation.")
                    break # Exit generation loop

                # Try finding a new valid context
                possible_starts = [s for s in self.s if s in self.m and self.m[s]] # Filter for valid starts
                if not possible_starts:
                     # If even starts are bad, try any valid context
                     possible_starts = [c for c in self.m.keys() if self.m[c]]

                if not possible_starts:
                    print("\nWarning: No valid contexts found to continue generation. Stopping.")
                    break # Exit generation loop

                context = random.choice(possible_starts)
                # Avoid adding the whole context if it might make output too long abruptly
                # result.extend(context) # Consider removing this line if it causes issues
                print(f"Info: Resetting context to {context} (Retry {retry_count}/{max_retries})")
                continue # Try again with the new context

            # Get weighted options using symbolic probability
            words, weights = self._symbolic_probability(context, self.m[context])

            # Make sure we have valid options
            if not words or not weights or len(words) != len(weights) or sum(weights) == 0:
                retry_count += 1
                if retry_count >= max_retries:
                    print(f"\nWarning: Max retries ({max_retries}) reached finding valid next words. Stopping generation.")
                    break

                # Try finding a new valid context (same logic as above)
                possible_starts = [s for s in self.s if s in self.m and self.m[s]]
                if not possible_starts:
                    possible_starts = [c for c in self.m.keys() if self.m[c]]
                if not possible_starts:
                     print("\nWarning: No valid contexts found to continue generation. Stopping.")
                     break

                context = random.choice(possible_starts)
                print(f"Info: Resetting context to {context} due to no valid next words (Retry {retry_count}/{max_retries})")

                continue # Try again


             # Get current window for filtering
            window_words = result[-min(len(result), window_size):]

            next_word = None # Initialize next_word

            # Apply word filtering if provided
            if word_filter is not None:
                available_words = words.copy()
                available_weights = weights.copy()
                found_valid_word = False
                filter_attempts = 0

                while available_words and filter_attempts < max_filter_attempts:
                     # Normalize remaining weights
                    current_total_weight = sum(available_weights)
                    if current_total_weight <= 0: # Check for non-positive weights
                        break # Cannot sample

                    normalized_weights = [w / current_total_weight for w in available_weights]

                    # Random weighted choice from available options
                    try:
                        chosen_idx = random.choices(
                            range(len(available_words)),
                            weights=normalized_weights,
                            k=1
                        )[0]
                    except ValueError as e:
                        print(f"Warning: Error during weighted choice (filter loop): {e}. Weights: {normalized_weights}")
                        break # Exit sampling loop

                    candidate_word = available_words[chosen_idx]

                    # Check filter
                    if word_filter(candidate_word, window_words):
                        next_word = candidate_word
                        found_valid_word = True
                        break # Found a valid word

                    # Remove this word from consideration for next attempt
                    available_words.pop(chosen_idx)
                    available_weights.pop(chosen_idx)
                    filter_attempts += 1

                # If filter failed after attempts, choose from original list respecting weights
                if not found_valid_word:
                    try:
                        next_word = random.choices(words, weights=weights, k=1)[0]
                    except ValueError as e:
                         print(f"Warning: Error during weighted choice (fallback): {e}. Weights: {weights}. Choosing uniformly.")
                         if words: # Check if words list is not empty
                              next_word = random.choice(words)
                         else: # Should not happen if initial check passed, but safety first
                              print("Error: Cannot select fallback word, no options available.")
                              retry_count+=1 # Count as retry and try new context next iteration
                              continue

            else:
                # No filter, choose based on original weights
                try:
                     next_word = random.choices(words, weights=weights, k=1)[0]
                except ValueError as e:
                    print(f"Warning: Error during weighted choice (no filter): {e}. Weights: {weights}. Choosing uniformly.")
                    if words:
                         next_word = random.choice(words)
                    else:
                        print("Error: Cannot select word, no options available.")
                        retry_count+=1
                        continue


            # Check if a word was actually selected
            if next_word is None:
                 print("Warning: Failed to select a next word. Resetting context.")
                 retry_count += 1
                 # (Context reset logic will trigger on next loop iteration if needed)
                 continue


            # Add the chosen word and update context
            result.append(next_word)
            context = tuple(result[-self.n:])
            retry_count = 0 # Reset retry counter on successful word generation

        # Return exactly 'count' words (or fewer if generation stopped early)
        # Slicing from the end ensures the most recently generated words are kept
        return ' '.join(result[-count:])


# Example usage
def no_repetition_filter(word, window_words):
    """Prevents a word from being repeated within the window"""
    return word not in window_words

if __name__ == "__main__":
    # Configuration
    CONFIG = {
        'input_filename': "input.txt",  # Default, will be overridden by input
        'ngram_size': 2, # Default n-gram size (can be changed)
        'words_to_generate': 150, # Default generation length
        'window_size': 50  # Default window size for filter
    }

    print(f"--- Symbolic Markov Text Generator (Training: ∀λ±ε+L | Generation: ⊆⊗∃·Λρ∑ω·Σø²) ---")

    # Get filename from user input
    try:
        filename_to_use = input(f"Enter input filename (default: {CONFIG['input_filename']}): ")
        if not filename_to_use:
            filename_to_use = CONFIG['input_filename']
        CONFIG['input_filename'] = filename_to_use

        with open(CONFIG['input_filename'], 'r', encoding='utf-8') as file:
            # Read, lower, and split robustly
            txt = ' '.join(file.read().lower().split())
            if not txt:
                 print(f"Error: Input file '{CONFIG['input_filename']}' is empty.")
                 sys.exit(1)

    except FileNotFoundError:
        print(f"Error: Input file '{CONFIG['input_filename']}' not found.")
        # Simple fallback text for demonstration if file fails
        print("Using sample text for demonstration.")
        txt = ("this is a simple sample text for the markov chain generator it demonstrates "
               "basic functionality and provides enough words for training with small ngrams "
               "repeating some words is necessary for the model to learn transitions this sample "
               "is quite short so results may be limited a larger corpus will yield better output")
    except Exception as e:
         print(f"Error reading file: {e}")
         sys.exit(1)


    # Allow user to override n-gram size
    try:
        ngram_input = input(f"Enter n-gram size (default: {CONFIG['ngram_size']}): ")
        if ngram_input:
            CONFIG['ngram_size'] = int(ngram_input)
            if CONFIG['ngram_size'] < 1:
                print(f"Invalid n-gram size: {CONFIG['ngram_size']}. Using default: 2")
                CONFIG['ngram_size'] = 2
    except ValueError:
        print(f"Invalid n-gram size input. Using default: {CONFIG['ngram_size']}")


    # Initialize and train model
    try:
        model = SymbolicMarkov(CONFIG['ngram_size'])
        model.train(txt)
    except ValueError as e:
        print(f"Error during initialization or training: {e}")
        sys.exit(1)
    except Exception as e: # Catch other potential errors
        print(f"An unexpected error occurred during training: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

    # Interactive generation loop
    if not model.m or not model.s:
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

                # Generate text with word filtering
                generated = model.gen(
                    seed=user_input or None,
                    count=CONFIG['words_to_generate'],
                    window_size=CONFIG['window_size'],
                    word_filter=no_repetition_filter
                )

                # Display result
                print("\n--- Generated Text ---")
                if user_input:
                    print(f"(Seed: '{user_input}')")
                print(generated)
                print("-" * 60)

            except KeyboardInterrupt:
                print("\nExiting.")
                break
            except ValueError as e: # Catch generation errors
                print(f"Generation Error: {e}")
                # Allow the user to try again
            except Exception as e:
                print(f"An unexpected error occurred during generation: {e}")
                import traceback
                traceback.print_exc()
                # Allow trying again unless it's critical

    print("\n--- Generation Complete ---")