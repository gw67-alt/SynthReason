import random
import math
import sys
from tqdm import tqdm
from collections import defaultdict
KB_limit = -1
class IsoMarkov:
    # Static configuration variables
    MAX_EXPONENT = 5.0
    MIN_EXPONENT = 0.1
    SENTENCE_END_CHARS = set('.!?')
    ISO_MODULUS = 5
    ISO_DIVISOR = 10.0

    def __init__(self, n=2):
        """
        Initializes the Markov chain model with n-gram size.
        """
        if not isinstance(n, int) or n < 1:
            raise ValueError("n-gram size 'n' must be a positive integer.")
        self.n = n
        self.m = {}  # Transitions: {context_tuple: {next_word: frequency}}
        self.s = []  # Sentence starting n-grams
        self.all_words = set()  # Store all words seen during training

    def train(self, t):
        """
        Trains the Markov model on the provided text with isohedral re-weighting.
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

        # Build frequency dictionary using defaultdict to simplify code
        temp_m = defaultdict(lambda: defaultdict(int))
        temp_s = set()  # Use set for faster lookup
        
        # Track all words seen during training
        self.all_words = set(words)

        # First pass: collect base frequencies with list comprehension and tqdm
        for i in tqdm(range(num_words - self.n), desc="Collecting Frequencies"):
            g = tuple(words[i:i+self.n])
            next_word = words[i+self.n]
            
            # Record sentence starts inline
            if i == 0 or any(c in self.SENTENCE_END_CHARS for c in words[i-1][-1:]):
                temp_s.add(g)
                
            # Increment frequency directly with defaultdict
            temp_m[g][next_word] += 1

        # Create a list of words from temp_s for later use
        # This step ensures we're working with individual words, not tuples
        temp_s_words = []
        for g in temp_s:
            temp_s_words.extend(g)
        temp_s_words = list(set(temp_s_words))  # Deduplicate
        
        # Second pass: apply isohedral re-weighting using dictionary comprehensions
        print(f"Applying isohedral re-weighting to {len(temp_m)} contexts...")
        
        # Calculate iso_factors for words in temp_s (sentence starters)
        iso_factors = {}
        for word in temp_s_words:
            iso_factors[word] = (len(word) % self.ISO_MODULUS) / self.ISO_DIVISOR
        
        # Also calculate iso_factors for contexts to handle any missing words
        for g in temp_m.keys():
            if g not in iso_factors:
                iso_factors[g] = (sum(len(w) for w in g) % self.ISO_MODULUS) / self.ISO_DIVISOR

        # Build final model with all re-weighted frequencies
        self.m = {}
        for g in tqdm(temp_m.keys(), desc="Re-weighting"):
            self.m[g] = {}
            
            for next_word, freq in temp_m[g].items():
                # Use the iso_factor for the word if available, otherwise use a default
                factor = iso_factors.get(next_word, 0.5)
                self.m[g][next_word] = max(1, int(freq * factor))
                
        # Store valid sentence starts using set operations
        self.s = list(temp_s.intersection(self.m.keys()))
        
        # If no valid starts, use random contexts
        if not self.s and self.m:
            self.s = random.sample(list(self.m.keys()), k=min(len(self.m), 100))
            
        print(f"Training complete. Model has {len(self.m)} contexts and {len(self.s)} sentence starts.")

    def _get_weights(self, options, context, iso_bias_strength):
        """Helper method to calculate weighted probabilities for next word selection"""
        words = list(options.keys())
        freqs = list(options.values())
        
        # Early return if no bias or empty lists
        if iso_bias_strength <= 0 or not words:
            return words, [f/sum(freqs) for f in freqs] if sum(freqs) > 0 else []
            
        # Calculate isohedral factor for current context
        current_f = (sum(len(w) for w in context) % self.ISO_MODULUS) / self.ISO_DIVISOR
        
        # Calculate exponent with bounded range
        exponent = 1.0 + (current_f * iso_bias_strength)
        exponent = max(self.MIN_EXPONENT, min(exponent, self.MAX_EXPONENT))
        
        # Calculate weights with bias applied
        biased = [(f + 1e-9) ** exponent for f in freqs]
        total = sum(biased)
        
        return words, [w/total for w in biased] if total > 0 else []
        
    def gen(self, seed=None, count=100, iso_bias_strength=2.0, window_size=20, word_filter=None):
        """
        Generates text using the trained Markov model with isohedral bias and word filtering.
        Args:
            seed (str, optional): Starting sequence of words.
            count (int): Number of words to generate.
            iso_bias_strength (float): Controls isohedral bias strength.
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
            raise ValueError("No valid starting contexts available.")

        # Process seed if provided
        if seed:
            seed_words = seed.lower().split()
            # Use seed if long enough, otherwise use a random start
            context = tuple(seed_words[-self.n:]) if len(seed_words) >= self.n else random.choice(self.s)
            result = seed_words if len(seed_words) >= self.n else list(context)
        else:
            # No seed, use a random start context
            context = random.choice(self.s)
            result = list(context)

        # Generate text using inline logic
        retry_count = 0
        max_retries = 10
        max_filter_attempts = 5  # Max attempts to find a word that passes the filter
        
        while len(result) < count and retry_count < max_retries:
            # Reset context if invalid and continue
            if context not in self.m or not self.m[context]:
                context = random.choice(self.s)
                result.extend(list(context))
                retry_count += 1
                continue
                
            # Get weighted options
            words, weights = self._get_weights(self.m[context], context, iso_bias_strength)
            
            # Make sure we have valid options
            if not words or not weights:
                context = random.choice(self.s)
                result.extend(list(context))
                retry_count += 1
                continue
            
            # Get current window for filtering
            window_words = result[-min(len(result), window_size):]
            
            # Apply word filtering if provided
            if word_filter is not None:
                # Try a few times to find a word that passes the filter
                found_valid_word = False
                filter_attempts = 0
                
                # Create a copy of words and weights for sampling without replacement
                available_words = words.copy()
                available_weights = weights.copy()
                
                while available_words and filter_attempts < max_filter_attempts:
                    # Choose next word from available options
                    if not available_weights:
                        break
                        
                    # Normalize weights
                    total_weight = sum(available_weights)
                    normalized_weights = [w/total_weight for w in available_weights] if total_weight > 0 else None
                    
                    if normalized_weights:
                        candidate_idx = random.choices(range(len(available_words)), weights=normalized_weights, k=1)[0]
                        candidate_word = available_words[candidate_idx]
                        
                        # Check if word passes the filter
                        if word_filter(candidate_word, window_words):
                            next_word = candidate_word
                            found_valid_word = True
                            break
                            
                        # Remove this word from consideration
                        available_words.pop(candidate_idx)
                        available_weights.pop(candidate_idx)
                    
                    filter_attempts += 1
                
                # If no word passes the filter after max attempts, choose randomly from original list
                if not found_valid_word:
                    next_word = random.choices(words, weights=weights, k=1)[0]
            else:
                # No filter, choose based on original weights
                next_word = random.choices(words, weights=weights, k=1)[0]
            
            # Add the chosen word and update context
            result.append(next_word)
            retry_count = 0  # Reset retry counter on success
            
            # Slide context window
            context = tuple(result[-self.n:])
        
        # Return exactly 'count' words
        return ' '.join(result[-count:])


if __name__ == "__main__":
    # Static configuration
    CONFIG = {
        'input_filename': "xaa",
        'ngram_size': 3,
        'words_to_generate': 250,
        'iso_bias_strength': 2.0,
        'window_size': 200  # Size of moving window for word filtering
    }
    
    print(f"--- IsoMarkov Text Generator with Word Filtering ---")
    print(f"Reading training file: {CONFIG['input_filename']}")
    
    # Read training file
    try:
        with open(CONFIG['input_filename'], 'r', encoding='utf-8') as file:
            txt = ' '.join(file.read().lower().split()[:KB_limit])
    except Exception as e:
        print(f"Error reading file: {e}")
        sys.exit(1)
    
    # Initialize and train model
    model = IsoMarkov(CONFIG['ngram_size'])
    model.train(txt)
    
    # Define a simple word filter function
    # This example prevents word repetition within the window
    def no_repetition_filter(word, window_words):
        return word not in window_words
    
    # Interactive generation loop
    print(f"\n--- Ready to generate (n={CONFIG['ngram_size']}) ---")
    print(f"Enter seed text or press Enter for random start. Type 'exit' to quit.")
    print(f"Using word filter: no_repetition_filter with window size: {CONFIG['window_size']}")
    
    while True:
        try:
            user_input = input("\nSEED: ")
            if user_input.lower() in ['quit', 'exit']:
                break
                
            # Generate text with inline conditional and word filtering
            generated = model.gen(
                seed=user_input or None,
                count=CONFIG['words_to_generate'],
                iso_bias_strength=CONFIG['iso_bias_strength'],
                window_size=CONFIG['window_size'],
                word_filter=no_repetition_filter
            )
            
            # Display result
            print("\n--- Generated Text ---")
            if user_input:
                print(f"(Seed: '{user_input}')")
            print(generated)
            print("-" * 40)
            
        except KeyboardInterrupt:
            print("\nExiting.")
            break
        except Exception as e:
            print(f"Error: {e}")
            import traceback
            traceback.print_exc()
    
    print("--- Generation Complete ---")
