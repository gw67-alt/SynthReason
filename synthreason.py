import random
import math
import sys
from tqdm import tqdm
from collections import defaultdict

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

        # First pass: collect base frequencies with list comprehension and tqdm
        for i in tqdm(range(num_words - self.n), desc="Collecting Frequencies"):
            g = tuple(words[i:i+self.n])
            next_word = words[i+self.n]
            
            # Record sentence starts inline
            if i == 0 or any(c in self.SENTENCE_END_CHARS for c in words[i-1][-1:]):
                temp_s.add(g)
                
            # Increment frequency directly with defaultdict
            temp_m[g][next_word] += 1

        # Second pass: apply isohedral re-weighting using dictionary comprehensions
        print(f"Applying isohedral re-weighting to {len(temp_m)} contexts...")
        
        # Calculate all isohedral factors at once
        iso_factors = {
            g: (sum(len(w) for w in g) % self.ISO_MODULUS) / self.ISO_DIVISOR 
            for g in temp_m.keys()
        }

        # Build final model with all re-weighted frequencies
        self.m = {
            g: {
                w: max(1, int(freq * iso_factors[g]))  # Use iso_factors[g] directly
                for w, freq in temp_m[g].items()
            }
            for g in tqdm(temp_m.keys(), desc="Re-weighting")
        }
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
        
        # Early return if no bias
        if iso_bias_strength <= 0:
            return words, [f/sum(freqs) for f in freqs]
            
        # Calculate isohedral factor for current context
        current_f = (sum(len(w) for w in context) % self.ISO_MODULUS) / self.ISO_DIVISOR
        
        # Calculate exponent with bounded range
        exponent = 1.0 + (current_f * iso_bias_strength)
        exponent = max(self.MIN_EXPONENT, min(exponent, self.MAX_EXPONENT))
        
        # Calculate weights with bias applied
        biased = [(f + 1e-9) ** exponent for f in freqs]
        total = sum(biased)
        
        return words, [w/total for w in biased]
        
    def gen(self, seed=None, count=100, iso_bias_strength=2.0):
        """
        Generates text using the trained Markov model with isohedral bias.
        Args:
            seed (str, optional): Starting sequence of words.
            count (int): Number of words to generate.
            iso_bias_strength (float): Controls isohedral bias strength.
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
        while len(result) < count:
            # Reset context if invalid and continue
            if context not in self.m or not self.m[context]:
                context = random.choice(self.s)
                result.extend(list(context))
                continue
                
            # Get weighted options
            words, weights = self._get_weights(self.m[context], context, iso_bias_strength)
            
            # Choose and append next word
            next_word = random.choices(words, weights=weights, k=1)[0]
            result.append(next_word)
            
            # Slide context window
            context = tuple(result[-self.n:])
        
        # Return exactly 'count' words
        return ' '.join(result[-count:])


if __name__ == "__main__":
    # Static configuration
    CONFIG = {
        'input_filename': "test.txt",
        'ngram_size': 3,
        'words_to_generate': 250,
        'iso_bias_strength': 2.0
    }
    
    print(f"--- IsoMarkov Text Generator ---")
    print(f"Reading training file: {CONFIG['input_filename']}")
    
    # Read training file
    try:
        with open(CONFIG['input_filename'], 'r', encoding='utf-8') as file:
            txt = file.read().lower()
    except Exception as e:
        print(f"Error reading file: {e}")
        sys.exit(1)
    
    # Initialize and train model
    model = IsoMarkov(CONFIG['ngram_size'])
    model.train(txt)
    
    # Interactive generation loop
    print(f"\n--- Ready to generate (n={CONFIG['ngram_size']}) ---")
    print(f"Enter seed text or press Enter for random start. Type 'exit' to quit.")
    
    while True:
        try:
            user_input = input("\nSEED: ")
            if user_input.lower() in ['quit', 'exit']:
                break
                
            # Generate text with inline conditional
            generated = model.gen(
                seed=user_input or None,
                count=CONFIG['words_to_generate'],
                iso_bias_strength=CONFIG['iso_bias_strength']
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
    
    print("--- Generation Complete ---")
