import random
import math
import sys
from tqdm import tqdm

class IsoMarkov:
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

        # Build frequency dictionary
        temp_m = {}
        temp_s = []

        # First pass: collect base frequencies
        for i in tqdm(range(num_words - self.n), desc="Collecting Frequencies"):
            g = tuple(words[i:i+self.n])
            next_word = words[i+self.n]

            # Record sentence starts
            if (i == 0) or (words[i-1][-1] in '.!?'):
                if g not in temp_s:
                    temp_s.append(g)

            # Record transition
            temp_m.setdefault(g, {})
            temp_m[g][next_word] = temp_m[g].get(next_word, 0) + 1

        # Second pass: apply isohedral re-weighting
        print(f"Applying isohedral re-weighting to {len(temp_m)} contexts...")
        
        for g, options in tqdm(temp_m.items(), desc="Re-weighting"):
            # Calculate isohedral factor
            f = (sum(len(w) for w in g) % 5) / 10.0  # f is 0.0 to 0.4
            
            # Create entry in final dictionary
            self.m[g] = {}
            
            # Apply re-weighting to each transition
            for w, freq in options.items():
                reweighted_freq = max(1, int(freq * (1 + f)))  # Ensure at least 1
                self.m[g][w] = reweighted_freq

        # Store valid sentence starts
        self.s = [start for start in temp_s if start in self.m]
        
        # If no valid starts, use random contexts
        if not self.s and self.m:
            self.s = random.sample(list(self.m.keys()), k=min(len(self.m), 100))
            
        print(f"Training complete. Model has {len(self.m)} contexts and {len(self.s)} sentence starts.")

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

        # Initialize generation
        result = []
        context = None
        
        # Process seed if provided
        if seed:
            seed_words = seed.lower().split()
            if len(seed_words) >= self.n:
                context = tuple(seed_words[-self.n:])
                result = seed_words
            else:
                # Seed too short, use a random start context
                context = random.choice(self.s)
                result = list(context)
        else:
            # No seed, use a random start context
            context = random.choice(self.s)
            result = list(context)

        # Generate text
        generated = len(result)
        while generated < count:
            # Ensure context exists in our model
            if context not in self.m:
                context = random.choice(self.s)
                result.extend(list(context))
                generated = len(result)
                continue
                
            options = self.m[context]
            if not options:
                context = random.choice(self.s)
                result.extend(list(context))
                generated = len(result)
                continue
            
            # Calculate isohedral factor for current context
            current_f = (sum(len(w) for w in context) % 5) / 10.0  # f is 0.0 to 0.4
            
            # Get words and their frequencies
            words = list(options.keys())
            freqs = list(options.values())
            
            # Apply isohedral bias to weights
            if iso_bias_strength > 0:
                exponent = 1.0 + (current_f * iso_bias_strength)
                exponent = max(0.1, min(exponent, 5.0))  # Limit exponent range
                weights = [(f + 1e-9) ** exponent for f in freqs]
                
                # Normalize weights
                total = sum(weights)
                weights = [w / total for w in weights]
            else:
                # No bias, use original frequencies
                total = sum(freqs)
                weights = [f / total for f in freqs]
            
            # Choose next word
            next_word = random.choices(words, weights=weights, k=1)[0]
            result.append(next_word)
            generated += 1
            
            # Update context window
            context = tuple(result[-self.n:])
        
        # Return exactly 'count' words
        return ' '.join(result[-count:])


if __name__ == "__main__":
    # Configuration
    input_filename = "test.txt"
    ngram_size = 3
    words_to_generate = 250
    
    print(f"--- IsoMarkov Text Generator ---")
    print(f"Reading training file: {input_filename}")
    
    try:
        with open(input_filename, 'r', encoding='utf-8') as file:
            txt = file.read().lower()
    except Exception as e:
        print(f"Error reading file: {e}")
        sys.exit(1)
    
    # Initialize and train model
    model = IsoMarkov(ngram_size)
    model.train(txt)
    
    # Interactive generation loop
    print(f"\n--- Ready to generate (n={ngram_size}) ---")
    print(f"Enter seed text or press Enter for random start. Type 'exit' to quit.")
    
    while True:
        try:
            user_input = input("\nSEED: ")
            if user_input.lower() in ['quit', 'exit']:
                break
                
            # Generate text
            generated = model.gen(
                seed=user_input if user_input else None,
                count=words_to_generate,
                iso_bias_strength=2.0
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