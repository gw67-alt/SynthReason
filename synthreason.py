import random
import math # Needed for exponentiation if used

class IsoMarkov:
    def __init__(self, n=2):
        """
        Initializes the Markov chain model.
        Args:
            n (int): The size of the n-gram (number of context words).
        """
        if not isinstance(n, int) or n < 1:
            raise ValueError("n-gram size 'n' must be a positive integer.")
        self.n = n
        self.m = {}  # Dictionary to store transitions: {context_tuple: {next_word: frequency}}
        self.s = []  # List to store potential sentence starting n-grams

    def train(self, t):
        """
        Trains the Markov model on the provided text, including the
        original 'isohedral' re-weighting based on context word lengths.
        Args:
            t (str): The training text.
        """
        if not isinstance(t, str):
             raise TypeError("Training data 't' must be a string.")
        if not t:
             print("Warning: Training data is empty.")
             return

        words = t.split()

        if len(words) <= self.n:
            print(f"Warning: Training data has {len(words)} words, which is not more than n-gram size {self.n}. Model cannot be trained effectively.")
            return

        print(f"Training started with n={self.n} on {len(words)} words...")

        temp_m = {} # Use a temporary dictionary to build frequencies first
        temp_s = []

        for i in range(len(words) - self.n):
            g = tuple(words[i:i+self.n])
            next_word = words[i+self.n]

            is_start = (i == 0) or (words[i-1][-1] in '.!?')
            if is_start and g not in temp_s:
                temp_s.append(g)

            temp_m.setdefault(g, {})
            temp_m[g][next_word] = temp_m[g].get(next_word, 0) + 1

        print(f"Base frequencies collected. Found {len(temp_m)} unique {self.n}-gram contexts.")
        print("Applying 'isohedral' re-weighting...")

        # Apply isohedral formula: balance transitions (as in original code)
        self.m = {} # Reset self.m to store the re-weighted values
        for g, options in temp_m.items():
            # Ensure context tuple g is not empty before calculating sum
            f = 0.0
            if g:
                try:
                    # Calculate Tessellation factor based on word pattern
                    f = (sum(len(w) for w in g) % 5) / 10.0 # f is 0.0 to 0.4
                except TypeError:
                    print(f"Warning: Skipping isohedral calculation for context {g} due to non-string element.")
                    f = 0.0 # Default factor if calculation fails


            self.m[g] = {} # Initialize context in the final map
            if not options: continue # Skip if no next words recorded

            for w, freq in options.items():
                # Re-weight according to isohedral pattern and convert to int
                # Multiply original frequency by (1 + f)
                reweighted_freq = int(freq * (1 + f))
                # Ensure frequency is at least 1 if it was positive before,
                # as int() can round down to 0. This preserves transitions.
                if freq > 0 and reweighted_freq == 0:
                     reweighted_freq = 1
                if reweighted_freq > 0:
                    self.m[g][w] = reweighted_freq

        self.s = temp_s # Assign collected sentence starts

        # Clean up contexts that might have become empty after re-weighting
        empty_contexts = [g for g, options in self.m.items() if not options]
        for g in empty_contexts:
            del self.m[g]

        print(f"Training complete with re-weighting. {len(self.m)} contexts remain.")
        if not self.s:
             print("Warning: No sentence starts detected. Will use random contexts.")
             if self.m:
                 self.s = random.sample(list(self.m.keys()), k=min(len(self.m), 100))


    def _get_random_start_context(self):
        """Helper function to get a starting context."""
        if self.s:
            return random.choice(self.s)
        elif self.m:
            print("Warning: Starting generation from a random context (no sentence starts identified).")
            return random.choice(list(self.m.keys()))
        else:
            return None

    def gen(self, seed=None, count=100, iso_bias_strength=2.0):
        """
        Generates text using the trained Markov model, with selection bias
        influenced by the 'isohedral factor' of the current context.

        Args:
            seed (str, optional): A starting sequence of words.
            count (int): The desired number of words.
            iso_bias_strength (float): Controls how much the context's 'f' factor
                                     biases the choice towards frequent words.
                                     0 = no extra bias (uses training weights directly).
                                     Higher values = stronger bias when f is high.

        Returns:
            str: The generated text, or an error message.
        """
        if not self.m:
            return "Model not trained. Please call train() first."
        if not isinstance(count, int) or count < 1:
            raise ValueError("'count' must be a positive integer.")

        # --- Seed Processing (Simplified from previous version for clarity) ---
        result = []
        context = None
        if seed:
            seed_words = seed.split()
            if len(seed_words) >= self.n:
                context = tuple(seed_words[-self.n:])
                result = seed_words
            # Basic handling if seed is too short (can be enhanced)
        if not context:
            context = self._get_random_start_context()
            if context:
                result = list(context)
            else:
                 return "Error: Cannot start generation - no valid contexts."
        # --- End Seed Processing ---

        while len(result) < count:
            if context in self.m:
                options = self.m[context] # Frequencies here are already re-weighted from train()

                if not options: # Context exists but has no next words after re-weighting
                    context = self._get_random_start_context()
                    if not context: break
                    result.extend(list(context))
                    continue

                # --- Dynamic Isohedral Bias during Generation ---
                current_f = 0.0
                if context:
                     try:
                         current_f = (sum(len(w) for w in context) % 5) / 10.0 # f = 0.0 to 0.4
                     except TypeError:
                         current_f = 0.0 # Default if calculation fails

                words = list(options.keys())
                base_freqs = list(options.values()) # These are the frequencies from self.m

                if sum(base_freqs) == 0: # Safety check
                    context = self._get_random_start_context()
                    if not context: break
                    result.extend(list(context))
                    continue

                if iso_bias_strength > 0:
                    # Apply *additional* bias based on current_f and strength
                    # Higher f -> higher exponent -> exaggerates differences in base_freqs
                    # Makes high-frequency words even more likely when f is high
                    exponent = 1.0 + (current_f * iso_bias_strength)
                    biased_weights = [(freq + 1e-9) ** exponent for freq in base_freqs]

                    if sum(biased_weights) > 0:
                         next_word = random.choices(words, weights=biased_weights, k=1)[0]
                    else: # Fallback if weights become zero (e.g., underflow)
                         next_word = random.choices(words, weights=base_freqs, k=1)[0]
                else:
                    # No dynamic bias, use the frequencies directly from training
                    next_word = random.choices(words, weights=base_freqs, k=1)[0]
                # --- End Dynamic Isohedral Bias ---

                result.append(next_word)
                context = tuple(list(context)[1:] + [next_word])

            else: # Context not found
                context = self._get_random_start_context()
                if not context: break
                result.extend(list(context))

        return ' '.join(result[:count])


# Example usage function
def demo(input_filename="xaa", ngram_size=3, words_to_generate=100):
    """ Demonstrates training and generating text. """
    print(f"--- Starting Demo (n={ngram_size}, file='{input_filename}') ---")
    try:
        with open(input_filename, 'r', encoding='utf-8') as file:
            txt = file.read().lower()
            print(f"Successfully read training file '{input_filename}'.")
    except FileNotFoundError:
         print(f"Error: Training file '{input_filename}' not found.")
         return f"Training file '{input_filename}' not found."
    except Exception as e:
         print(f"Error reading file '{input_filename}': {e}")
         return f"Error reading training file: {e}"

    if not txt: return "Training file empty."

    m = IsoMarkov(ngram_size)
    m.train(txt)

    if not m.m: return "Model training failed."

    print("\n--- Generation Example (with Isohedral Bias) ---")

    return m.gen(None, words_to_generate, iso_bias_strength=2.0) # Return one sample

# Main execution block
if __name__ == "__main__":
    generated_text = demo(input_filename="xaa", ngram_size=3, words_to_generate=1000)
    print("\n--- Final Sample Output (Iso Bias 2.0) ---")
    print(generated_text)

    print("\n--- Demo Complete ---")
