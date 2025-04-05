import random
import math # Needed for exponentiation if used
import sys # Added previously, good practice

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
            # Ensure g contains actual words before adding to starts
            if is_start and g and all(isinstance(word, str) and word for word in g):
                if g not in temp_s:
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
            valid_context = g and all(isinstance(w, str) for w in g) # Check if context words are strings
            if valid_context:
                try:
                    # Calculate Tessellation factor based on word pattern
                    f = (sum(len(w) for w in g) % 5) / 10.0 # f is 0.0 to 0.4
                except TypeError:
                    print(f"Warning: Skipping isohedral calculation for context {g} due to non-string element.")
                    f = 0.0 # Default factor if calculation fails
            else:
                 f = 0.0 # Default factor for invalid contexts

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
             if g in self.m: # Check existence before deleting
                del self.m[g]
        # Clean up sentence starts that may no longer be valid contexts
        self.s = [start for start in self.s if start in self.m]


        print(f"Training complete with re-weighting. {len(self.m)} contexts remain.")
        if not self.s:
             print("Warning: No sentence starts detected or valid after re-weighting. Will use random contexts.")
             # Try to populate self.s with *valid* keys if empty
             if self.m:
                 valid_keys = list(self.m.keys())
                 self.s = random.sample(valid_keys, k=min(len(valid_keys), 100))


    def _get_random_start_context(self):
        """Helper function to get a starting context."""
        if self.s:
            return random.choice(self.s)
        elif self.m:
            print("Warning: Starting generation from a random context (no sentence starts identified/valid).")
            # Ensure we pick from keys that actually exist
            valid_keys = list(self.m.keys())
            if valid_keys:
                return random.choice(valid_keys)
        return None # Return None if no options available

    def gen(self, seed=None, count=100, iso_bias_strength=2.0):
        """
        Generates text using the trained Markov model, with selection bias
        influenced by the 'isohedral factor' of the current context.

        Args:
            seed (str, optional): A starting sequence of words.
            count (int): The desired number of words.
            iso_bias_strength (float): Controls how much the context's 'f' factor
                                     biases the choice towards frequent words.

        Returns:
            str: The generated text, or an error message.
        """
        if not self.m:
            return "Model not trained or no valid contexts remain after training. Please call train() first."
        if not isinstance(count, int) or count < 1:
            raise ValueError("'count' must be a positive integer.")

        # --- Seed Processing ---
        result = []
        context = None

        if seed:
            seed_words_list = seed.lower().split() # Process seed same way as training data
            if len(seed_words_list) >= self.n:
                # Use last n words directly as context
                context = tuple(seed_words_list[-self.n:])
                result = seed_words_list # Start result with the full seed
            else:
                # Seed is shorter than n, try to pad
                print(f"Warning: Seed '{seed}' is shorter than n={self.n}. Trying to pad...")
                start_context_full = self._get_random_start_context()
                if start_context_full:
                    num_padding_words = self.n - len(seed_words_list)
                    # Take first part of a random start context for padding
                    padding = list(start_context_full)[:num_padding_words]
                    context = tuple(padding + seed_words_list)
                    result = padding + seed_words_list
                    print(f"Padded seed context: {context}")
                else:
                     # Cannot pad if no random starts available
                     print("Error: Cannot pad seed - no valid starting contexts available.")
                     context = None # Fallback to random start if possible below

        # If no seed provided OR padding failed OR seed context invalid
        if not context or context not in self.m:
            if context: # only print warning if context was attempted but failed
                 print(f"Warning: Seed context {context} not found in model. Starting randomly.")
            context = self._get_random_start_context()
            if context:
                result = list(context) # Start result with the random context
            else:
                 # Critical error: No valid seed, no valid random start
                 return "Error: Cannot start generation - no valid contexts found in model."
        # --- End Seed Processing ---

        while len(result) < count:
            # Check if current context is valid before proceeding
            if context not in self.m:
                print(f"\nWarning: Fell into unknown context {context}. Attempting to restart...")
                context = self._get_random_start_context()
                if not context:
                    print("Error: Cannot recover from unknown context - no valid contexts left.")
                    break # Stop generation
                result.extend(list(context)) # Add the restart context words
                print(f"Restarted with context: {context}")
                continue # Try generating from new context

            options = self.m[context] # Frequencies here are already re-weighted from train()

            if not options: # Context exists but has no next words
                print(f"\nWarning: Context {context} has no recorded next words. Attempting to restart...")
                context = self._get_random_start_context()
                if not context:
                    print("Error: Cannot recover from dead-end context - no valid contexts left.")
                    break # Stop generation
                result.extend(list(context)) # Add the restart context words
                print(f"Restarted with context: {context}")
                continue # Try generating from new context

            # --- Dynamic Isohedral Bias during Generation ---
            current_f = 0.0
            valid_context = context and all(isinstance(w, str) for w in context)
            if valid_context:
                 try:
                     current_f = (sum(len(w) for w in context) % 5) / 10.0 # f = 0.0 to 0.4
                 except TypeError:
                     current_f = 0.0 # Default if calculation fails
            else:
                current_f = 0.0 # Default for invalid contexts

            words = list(options.keys())
            base_freqs = list(options.values()) # These are the frequencies from self.m

            # Ensure base_freqs sum is > 0 before proceeding
            if sum(base_freqs) == 0:
                 print(f"\nWarning: Context {context} has zero total frequency for next words. Attempting restart.")
                 context = self._get_random_start_context()
                 if not context:
                     print("Error: Cannot recover from zero-frequency context.")
                     break
                 result.extend(list(context))
                 print(f"Restarted with context: {context}")
                 continue

            next_word = None # Initialize next_word
            if iso_bias_strength > 0:
                exponent = 1.0 + (current_f * iso_bias_strength)
                biased_weights = [(freq + 1e-9) ** exponent for freq in base_freqs]

                if sum(biased_weights) > 1e-9: # Check if weights sum is effectively > 0
                     try:
                         next_word = random.choices(words, weights=biased_weights, k=1)[0]
                     except ValueError:
                          print(f"\nWarning: random.choices failed with biased weights for {context}. Weights: {biased_weights[:5]}... Using fallback.")
                          # Fallback to standard weights if biased choice fails
                          next_word = random.choices(words, weights=base_freqs, k=1)[0]
                else: # Fallback if biased weights sum is near zero
                     next_word = random.choices(words, weights=base_freqs, k=1)[0]
            else:
                # No dynamic bias, use the frequencies directly from training
                next_word = random.choices(words, weights=base_freqs, k=1)[0]
            # --- End Dynamic Isohedral Bias ---

            result.append(next_word)
            # Ensure the chosen word is a string before adding to context
            if isinstance(next_word, str):
                 context = tuple(list(context)[1:] + [next_word])
            else:
                 print(f"\nError: Generated non-string word '{next_word}'. Stopping generation.")
                 break # Stop if something went wrong

        return ' '.join(result[:count])

# Main execution block
if __name__ == "__main__":
    """ Demonstrates training and generating text, potentially using a seed. """
    input_filename="xaa"
    with open(input_filename, 'r', encoding='utf-8') as file:
        txt = file.read().lower()
 
    m = IsoMarkov(3)
    m.train(txt)


    words_to_generate = 250 # Generate 250 words as per previous example

    while True:
        seed_text = input("USER: ")

        # Pass the seed text to the demo function
        generated_text = print(m.gen(seed=seed_text, count=words_to_generate, iso_bias_strength=2.0))
