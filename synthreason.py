import random
import math
import sys
from collections import defaultdict
from tqdm import tqdm

class SymbolicMarkov:
    """
    Markov chain text generator using symbolic probability distribution ⊆⊗∃·Λρ∑ω·Σø²
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
        self.m = {}  # Transitions: {context_tuple: {next_word: frequency}}
        self.s = []  # Sentence starting n-grams
        self.all_words = set()  # Store all words seen during training
    
    def train(self, t):
        """
        Trains the Markov model on the provided text.
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

        # First pass: collect base frequencies with tqdm for progress indication
        for i in tqdm(range(num_words - self.n), desc="Collecting Frequencies"):
            g = tuple(words[i:i+self.n])
            next_word = words[i+self.n]
            
            # Record sentence starts
            if i == 0 or any(c in self.SENTENCE_END_CHARS for c in words[i-1][-1:]):
                temp_s.add(g)
                
            # Increment frequency directly with defaultdict
            temp_m[g][next_word] += 1
                
        # Convert to regular dictionaries for the final model
        self.m = {g: dict(next_words) for g, next_words in temp_m.items()}
        
        # Store valid sentence starts
        self.s = list(filter(lambda g: g in self.m, temp_s))
        
        # If no valid starts, use random contexts
        if not self.s and self.m:
            self.s = random.sample(list(self.m.keys()), k=min(len(self.m), 100))
            
        print(f"Training complete. Model has {len(self.m)} contexts and {len(self.s)} sentence starts.")

    def _symbolic_probability(self, context, options):
        """
        Applies the symbolic probability distribution ⊆⊗∃·Λρ∑ω·Σø² to select the next word.
        
        Args:
            context (tuple): The current context (n-gram)
            options (dict): Possible next words with their frequencies
            
        Returns:
            tuple: (words, weights) Lists of candidate words and their probabilities
        """
        words = list(options.keys())
        freqs = list(options.values())
        
        # Early return if no options
        if not words:
            return [], []
            
        # ⊆ (subset) - Filter options to focus on more frequent words
        # Keep words with frequency > 1 or all if that would leave us with nothing
        subset_indices = [i for i, f in enumerate(freqs) if f > 1]
        
        # If nothing would remain, use all words
        if not subset_indices:
            subset_indices = list(range(len(words)))
            
        subsetWords = [words[i] for i in subset_indices]
        subsetFreqs = [freqs[i] for i in subset_indices]
        
        # If we still have nothing, return empty lists
        if not subsetWords:
            return [], []
            
        # ⊗ (tensor product) - Create relationship between context and options
        tensorValues = []
        for word in subsetWords:
            tensorValue = 1
            for contextWord in context:
                # Find character overlap between context word and candidate
                overlap = len(set(contextWord) & set(word))
                # Use character overlap as a tensor factor (add 1 to avoid zeros)
                tensorValue *= (overlap + 1)
            tensorValues.append(tensorValue)
            
        # ∃ (existential) - Check if there's at least one high probability option
        maxFreq = max(subsetFreqs)
        existsHighProb = any(freq > maxFreq * 0.8 for freq in subsetFreqs)
        
        # Λ (wedge product) - Combine context influence
        # Use last letters of context words to influence selection
        lastLetters = [w[-1] if w else '' for w in context]
        uniqueLastLetters = len(set(lastLetters))
        contextInfluence = math.pow(uniqueLastLetters + 1, 13.5)  # Add 1 to avoid zeros
        
        # ρ (rho) - Base density/distribution
        totalFreq = sum(subsetFreqs)
        baseDistribution = [freq / totalFreq for freq in subsetFreqs]
        
        # ∑ω (sum of weights) - Apply weights based on word properties
        wordWeights = []
        for word in subsetWords:
            # Length factor (longer words get slightly higher weight)
            lengthFactor = math.log(len(word) + 1)
            
            # Vowel-consonant ratio
            vowels = sum(1 for c in word if c.lower() in 'aeiou')
            consonants = len(word) - vowels
            vcRatio = (vowels + 0.1) / (consonants + 0.1)  # Add 0.1 to avoid division by zero
            
            # First letter influence
            firstLetterCode = ord(word[0]) % 10 if word else 0
            
            # Combine factors
            wordWeight = lengthFactor * (vcRatio + 0.5) * (firstLetterCode + 1)
            wordWeights.append(wordWeight)
            
        # Σø² (sum of squared empty set) - Entropy adjustment
        adjustedWeights = []
        for i in range(len(subsetWords)):
            # Combine all the factors with appropriate scaling
            combined = baseDistribution[i] * 5  # Base importance
            combined *= math.pow(tensorValues[i], 0.3)  # Tensor relationship (reduced impact)
            combined *= wordWeights[i] * 0.8  # Word intrinsic properties
            combined *= math.pow(contextInfluence, 0.4)  # Context influence
            
            # If high probability exists, boost diversity of selection (entropy)
            if existsHighProb and baseDistribution[i] < 0.3:
                combined *= 1.5  # Boost low probability options
                
            # Apply the square as indicated by ²
            adjustedWeights.append(math.pow(combined, 2))
            
        # Normalize the final weights
        totalWeight = sum(adjustedWeights)
        if totalWeight == 0:
            normalizedWeights = [1.0 / len(adjustedWeights) for _ in adjustedWeights]
        else:
            normalizedWeights = [w / totalWeight for w in adjustedWeights]
            
        return subsetWords, normalizedWeights

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
            raise ValueError("No valid starting contexts available.")

        # Process seed if provided
        if seed:
            seed_words = seed.lower().split()
            # Use seed if long enough, otherwise use a random start
            if len(seed_words) >= self.n:
                context = tuple(seed_words[-self.n:])
                result = seed_words
            else:
                context = random.choice(self.s)
                result = list(context)
        else:
            # No seed, use a random start context
            context = random.choice(self.s)
            result = list(context)

        # Generate text
        retry_count = 0
        max_retries = 10
        max_filter_attempts = 5  # Max attempts to find a word that passes the filter
        
        while len(result) < count and retry_count < max_retries:
            # Reset context if invalid and continue
            if context not in self.m or not self.m[context]:
                context = random.choice(self.s)
                result.extend(context)
                retry_count += 1
                continue
                
            # Get weighted options using symbolic probability
            words, weights = self._symbolic_probability(context, self.m[context])
            
            # Make sure we have valid options
            if not words or not weights:
                context = random.choice(self.s)
                result.extend(context)
                retry_count += 1
                continue
            
            # Get current window for filtering
            window_words = result[-min(len(result), window_size):]
            
            # Apply word filtering if provided
            if word_filter is not None:
                # Try a few times to find a word that passes the filter
                found_valid_word = False
                filter_attempts = 0
                
                # Create copies for sampling without replacement
                available_words = words.copy()
                available_weights = weights.copy()
                
                while available_words and filter_attempts < max_filter_attempts:
                    # Choose next word from available options
                    if not available_weights:
                        break
                        
                    # Normalize weights
                    total_weight = sum(available_weights)
                    if total_weight == 0:
                        normalized_weights = None
                    else:
                        normalized_weights = [w/total_weight for w in available_weights]
                    
                    if normalized_weights:
                        # Random weighted choice
                        candidate_idx = random.choices(
                            range(len(available_words)), 
                            weights=normalized_weights, 
                            k=1
                        )[0]
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

# Example usage
def no_repetition_filter(word, window_words):
    """Prevents a word from being repeated within the window"""
    return word not in window_words

if __name__ == "__main__":
    # Configuration
    CONFIG = {
        'input_filename': "input.txt",  # Replace with your input file
        'ngram_size': 1,
        'words_to_generate': 230,
        'window_size': 200  # Size of moving window for word filtering
    }
    
    print(f"--- Symbolic Markov Text Generator ---")
    
    # Handle command line arguments if provided
    if len(sys.argv) > 1:
        CONFIG['input_filename'] = sys.argv[1]
    
    
    # Provide a sample text for demonstration if file reading fails
    with open(input("Filename: "), 'r', encoding='utf-8') as file:
        txt = ' '.join(file.read().lower().split())
    
    # Initialize and train model
    model = SymbolicMarkov(CONFIG['ngram_size'])
    model.train(txt)
    
    # Interactive generation loop
    print(f"\n--- Ready to generate text using symbolic probability distribution ⊆⊗∃·Λρ∑ω·Σø² ---")
    print(f"Enter seed text or press Enter for random start. Type 'exit' to quit.")
    print(f"Using word filter: no_repetition_filter with window size: {CONFIG['window_size']}")
    
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
        except Exception as e:
            print(f"Error: {e}")
            import traceback
            traceback.print_exc()
    
    print("--- Generation Complete ---")
