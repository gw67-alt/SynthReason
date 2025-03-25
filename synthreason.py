import random
import numpy as np
from collections import defaultdict, Counter
from typing import List, Tuple, Dict, Optional, Union, Set, Any
import random
import numpy as np
from collections import defaultdict, Counter
from typing import List, Tuple, Dict, Optional, Union, Set, Any, Callable

class NGramTextGenerator:
    def __init__(self, n=3):
        """
        Initialize the N-gram text generator.
        
        Args:
            n (int): Order of the n-gram model (default is trigram).
        """
        self.n = n
        self.word_to_index = {}
        self.index_to_word = {}
        self.ngram_probs = defaultdict(lambda: defaultdict(float))
    
    def _tokenize(self, text):
        """ Tokenizes text into words and lowercases them. """
        return text.lower().split()
    
    def build_vocabulary(self, corpus):
        """
        Builds the vocabulary from a given corpus.
        
        Args:
            corpus (list of str): List of sentences to process.
        """
        unique_words = set()
        for sentence in corpus:
            unique_words.update(self._tokenize(sentence))
        
        # Add special tokens to vocabulary
        special_tokens = ["<START>", "<END>"]
        for token in special_tokens:
            if token not in unique_words:
                unique_words.add(token)
        
        # Assign indices to words
        self.word_to_index = {word: idx for idx, word in enumerate(unique_words, start=1)}
        self.index_to_word = {idx: word for word, idx in self.word_to_index.items()}
    
    def f(self, *words: str) -> Optional[str]:
        """
        Formal function f :⊆ (σ∗)n → r
        
        This is a partial function that maps from n strings (words) to a result string,
        representing the most likely next word given the input context.
        
        Args:
            *words: Variable number of input strings (words) representing the context
            
        Returns:
            Optional[str]: The most likely next word, or None if no prediction can be made
        """
        # Check if we have enough context words based on n-gram order
        if len(words) < self.n - 1:
            return None  # Not enough context for prediction (partial function)
        
        # Get the relevant context (last n-1 words)
        context_words = words[-(self.n-1):]
        
        # Check if all context words are in vocabulary
        for word in context_words:
            if word not in self.word_to_index:
                return None  # Word not in vocabulary (partial function)
        
        # Convert words to indices
        if self.n == 2:  # Bigram case
            context_idx = self.word_to_index[context_words[0]]
            if context_idx not in self.ngram_probs:
                return None  # Context not found in training data (partial function)
                
            # Find most likely next word
            next_word_idx = max(self.ngram_probs[context_idx].items(), 
                              key=lambda x: x[1])[0]
        else:  # Trigram or higher order
            # Create context tuple from indices
            context_indices = tuple(self.word_to_index[w] for w in context_words)
            
            if context_indices not in self.ngram_probs:
                return None  # Context not found in training data (partial function)
                
            # Find most likely next word
            next_word_idx = max(self.ngram_probs[context_indices].items(), 
                              key=lambda x: x[1])[0]
        
        # Convert back to word
        return self.index_to_word[next_word_idx]
    
    def build_ngram_model(self, corpus):
        """
        Builds an n-gram probability model from the given corpus.
        
        Args:
            corpus (list of str): List of sentences.
        """
        if self.n == 2:
            self._build_bigram_model(corpus)
        elif self.n == 3:
            self._build_trigram_model(corpus)
        else:
            raise ValueError(f"N-gram order {self.n} not supported. Use 2 or 3.")
    
    def _build_bigram_model(self, corpus):
        """
        Builds a bigram probability model from the given corpus.
        
        Args:
            corpus (list of str): List of sentences.
        """
        word_counts = defaultdict(lambda: defaultdict(int))

        for sentence in corpus:
            words = ["<START>"] + self._tokenize(sentence) + ["<END>"]
            for i in range(len(words) - 1):
                if words[i] in self.word_to_index and words[i+1] in self.word_to_index:
                    word_counts[words[i]][words[i + 1]] += 1
        
        # Convert counts to probabilities
        for word, next_words in word_counts.items():
            total_count = sum(next_words.values())
            self.ngram_probs[self.word_to_index[word]] = {
                self.word_to_index[next_word]: count / total_count
                for next_word, count in next_words.items()
            }
    
    def _build_trigram_model(self, corpus):
        """
        Builds a trigram probability model from the given corpus.
        
        Args:
            corpus (list of str): List of sentences.
        """
        # For trigrams, we count sequences of three words
        trigram_counts = defaultdict(lambda: defaultdict(int))

        for sentence in corpus:
            words = ["<START>", "<START>"] + self._tokenize(sentence) + ["<END>"]
            
            # Process trigrams
            for i in range(len(words) - 2):
                if (words[i] in self.word_to_index and 
                    words[i+1] in self.word_to_index and 
                    words[i+2] in self.word_to_index):
                    
                    # Create a context key from the first two words
                    context = (self.word_to_index[words[i]], self.word_to_index[words[i+1]])
                    next_word = self.word_to_index[words[i+2]]
                    
                    # Count this trigram
                    trigram_counts[context][next_word] += 1
        
        # Convert counts to probabilities
        for context, next_words in trigram_counts.items():
            total_count = sum(next_words.values())
            self.ngram_probs[context] = {
                next_word: count / total_count
                for next_word, count in next_words.items()
            }
    
    def generate_text(self, seed=None, length=50, temperature=1.0):
        """
        Generate text using the n-gram model with temperature control.
        Stops generation when <END> token is encountered.
        
        Args:
            seed (str, optional): The seed text to start generation.
            length (int): Maximum number of words to generate.
            temperature (float): Controls randomness (higher = more random).
        
        Returns:
            str: The generated text.
        """
        if not self.word_to_index:
            raise ValueError("Vocabulary is empty, cannot generate text.")
        
        valid_indices = list(self.index_to_word.keys())
        if not valid_indices:
            raise ValueError("No valid words in vocabulary.")
            
        # Get END token index if it exists
        end_token_idx = self.word_to_index.get("<END>")

        generated_indices = []

        # Process seed text if provided
        if seed:
            seed_words = self._tokenize(seed)
            seed_indices = []
            unknown_words = []
            
            for word in seed_words:
                if word in self.word_to_index:
                    seed_indices.append(self.word_to_index[word])
                else:
                    unknown_words.append(word)
            
            if unknown_words:
                unknown_str = ", ".join(f"'{word}'" for word in unknown_words)
                print(f"Warning: The following seed words are not in vocabulary: {unknown_str}")
            
            if not seed_indices:
                raise ValueError("No valid words from the seed exist in the vocabulary.")
            
            generated_indices = seed_indices
            
            # Set up the context based on n-gram order
            if self.n == 2:
                current_idx = seed_indices[-1]
            else:  # For trigram or higher
                # We need at least n-1 words for context
                if len(seed_indices) < self.n - 1:
                    # If not enough words in seed, prepend with START
                    num_pad = self.n - 1 - len(seed_indices)
                    start_idx = self.word_to_index.get("<START>", random.choice(valid_indices))
                    context = tuple([start_idx] * num_pad + seed_indices)
                else:
                    # Use the last n-1 words as context
                    context = tuple(seed_indices[-(self.n-1):])
        else:
            # No seed provided
            if self.n == 2:
                # For bigram, just pick a random starting word
                current_idx = random.choice(valid_indices)
                generated_indices = [current_idx]
            else:
                # For trigram or higher, initialize with START tokens or random words
                start_idx = self.word_to_index.get("<START>", random.choice(valid_indices))
                context = tuple([start_idx] * (self.n - 1))
                generated_indices = list(context)

        # Generate the text sequence
        remaining_length = max(0, length - len(generated_indices))
        
        for _ in range(remaining_length):
            # Different handling based on n-gram order
            if self.n == 2:
                # Bigram generation
                if current_idx not in self.ngram_probs:
                    next_idx = random.choice(valid_indices)
                else:
                    candidates = list(self.ngram_probs[current_idx].keys())
                    probabilities = np.array(list(self.ngram_probs[current_idx].values()))
                    
                    # Apply temperature
                    adjusted_probs = probabilities ** (1 / max(0.1, temperature))
                    total = adjusted_probs.sum()
                    
                    if total > 0:
                        adjusted_probs = adjusted_probs / total
                        next_idx = np.random.choice(candidates, p=adjusted_probs)
                    else:
                        next_idx = random.choice(candidates)
                
                generated_indices.append(next_idx)
                
                # Stop if END token is generated
                if end_token_idx and next_idx == end_token_idx:
                    break
                    
                current_idx = next_idx
            else:
                # Trigram or higher generation
                if context not in self.ngram_probs:
                    next_idx = random.choice(valid_indices)
                else:
                    candidates = list(self.ngram_probs[context].keys())
                    probabilities = np.array(list(self.ngram_probs[context].values()))
                    
                    # Apply temperature
                    adjusted_probs = probabilities ** (1 / max(0.1, temperature))
                    total = adjusted_probs.sum()
                    
                    if total > 0:
                        adjusted_probs = adjusted_probs / total
                        next_idx = np.random.choice(candidates, p=adjusted_probs)
                    else:
                        next_idx = random.choice(candidates)
                
                generated_indices.append(next_idx)
                
                # Stop if END token is generated
                if end_token_idx and next_idx == end_token_idx:
                    break
                    
                # Slide the context window forward
                context = tuple(list(context)[1:] + [next_idx])

        # Convert indices back to words
        generated_words = [self.index_to_word[idx] for idx in generated_indices]
        
        # Remove START tokens from output but keep the END token (since we want to show where it ended)
        if "<START>" in self.word_to_index:
            generated_words = [w for w in generated_words if w != "<START>"]
            
        return " ".join(generated_words)

class SelfMeasuredNGramTextGenerator(NGramTextGenerator):
    def __init__(self, n=3):
        """
        Initialize the Self-Measured N-gram text generator.
        
        Args:
            n (int): Order of the n-gram model (default is trigram).
        """
        super().__init__(n)
        
        # Self-measurement attributes
        self.generation_history = []
        self.self_measure_functions = {
            'entropy': self._entropy_self_measure,
            'uniqueness': self._uniqueness_self_measure,
            'repetitiveness': self._repetitiveness_self_measure
        }
    
    def c(self, measure_type: str = 'entropy') -> float:
        """
        Self-measurement function c(x)
        
        Args:
            measure_type (str): Type of self-measurement to perform
        
        Returns:
            float: Measurement value representing the current self
        """
        if not self.generation_history:
            return 0.0
        
        # Select the appropriate measurement function
        measure_func = self.self_measure_functions.get(measure_type)
        
        if measure_func is None:
            raise ValueError(f"Unknown self-measurement type: {measure_type}")
        
        return measure_func()
    
    def _entropy_self_measure(self) -> float:
        """
        Calculate entropy of generated text as a self-measurement.
        
        Returns:
            float: Entropy value representing text diversity
        """
        # Flatten the generation history
        all_words = [word for text in self.generation_history for word in self._tokenize(text)]
        
        # Count word frequencies
        word_counts = Counter(all_words)
        total_words = len(all_words)
        
        # Calculate entropy
        entropy = 0.0
        for count in word_counts.values():
            prob = count / total_words
            entropy -= prob * np.log2(prob)
        
        return entropy
    
    def _uniqueness_self_measure(self) -> float:
        """
        Measure the uniqueness of generated text.
        
        Returns:
            float: Uniqueness score (ratio of unique words to total words)
        """
        # Flatten the generation history
        all_words = [word for text in self.generation_history for word in self._tokenize(text)]
        
        # Calculate uniqueness
        unique_words = len(set(all_words))
        total_words = len(all_words)
        
        return unique_words / total_words if total_words > 0 else 0.0
    
    def _repetitiveness_self_measure(self) -> float:
        """
        Measure the repetitiveness of generated text.
        
        Returns:
            float: Repetitiveness score (ratio of repeated words)
        """
        # Flatten the generation history
        all_words = [word for text in self.generation_history for word in self._tokenize(text)]
        
        # Count word frequencies
        word_counts = Counter(all_words)
        
        # Calculate repetitiveness (proportion of words appearing more than once)
        repeated_words = sum(1 for count in word_counts.values() if count > 1)
        total_unique_words = len(word_counts)
        
        return repeated_words / total_unique_words if total_unique_words > 0 else 0.0
    
    def generate_text(self, seed=None, length=50, temperature=1.0, 
                      self_measure_type: Optional[str] = None) -> str:
        """
        Generate text with optional self-measurement.
        
        Args:
            seed (str, optional): The seed text to start generation.
            length (int): Maximum number of words to generate.
            temperature (float): Controls randomness.
            self_measure_type (str, optional): Type of self-measurement to apply.
        
        Returns:
            str: The generated text.
        """
        # Generate text using the parent class method
        generated_text = super().generate_text(seed, length, temperature)
        
        # Record generated text in history
        self.generation_history.append(generated_text)
        
        # Apply self-measurement if specified
        if self_measure_type:
            # Modify generation based on self-measurement
            current_self = self.c(self_measure_type)
            
            # Example modification: adjust temperature based on self-measurement
            # 1 - c(x) to invert the measure
            adjusted_temperature = temperature * (1 - current_self)
            
            # Regenerate with adjusted temperature
            generated_text = super().generate_text(seed, length, adjusted_temperature)
            
            # Update generation history with modified text
            self.generation_history[-1] = generated_text
        
        return generated_text
    
    def reset_self_measurement(self):
        """
        Reset the generation history and self-measurement state.
        """
        self.generation_history = []

# Example usage
if __name__ == "__main__":
    # Read corpus from file
    with open("test.txt", 'r', encoding="utf-8") as file:
        corpus = file.read().lower().split(".")
    
    # Initialize and train generator
    gen = SelfMeasuredNGramTextGenerator(n=3)
    gen.build_vocabulary(corpus)
    gen.build_ngram_model(corpus)
    
    while True:
        generated_text = gen.generate_text(
        seed=input("USER: "), 
        length=50, 
        temperature=0.8, 
        self_measure_type='uniqueness'
        )
        print(f"{generated_text}")
