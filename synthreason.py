import random
import numpy as np
from collections import defaultdict
from typing import List, Tuple, Dict, Optional, Union, Set, Any

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
    
    def generate_text(self, seed=None, length=50, temperature=1.0, elasticity_factor=1.5):
        """
        Generate text using the n-gram model with temperature control.
        Stops generation when <END> token is encountered.
        
        Args:
            seed (str, optional): The seed text to start generation.
            length (int): Maximum number of words to generate.
            temperature (float): Controls randomness (higher = more random).
            elasticity_factor (float): Bias towards shorter words.
        
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

# Example usage with end token stopping
if __name__ == "__main__":
    with open("test.txt", 'r', encoding="utf-8") as file:
        corpus = file.read().lower().split(".")
        
    gen = NGramTextGenerator(n=3)
    gen.build_vocabulary(corpus)
    gen.build_ngram_model(corpus)
    
    print("\nDemonstration of formal function f :⊆ (σ∗)n → r")
    print("-------------------------------------------------")
            
    while True:
        try:
            user_input = input("\nEnter seed text (or 'q' to quit): ")
            if user_input.lower() == 'q':
                break
                
            generated_text = gen.generate_text(seed=user_input, length=50, temperature=0.8)
            print(generated_text)
        except ValueError as e:
            print(f"Error: {e}. Please try again with different input.")
