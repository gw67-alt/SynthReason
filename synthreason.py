import random
import numpy as np
from collections import defaultdict

class NGramTextGenerator:
    def __init__(self, n=2):
        """
        Initialize the N-gram text generator.
        
        Args:
            n (int): Order of the n-gram model (default is bigram).
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
        
        # Assign indices to words
        self.word_to_index = {word: idx for idx, word in enumerate(unique_words, start=1)}
        self.word_to_index["<UNK>"] = 0  # Unknown token
        self.index_to_word = {idx: word for word, idx in self.word_to_index.items()}
    
    def build_bigram_model(self, corpus):
        """
        Builds a bigram probability model from the given corpus.
        
        Args:
            corpus (list of str): List of sentences.
        """
        word_counts = defaultdict(lambda: defaultdict(int))

        for sentence in corpus:
            words = ["<START>"] + self._tokenize(sentence) + ["<END>"]
            for i in range(len(words) - 1):
                word_counts[words[i]][words[i + 1]] += 1
        
        # Convert counts to probabilities
        for word, next_words in word_counts.items():
            total_count = sum(next_words.values())
            self.ngram_probs[self.word_to_index.get(word, 0)] = {
                self.word_to_index.get(next_word, 0): count / total_count
                for next_word, count in next_words.items()
            }

    def generate_text(self, seed=None, length=50, temperature=1.0, elasticity_factor=1.5):
        """
        Generate text using a bigram model with elasticity towards smaller words.
        
        Args:
            seed (str, optional): The seed text to start generation.
            length (int): Number of words to generate.
            temperature (float): Controls randomness (higher = more random).
            elasticity_factor (float): Bias towards shorter words.
        
        Returns:
            str: The generated text.
        """
        if not self.word_to_index:
            raise ValueError("Vocabulary is empty, cannot generate text.")
        
        valid_indices = [idx for idx in self.index_to_word if idx > 0]
        if not valid_indices:
            raise ValueError("No valid words in vocabulary.")

        generated_indices = []

        # Process seed text if provided
        if seed:
            seed_words = self._tokenize(seed)
            seed_indices = [
                self.word_to_index.get(word, self.word_to_index["<UNK>"]) for word in seed_words
            ]
            generated_indices = seed_indices
            current_idx = seed_indices[-1]
        else:
            # Randomly select a starting word
            current_idx = random.choice(valid_indices)
            generated_indices = [current_idx]

        # Generate sequence
        for _ in range(length - len(generated_indices)):
            if current_idx not in self.ngram_probs:
                next_idx = random.choice(valid_indices)
            else:
                candidates = list(self.ngram_probs[current_idx].keys())
                probabilities = np.array([self.ngram_probs[current_idx][idx] for idx in candidates])

                # Apply temperature
                probabilities = probabilities ** (1 / max(0.1, temperature))
                probabilities /= probabilities.sum()  # Normalize

                next_idx = np.random.choice(candidates, p=probabilities)
            
            generated_indices.append(next_idx)
            current_idx = next_idx

        # Convert indices back to words
        generated_words = [self.index_to_word[idx] for idx in generated_indices]
        return " ".join(generated_words)

# Usage Example
with open("test.txt", 'r', encoding="utf-8") as file:
    corpus = file.read().lower().split(".")
gen = NGramTextGenerator(n=3)
gen.build_vocabulary(corpus)
gen.build_bigram_model(corpus)

# Generate text
while True:
    print(gen.generate_text(seed=input("USER: "), length=10, temperature=0.8))
