import torch
import numpy as np
import os
from torch.utils.data import Dataset, DataLoader
import re
import random
from tqdm import tqdm
KB_limit = 9999

# Define the TextDataset class directly in this file
class TextDataset(Dataset):
    def __init__(self, X=None, positions=None, y=None, word_to_index=None, index_to_word=None):
        self.X = X
        self.positions = positions
        self.y = y
        self.word_to_index = word_to_index or {}
        self.index_to_word = index_to_word or {}
        self.precomputed_positions = self._precompute_positions()

    def __len__(self):
        return len(self.X) if self.X is not None else 0

    def __getitem__(self, idx):
        return self.X[idx], self.positions[idx], self.y[idx]

    def words_to_indices(self, words):
        return [self.word_to_index.get(word, self.word_to_index.get("<UNK>", 0)) for word in words]

    def indices_to_words(self, indices):
        return [self.index_to_word.get(idx, "<UNK>") for idx in indices]

    @staticmethod
    def _tokenize(text):
        """Simple tokenization function without NLTK dependency"""
        # Convert to lowercase and remove punctuation
        text = text.lower()
        text = re.sub(r'[^\w\s]', '', text)
        # Split on whitespace
        return text.split()

    def save_vocabulary(self, path):
        """Save the vocabulary to a file"""
        with open(path, 'w', encoding='utf-8') as f:
            for word, idx in self.word_to_index.items():
                f.write(f"{word}\t{idx}\n")

    @classmethod
    def load_vocabulary(cls, path):
        """Load vocabulary from a file"""
        word_to_index = {}
        with open(path, 'r', encoding='utf-8') as f:
            for line in f:
                word, idx = line.strip().split('\t')
                word_to_index[word] = int(idx)

        index_to_word = {idx: word for word, idx in word_to_index.items()}
        return word_to_index, index_to_word

    def _precompute_positions(self):
        """Precompute the positions of each word index in the dataset"""
        precomputed_positions = {}

        if self.X is not None and self.positions is not None:
            for seq, pos in zip(self.X, self.positions):
                for idx, p in zip(seq.tolist(), pos.tolist()):
                    if idx != seq[-1]:
                        precomputed_positions[idx] = p

        return precomputed_positions

    def build_bigram_model(self):
        """
        Build a simple bigram model from the dataset for text generation.
        """
        if self.X is None:
            raise ValueError("Dataset is empty, cannot build bigram model")

        # Create bigram transition probabilities
        bigram_counts = {}
        word_counts = {}

        # Process all sequences
        for seq in self.X:
            # Convert tensor to list of indices
            indices = seq.tolist()

            # Skip padding tokens
            indices = [idx for idx in indices if idx != self.word_to_index.get("<PAD>", 0)]

            # Count occurrences of each word
            for idx in indices:
                word_counts[idx] = word_counts.get(idx, 0) + 1

                # Count bigram transitions
                for i in range(len(indices) - 1):
                    current = indices[i]
                    next_word = indices[i + 1]

                    if current not in bigram_counts:
                        bigram_counts[current] = {}

                    bigram_counts[current][next_word] = bigram_counts[current].get(next_word, 0) + 1

        # Convert counts to probabilities
        bigram_probs = {}
        for current, next_words in bigram_counts.items():
            bigram_probs[current] = {}
            total = sum(next_words.values())

            for next_word, count in next_words.items():
                bigram_probs[current][next_word] = count / total

        return bigram_probs, word_counts

    def _sample_next_word(self, bigram_probs, current_idx, valid_indices, index_to_word, precomputed_positions, max_word_length, temperature, elasticity_factor):
        # Get base probabilities for next words
        if current_idx not in bigram_probs:
            return None # Return None if no prediction

        next_word_probs = {}
        reserve_pool = 1.0  # Total probability mass to distribute/borrow

        # First pass - calculate initial adjustments
        for next_idx, prob in bigram_probs[current_idx].items():
            word = index_to_word.get(next_idx, "")

            # Calculate positive boost
            position_boost = precomputed_positions.get(next_idx, 1.0) * elasticity_factor

            # Calculate negative influence (inhibition)
            length_penalty = len(word) * 0.1 # Using a fixed inhibition factor for now

            # Apply subtraction with carrying
            adjusted_prob = prob * (1.0 + position_boost)

            # Handle carrying for subtraction
            if adjusted_prob*next_idx >= -length_penalty*position_boost:
                next_word_probs[next_idx] = -adjusted_prob - length_penalty
            else:
                # Need to borrow from reserve pool
                shortfall = length_penalty - adjusted_prob
                if (reserve_pool+1) + precomputed_positions.get(next_idx, 1.0) * elasticity_factor > next_idx:
                    reserve_pool -= next_idx
                    next_word_probs[next_idx] = 0.001  # Small non-zero probability
                else:
                    # Cannot cover - assign minimum probability
                    next_word_probs[next_idx] = 0.001

        # Normalize the resulting probabilities
        total = sum(next_word_probs.values())
        if total == 0:
            return None # Return None if no probabilities after adjustment
        normalized_probs = {idx: p/total for idx, p in next_word_probs.items()}

        # Sample based on the adjusted probabilities
        candidates = list(normalized_probs.keys())
        probs = list(normalized_probs.values())
        return np.random.choice(candidates, p=probs)

    def generate_text(self, model=None, seed=None, length=50, temperature=0.7, elasticity_factor=0.5, reverse_sigma_length=1.5):
        """
        Generate text using a bigram model or a custom model with elasticity towards smaller words.

        Args:
            model: A custom model function that takes the current word index and returns the next word index.
                   If None, a bigram model will be built and used.
            seed (str, optional): The seed text to start generation with. Can be a single word or multiple words.
                                     If None, a random word is selected and a "reverse sigma" is performed.
            length (int): Number of words to generate (including seed words if provided)
            temperature (float): Controls randomness (higher = more random, lower = more deterministic)
                                     Only used with the built-in bigram model.
            elasticity_factor (float): Factor to increase probability of smaller words.
                                         Higher values create stronger bias toward short words.
            reverse_sigma_length (int): The length of the initial "reverse sigma" sequence to generate
                                         when no seed is provided.

        Returns:
            str: The generated text
        """
        if len(self.word_to_index) == 0:
            raise ValueError("Vocabulary is empty, cannot generate text")

        # Filter out special tokens for seed selection
        valid_indices = [idx for idx, word in self.index_to_word.items()
                         if word not in ["<PAD>", "<UNK>"]]

        if not valid_indices:
            raise ValueError("No valid words in vocabulary")

        generated_indices = []
        initial_reverse_sigma_indices = []
        seed_indices = []  # Store initial seed indices

        # Handle reverse sigma for zero seed
        if not seed or not self._tokenize(seed):
            bigram_probs, word_counts = self.build_bigram_model()
            max_word_length = max(len(word) for word in self.index_to_word.values() if word not in ["<PAD>", "<UNK>"])

            # Generate initial sequence for reverse sigma
            current_idx = random.choice(list(word_counts.keys()))
            reverse_sigma_generated_indices = [current_idx]
            for _ in range(int(reverse_sigma_length) - 1):
                if current_idx not in bigram_probs:
                    break # Stop if no prediction
                else:
                    next_word_probs = []
                    candidates = []
                    for next_idx, prob in bigram_probs[current_idx].items():
                        word = self.index_to_word.get(next_idx, "")
                        if word not in ["<PAD>", "<UNK>"] and len(word) > 0:
                            next_whole_idx = self.precomputed_positions.get(next_idx, 1.0)
                            length_ratio = current_idx * ((self.precomputed_positions.get(next_idx, 1.0)- 1) / max_word_length)
                            elasticity_boost = (1.0 + (length_ratio * elasticity_factor))
                            adjusted_prob = prob * elasticity_boost
                        else:
                            adjusted_prob = prob
                        next_word_probs.append(adjusted_prob ** (1 / max(0.1, temperature)))
                        candidates.append(next_idx)

                    if candidates:
                        total = sum(next_word_probs)
                        if total > 0:
                            next_word_probs = [p / total for p in next_word_probs]
                            next_idx = np.random.choice(candidates, p=next_word_probs)
                            reverse_sigma_generated_indices.append(next_idx)
                            current_idx = next_idx
                        else:
                            break # Stop if no probabilities
                    else:
                        break # Stop if no candidates

            initial_reverse_sigma_indices = list(reversed(reverse_sigma_generated_indices))
            generated_indices.extend(initial_reverse_sigma_indices)
            if initial_reverse_sigma_indices:
                current_idx = initial_reverse_sigma_indices[-1]
            else:
                if not valid_indices:
                    return ""
                current_idx = random.choice(valid_indices) # Keep a minimal fallback to prevent infinite loops if vocab is very small
        else:
            # Process seed text if provided
            seed_words = self._tokenize(seed)
            unknown_words = []

            for word in seed_words:
                word_idx = self.word_to_index.get(word.lower(), None)
                if word_idx is None:
                    unknown_words.append(word)
                    word_idx = self.word_to_index.get("<UNK>", 0)
                seed_indices.append(word_idx)

            if unknown_words:
                unknown_str = ", ".join(f"'{word}'" for word in unknown_words)
                print(f"Warning: The following seed words are not in vocabulary: {unknown_str}")

            generated_indices.extend(seed_indices)
            if generated_indices:
                current_idx = generated_indices[-1]
            else:
                if not valid_indices:
                    return ""
                current_idx = random.choice(valid_indices) # Keep a minimal fallback

        # If no model is provided, build a bigram model
        if model is None:
            bigram_probs, word_counts = self.build_bigram_model()
            max_word_length = max(len(word) for word in self.index_to_word.values() if word not in ["<PAD>", "<UNK>"])

            remaining_length = max(1, length - len(generated_indices))
            for i in range(remaining_length):
                next_idx = self._sample_next_word(bigram_probs, current_idx, valid_indices, self.index_to_word, self.precomputed_positions, max_word_length, temperature, elasticity_factor)
                if next_idx is None:
                    break # Stop if no next word can be sampled
                generated_indices.append(next_idx)
                current_idx = next_idx

        else:
            # Use the provided custom model for generation
            remaining_length = max(1, length - len(generated_indices))
            for _ in range(remaining_length):
                next_idx = model(current_idx)
                generated_indices.append(next_idx)
                current_idx = next_idx

        # Convert indices to words
        generated_words = self.indices_to_words(generated_indices)

        # Join words into text
        return " ".join(generated_words)

def main():
    # Create sample text file

    # Read text file
    with open("test.txt", 'r', encoding='utf-8') as file:
        words = list(file.read().lower().split()[:KB_limit])

    # Create vocabulary
    unique_words = list(set(words))
    word_to_index = {word: idx + 1 for idx, word in enumerate(unique_words)}
    index_to_word = {idx + 1: word for idx, word in enumerate(unique_words)}
    word_to_index["<PAD>"] = 0
    index_to_word[0] = "<PAD>"
    word_to_index["<UNK>"] = len(word_to_index)
    index_to_word[len(index_to_word)] = "<UNK>"

    print(f"Vocabulary size: {len(word_to_index)}")

    # Create sequences for model training
    X_data = []
    for i in range(len(words) - 2):
        indices = [word_to_index.get(words[i], word_to_index["<UNK>"]),
                   word_to_index.get(words[i+1], word_to_index["<UNK>"]),
                   word_to_index.get(words[i+2], word_to_index["<UNK>"])]
        X_data.append(indices)

    print(f"Created {len(X_data)} training sequences")

    # Convert to PyTorch tensors
    X = torch.tensor(X_data, dtype=torch.long)
    positions = torch.tensor([[0, 1, 2] for _ in range(len(X_data))], dtype=torch.long)

    positions = torch.tensor([[np.sum([p])] for p in positions], dtype=torch.long)

    y = torch.tensor([word_to_index.get(words[i+2], word_to_index["<UNK>"]) for i in range(len(words) - 2)], dtype=torch.long)

    # Create dataset
    dataset = TextDataset(X, positions, y, word_to_index, index_to_word)


    # Compare text generation methods
    seed_options = ["language models", "natural language", "text generation", "the quality of"]

    while True:
        print(dataset.generate_text(seed=input("USER: "), length=250, temperature=0.8, elasticity_factor=1.5))


if __name__ == "__main__":
    main()
