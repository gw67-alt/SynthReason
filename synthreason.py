import torch
import numpy as np
import os
from torch.utils.data import Dataset, DataLoader
import re
import random
from tqdm import tqdm
KB_limit = -1

class TextDataset(Dataset):
    def __init__(self, X=None, positions=None, y=None, word_to_index=None, index_to_word=None):
        self.X = X
        self.positions = positions
        self.y = y
        self.word_to_index = word_to_index or {}
        self.index_to_word = index_to_word or {}
        # Create a separate dictionary for word vectors instead of trying to modify positions tensor
        self.word_vectors = {}
        self._initialize_word_vectors()
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
        
    def _initialize_word_vectors(self):
        """Initialize word vectors using a simplified embedding approach"""
        # Use a simple approach to create word vectors based on character distributions
        # Store in separate word_vectors dictionary instead of trying to modify positions tensor
        for idx, word in self.index_to_word.items():
            if word in [",", "."] or word in ["<PAD>", "<UNK>"]:
                # Special tokens get random vectors
                self.word_vectors[idx] = np.random.randn(10)
            else:
                # Create a simple vector based on character distributions
                vec = np.zeros(10)  # Reduced dimension for simplicity
                for i, char in enumerate(word):
                    # Use character code and position to influence the vector
                    pos_factor = i / (len(word) + i + 1)  # Avoid division by zero
                    vec[i % 10] += ord(char) * pos_factor
                # Normalize
                norm = np.linalg.norm(vec)
                if norm > 0:
                    vec = vec / norm
                self.word_vectors[idx] = vec

    def _precompute_positions(self):
        """Precompute the positions of each word index in the dataset"""
        precomputed_positions = {}

        if self.X is not None and self.positions is not None:
            for seq, pos in zip(self.X, self.positions):
                for idx, p in zip(seq.tolist(), pos.tolist()):
                    if idx != seq[-1]:
                        precomputed_positions[idx] = float(p)  # Convert tensor value to float

        return precomputed_positions

    def get_euclidean_distance(self, idx1, idx2):
        """Calculate Euclidean distance between two word vectors"""
        # Get vectors from word_vectors dictionary
        vec1 = self.word_vectors.get(idx1, np.random.randn(10))
        vec2 = self.word_vectors.get(idx2, np.random.randn(10))
        # Calculate Euclidean distance
        return np.sqrt(np.sum((vec1 - vec2) ** 2))

    def _sample_next_word_euclidean(self, bigram_probs, current_idx, valid_indices, index_to_word, precomputed_positions, max_word_length, temperature, elasticity_factor):
        """
        Sample the next word using a constant flow rate based on Euclidean distance.
        This implementation uses a more consistent probability distribution.
        """
        if current_idx not in bigram_probs:
            return random.choice(valid_indices)
        else:
            next_word_probs = []
            candidates = []
            
            # Calculate Euclidean distances for normalization
            distances = []
            for next_idx, prob in bigram_probs[current_idx].items():
                word = index_to_word.get(next_idx, "")
                if word not in ["<PAD>", "<UNK>"] and len(word) > 0:
                    # Use word vectors to calculate distance instead of positions
                    distance = self.get_euclidean_distance(current_idx, next_idx)
                    distances.append((next_idx, distance, prob))

            # Sort by distance for constant flow
            if distances:
                # Normalize distances to create constant flow rate
                sorted_distances = sorted(distances, key=lambda x: x[0])
                total_distance = sum(d[1] for d in sorted_distances) or 1.0
                
                # Apply constant flow rate adjustment
                for next_idx, distance, prob in sorted_distances:
                    # Normalize distance to [0,1] range
                    normalized_distance = distance / total_distance
                    
                    # Apply constant flow rate formula
                    # Higher distances get higher probabilities for more diversity
                    flow_rate = normalized_distance  # Reverse to prefer high distance
                    flow_adjusted_prob = prob * (1.0 + (flow_rate * elasticity_factor))
                    
                    # Apply temperature
                    adjusted_prob = flow_adjusted_prob ** (1 / max(0.1, temperature))
                    
                    next_word_probs.append(adjusted_prob)
                    candidates.append(next_idx)
                    
                if candidates:
                    total = sum(next_word_probs)
                    if total > 0:
                        next_word_probs = [p / total for p in next_word_probs]
                        
                        return np.random.choice(candidates, p=next_word_probs)
                    else:
                        return random.choice(candidates)
            
            # Fallback to random choice if no valid candidates
            return random.choice(valid_indices)

    def generate_text_with_constant_flow(self, seed=None, length=50, temperature=1.0, elasticity_factor=1.5, reverse_sigma_length=1.5):
        """
        Generate text using a bigram model with constant flow rate based on Euclidean distances.
        Modified to create high contextual Euclidean distance between inputs and outputs.
        
        Args:
            seed (str, optional): The seed text to start generation with.
            length (int): Number of words to generate (including seed words if provided)
            temperature (float): Controls randomness (higher = more random, lower = more deterministic)
            elasticity_factor (float): Factor to adjust the constant flow rate effect
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
        
        # Process seed if provided
        if seed and self._tokenize(seed):
            seed_words = self._tokenize(seed)
            for word in seed_words:
                word_idx = self.word_to_index.get(word.lower(), self.word_to_index.get("<UNK>", 0))
                generated_indices.append(word_idx)
        else:
            # Start with a random word
            generated_indices.append(random.choice(valid_indices))
        
        # Build the bigram model
        bigram_probs, word_counts = self.build_bigram_model()
        max_word_length = max(len(word) for word in self.index_to_word.values() if word not in ["<PAD>", "<UNK>"])
        
        # Generate text with constant flow rate
        remaining_length = max(1, length - len(generated_indices))
        
        # Track distances for reporting
        distances = []
        
        for i in range(remaining_length):
            current_idx = generated_indices[-1]
            
            # Set higher elasticity factor every 3rd word to create more diversity patterns
            current_elasticity = elasticity_factor * 2.0 if i % 3 == 0 else elasticity_factor
            
            next_idx = self._sample_next_word_euclidean(
                bigram_probs, 
                current_idx, 
                valid_indices, 
                self.index_to_word, 
                self.precomputed_positions, 
                max_word_length, 
                temperature, 
                current_elasticity
            )
            
            # Calculate and store distance
            distance = self.get_euclidean_distance(current_idx, next_idx)
            distances.append(distance)
            
            generated_indices.append(next_idx)
        
        # Report average distance for analysis
        if distances:
            avg_distance = sum(distances) / len(distances)
            print(f"Average Euclidean distance between words: {avg_distance:.4f}")
        
        # Convert indices to words
        generated_words = self.indices_to_words(generated_indices)
        
        # Join words into text
        return " ".join(generated_words)

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
                
def create_sample_text_file(filename, text=None):
    """Create a sample text file for demonstration if it doesn't exist"""
    if not os.path.exists(filename):
        if text is None:
            # Default sample text
            text = """
            In the field of natural language processing, text generation has become an increasingly important 
            research area. Language models can be trained to generate coherent and meaningful text based on 
            statistical patterns found in training data. The quality of generated text depends on various factors 
            including the model architecture, training data quality, and the specific algorithms used for sampling 
            from probability distributions. Advanced techniques like Euclidean flow models aim to improve the 
            coherence and flow of generated text by considering the relationships between words in vector spaces. 
            These approaches often produce more natural sounding output by maintaining consistent thematic and 
            stylistic elements throughout the generated sequence. Researchers continue to explore new methods 
            for improving text generation across different domains and applications.
            """
        with open(filename, 'w', encoding='utf-8') as f:
            f.write(text)
    return filename
    
def main():
    # Create sample text file
    file_path = create_sample_text_file("test.txt")
    
    print(f"Using text file: {file_path}")
    
    # Read text file
    with open(file_path, 'r', encoding='utf-8') as file:
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
    y = torch.tensor([word_to_index.get(words[i+2], word_to_index["<UNK>"]) for i in range(len(words) - 2)], dtype=torch.long)
    
    # Create dataset
    dataset = TextDataset(X, positions, y, word_to_index, index_to_word)
    
    # Text generation loop
    while True:
        seed = input("USER: ")
        print(dataset.generate_text_with_constant_flow(
            seed=seed, 
            length=250, 
            temperature=0.8, 
            elasticity_factor=2.0  # Higher elasticity factor for more diverse transitions
        ))

if __name__ == "__main__":
    main()
