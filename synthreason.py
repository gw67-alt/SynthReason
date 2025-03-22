from torch.utils.data import Dataset
import torch
import re
import os
import random

class TextDataset(Dataset):
    def __init__(self, X=None, positions=None, y=None, word_to_index=None, index_to_word=None, max_seq_length=100):
        self.X = X
        self.positions = positions
        self.y = y
        self.word_to_index = word_to_index or {}
        self.index_to_word = index_to_word or {}
        self.max_seq_length = max_seq_length
        
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
    
    @classmethod
    def from_text_file(cls, file_path, label=None, max_seq_length=100, min_word_freq=2):
        """
        Create a dataset from a text file.
        
        Args:
            file_path (str): Path to the text file
            label (int, optional): Label for all sequences in this file
            max_seq_length (int): Maximum sequence length
            min_word_freq (int): Minimum word frequency to include in vocabulary
        
        Returns:
            TextDataset: Dataset created from the text file
        """
        # Read file and tokenize
        with open(file_path, 'r', encoding='utf-8') as f:
            text = f.read()
        
        # Tokenize into words
        words = cls._tokenize(text)
        
        # Build vocabulary with frequency threshold
        word_freq = {}
        for word in words:
            word_freq[word] = word_freq.get(word, 0) + 1
        
        # Filter words by frequency
        filtered_words = [word for word in word_freq if word_freq[word] >= min_word_freq]
        
        # Create word mappings
        word_to_index = {"<PAD>": 0, "<UNK>": 1}
        for idx, word in enumerate(filtered_words):
            word_to_index[word] = idx + 2  # +2 for PAD and UNK tokens
        
        index_to_word = {idx: word for word, idx in word_to_index.items()}
        
        # Create sequences
        sequences = []
        positions = []
        labels = []
        
        for i in range(0, len(words) - max_seq_length + 1, max_seq_length // 2):  # Overlapping sequences
            seq = words[i:i + max_seq_length]
            if len(seq) < max_seq_length // 2:  # Skip very short sequences
                continue
                
            # Convert to indices
            seq_indices = [word_to_index.get(word, word_to_index["<UNK>"]) for word in seq]
            
            # Pad if necessary
            if len(seq_indices) < max_seq_length:
                seq_indices = seq_indices + [word_to_index["<PAD>"]] * (max_seq_length - len(seq_indices))
            
            # Create position indices
            pos_indices = list(range(len(seq_indices)))
            
            sequences.append(seq_indices)
            positions.append(pos_indices)
            labels.append(label if label is not None else 0)
        
        # Convert to PyTorch tensors
        X = torch.tensor(sequences, dtype=torch.long)
        positions = torch.tensor(positions, dtype=torch.long)
        y = torch.tensor(labels, dtype=torch.long)
        
        return cls(X, positions, y, word_to_index, index_to_word, max_seq_length)
    
    @classmethod
    def from_directory(cls, dir_path, label_map=None, max_seq_length=100, min_word_freq=2):
        """
        Create a dataset from a directory of text files.
        
        Args:
            dir_path (str): Path to directory containing text files
            label_map (dict, optional): Mapping from subdirectory names to labels
            max_seq_length (int): Maximum sequence length
            min_word_freq (int): Minimum word frequency to include in vocabulary
        
        Returns:
            TextDataset: Dataset created from all text files in the directory
        """
        if not os.path.isdir(dir_path):
            raise ValueError(f"{dir_path} is not a valid directory")
        
        all_words = []
        all_files = []
        
        # First pass: collect all words to build vocabulary
        for root, dirs, files in os.walk(dir_path):
            for file in files:
                if file.endswith('.txt'):
                    file_path = os.path.join(root, file)
                    with open(file_path, 'r', encoding='utf-8') as f:
                        text = f.read()
                        words = cls._tokenize(text)
                        all_words.extend(words)
                        
                    # Determine label
                    label = 0
                    if label_map:
                        subdir = os.path.basename(root)
                        label = label_map.get(subdir, 0)
                    
                    all_files.append((file_path, label))
        
        # Build vocabulary with frequency threshold
        word_freq = {}
        for word in all_words:
            word_freq[word] = word_freq.get(word, 0) + 1
        
        # Filter words by frequency
        filtered_words = [word for word in word_freq if word_freq[word] >= min_word_freq]
        
        # Create word mappings
        word_to_index = {"<PAD>": 0, "<UNK>": 1}
        for idx, word in enumerate(filtered_words):
            word_to_index[word] = idx + 2  # +2 for PAD and UNK tokens
        
        index_to_word = {idx: word for word, idx in word_to_index.items()}
        
        # Second pass: process each file and create dataset
        all_sequences = []
        all_positions = []
        all_labels = []
        
        for file_path, label in all_files:
            with open(file_path, 'r', encoding='utf-8') as f:
                text = f.read()
                words = cls._tokenize(text)
                
                for i in range(0, len(words) - max_seq_length + 1, max_seq_length // 2):
                    seq = words[i:i + max_seq_length]
                    if len(seq) < max_seq_length // 2:
                        continue
                        
                    # Convert to indices
                    seq_indices = [word_to_index.get(word, word_to_index["<UNK>"]) for word in seq]
                    
                    # Pad if necessary
                    if len(seq_indices) < max_seq_length:
                        seq_indices = seq_indices + [word_to_index["<PAD>"]] * (max_seq_length - len(seq_indices))
                    
                    # Create position indices
                    pos_indices = list(range(len(seq_indices)))
                    
                    all_sequences.append(seq_indices)
                    all_positions.append(pos_indices)
                    all_labels.append(label)
        
        # Convert to PyTorch tensors
        X = torch.tensor(all_sequences, dtype=torch.long)
        positions = torch.tensor(all_positions, dtype=torch.long)
        y = torch.tensor(all_labels, dtype=torch.long)
        
        return cls(X, positions, y, word_to_index, index_to_word, max_seq_length)
    
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
    
    def generate_text(self, model=None, seed=None, length=50, temperature=1.0):
        """
        Generate text using a bigram model or a custom model.
        
        Args:
            model: A custom model function that takes the current word index and returns the next word index.
                  If None, a bigram model will be built and used.
            seed (str, optional): The seed text to start generation with. Can be a single word or multiple words.
                                 If None, a random word is selected.
            length (int): Number of words to generate (including seed words if provided)
            temperature (float): Controls randomness (higher = more random, lower = more deterministic)
                               Only used with the built-in bigram model.
        
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
        
        # Initialize generated_indices
        generated_indices = []
        
        # Process seed text if provided
        if seed:
            # Tokenize seed text
            seed_words = self._tokenize(seed)
            
            # Convert seed words to indices
            seed_indices = []
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
            
            # Use seed indices as the beginning of our generated text
            generated_indices = seed_indices
            
            # Set the last seed word as our current index
            current_idx = seed_indices[-1]
        else:
            # If no seed is provided, select a random starting word
            current_idx = random.choice(valid_indices)
            generated_indices = [current_idx]
        
        # If no model is provided, build a bigram model
        if model is None:
            bigram_probs, word_counts = self.build_bigram_model()
            
            # Generate the sequence
            remaining_length = max(1, length - len(generated_indices))
            for _ in range(remaining_length):
                if current_idx not in bigram_probs:
                    # If current word has no observed transitions, pick a random word
                    next_idx = random.choice(valid_indices)
                else:
                    # Apply temperature to the probability distribution
                    next_word_probs = []
                    candidates = []
                    
                    for next_idx, prob in bigram_probs[current_idx].items():
                        next_word_probs.append(prob ** (1 / max(0.1, temperature)))
                        candidates.append(next_idx)
                    
                    # Normalize probabilities
                    total = sum(next_word_probs)
                    next_word_probs = [p / total for p in next_word_probs]
                    
                    # Sample based on probabilities
                    next_idx = random.choices(candidates, weights=next_word_probs, k=1)[0]
                
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
    
# Example usage
if __name__ == "__main__":
    # Example data
    with open("test.txt", 'r', encoding="utf-8") as file:
        words = file.read().split()[:9999]
    
    # Create vocabulary
    unique_words = list(set(words))
    word_to_index = {word: idx + 1 for idx, word in enumerate(unique_words)}
    index_to_word = {idx + 1: word for idx, word in enumerate(unique_words)}
    
    # Create sequences for model training
    X_data = []
    for i in range(len(words) - 2):
        X_data.append([word_to_index[words[i]], word_to_index[words[i+1]], word_to_index[words[i+2]]])
    
    # Convert to PyTorch tensors
    X = torch.tensor(X_data, dtype=torch.long)
    positions = torch.tensor([[0, 1, 2] for _ in range(len(X_data))], dtype=torch.long)
    y = torch.tensor([word_to_index[words[i+2]] for i in range(len(words) - 2)], dtype=torch.long)
    
    # Create dataset
    dataset = TextDataset(X, positions, y, word_to_index, index_to_word)
    
    # Generate text using the built-in bigram model
    print("\nText generation with bigram model:")
    while True:
        seed_text = input("USER: ")
        generated_text = dataset.generate_text(seed=seed_text, length=250, temperature=0.8)
        print(generated_text)
