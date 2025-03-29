from torch.utils.data import Dataset
import torch
import re
import os
import random
import numpy as np

KB_limit = 10000

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
                    if idx != self.word_to_index.get("<PAD>", 0):
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

    def generate_text(self, model=None, seed=None, length=50, temperature=1.0, elasticity_factor=1.5, reverse_sigma_length=5):
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
            current_idx = random.choice(valid_indices)
            reverse_sigma_generated_indices = [current_idx]
            for _ in range(reverse_sigma_length - 1):
                if current_idx not in bigram_probs:
                    next_idx = random.choice(valid_indices)
                else:
                    next_word_probs = []
                    candidates = []
                    for next_idx, prob in bigram_probs[current_idx].items():
                        word = self.index_to_word.get(next_idx, "")
                        if word not in ["<PAD>", "<UNK>"] and len(word) > 0:
                            next_whole_idx = self.precomputed_positions.get(next_idx, 1.0)
                            length_ratio = next_whole_idx - ((self.precomputed_positions.get(next_idx, 1.0)- 1) / max_word_length)
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
                        else:
                            next_idx = random.choice(candidates)
                    else:
                        next_idx = random.choice(valid_indices)
                reverse_sigma_generated_indices.append(next_idx)

            initial_reverse_sigma_indices = list(reversed(reverse_sigma_generated_indices))
            generated_indices.extend(initial_reverse_sigma_indices)
            if initial_reverse_sigma_indices:
                current_idx = initial_reverse_sigma_indices[-1]
            else:
                current_idx = random.choice(valid_indices)
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
                current_idx = random.choice(valid_indices)

        # If no model is provided, build a bigram model
        if model is None:
            bigram_probs, word_counts = self.build_bigram_model()
            max_word_length = max(len(word) for word in self.index_to_word.values() if word not in ["<PAD>", "<UNK>"])

            remaining_length = max(1, length - len(generated_indices))
            for i in range(remaining_length):
                if seed_indices and i == 0 and len(seed_indices) > 1:  # Apply implicit union after the seed
                    combined_next_word_probs = {}
                    for seed_index in seed_indices:
                        if seed_index in bigram_probs:
                            for next_idx, prob in bigram_probs[seed_index].items():
                                combined_next_word_probs[next_idx] = combined_next_word_probs.get(next_idx, 0) + prob

                    if combined_next_word_probs:
                        total_prob = sum(combined_next_word_probs.values())
                        if total_prob > 0:
                            normalized_probs = {idx: prob / total_prob for idx, prob in combined_next_word_probs.items()}
                            candidates = list(normalized_probs.keys())
                            probabilities = list(normalized_probs.values())
                            next_idx = np.random.choice(candidates, p=probabilities)
                            generated_indices.append(next_idx)
                            current_idx = next_idx
                        else:
                            # Fallback to the last seed word if no combined probabilities
                            current_idx = seed_indices[-1]
                            if current_idx in bigram_probs:
                                next_idx = self._sample_next_word(bigram_probs, current_idx, valid_indices, self.index_to_word, self.precomputed_positions, max_word_length, temperature, elasticity_factor)
                                generated_indices.append(next_idx)
                                current_idx = next_idx
                            else:
                                next_idx = random.choice(valid_indices)
                                generated_indices.append(next_idx)
                                current_idx = next_idx
                    else:
                        # Fallback if no next words found for any seed
                        current_idx = seed_indices[-1]
                        if current_idx in bigram_probs:
                            next_idx = self._sample_next_word(bigram_probs, current_idx, valid_indices, self.index_to_word, self.precomputed_positions, max_word_length, temperature, elasticity_factor)
                            generated_indices.append(next_idx)
                            current_idx = next_idx
                        else:
                            next_idx = random.choice(valid_indices)
                            generated_indices.append(next_idx)
                            current_idx = next_idx

                elif generated_indices:
                    current_idx = generated_indices[-1]
                    if current_idx in bigram_probs:
                        next_idx = self._sample_next_word(bigram_probs, current_idx, valid_indices, self.index_to_word, self.precomputed_positions, max_word_length, temperature, elasticity_factor)
                        generated_indices.append(next_idx)
                        current_idx = next_idx
                    else:
                        next_idx = random.choice(valid_indices)
                        generated_indices.append(next_idx)
                        current_idx = next_idx
                else:
                    # Should not happen if reverse sigma or random start worked
                    generated_indices.append(random.choice(valid_indices))
                    current_idx = generated_indices[-1]

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

    def _sample_next_word(self, bigram_probs, current_idx, valid_indices, index_to_word, precomputed_positions, max_word_length, temperature, elasticity_factor):
        if current_idx not in bigram_probs:
            return random.choice(valid_indices)
        else:
            next_word_probs = []
            candidates = []

            for next_idx, prob in bigram_probs[current_idx].items():
                word = index_to_word.get(next_idx, "")
                if word not in ["<PAD>", "<UNK>"] and len(word) > 0:
                    next_whole_idx = precomputed_positions.get(next_idx, 1.0)
                    length_ratio = next_whole_idx - ((precomputed_positions.get(next_idx, 1.0)- 1) / max_word_length)
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
                    return np.random.choice(candidates, p=next_word_probs)
                else:
                    return random.choice(candidates)
            else:
                return random.choice(valid_indices)

    def apply_pointwise_transform(self, transform_fn, apply_to=None):
        """
        Apply a pointwise transformation function to dataset elements.
        
        Args:
            transform_fn (callable): A function that takes an input value and returns a transformed value.
                                   The function should handle the specific data type (tensor, list, etc.) correctly.
            apply_to (str or list, optional): Specifies which dataset attribute(s) to transform.
                                             Can be 'X', 'y', 'positions', or a list containing these strings.
                                             If None, applies to all three attributes. Default is None.
        
        Returns:
            TextDataset: A new dataset with transformed values.
        
        Example:
            # Double all values in X
            new_dataset = dataset.apply_pointwise_transform(lambda x: x * 2, apply_to='X')
            
            # Apply custom normalization to all attributes
            def normalize(x):
                if torch.is_tensor(x) and x.dtype.is_floating_point:
                    return (x - x.mean()) / (x.std() + 1e-8)
                return x
            new_dataset = dataset.apply_pointwise_transform(normalize)
        """
        # Validate the apply_to parameter
        valid_attributes = ['X', 'y', 'positions']
        if apply_to is None:
            apply_to = valid_attributes
        elif isinstance(apply_to, str):
            if apply_to not in valid_attributes:
                raise ValueError(f"apply_to must be one of {valid_attributes}, got '{apply_to}'")
            apply_to = [apply_to]
        elif not all(attr in valid_attributes for attr in apply_to):
            invalid = [attr for attr in apply_to if attr not in valid_attributes]
            raise ValueError(f"Invalid attribute(s) in apply_to: {invalid}")
        
        # Create a new dataset with the transformed data
        new_X = self.X
        new_positions = self.positions
        new_y = self.y
        
        if 'X' in apply_to and self.X is not None:
            if torch.is_tensor(self.X):
                # Element-wise transform for tensor
                new_X = torch.stack([transform_fn(x) for x in self.X])
            else:
                # For non-tensor data types
                new_X = [transform_fn(x) for x in self.X]
        
        if 'positions' in apply_to and self.positions is not None:
            if torch.is_tensor(self.positions):
                new_positions = torch.stack([transform_fn(p) for p in self.positions])
            else:
                new_positions = [transform_fn(p) for p in self.positions]
        
        if 'y' in apply_to and self.y is not None:
            if torch.is_tensor(self.y):
                if self.y.dim() > 0:  # If y is not scalar
                    new_y = torch.stack([transform_fn(y_i) for y_i in self.y])
                else:
                    new_y = transform_fn(self.y)
            else:
                if hasattr(self.y, '__iter__') and not isinstance(self.y, (str, bytes)):
                    new_y = [transform_fn(y_i) for y_i in self.y]
                else:
                    new_y = transform_fn(self.y)
        
        # Create new dataset with transformed data
        return TextDataset(
            X=new_X,
            positions=new_positions,
            y=new_y,
            word_to_index=self.word_to_index,
            index_to_word=self.index_to_word
        )

    def apply_batch_transform(self, transform_fn, apply_to=None):
        """
        Apply a transformation function to entire batches of data.
        
        Args:
            transform_fn (callable): A function that takes a batch of data and returns a transformed batch.
            apply_to (str or list, optional): Specifies which dataset attribute(s) to transform.
                                             Can be 'X', 'y', 'positions', or a list containing these strings.
                                             If None, applies to all three attributes. Default is None.
        
        Returns:
            TextDataset: A new dataset with transformed values.
        
        Example:
            # Normalize the X values as a batch
            def batch_normalize(batch):
                mean = batch.mean(dim=0, keepdim=True)
                std = batch.std(dim=0, keepdim=True) + 1e-8
                return (batch - mean) / std
            
            normalized_dataset = dataset.apply_batch_transform(batch_normalize, apply_to='X')
        """
        # Validate the apply_to parameter
        valid_attributes = ['X', 'y', 'positions']
        if apply_to is None:
            apply_to = valid_attributes
        elif isinstance(apply_to, str):
            if apply_to not in valid_attributes:
                raise ValueError(f"apply_to must be one of {valid_attributes}, got '{apply_to}'")
            apply_to = [apply_to]
        elif not all(attr in valid_attributes for attr in apply_to):
            invalid = [attr for attr in apply_to if attr not in valid_attributes]
            raise ValueError(f"Invalid attribute(s) in apply_to: {invalid}")
        
        # Apply batch transformations
        new_X = self.X if 'X' not in apply_to or self.X is None else transform_fn(self.X)
        new_positions = self.positions if 'positions' not in apply_to or self.positions is None else transform_fn(self.positions)
        new_y = self.y if 'y' not in apply_to or self.y is None else transform_fn(self.y)
        
        # Create new dataset with transformed data
        return TextDataset(
            X=new_X,
            positions=new_positions,
            y=new_y,
            word_to_index=self.word_to_index,
            index_to_word=self.index_to_word
        )

    def map_vocabulary(self, vocab_transform_fn):
        """
        Apply a transformation function to the dataset's vocabulary.
        
        Args:
            vocab_transform_fn (callable): A function that takes a word and returns a transformed word.
            
        Returns:
            TextDataset: A new dataset with transformed vocabulary.
            
        Example:
            # Lowercase all vocabulary words
            new_dataset = dataset.map_vocabulary(lambda word: word.lower())
            
            # Add a prefix to all words
            new_dataset = dataset.map_vocabulary(lambda word: f"prefix_{word}")
        """
        new_word_to_index = {}
        
        # Handle special tokens separately to maintain their original indices
        special_tokens = {"<PAD>", "<UNK>"}
        special_indices = {self.word_to_index.get(token) for token in special_tokens if token in self.word_to_index}
        
        # Process special tokens first
        for token in special_tokens:
            if token in self.word_to_index:
                new_word_to_index[token] = self.word_to_index[token]
        
        # Apply transformation to all non-special tokens
        for word, idx in self.word_to_index.items():
            if word not in special_tokens:
                transformed_word = vocab_transform_fn(word)
                if transformed_word in new_word_to_index:
                    print(f"Warning: Vocabulary collision after transform: '{word}' and another word both transform to '{transformed_word}'")
                new_word_to_index[transformed_word] = idx
        
        # Generate the new index_to_word mapping
        new_index_to_word = {idx: word for word, idx in new_word_to_index.items()}
        
        # Create new dataset with transformed vocabulary
        return TextDataset(
            X=self.X,
            positions=self.positions,
            y=self.y,
            word_to_index=new_word_to_index,
            index_to_word=new_index_to_word
        )

    def augment_data(self, augmentation_fn, num_augmentations=1):
        """
        Augment the dataset by creating modified copies of the original data.
        
        Args:
            augmentation_fn (callable): A function that takes a tuple of (X, positions, y) and 
                                       returns a modified tuple (X', positions', y').
            num_augmentations (int): Number of augmented copies to create for each original item.
            
        Returns:
            TextDataset: A new dataset with both original and augmented data.
            
        Example:
            # Add random noise to word indices
            def add_noise(item):
                x, pos, y = item
                noise = torch.randint(-1, 2, x.shape)
                return x + noise, pos, y
                
            augmented_dataset = dataset.augment_data(add_noise, num_augmentations=2)
        """
        if self.X is None or len(self.X) == 0:
            return self
        
        # Start with original data
        all_X = list(self.X)
        all_positions = list(self.positions)
        all_y = list(self.y)
        
        # Generate augmented data
        for i in range(len(self.X)):
            original_item = (self.X[i], self.positions[i], self.y[i])
            
            for _ in range(num_augmentations):
                augmented_X, augmented_pos, augmented_y = augmentation_fn(original_item)
                all_X.append(augmented_X)
                all_positions.append(augmented_pos)
                all_y.append(augmented_y)
        
        # Convert lists back to tensors
        if torch.is_tensor(self.X):
            new_X = torch.stack(all_X)
        else:
            new_X = all_X
            
        if torch.is_tensor(self.positions):
            new_positions = torch.stack(all_positions)
        else:
            new_positions = all_positions
            
        if torch.is_tensor(self.y):
            new_y = torch.stack(all_y) if all_y[0].dim() > 0 else torch.tensor(all_y)
        else:
            new_y = all_y
        
        # Create new dataset with augmented data
        return TextDataset(
            X=new_X,
            positions=new_positions,
            y=new_y,
            word_to_index=self.word_to_index,
            index_to_word=self.index_to_word
        )


# Example usage
if __name__ == "__main__":
    # Example data
    file_path = "test.txt"
    if not os.path.exists(file_path):
        with open(file_path, 'w', encoding="utf-8") as f:
            f.write("This is a test file. It contains some text for testing purposes. This text will be used to build a vocabulary and train a simple bigram model. The model will then be used to generate new text based on a seed or without a seed using a reverse sigma approach.")

    with open(file_path, 'r', encoding="utf-8") as file:
        words = list(file.read().lower().split()[:KB_limit])

    # Create vocabulary
    unique_words = list(set(words))
    word_to_index = {word: idx + 1 for idx, word in enumerate(unique_words)}
    index_to_word = {idx + 1: word for idx, word in enumerate(unique_words)}
    word_to_index["<PAD>"] = 0
    index_to_word[0] = "<PAD>"
    word_to_index["<UNK>"] = len(word_to_index)
    index_to_word[len(index_to_word)] = "<UNK>"

    # Create sequences for model training
    X_data = []
    for i in range(len(words) - 2):
        indices = [word_to_index.get(words[i], word_to_index["<UNK>"]),
                   word_to_index.get(words[i+1], word_to_index["<UNK>"]),
                   word_to_index.get(words[i+2], word_to_index["<UNK>"])]
        X_data.append(indices)

    # Convert to PyTorch tensors
    X = torch.tensor(X_data, dtype=torch.long)
    positions = torch.tensor([[0, 1, 2] for _ in range(len(X_data))], dtype=torch.long)
    y = torch.tensor([word_to_index.get(words[i+2], word_to_index["<UNK>"]) for i in range(len(words) - 2)], dtype=torch.long)

    # Create dataset
    dataset = TextDataset(X, positions, y, word_to_index, index_to_word)

    # Example of using pointwise transform
    print("\nExample of pointwise transform - adding +1 to all X values:")
    transformed_dataset = dataset.apply_pointwise_transform(
        lambda x: x + 1 if torch.is_tensor(x) else x,
        apply_to='positions'
    )
    print(f"Original X[0]: {dataset.X[0]}")
    print(f"Transformed X[0]: {transformed_dataset.X[0]}")
    
    # Example of batch transform
    print("\nExample of batch transform - normalizing positions to floating point:")
    def normalize_positions(batch):
        if torch.is_tensor(batch):
            # Convert to float and normalize
            batch_float = batch.float()
            return (batch_float - batch_float.mean()) / (batch_float.std() + 1e-8)
        return batch
    
    normalized_dataset = dataset.apply_batch_transform(normalize_positions, apply_to='positions')
    print(f"Original positions[0]: {dataset.positions[0]}")
    print(f"Normalized positions[0]: {normalized_dataset.positions[0]}")
    
    # Example of vocabulary mapping
    print("\nExample of vocabulary mapping - adding prefix:")
    prefixed_dataset = dataset.map_vocabulary(lambda word: f"prefix_{word}" if word not in ["<PAD>", "<UNK>"] else word)
    sample_idx = 1
    print(f"Original word at index {sample_idx}: {dataset.index_to_word.get(sample_idx)}")
    print(f"Transformed word at index {sample_idx}: {prefixed_dataset.index_to_word.get(sample_idx)}")
    
    # Example of data augmentation
    print("\nExample of data augmentation - random replacements:")
    def augment_with_replacements(item):
        x, pos, y = item
        # Make a copy to avoid modifying the original
        x_copy = x.clone()
        
        # Replace one token with a random valid token
        if len(x_copy) > 0:
            valid_indices = [idx for idx in range(1, len(dataset.word_to_index)) 
                            if dataset.index_to_word[idx] not in ["<PAD>", "<UNK>"]]
            if valid_indices:
                replace_pos = random.randint(0, len(x_copy) - 2)
                x_copy[replace_pos+1] = random.choice(valid_indices)
        
        return x_copy, pos, y
    
    augmented_dataset = dataset.augment_data(augment_with_replacements, num_augmentations=2)
    print(f"Original dataset size: {len(dataset)}")
    print(f"Augmented dataset size: {len(augmented_dataset)}")
    
    # Generate text using the built-in bigram model
    print("\nText generation with bigram model:")
    while True:
        seed_text = input("USER: ")
        if seed_text.lower() == "exit":
            break
        generated_text = dataset.generate_text(seed=seed_text, length=250, temperature=0.8)
        print(generated_text)
        print()
