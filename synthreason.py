import numpy as np
from collections import Counter, defaultdict
import random
import os
from typing import Dict, List, Tuple, Optional, Any
import torch
import torch.nn as nn
import torch.optim as optim
from concurrent.futures import ThreadPoolExecutor
import pickle
import json

KB_LEN = -1

class EnhancedInterstitialMarkovianPredictor:
    def __init__(self, n_threads: Optional[int] = None):
        self.bigram_frequencies: Dict[Tuple[str, str], int] = {}
        self.trigram_frequencies: Dict[Tuple[str, str, str], int] = {}
        self.transition_matrix: Dict[str, Dict[str, float]] = defaultdict(lambda: defaultdict(float))
        self.trigram_transition_matrix: Dict[Tuple[str, str], Dict[str, float]] = defaultdict(lambda: defaultdict(float))
        self.word_to_idx: Dict[str, int] = {}
        self.idx_to_word: Dict[int, str] = {}
        self.unigram_counts: Dict[str, int] = Counter()
        self.vocab_size = 0
        self.n_threads = n_threads or min(8, os.cpu_count())
        self.model_features_shape = None
        self.feature_mean = None
        self.feature_std = None
        self.predictor_model: Optional[nn.Module] = None

    def save_model(self, filepath: str) -> None:
        """Save the complete model state including neural network and training data"""
        save_data = {
            'bigram_frequencies': dict(self.bigram_frequencies),
            'trigram_frequencies': dict(self.trigram_frequencies),
            'transition_matrix': dict(self.transition_matrix),
            'trigram_transition_matrix': dict(self.trigram_transition_matrix),
            'word_to_idx': self.word_to_idx,
            'idx_to_word': self.idx_to_word,
            'unigram_counts': dict(self.unigram_counts),
            'vocab_size': self.vocab_size,
            'model_features_shape': self.model_features_shape,
            'feature_mean': self.feature_mean.tolist() if self.feature_mean is not None else None,
            'feature_std': self.feature_std.tolist() if self.feature_std is not None else None,
        }
        
        # Save the data as JSON
        with open(f"{filepath}_data.json", 'w') as f:
            json.dump(save_data, f, indent=2)
        
        # Save the neural network model if it exists
        if self.predictor_model is not None:
            torch.save({
                'model_state_dict': self.predictor_model.state_dict(),
                'model_features_shape': self.model_features_shape,
            }, f"{filepath}_model.pth")
            print(f"Model saved successfully to {filepath}_data.json and {filepath}_model.pth")
        else:
            print(f"Data saved to {filepath}_data.json (no trained model to save)")

    def load_model(self, filepath: str) -> None:
        """Load the complete model state including neural network and training data"""
        try:
            # Load the data
            with open(f"{filepath}_data.json", 'r') as f:
                save_data = json.load(f)
            
            # Restore data structures
            self.bigram_frequencies = {tuple(k.split('|')): v for k, v in save_data['bigram_frequencies'].items()}
            self.trigram_frequencies = {tuple(k.split('|')): v for k, v in save_data['trigram_frequencies'].items()}
            
            # Restore transition matrices
            self.transition_matrix = defaultdict(lambda: defaultdict(float))
            for k, v in save_data['transition_matrix'].items():
                self.transition_matrix[k] = defaultdict(float, v)
            
            self.trigram_transition_matrix = defaultdict(lambda: defaultdict(float))
            for k, v in save_data['trigram_transition_matrix'].items():
                key_tuple = tuple(k.split('|'))
                self.trigram_transition_matrix[key_tuple] = defaultdict(float, v)
            
            self.word_to_idx = save_data['word_to_idx']
            self.idx_to_word = {int(k): v for k, v in save_data['idx_to_word'].items()}
            self.unigram_counts = Counter(save_data['unigram_counts'])
            self.vocab_size = save_data['vocab_size']
            self.model_features_shape = save_data['model_features_shape']
            
            if save_data['feature_mean'] is not None:
                self.feature_mean = np.array(save_data['feature_mean'])
                self.feature_std = np.array(save_data['feature_std'])
            
            # Load the neural network model if it exists
            model_path = f"{filepath}_model.pth"
            if os.path.exists(model_path) and self.model_features_shape is not None:
                self._create_model(self.model_features_shape)
                checkpoint = torch.load(model_path, weights_only=True)
                self.predictor_model.load_state_dict(checkpoint['model_state_dict'])
                self.predictor_model.eval()
                print(f"Model loaded successfully from {filepath}_data.json and {filepath}_model.pth")
            else:
                print(f"Data loaded from {filepath}_data.json (no trained model found)")
                
        except FileNotFoundError as e:
            print(f"Error loading model: {e}")
        except Exception as e:
            print(f"Error loading model: {e}")

    def save_checkpoint(self, filepath: str, epoch: int, loss: float, optimizer: Optional[optim.Optimizer] = None) -> None:
        """Save training checkpoint for resuming training"""
        if self.predictor_model is None:
            print("No model to save checkpoint for")
            return
            
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.predictor_model.state_dict(),
            'loss': loss,
            'model_features_shape': self.model_features_shape,
            'feature_mean': self.feature_mean.tolist() if self.feature_mean is not None else None,
            'feature_std': self.feature_std.tolist() if self.feature_std is not None else None,
        }
        
        if optimizer is not None:
            checkpoint['optimizer_state_dict'] = optimizer.state_dict()
            
        torch.save(checkpoint, f"{filepath}_checkpoint.pth")
        print(f"Checkpoint saved to {filepath}_checkpoint.pth")

    def load_checkpoint(self, filepath: str, optimizer: Optional[optim.Optimizer] = None) -> Tuple[int, float]:
        """Load training checkpoint for resuming training"""
        try:
            checkpoint = torch.load(f"{filepath}_checkpoint.pth", weights_only=True)
            
            # Restore model
            if checkpoint['model_features_shape'] is not None:
                self._create_model(checkpoint['model_features_shape'])
                self.predictor_model.load_state_dict(checkpoint['model_state_dict'])
                self.model_features_shape = checkpoint['model_features_shape']
                
                if checkpoint['feature_mean'] is not None:
                    self.feature_mean = np.array(checkpoint['feature_mean'])
                    self.feature_std = np.array(checkpoint['feature_std'])
                
                # Restore optimizer if provided
                if optimizer is not None and 'optimizer_state_dict' in checkpoint:
                    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
                
                epoch = checkpoint['epoch']
                loss = checkpoint['loss']
                print(f"Checkpoint loaded from {filepath}_checkpoint.pth")
                return epoch, loss
            else:
                print("Invalid checkpoint: missing model features shape")
                return 0, float('inf')
                
        except FileNotFoundError:
            print(f"Checkpoint file {filepath}_checkpoint.pth not found")
            return 0, float('inf')
        except Exception as e:
            print(f"Error loading checkpoint: {e}")
            return 0, float('inf')

    # [Rest of your existing methods remain the same...]
    def extract_transition_probabilities(self, text: str) -> None:
        """Extract and normalize transition probabilities from text (bigrams and trigrams)"""
        words = text.lower().split()
        if len(words) < 2:
            return

        # Build vocabulary
        unique_words = list(set(words))
        self.vocab_size = len(unique_words)
        self.word_to_idx = {word: idx for idx, word in enumerate(unique_words)}
        self.idx_to_word = {idx: word for word, idx in self.word_to_idx.items()}
        self.unigram_counts = Counter(words)

        # Count bigrams
        self.bigram_frequencies = dict(Counter((words[i], words[i+1]) for i in range(len(words)-1)))
        # Count trigrams
        self.trigram_frequencies = dict(Counter((words[i], words[i+1], words[i+2]) for i in range(len(words)-2)))

        # Build bigram transition matrix
        word_totals = defaultdict(float)
        for (w1, w2), count in self.bigram_frequencies.items():
            word_totals[w1] += count
        for (w1, w2), count in self.bigram_frequencies.items():
            if word_totals[w1] > 0:
                self.transition_matrix[w1][w2] = count / word_totals[w1]

        # Build trigram (two-step) transition matrix
        trigram_totals = defaultdict(float)
        for (w1, w2, w3), count in self.trigram_frequencies.items():
            trigram_totals[(w1, w2)] += count
        for (w1, w2, w3), count in self.trigram_frequencies.items():
            if trigram_totals[(w1, w2)] > 0:
                self.trigram_transition_matrix[(w1, w2)][w3] = count / trigram_totals[(w1, w2)]

    def _calculate_interstitial_value(self, state: Tuple[str, str]) -> float:
        """Calculate interstitial value for a two-word state for feature extraction"""
        w1, w2 = state
        current_transitions = dict(self.transition_matrix.get(w2, {}))
        next_transitions = dict(self.trigram_transition_matrix.get(state, {}))

        # Features: entropy of current and historical transitions, connectivity, frequency
        probs = list(current_transitions.values())
        entropy = -sum(p * np.log2(p + 1e-10) for p in probs if p > 0) if probs else 0.0
        probs = list(next_transitions.values())
        trigram_entropy = -sum(p * np.log2(p + 1e-10) for p in probs if p > 0) if probs else 0.0
        connectivity = len(next_transitions) / self.vocab_size if self.vocab_size > 0 else 0.0
        word_freq = self.unigram_counts[w2] / sum(self.unigram_counts.values()) if sum(self.unigram_counts.values()) > 0 else 0.0
        freq_adjusted_connectivity = connectivity * (1 - word_freq)

        features = [entropy, trigram_entropy, connectivity, freq_adjusted_connectivity]
        weights = [0.4, 0.4, 0.1, 0.1]
        interstitial_value = sum(w * f for w, f in zip(weights, features))
        return interstitial_value

    def _process_word_features(self, word_data: Tuple[str, int]) -> Tuple[List[float], float]:
        """Process features for a single word (thread-safe)"""
        word1, w1_idx = word_data
        prev_word = self.idx_to_word.get(w1_idx-1, "") if w1_idx > 0 else ""
        state = (prev_word, word1)
        
        feature_vector = [
            self._calculate_interstitial_value(state),
            len(self.transition_matrix[word1]) / self.vocab_size if self.vocab_size > 0 else 0.0,
            self.unigram_counts[word1] / sum(self.unigram_counts.values()) if sum(self.unigram_counts.values()) > 0 else 0.0,
        ]
        
        target = self._calculate_interstitial_value(state)
        return feature_vector, target

    def create_interstitial_features(self) -> Tuple[np.ndarray, np.ndarray]:
        """Create features that capture interstitial markovian relationships using multithreading"""
        words_to_process = [(word, self.word_to_idx[word]) 
                          for word in self.word_to_idx.keys() 
                          if word in self.word_to_idx]
        if not words_to_process:
            return np.array([]), np.array([])

        # Process features in parallel
        features = []
        targets = []
        chunk_size = max(1, len(words_to_process) // self.n_threads)
        chunks = [words_to_process[i:i + chunk_size] 
                 for i in range(0, len(words_to_process), chunk_size)]

        def process_chunk(chunk):
            chunk_features = []
            chunk_targets = []
            for word_data in chunk:
                feature_vector, target = self._process_word_features(word_data)
                chunk_features.append(feature_vector)
                chunk_targets.append(target)
            return chunk_features, chunk_targets

        with ThreadPoolExecutor(max_workers=self.n_threads) as executor:
            futures = [executor.submit(process_chunk, chunk) for chunk in chunks]
            for future in futures:
                chunk_features, chunk_targets = future.result()
                features.extend(chunk_features)
                targets.extend(chunk_targets)

        return np.array(features), np.array(targets)

    def _create_model(self, input_size: int):
        """Create the neural network model for interstitial prediction"""
        class InterstitialNet(nn.Module):
            def __init__(self, input_size):
                super().__init__()
                hidden_size = max(128, ((input_size + 15) // 16) * 16)
                conv_channels = hidden_size // 8
                conv_length = 8
                
                self.layers = nn.Sequential(
                    nn.Linear(input_size, hidden_size),
                    nn.ReLU(),
                    nn.Dropout(0.2),
                    nn.Unflatten(1, (conv_channels, conv_length)),
                    nn.ConvTranspose1d(conv_channels, conv_channels*2, 3, stride=2, padding=1, output_padding=1),
                    nn.ReLU(),
                    nn.BatchNorm1d(conv_channels*2),
                    nn.ConvTranspose1d(conv_channels*2, conv_channels, 3, stride=2, padding=1, output_padding=1),
                    nn.ReLU(),
                    nn.Flatten(),
                    nn.Linear(conv_channels * conv_length * 4, 64),
                    nn.ReLU(),
                    nn.Dropout(0.2),
                    nn.Linear(64, 1),
                    nn.Sigmoid()
                )
                
            def forward(self, x):
                return self.layers(x)
                
        self.predictor_model = InterstitialNet(input_size)

    def train_interstitial_predictor(self, epochs: int = 100, save_checkpoints: bool = False, checkpoint_path: str = "model_checkpoint"):
        """Train neural network to predict interstitial markovian values with optional checkpointing"""
        features, targets = self.create_interstitial_features()
        if len(features) == 0:
            print("No features created for training")
            return

        # Store feature normalization parameters
        self.feature_mean = np.mean(features, axis=0)
        self.feature_std = np.std(features, axis=0) + 1e-8
        features_normalized = (features - self.feature_mean) / self.feature_std
        self.model_features_shape = features.shape[1]

        # Convert to tensors
        X_tensor = torch.FloatTensor(features_normalized)
        y_tensor = torch.FloatTensor(targets).unsqueeze(1)

        # Create model
        self._create_model(features.shape[1])
        criterion = nn.MSELoss()
        optimizer = optim.Adam(self.predictor_model.parameters(), lr=0.001)

        best_loss = float('inf')
        
        # Training loop
        for epoch in range(epochs):
            optimizer.zero_grad()
            outputs = self.predictor_model(X_tensor)
            loss = criterion(outputs, y_tensor)
            loss.backward()
            optimizer.step()

            # Save checkpoint if loss improved
            if save_checkpoints and loss.item() < best_loss:
                best_loss = loss.item()
                self.save_checkpoint(checkpoint_path, epoch, loss.item(), optimizer)

            if (epoch + 1) % 10 == 0:
                print(f"Epoch {epoch+1}/{epochs}, Loss: {loss.item():.6f}")
        
        print("Training completed!")

    def generate_next_word(self, prev_word: str, current_word: str) -> str:
        """Generate next word using weighted interstitial and transition probabilities"""
        state = (prev_word, current_word)
        transitions = dict(self.trigram_transition_matrix.get(state, {}))
        if not transitions:
            transitions = dict(self.transition_matrix.get(current_word, {}))
            if not transitions:
                return random.choice(list(self.unigram_counts.keys())) if self.unigram_counts else ""

        # Get interstitial values for each possible next word
        next_words = list(transitions.keys())
        if self.predictor_model and self.feature_mean is not None:
            features = []
            for w in next_words:
                feature_vector = [
                    self._calculate_interstitial_value((current_word, w)),
                    len(self.transition_matrix[w]) / self.vocab_size if self.vocab_size > 0 else 0.0,
                    self.unigram_counts[w] / sum(self.unigram_counts.values()) if sum(self.unigram_counts.values()) > 0 else 0.0,
                ]
                feature_vector = (np.array(feature_vector) - self.feature_mean) / self.feature_std
                features.append(feature_vector)
            
            features_tensor = torch.FloatTensor(np.array(features))
            with torch.no_grad():
                interstitial_values = self.predictor_model(features_tensor).squeeze().numpy()
                if interstitial_values.ndim == 0:
                    interstitial_values = np.array([interstitial_values])
        else:
            interstitial_values = np.array([self._calculate_interstitial_value((current_word, w)) for w in next_words])

        transition_probs = np.array([transitions[w] for w in next_words])
        combined_probs = transition_probs * (interstitial_values + 1e-8)
        combined_probs = combined_probs / np.sum(combined_probs) if np.sum(combined_probs) > 0 else np.ones_like(combined_probs) / len(combined_probs)
        
        next_word = np.random.choice(next_words, p=combined_probs)
        return next_word

    def generate_text(self, length: int = 50, seed: Optional[str] = None) -> str:
        """Generate text using interstitial Markovian values and two-step transitions"""
        if not self.unigram_counts:
            return "No data to generate text"
        
        try:
            if seed and len(seed.split()) >= 2:
                prev_word, current_word = seed.split()[-2], seed.split()[-1]
            else:
                words_list = list(self.unigram_counts.keys())
                prev_word = random.choice(words_list)
                current_word = random.choice(words_list)
            
            words = [prev_word, current_word]
            for _ in range(length):
                next_word = self.generate_next_word(prev_word, current_word)
                words.append(next_word)
                prev_word, current_word = current_word, next_word
            
            return ' '.join(words)
        except Exception as e:
            print(f"Error generating text: {e}")
            return "Error in text generation"

# Example usage with save/load functionality
if __name__ == "__main__":
    predictor = EnhancedInterstitialMarkovianPredictor()
    
    try:
        # Try to load existing model
        model_name = input("Model name (or press Enter for new): ").strip()
        if model_name and os.path.exists(f"{model_name}_data.json"):
            predictor.load_model(model_name)
            print("Model loaded successfully!")
        else:
            # Train new model
            filename = input("Training filename: ")
            with open(filename, 'r', encoding='utf-8') as f:
                content = ' '.join(f.read().split()[:KB_LEN])
            
            predictor.extract_transition_probabilities(content)
            predictor.train_interstitial_predictor(epochs=50, save_checkpoints=True, checkpoint_path=model_name or "model")
            
            # Save the trained model
            save_name = model_name or input("Save model as: ")
            predictor.save_model(save_name)
        
        # Interactive text generation
        while True:
            seed_input = input("USER: ")
            if seed_input.lower() == 'quit':
                break
            generated_text = predictor.generate_text(length=50, seed=seed_input)
            print("Generated text:", generated_text)
            print()
            
    except KeyboardInterrupt:
        print("\nExiting...")
    except Exception as e:
        print(f"Error: {e}")
