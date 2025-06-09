import numpy as np
from collections import Counter, defaultdict
import random
import os
import pickle
import json
from typing import Dict, List, Tuple, Optional, Any
import torch
import torch.nn as nn
import torch.optim as optim
from concurrent.futures import ThreadPoolExecutor

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
        self.text_data_length = 0  # Track text data length for proper split

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

    def generate_sine_wave_features(self, length: int = 1000, frequency: float = 0.1, 
                                   amplitude: float = 1.0, phase: float = 0.0) -> np.ndarray:
        """Generate sine wave features for training data augmentation"""
        t = np.linspace(0, length, length)
        sine_wave = amplitude * np.sin(2 * np.pi * frequency * t + phase)
        
        # Create features from sine wave properties
        features = []
        for i in range(len(sine_wave) - 1):
            current_val = sine_wave[i]
            next_val = sine_wave[i + 1]
            
            # Extract meaningful features from sine wave
            slope = next_val - current_val
            magnitude = abs(current_val)
            phase_position = (i % (1/frequency)) / (1/frequency) if frequency > 0 else 0
            
            feature_vector = [current_val, slope, magnitude, phase_position]
            features.append(feature_vector)
        
        return np.array(features)

    def _process_word_features(self, word_data: Tuple[str, int]) -> Tuple[List[float], float]:
        """Process features for a single word (thread-safe)"""
        word1, w1_idx = word_data
        # Use previous word if available, otherwise empty string
        prev_word = self.idx_to_word.get(w1_idx-1, "") if w1_idx > 0 else ""
        state = (prev_word, word1)
        
        feature_vector = [
            self._calculate_interstitial_value(state),
            len(self.transition_matrix[word1]) / self.vocab_size if self.vocab_size > 0 else 0.0,
            self.unigram_counts[word1] / sum(self.unigram_counts.values()) if sum(self.unigram_counts.values()) > 0 else 0.0,
        ]
        
        # Fixed target calculation - use a simple metric instead of recursive call
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

    def create_enhanced_interstitial_features(self) -> Tuple[np.ndarray, np.ndarray]:
        """Enhanced version that includes sine wave data"""
        # Get original text-based features
        text_features, text_targets = self.create_interstitial_features()
        
        if len(text_features) == 0:
            return np.array([]), np.array([])
        
        # Store text data length for proper split tracking
        self.text_data_length = len(text_features)
        
        # Generate sine wave features with multiple frequencies
        sine_features_list = []
        sine_targets_list = []
        
        frequencies = [0.05, 0.1, 0.2, 0.5]  # Different sine wave frequencies
        for freq in frequencies:
            sine_features = self.generate_sine_wave_features(
                length=len(text_features), 
                frequency=freq,
                amplitude=np.random.uniform(0.5, 2.0),
                phase=np.random.uniform(0, 2*np.pi)
            )
            
            # Create targets based on sine wave predictability
            sine_targets = np.array([
                abs(np.sin(2 * np.pi * freq * i)) for i in range(len(sine_features))
            ])
            
            sine_features_list.append(sine_features)
            sine_targets_list.append(sine_targets)
        
        # Combine all sine wave features
        all_sine_features = np.vstack(sine_features_list)
        all_sine_targets = np.hstack(sine_targets_list)
        
        # Pad features to match dimensions
        text_feature_dim = text_features.shape[1]
        sine_feature_dim = all_sine_features.shape[1]
        
        if text_feature_dim > sine_feature_dim:
            # Pad sine features
            padding = np.zeros((all_sine_features.shape[0], text_feature_dim - sine_feature_dim))
            all_sine_features = np.hstack([all_sine_features, padding])
        elif sine_feature_dim > text_feature_dim:
            # Pad text features
            padding = np.zeros((text_features.shape[0], sine_feature_dim - text_feature_dim))
            text_features = np.hstack([text_features, padding])
        
        # Combine text and sine wave features
        combined_features = np.vstack([text_features, all_sine_features])
        combined_targets = np.hstack([text_targets, all_sine_targets])
        
        return combined_features, combined_targets

    def _create_enhanced_model(self, input_size: int):
        """Enhanced model that can handle both text and sine wave features"""
        class EnhancedInterstitialNet(nn.Module):
            def __init__(self, input_size):
                super().__init__()
                
                # Separate pathways for different feature types
                self.text_pathway = nn.Sequential(
                    nn.Linear(input_size, 128),
                    nn.ReLU(),
                    nn.Dropout(0.2)
                )
                
                self.sine_pathway = nn.Sequential(
                    nn.Linear(input_size, 64),
                    nn.ReLU(),
                    nn.Dropout(0.1)
                )
                
                # Combined processing
                self.combined_layers = nn.Sequential(
                    nn.Linear(128 + 64, 256),
                    nn.ReLU(),
                    nn.BatchNorm1d(256),
                    nn.Dropout(0.3),
                    
                    # Convolutional processing for pattern recognition
                    nn.Unflatten(1, (16, 16)),
                    nn.Conv1d(16, 32, 3, padding=1),
                    nn.ReLU(),
                    nn.Conv1d(32, 16, 3, padding=1),
                    nn.ReLU(),
                    nn.Flatten(),
                    
                    nn.Linear(16 * 16, 64),
                    nn.ReLU(),
                    nn.Linear(64, 1),
                    nn.Sigmoid()
                )
                
            def forward(self, x):
                text_out = self.text_pathway(x)
                sine_out = self.sine_pathway(x)
                combined = torch.cat([text_out, sine_out], dim=1)
                return self.combined_layers(combined)
        
        self.predictor_model = EnhancedInterstitialNet(input_size)

    def _create_model(self, input_size: int):
        """Create the neural network model for interstitial prediction"""
        class InterstitialNet(nn.Module):
            def __init__(self, input_size):
                super().__init__()
                # Calculate dimensions for proper reshaping
                hidden_size = max(128, ((input_size + 15) // 16) * 16)  # Round up to nearest 16
                conv_channels = hidden_size // 8
                conv_length = 8
                
                self.layers = nn.Sequential(
                    # Dense encoder
                    nn.Linear(input_size, hidden_size),
                    nn.ReLU(),
                    nn.Dropout(0.2),
                    
                    # Reshape for convolution
                    nn.Unflatten(1, (conv_channels, conv_length)),
                    
                    # Transposed convolutions
                    nn.ConvTranspose1d(conv_channels, conv_channels*2, 3, stride=2, padding=1, output_padding=1),
                    nn.ReLU(),
                    nn.BatchNorm1d(conv_channels*2),
                    
                    nn.ConvTranspose1d(conv_channels*2, conv_channels, 3, stride=2, padding=1, output_padding=1),
                    nn.ReLU(),
                    
                    # Flatten and final layers
                    nn.Flatten(),
                    nn.Linear(conv_channels * conv_length * 4, 64),  # 4x upsampling from convolutions
                    nn.ReLU(),
                    nn.Dropout(0.2),
                    nn.Linear(64, 1),
                    nn.Sigmoid()
                )
                
            def forward(self, x):
                return self.layers(x)
                
        self.predictor_model = InterstitialNet(input_size)

    def train_enhanced_interstitial_predictor(self, epochs: int = 100, sine_weight: float = 0.3):
        """Train with both text and sine wave features - FIXED VERSION"""
        features, targets = self.create_enhanced_interstitial_features()
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

        # Create enhanced model
        self._create_enhanced_model(features.shape[1])
        
        # Use different loss weights for text vs sine wave data
        criterion = nn.MSELoss()
        optimizer = optim.Adam(self.predictor_model.parameters(), lr=0.001, weight_decay=1e-5)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=10, factor=0.5)

        # Training loop with sine wave integration
        for epoch in range(epochs):
            optimizer.zero_grad()
            outputs = self.predictor_model(X_tensor)
            
            # Calculate loss with potential weighting
            loss = criterion(outputs, y_tensor)
            
            # FIXED: Add regularization for sine wave learning with proper tensor sizing
            if len(targets) > self.text_data_length:
                # Calculate the split point using stored text_data_length
                sine_data_start = self.text_data_length
                sine_data_end = len(targets)
                sine_data_length = sine_data_end - sine_data_start
                
                if sine_data_length > 0:
                    # Create sine wave target with exact matching size
                    sine_outputs = outputs[sine_data_start:sine_data_end]
                    sine_reference = torch.sin(torch.linspace(0, 4*np.pi, sine_data_length)).unsqueeze(1)
                    
                    # Ensure tensors have the same size
                    if sine_outputs.shape[0] == sine_reference.shape[0]:
                        sine_reg = torch.mean(torch.abs(sine_outputs - sine_reference))
                        loss += sine_weight * sine_reg
            
            loss.backward()
            optimizer.step()
            scheduler.step(loss)

            if (epoch + 1) % 10 == 0:
                print(f"Epoch {epoch+1}/{epochs}, Loss: {loss.item():.6f}")
        
        print("Enhanced training completed!")

    def train_interstitial_predictor(self, epochs: int = 100):
        """Train neural network to predict interstitial markovian values"""
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

        # Training loop
        for epoch in range(epochs):
            optimizer.zero_grad()
            outputs = self.predictor_model(X_tensor)
            loss = criterion(outputs, y_tensor)
            loss.backward()
            optimizer.step()

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
            # Use model prediction if available
            features = []
            for w in next_words:
                feature_vector = [
                    self._calculate_interstitial_value((current_word, w)),
                    len(self.transition_matrix[w]) / self.vocab_size if self.vocab_size > 0 else 0.0,
                    self.unigram_counts[w] / sum(self.unigram_counts.values()) if sum(self.unigram_counts.values()) > 0 else 0.0,
                ]
                
                # Debug: Check dimensions before normalization
                feature_vector = np.array(feature_vector)
                
                # Ensure dimensions match
                if feature_vector.shape[0] != self.feature_mean.shape[0]:
                    # Pad or truncate to match expected dimensions
                    expected_dim = self.feature_mean.shape[0]
                    if feature_vector.shape[0] < expected_dim:
                        # Pad with zeros
                        padding = np.zeros(expected_dim - feature_vector.shape[0])
                        feature_vector = np.concatenate([feature_vector, padding])
                    else:
                        # Truncate
                        feature_vector = feature_vector[:expected_dim]
                
                feature_vector = (feature_vector - self.feature_mean) / self.feature_std
                features.append(feature_vector)
            
            try:
                features_tensor = torch.FloatTensor(np.array(features))
                with torch.no_grad():
                    interstitial_values = self.predictor_model(features_tensor).squeeze().numpy()
                    if interstitial_values.ndim == 0:  # Handle single prediction
                        interstitial_values = np.array([interstitial_values])
            except Exception as e:
                print(f"Model prediction failed: {e}")
                # Fallback to direct calculation
                interstitial_values = np.array([self._calculate_interstitial_value((current_word, w)) for w in next_words])
        else:
            # Fallback to direct calculation
            interstitial_values = np.array([self._calculate_interstitial_value((current_word, w)) for w in next_words])

        transition_probs = np.array([transitions[w] for w in next_words])
        combined_probs = transition_probs * (interstitial_values + 1e-8)  # Add small epsilon
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
                # Start with random words if seed is insufficient
                words_list = list(self.unigram_counts.keys())
                prev_word = random.choice(words_list)
                current_word = random.choice(words_list)
            
            # Generate words
            words = [prev_word, current_word]  # Include seed words
            for _ in range(length):
                next_word = self.generate_next_word(prev_word, current_word)
                words.append(next_word)
                prev_word, current_word = current_word, next_word
            
            return ' '.join(words)
        except Exception as e:
            print(f"Error generating text: {e}")
            return "Error in text generation"


    def save_model(self, filepath: str) -> bool:
        """Save the complete model state to disk"""
        try:
            # Convert bigram frequencies (tuple keys to strings)
            bigram_frequencies_str = {}
            for (w1, w2), count in self.bigram_frequencies.items():
                key_str = f"{w1}|||{w2}"
                bigram_frequencies_str[key_str] = count
            
            # Convert trigram frequencies (tuple keys to strings)
            trigram_frequencies_str = {}
            for (w1, w2, w3), count in self.trigram_frequencies.items():
                key_str = f"{w1}|||{w2}|||{w3}"
                trigram_frequencies_str[key_str] = count
            
            # Convert defaultdicts to regular dicts for JSON serialization
            transition_matrix_dict = {}
            for k, v in self.transition_matrix.items():
                transition_matrix_dict[k] = dict(v)
            
            trigram_transition_matrix_dict = {}
            for k, v in self.trigram_transition_matrix.items():
                # Convert tuple keys to strings for JSON
                key_str = f"{k[0]}|||{k[1]}"
                trigram_transition_matrix_dict[key_str] = dict(v)
            
            # Prepare data for saving
            save_data = {
                'bigram_frequencies': bigram_frequencies_str,
                'trigram_frequencies': trigram_frequencies_str,
                'transition_matrix': transition_matrix_dict,
                'trigram_transition_matrix': trigram_transition_matrix_dict,
                'word_to_idx': self.word_to_idx,
                'idx_to_word': self.idx_to_word,
                'unigram_counts': dict(self.unigram_counts),
                'vocab_size': self.vocab_size,
                'model_features_shape': self.model_features_shape,
                'feature_mean': self.feature_mean.tolist() if self.feature_mean is not None else None,
                'feature_std': self.feature_std.tolist() if self.feature_std is not None else None,
                'n_threads': self.n_threads,
                'text_data_length': self.text_data_length
            }
            
            # Save main data as JSON
            with open(f"{filepath}.json", 'w', encoding='utf-8') as f:
                json.dump(save_data, f, indent=2, ensure_ascii=False)
            
            # Save PyTorch model separately if it exists
            if self.predictor_model is not None:
                torch.save(self.predictor_model.state_dict(), f"{filepath}_model.pth")
            
            print(f"Model saved successfully to {filepath}")
            return True
            
        except Exception as e:
            print(f"Error saving model: {e}")
            return False

    def load_model(self, filepath: str) -> bool:
        """Load a complete model state from disk"""
        try:
            # Load main data
            with open(f"{filepath}.json", 'r', encoding='utf-8') as f:
                save_data = json.load(f)
            
            # Restore basic attributes
            self.vocab_size = save_data['vocab_size']
            self.word_to_idx = save_data['word_to_idx']
            self.idx_to_word = save_data['idx_to_word']
            self.unigram_counts = Counter(save_data['unigram_counts'])
            self.model_features_shape = save_data['model_features_shape']
            self.n_threads = save_data.get('n_threads', min(8, os.cpu_count()))
            self.text_data_length = save_data.get('text_data_length', 0)
            
            # Restore numpy arrays
            self.feature_mean = np.array(save_data['feature_mean']) if save_data['feature_mean'] is not None else None
            self.feature_std = np.array(save_data['feature_std']) if save_data['feature_std'] is not None else None
            
            # Restore bigram frequencies (convert string keys back to tuples)
            self.bigram_frequencies = {}
            for key_str, count in save_data['bigram_frequencies'].items():
                parts = key_str.split('|||')
                if len(parts) == 2:
                    self.bigram_frequencies[(parts[0], parts[1])] = count
            
            # Restore trigram frequencies (convert string keys back to tuples)
            self.trigram_frequencies = {}
            for key_str, count in save_data['trigram_frequencies'].items():
                parts = key_str.split('|||')
                if len(parts) == 3:
                    self.trigram_frequencies[(parts[0], parts[1], parts[2])] = count
            
            # Restore transition matrices
            self.transition_matrix = defaultdict(lambda: defaultdict(float))
            for k, v in save_data['transition_matrix'].items():
                self.transition_matrix[k] = defaultdict(float, v)
            
            self.trigram_transition_matrix = defaultdict(lambda: defaultdict(float))
            for k_str, v in save_data['trigram_transition_matrix'].items():
                # Convert string keys back to tuples
                k_parts = k_str.split('|||')
                if len(k_parts) == 2:
                    k = (k_parts[0], k_parts[1])
                    self.trigram_transition_matrix[k] = defaultdict(float, v)
            
            # Load PyTorch model if it exists
            model_path = f"{filepath}_model.pth"
            if os.path.exists(model_path) and self.model_features_shape is not None:
                self._create_enhanced_model(self.model_features_shape)
                self.predictor_model.load_state_dict(torch.load(model_path))
                self.predictor_model.eval()
            
            print(f"Model loaded successfully from {filepath}")
            return True
            
        except Exception as e:
            print(f"Error loading model: {e}")
            return False

    def get_model_info(self) -> Dict[str, Any]:
        """Get information about the current model state"""
        return {
            'vocab_size': self.vocab_size,
            'bigram_count': len(self.bigram_frequencies),
            'trigram_count': len(self.trigram_frequencies),
            'has_neural_model': self.predictor_model is not None,
            'model_trained': self.feature_mean is not None,
            'features_shape': self.model_features_shape,
            'n_threads': self.n_threads,
            'text_data_length': self.text_data_length
        }

# Example usage
if __name__ == "__main__":
    predictor = EnhancedInterstitialMarkovianPredictor()
    
    try:
        # Check if user wants to load existing model
        load_choice = input("Load existing model? (y/n): ").lower().strip()
        
        if load_choice == 'y':
            model_path = input("Enter model path (without extension): ").strip()
            if predictor.load_model(model_path):
                print("Model loaded successfully!")
                print("Model info:", predictor.get_model_info())
            else:
                print("Failed to load model. Starting fresh...")
                predictor = EnhancedInterstitialMarkovianPredictor()
        
        # If no model loaded or loading failed, train new model
        if not predictor.unigram_counts:
            filename = input("Filename: ")
            with open(filename, 'r', encoding='utf-8') as f:
                content = ' '.join(f.read().split()[:KB_LEN])
            predictor.extract_transition_probabilities(content)
            
            # Ask user if they want to use enhanced training with sine waves

            predictor.train_enhanced_interstitial_predictor(epochs=50, sine_weight=0.2)
          
            
            # Ask if user wants to save the model
            save_choice = input("Save trained model? (y/n): ").lower().strip()
            if save_choice == 'y':
                save_path = input("Enter save path (without extension): ").strip()
                predictor.save_model(save_path)
        
        # Interactive text generation
        while True:
            seed_input = input("USER: ")
            if seed_input.lower() == 'quit':
                break
            elif seed_input.lower() == 'save':
                save_path = input("Enter save path (without extension): ").strip()
                predictor.save_model(save_path)
                continue
            elif seed_input.lower() == 'info':
                print("Model info:", predictor.get_model_info())
                continue
                
            generated_text = predictor.generate_text(length=250, seed=seed_input)
            print("Generated text:", generated_text)
            print()
            
    except KeyboardInterrupt:
        print("\nExiting...")
    except Exception as e:
        print(f"Error: {e}")
