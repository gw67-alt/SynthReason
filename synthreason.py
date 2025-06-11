import numpy as np
from collections import defaultdict, deque
from typing import Dict, List, Tuple, Optional, Any, Callable
import random
import math
import torch
import torch.nn as nn
import snntorch as snn
from snntorch import surrogate
from snntorch import utils
from collections import Counter, defaultdict
import re
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import snntorch
import os
import pickle
from datetime import datetime

KB_LEN = 9999

# Check for optional dependencies
try:
    import tensorflow as tf
    TENSORFLOW_AVAILABLE = True
except ImportError:
    TENSORFLOW_AVAILABLE = False
    print("TensorFlow not available. Neural network training will be disabled.")

try:
    from datasets import load_dataset
    DATASETS_AVAILABLE = True
except ImportError:
    DATASETS_AVAILABLE = False
    print("Hugging Face datasets not available. HF dataset loading will be disabled.")



class SpikingFrequencyPredictor:
    def __init__(self):
        print("VERBOSE: SpikingFrequencyPredictor initialized.")
        self.bigram_frequencies: Dict[Tuple[str, str], int] = {}
        self.frequency_features: List[List[float]] = []
        self.snn_model: Optional[nn.Module] = None
        self.scaler = StandardScaler()
        self.sorted_bigrams: List[Tuple[str, str]] = []
        self.unigram_counts: Dict[str, int] = Counter()
        self.num_base_features: int = 16  # Updated for basic features
        self.feature_operations: Optional[List[Optional[Callable[[np.ndarray], np.ndarray]]]] = None
        
        # Spiking neural network parameters
        self.num_steps = 4  # Number of time steps for SNN simulation
        self.beta = 0.1  # Neuron decay rate
        self.spike_grad = surrogate.fast_sigmoid()  # Surrogate gradient function
        self.current_text = ""  # Store current text for access
        
        # Model save/load functionality
        self.model_save_path = "models/"
        self.ensure_model_directory()

    def ensure_model_directory(self):
        """Create model directory if it doesn't exist."""
        if not os.path.exists(self.model_save_path):
            os.makedirs(self.model_save_path)
            print(f"VERBOSE: Created model directory: {self.model_save_path}")

    def save_model(self, model_name: str = None, include_training_data: bool = True) -> str:
        """
        Save the complete model state including SNN, features, and metadata.
        
        Args:
            model_name: Custom name for the model. If None, uses timestamp.
            include_training_data: Whether to save training features and bigram data.
        
        Returns:
            Path to the saved model file.
        """
        if model_name is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            model_name = f"spiking_model_{timestamp}"
        
        model_path = os.path.join(self.model_save_path, f"{model_name}.pth")
        
        # Prepare checkpoint data
        checkpoint = {
            'model_metadata': {
                'model_name': model_name,
                'save_timestamp': datetime.now().isoformat(),
                'num_base_features': self.num_base_features,
                'num_steps': self.num_steps,
                'beta': self.beta,
            },
            'model_state_dict': self.snn_model.state_dict() if self.snn_model else None,
            'scaler_state': {
                'mean_': self.scaler.mean_ if hasattr(self.scaler, 'mean_') else None,
                'scale_': self.scaler.scale_ if hasattr(self.scaler, 'scale_') else None,
                'var_': self.scaler.var_ if hasattr(self.scaler, 'var_') else None,
                'n_features_in_': self.scaler.n_features_in_ if hasattr(self.scaler, 'n_features_in_') else None,
            }
        }
        
        if include_training_data:
            checkpoint.update({
                'bigram_frequencies': self.bigram_frequencies,
                'frequency_features': self.frequency_features,
                'sorted_bigrams': self.sorted_bigrams,
                'unigram_counts': dict(self.unigram_counts),
                'current_text': self.current_text,
                'feature_operations': self.feature_operations,
            })
        
        # Save the checkpoint
        torch.save(checkpoint, model_path)
        print(f"VERBOSE: Model saved successfully to {model_path}")
        
        # Also save a human-readable summary
        summary_path = os.path.join(self.model_save_path, f"{model_name}_summary.txt")
        self._save_model_summary(checkpoint, summary_path)
        
        return model_path

    def _save_model_summary(self, checkpoint: dict, summary_path: str):
        """Save a human-readable summary of the model."""
        with open(summary_path, 'w') as f:
            f.write("SPIKING NEURAL NETWORK MODEL SUMMARY\n")
            f.write("=" * 50 + "\n\n")
            
            metadata = checkpoint.get('model_metadata', {})
            f.write(f"Model Name: {metadata.get('model_name', 'Unknown')}\n")
            f.write(f"Save Date: {metadata.get('save_timestamp', 'Unknown')}\n")
            f.write(f"Base Features: {metadata.get('num_base_features', 'Unknown')}\n")
            f.write(f"Time Steps: {metadata.get('num_steps', 'Unknown')}\n")
            f.write(f"Beta (Decay): {metadata.get('beta', 'Unknown')}\n\n")
            
            if 'bigram_frequencies' in checkpoint:
                bigrams = checkpoint['bigram_frequencies']
                f.write(f"Bigram Count: {len(bigrams)}\n")
                f.write(f"Feature Count: {len(checkpoint.get('frequency_features', []))}\n")
                f.write(f"Text Length: {len(checkpoint.get('current_text', ''))}\n\n")
                
                if bigrams:
                    f.write("Top 10 Bigrams:\n")
                    sorted_bigrams = sorted(bigrams.items(), key=lambda x: x[1], reverse=True)
                    for i, ((w1, w2), freq) in enumerate(sorted_bigrams[:10]):
                        f.write(f"  {i+1}. ({w1}, {w2}): {freq}\n")
            
            f.write(f"\nModel Architecture: {'Loaded' if checkpoint.get('model_state_dict') else 'Not Available'}\n")
            f.write(f"Scaler: {'Fitted' if checkpoint.get('scaler_state', {}).get('mean_') is not None else 'Not Fitted'}\n")

    def load_model(self, model_path: str, load_training_data: bool = True) -> bool:
        """
        Load a complete model state from file.
        
        Args:
            model_path: Path to the saved model file.
            load_training_data: Whether to load training features and bigram data.
        
        Returns:
            True if loading was successful, False otherwise.
        """
        try:
            print(f"VERBOSE: Loading model from {model_path}")
            
            # Load checkpoint
            checkpoint = torch.load(model_path, map_location='cpu', weights_only=False)
            
            # Restore metadata
            metadata = checkpoint.get('model_metadata', {})
            self.num_base_features = metadata.get('num_base_features', self.num_base_features)
            self.num_steps = metadata.get('num_steps', self.num_steps)
            self.beta = metadata.get('beta', self.beta)
            
            print(f"VERBOSE: Loaded model metadata: {metadata.get('model_name', 'Unknown')}")
            print(f"VERBOSE: Model saved on: {metadata.get('save_timestamp', 'Unknown')}")
            
            # Restore scaler state
            scaler_state = checkpoint.get('scaler_state', {})
            if scaler_state.get('mean_') is not None:
                self.scaler.mean_ = scaler_state['mean_']
                self.scaler.scale_ = scaler_state['scale_']
                self.scaler.var_ = scaler_state['var_']
                self.scaler.n_features_in_ = scaler_state['n_features_in_']
                print("VERBOSE: Scaler state restored")
            
            # Restore SNN model
            model_state_dict = checkpoint.get('model_state_dict')
            if model_state_dict is not None:
                # Determine input size from scaler or features
                input_size = self.num_base_features
                if hasattr(self.scaler, 'n_features_in_') and self.scaler.n_features_in_:
                    input_size = self.scaler.n_features_in_
                
                # Recreate the model architecture
                self.snn_model = self._create_spiking_network(input_size)
                self.snn_model.load_state_dict(model_state_dict)
                print("VERBOSE: SNN model architecture recreated and weights loaded")
            
            # Restore training data if requested
            if load_training_data:
                if 'bigram_frequencies' in checkpoint:
                    self.bigram_frequencies = checkpoint['bigram_frequencies']
                    print(f"VERBOSE: Loaded {len(self.bigram_frequencies)} bigram frequencies")
                
                if 'frequency_features' in checkpoint:
                    self.frequency_features = checkpoint['frequency_features']
                    print(f"VERBOSE: Loaded {len(self.frequency_features)} frequency features")
                
                if 'sorted_bigrams' in checkpoint:
                    self.sorted_bigrams = checkpoint['sorted_bigrams']
                
                if 'unigram_counts' in checkpoint:
                    self.unigram_counts = Counter(checkpoint['unigram_counts'])
                    print(f"VERBOSE: Loaded {len(self.unigram_counts)} unigram counts")
                
                if 'current_text' in checkpoint:
                    self.current_text = checkpoint['current_text']
                    print(f"VERBOSE: Loaded current text ({len(self.current_text)} characters)")
                
                if 'feature_operations' in checkpoint:
                    self.feature_operations = checkpoint['feature_operations']
                    print("VERBOSE: Loaded feature operations")
            
            print("VERBOSE: Model loading completed successfully")
            return True
            
        except Exception as e:
            print(f"VERBOSE: Error loading model: {e}")
            return False

    def list_saved_models(self) -> List[str]:
        """List all saved models in the models directory."""
        if not os.path.exists(self.model_save_path):
            return []
        
        model_files = [f for f in os.listdir(self.model_save_path) if f.endswith('.pth')]
        return sorted(model_files)

    def delete_model(self, model_name: str) -> bool:
        """Delete a saved model and its summary."""
        try:
            model_path = os.path.join(self.model_save_path, f"{model_name}.pth")
            summary_path = os.path.join(self.model_save_path, f"{model_name}_summary.txt")
            
            if os.path.exists(model_path):
                os.remove(model_path)
                print(f"VERBOSE: Deleted model file: {model_path}")
            
            if os.path.exists(summary_path):
                os.remove(summary_path)
                print(f"VERBOSE: Deleted summary file: {summary_path}")
            
            return True
        except Exception as e:
            print(f"VERBOSE: Error deleting model: {e}")
            return False

    def set_feature_operations(self, operations: Optional[List[Optional[Callable[[np.ndarray], np.ndarray]]]]) -> None:
        print(f"VERBOSE: Setting feature operations. Received {len(operations) if operations else 'None'} operations.")
        if operations and len(operations) != self.num_base_features:
            raise ValueError(f"Number of operations ({len(operations)}) must match the number of base features ({self.num_base_features})")
        self.feature_operations = operations
        if operations:
            print(f"VERBOSE: {sum(1 for op in operations if op is not None)} active feature operations set for {self.num_base_features} base features.")

    def _apply_feature_operations(self, X_data: np.ndarray) -> np.ndarray:
        print(f"VERBOSE: Attempting to apply feature operations on data with shape {X_data.shape}.")
        if not self.feature_operations:
            print("VERBOSE: No feature operations defined. Returning data as is.")
            return X_data

        if X_data.ndim != 2 or X_data.shape[1] != self.num_base_features:
            print(f"VERBOSE: Warning: X_data shape ({X_data.shape}) is not as expected for feature operations (expected {self.num_base_features} cols). Skipping transformations.")
            return X_data

        X_transformed = X_data.astype(float).copy()
        print(f"VERBOSE: Applying feature operations. Initial X_data shape: {X_data.shape}")

        for i in range(self.num_base_features):
            if i < len(self.feature_operations):
                operation = self.feature_operations[i]
                if operation:
                    try:
                        X_transformed[:, i] = operation(X_data[:, i].astype(float))
                    except Exception as e:
                        print(f"VERBOSE: Error applying operation to feature index {i}: {e}. Feature {i} remains as original.")
                        X_transformed[:, i] = X_data[:, i].astype(float)
        print(f"VERBOSE: Finished applying feature operations. Transformed X_data shape: {X_transformed.shape}")
        return X_transformed

    def load_text_file(self, file_path: str) -> str:
        print(f"VERBOSE: Attempting to load text from local file: {file_path}")
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
                words = content.lower().split()[:KB_LEN]
                print(f"VERBOSE: Successfully loaded {len(words)} words from {file_path}.")
                return ' '.join(words)
        except FileNotFoundError:
            print(f"VERBOSE: File {file_path} not found. Using internal sample text.")
            return self.get_sample_text()
        except Exception as e:
            print(f"VERBOSE: Error loading file {file_path}: {e}. Using internal sample text.")
            return self.get_sample_text()

    def get_sample_text(self) -> str:
        print("VERBOSE: Providing internal sample text.")
        sample = """
        the quick brown fox jumps over the lazy dog the fox is very agile and quick
        machine learning is revolutionizing technology artificial intelligence learns from data patterns
        natural language processing enables computers to understand human communication effectively
        data science combines statistics programming and domain expertise to extract insights
        python programming language offers powerful libraries for machine learning applications
        deep learning neural networks can model complex relationships in large datasets
        the quick brown fox returned the lazy dog slept the quick fox ran again
        machine learning models improve with more data data patterns are key a a a a a
        the a and of to in is you that it he was for on are as with his they I at
        be this have from or one had by word but not what all were we when your can said
        there use an each which she do how their if will up other about out many then them
        these so some her would make like him into time has look two more write go see number
        no way could people my than first water been call who oil its now find long down day
        did get come made may part this is a very long sentence with many common words to make
        the frequency distribution more varied and provide enough data for the neural network model
        to learn some patterns even from this relatively small sample text it is important to have
        diversity in the input for any machine learning task especially for natural language.
        """.lower()
        print(f"VERBOSE: Sample text has {len(sample.split())} words.")
        return sample

    def preprocess_text(self, text: str) -> List[str]:
        print(f"VERBOSE: Preprocessing text. Initial length: {len(text)} characters.")
        words = text.split()
        valid_words = [word for word in words if word]
        print(f"VERBOSE: Preprocessing complete. Number of words: {len(valid_words)} (after cleaning and removing empty strings).")
        return valid_words

    def extract_bigram_frequencies(self, text: str) -> Dict[Tuple[str, str], int]:
        print("VERBOSE: Starting bigram frequency extraction.")
        words = self.preprocess_text(text)
        if len(words) < 2:
            print("VERBOSE: Not enough words to form bigrams. Extracted 0 bigrams.")
            self.bigram_frequencies = {}
            self.sorted_bigrams = []
            self.unigram_counts = Counter(words)
            return {}

        self.unigram_counts = Counter(words)
        print(f"VERBOSE: Unigram counts calculated. Total unigrams: {len(self.unigram_counts)}, e.g., {list(self.unigram_counts.items())[:3]}")
        bigrams = [(words[i], words[i+1]) for i in range(len(words)-1)]
        self.bigram_frequencies = dict(Counter(bigrams))
        print(f"VERBOSE: Extracted {len(self.bigram_frequencies)} unique bigrams. Total bigram occurrences: {len(bigrams)}.")
        
        self.sorted_bigrams = [
            item[0] for item in sorted(self.bigram_frequencies.items(), key=lambda x: (x[1], x[0][0], x[0][0]), reverse=True)
        ]
        print(f"VERBOSE: Bigrams sorted by frequency. Top 3: {self.sorted_bigrams[:3] if len(self.sorted_bigrams) >=3 else self.sorted_bigrams}")
        return self.bigram_frequencies

    def _encode_features_to_spikes(self, features: np.ndarray) -> torch.Tensor:
        """Convert feature vectors to spike trains using rate coding."""
        print(f"VERBOSE: Encoding {features.shape} features to spike trains")
        
        # Normalize features to [0, 1] for spike rate encoding
        features_normalized = (features - features.min()) / (features.max() - features.min() + 1e-8)
        
        # Convert to torch tensor
        features_tensor = torch.FloatTensor(features_normalized)
        
        # Generate Poisson spike trains
        spike_data = torch.zeros(self.num_steps, features_tensor.shape[0], features_tensor.shape[1])
        
        for step in range(self.num_steps):
            # Generate spikes based on feature values as firing rates
            spike_probs = features_tensor
            spikes = torch.bernoulli(spike_probs)
            spike_data[step] = spikes
            
        print(f"VERBOSE: Generated spike data with shape {spike_data.shape}")
        return spike_data

    def _create_spiking_network(self, input_size: int) -> nn.Module:
        """Create a spiking neural network architecture."""
        print(f"VERBOSE: Creating spiking network with input size {input_size}")
        
        snn_model = nn.Sequential(
            # First spiking layer
            nn.Linear(input_size, 128),
            snn.Leaky(beta=self.beta, init_hidden=True, spike_grad=self.spike_grad),
            
            # Second spiking layer with dropout
            nn.Linear(128, 64),
            snn.Leaky(beta=self.beta, init_hidden=True, spike_grad=self.spike_grad),
            
            # Third spiking layer
            nn.Linear(64, 32),
            snn.Leaky(beta=self.beta, init_hidden=True, spike_grad=self.spike_grad),
            
            # Output layer
            nn.Linear(32, 1),
            snn.Leaky(beta=self.beta, init_hidden=True, spike_grad=self.spike_grad, output=True)
        )
        
        print("VERBOSE: Spiking neural network architecture created")
        return snn_model

    def _spiking_forward_pass(self, spike_data: torch.Tensor) -> torch.Tensor:
        """Perform forward pass through spiking network."""
        print(f"VERBOSE: Running spiking forward pass on data shape {spike_data.shape}")
        
        # Initialize spike and membrane potential recordings
        spk_rec = []
        mem_rec = []
        
        # Reset network state
        utils.reset(self.snn_model)
        
        # Process each time step
        for step in range(self.num_steps):
            spk_out, mem_out = self.snn_model(spike_data[step])
            spk_rec.append(spk_out)
            mem_rec.append(mem_out)
        
        # Stack recordings
        spk_rec = torch.stack(spk_rec)
        mem_rec = torch.stack(mem_rec)
        
        # Use rate coding: sum spikes over time as output
        output = torch.sum(spk_rec, dim=0)
        
        print(f"VERBOSE: Forward pass complete, output shape: {output.shape}")
        return output, spk_rec, mem_rec

    def create_bigram_frequency_features(self) -> List[List[float]]:
        """Create bigram frequency features using neural information."""
        print("VERBOSE: Creating bigram frequency features with neural information.")
        
        if not self.bigram_frequencies:
            print("VERBOSE: No bigram frequencies available.")
            return []
        
        # Initialize multilinear stream linker
        linker = SpikingMultilinearStreamLinker(self)
        
        # Get the text content
        text_content = self.current_text if self.current_text else self.get_sample_text()
        words = self.preprocess_text(text_content)
        
        # Create actual numerical neural features using the linker
        raw_neural_matrix = linker.create_multilinear_features(text_content)
        
        if raw_neural_matrix.size == 0:
            print("VERBOSE: No neural features generated, falling back to basic features.")
            return self._create_basic_bigram_features()
        
        neural_features = []
        
        for i in range(len(words) - 1):
            if i >= len(raw_neural_matrix) or (i + 1) >= len(raw_neural_matrix):
                continue
                
            # Extract numerical neural features for both words
            neuron_w1 = raw_neural_matrix[i]
            neuron_w2 = raw_neural_matrix[i + 1]
            
            # Ensure we have numerical arrays
            if not isinstance(neuron_w1, np.ndarray):
                neuron_w1 = np.array(neuron_w1, dtype=float)
            if not isinstance(neuron_w2, np.ndarray):
                neuron_w2 = np.array(neuron_w2, dtype=float)
            
            # Extract neural firing characteristics
            firing_rate_w1 = float(np.mean(neuron_w1))
            firing_variance_w1 = float(np.var(neuron_w1))
            firing_rate_w2 = float(np.mean(neuron_w2))
            firing_variance_w2 = float(np.var(neuron_w2))
            
            # Neural correlation patterns (handle potential NaN values)
            try:
                correlation_matrix = np.corrcoef(neuron_w1.flatten(), neuron_w2.flatten())
                cross_correlation = float(correlation_matrix[0, 1]) if not np.isnan(correlation_matrix[0, 1]) else 0.0
            except:
                cross_correlation = 0.0
            
            # Temporal neural dynamics
            activation_difference = float(np.linalg.norm(neuron_w2 - neuron_w1))
            
            # Get bigram frequency
            bigram = (words[i], words[i+1])
            freq = self.bigram_frequencies.get(bigram, 0)
            
            # Basic word features for compatibility
            word1, word2 = bigram
            
            # Construct neural-based feature vector
            neural_feature_vector = [
                float(freq),  # Target frequency
                firing_rate_w1, firing_rate_w2,  # Neural firing rates
                firing_variance_w1, firing_variance_w2,  # Neural variability
                cross_correlation,  # Neural synchronization
                activation_difference,  # Neural transition dynamics
                float(len(word1)), float(len(word2)),  # Word lengths
                float(word1.count('e')), float(word2.count('e')),  # Character counts
                float(self.unigram_counts.get(word1, 0)),  # Unigram counts
                float(self.unigram_counts.get(word2, 0)),
                1.0 if word1.endswith('ing') else 0.0,  # Morphological features
                1.0 if word2.endswith('ing') else 0.0,
                1.0 if word1.endswith('ed') else 0.0,
                1.0 if word2.endswith('ed') else 0.0,
            ]
            
            neural_features.append(neural_feature_vector)
        
        # Update the number of base features
        if neural_features:
            self.num_base_features = len(neural_features[0]) - 1  # Exclude frequency target
            print(f"VERBOSE: Updated num_base_features to {self.num_base_features}")
        
        self.frequency_features = neural_features
        print(f"VERBOSE: Created {len(neural_features)} neural-enhanced bigram features.")
        return neural_features

    def _create_basic_bigram_features(self) -> List[List[float]]:
        """Fallback method to create basic bigram features without neural data."""
        print("VERBOSE: Creating basic bigram features as fallback.")
        features = []
        
        for bigram, freq in self.bigram_frequencies.items():
            w1, w2 = bigram
            
            # Basic word features
            feature_vector = [
                float(freq),  # Target frequency
                float(len(w1)), float(len(w2)),  # Word lengths
                float(self.unigram_counts.get(w1, 1)),  # Unigram counts
                float(self.unigram_counts.get(w2, 1)),
                1.0 if w1.endswith('ing') else 0.0,  # Basic morphological features
                1.0 if w2.endswith('ed') else 0.0,
                float(w1.count('e')), float(w2.count('e')),  # Character counts
                float(w1.count('a')), float(w2.count('a')),
                1.0 if w1.startswith('un') else 0.0,  # Prefix features
                1.0 if w2.startswith('un') else 0.0,
                float(len(set(w1))), float(len(set(w2))),  # Unique character counts
                1.0 if w1.endswith('ly') else 0.0,  # Additional morphological
                1.0 if w2.endswith('ly') else 0.0,
            ]
            
            features.append(feature_vector)
        
        if features:
            self.num_base_features = len(features[0]) - 1
        
        return features

    def train_spiking_predictor(self) -> None:
        """Train the spiking neural network predictor."""
        print("VERBOSE: Starting spiking neural network training")
        
        if not self.frequency_features:
            print("VERBOSE: No frequency features available for SNN training")
            return
        
        # Prepare data
        X_raw = np.array([f[1:] for f in self.frequency_features])
        y = np.array([f[0] for f in self.frequency_features])
        
        print(f"VERBOSE: Training data - X shape: {X_raw.shape}, y shape: {y.shape}")
        
        # Apply feature transformations
        X_transformed = self._apply_feature_operations(X_raw)
        
        # Scale features
        X_scaled = self.scaler.fit_transform(X_transformed)
        
        # Convert to spike trains
        spike_data = self._encode_features_to_spikes(X_scaled)
        y_tensor = torch.FloatTensor(y).unsqueeze(1)
        
        # Create spiking network
        self.snn_model = self._create_spiking_network(X_scaled.shape[1])
        
        # Define loss function and optimizer
        criterion = nn.MSELoss()
        optimizer = torch.optim.Adam(self.snn_model.parameters(), lr=0.001)
        
        # Training loop
        num_epochs = 100
        print(f"VERBOSE: Training SNN for {num_epochs} epochs")
        
        for epoch in range(num_epochs):
            optimizer.zero_grad()
            
            # Forward pass
            output, spk_rec, mem_rec = self._spiking_forward_pass(spike_data)
            
            # Calculate loss
            loss = criterion(output, y_tensor)
            
            # Backward pass
            loss.backward()
            optimizer.step()
            
            if epoch % 20 == 0:
                print(f"VERBOSE: Epoch {epoch}, Loss: {loss.item():.4f}")
        
        print("VERBOSE: Spiking neural network training completed")

    def generate_spiking_predictions(self, num_variations: int = 1) -> List[Dict[Tuple[str, str], float]]:
        """Generate predictions using the trained spiking network."""
        print(f"VERBOSE: Generating {num_variations} predictions with SNN")
        
        if self.snn_model is None:
            print("VERBOSE: No trained SNN model available")
            return []
        
        new_frequency_sets = []
        
        # Prepare base features
        base_X = np.array([f[1:] for f in self.frequency_features])
        
        for variation in range(num_variations):
            print(f"VERBOSE: Generating SNN variation {variation + 1}")
            
            # Add noise to features
            noise_factor = 0.1 + (variation * 0.02)
            X_noised = base_X.copy()
            
            for j in range(X_noised.shape[1]):
                noise = np.random.normal(0, noise_factor * np.abs(X_noised[:, j] + 0.01))
                X_noised[:, j] = np.maximum(0, X_noised[:, j] + noise)
            
            # Transform and scale
            X_transformed = self._apply_feature_operations(X_noised)
            X_scaled = self.scaler.transform(X_transformed)
            
            # Convert to spikes
            spike_data = self._encode_features_to_spikes(X_scaled)
            
            # Generate predictions
            with torch.no_grad():
                predictions, _, _ = self._spiking_forward_pass(spike_data)
                predictions = predictions.numpy().flatten()
            
            # Convert to frequency dictionary
            predictions = np.maximum(predictions, 0.01)
            new_freq_dict = {
                bigram: float(predictions[i]) 
                for i, bigram in enumerate(self.sorted_bigrams) 
                if i < len(predictions)
            }
            
            new_frequency_sets.append(new_freq_dict)
        
        print(f"VERBOSE: Generated {len(new_frequency_sets)} SNN prediction sets")
        return new_frequency_sets

    def expand_text_from_bigrams(self,
                                 frequency_dict: Dict[Tuple[str, str], float],
                                 text_length: int = 100,
                                 seed_phrase: Optional[str] = None) -> str:
        print(f"VERBOSE: Starting text expansion. Target length: {text_length}. Seed: '{seed_phrase if seed_phrase else 'None'}'")
        if not frequency_dict:
            print("VERBOSE: Error: No frequency data provided for text expansion.")
            return "Error: No frequency data provided."

        transitions = defaultdict(list)
        for (w1, w2), count in frequency_dict.items():
            if count > 0: 
                transitions[w1].append((w2, count))
        
        if not transitions:
            print("VERBOSE: Error: Frequency data has no usable transitions.")
            return "Error: Frequency data has no usable transitions."

        generated_text_list = []
        current_word: Optional[str] = None
        num_words_to_generate = text_length
        start_word_selected_from_seed = False

        if seed_phrase:
            seed_words = self.preprocess_text(seed_phrase) 
            if seed_words:
                print(f"VERBOSE: Processed seed phrase: {seed_words}")
                potential_start_node = seed_words[-1]
                if potential_start_node in transitions and transitions[potential_start_node]:
                    generated_text_list.extend(seed_words)
                    current_word = potential_start_node
                    start_word_selected_from_seed = True
                    num_words_to_generate = text_length - len(generated_text_list)
                    print(f"VERBOSE: Started with seed. Current word: '{current_word}'. Words to generate: {num_words_to_generate}.")
                    if num_words_to_generate <= 0:
                        final_text = ' '.join(generated_text_list[:text_length])
                        print(f"VERBOSE: Seed phrase already meets/exceeds target length. Generated text: '{final_text[:50]}...'")
                        return final_text

        if not start_word_selected_from_seed:
            print("VERBOSE: Selecting a starting word (seed not used or invalid).")
            valid_starting_unigrams = {w:c for w,c in self.unigram_counts.items() if w in transitions and transitions[w]}
            if valid_starting_unigrams:
                sorted_starters = sorted(valid_starting_unigrams.items(), key=lambda item: item[1], reverse=True)
                starters = [item[0] for item in sorted_starters]
                weights = [item[1] for item in sorted_starters]
                current_word = random.choices(starters, weights=weights, k=1)[0]
                print(f"VERBOSE: Selected start word '{current_word}' based on weighted unigram counts.")
            elif any(transitions.values()):
                possible_starters = [w1 for w1, w2_list in transitions.items() if w2_list]
                if possible_starters:
                    current_word = random.choice(possible_starters)
                    print(f"VERBOSE: Selected start word '{current_word}' randomly from possible transition starters.")
                else:
                    print("VERBOSE: Error: Cannot determine any valid starting word from transitions.")
                    return "Error: Cannot determine any valid starting word."
            else:
                print("VERBOSE: Error: Cannot determine a starting word (no valid transitions).")
                return "Error: Cannot determine a starting word (no valid transitions)."
            
            if current_word:
                generated_text_list.append(current_word)
                num_words_to_generate = text_length - 1
            else:
                print("VERBOSE: Error: Failed to select a starting word.")
                return "Error: Failed to select a starting word."

        for i in range(max(0, num_words_to_generate)):
            if not current_word or current_word not in transitions or not transitions[current_word]:
                print(f"VERBOSE: Current word '{current_word}' has no further transitions. Attempting to restart.")
                valid_restart_candidates = [w for w, trans_list in transitions.items() if trans_list]
                if not valid_restart_candidates:
                    print("VERBOSE: No valid restart candidates found. Ending generation.")
                    break 
                
                restart_options = {w:c for w,c in self.unigram_counts.items() if w in valid_restart_candidates}
                if restart_options:
                    sorted_restart_options = sorted(restart_options.items(), key=lambda item: item[1], reverse=True)
                    starters = [item[0] for item in sorted_restart_options]
                    weights = [item[1] for item in sorted_restart_options]
                    current_word = random.choices(starters, weights=weights, k=1)[0]
                    print(f"VERBOSE: Restarted with word '{current_word}' (weighted choice).")
                else:
                    current_word = random.choice(valid_restart_candidates)
                    print(f"VERBOSE: Restarted with word '{current_word}' (random choice).")
                if not current_word:
                    print("VERBOSE: Failed to select a restart word. Ending generation.")
                    break 

            possible_next_words, weights = zip(*transitions[current_word])
            next_word = random.choices(possible_next_words, weights=weights, k=1)[0]
            generated_text_list.append(next_word)
            current_word = next_word

        final_text = ' '.join(generated_text_list)
        print(f"VERBOSE: Text expansion complete. Generated {len(generated_text_list)} words. Preview: '{final_text[:70]}...'")
        return final_text


class SpikingMultilinearStreamLinker:
    def __init__(self, predictor: SpikingFrequencyPredictor):
        """Initialize the spiking multilinear stream linker with a frequency predictor."""
        self.predictor = predictor
        self.stream_buffers: Dict[str, deque] = {}
        self.seed_positions: List[Tuple[int, str, float]] = []
        self.optimal_link_positions: List[int] = []
        self.multilinear_features: np.ndarray = None
        self.stream_weights: Dict[str, float] = {}
        self.spike_encoding_window = 10
        
    def initialize_streams(self, stream_names: List[str], buffer_size: int = 100):
        """Initialize multiple parallel streams for processing."""
        print(f"VERBOSE: Initializing {len(stream_names)} multilinear streams with buffer size {buffer_size}")
        for stream_name in stream_names:
            self.stream_buffers[stream_name] = deque(maxlen=buffer_size)
            self.stream_weights[stream_name] = 1.0 / len(stream_names)
        
    def extract_seed_positions(self, text: str, seed_phrases: List[str]) -> List[Tuple[int, str, float]]:
        """Extract positions of seed phrases in text with confidence scores."""
        words = self.predictor.preprocess_text(text)
        seed_positions = []
        
        for seed_phrase in seed_phrases:
            seed_words = self.predictor.preprocess_text(seed_phrase)
            if not seed_words:
                continue
                
            for i in range(len(words) - len(seed_words) + 1):
                if words[i:i+len(seed_words)] == seed_words:
                    confidence = self._calculate_seed_confidence(words, i, len(seed_words))
                    seed_positions.append((i, seed_phrase, confidence))
        
        self.seed_positions = sorted(seed_positions, key=lambda x: x[0])
        print(f"VERBOSE: Extracted {len(self.seed_positions)} seed positions")
        return self.seed_positions
    
    def _calculate_seed_confidence(self, words: List[str], position: int, length: int) -> float:
        """Calculate confidence score for seed placement based on local context."""
        context_window = 5
        start_context = max(0, position - context_window)
        end_context = min(len(words), position + length + context_window)
        
        context_words = words[start_context:end_context]
        total_freq = sum(self.predictor.unigram_counts.get(word, 1) for word in context_words)
        avg_freq = total_freq / len(context_words) if context_words else 1
        
        confidence = min(1.0, math.log(avg_freq + 1) / 10.0)
        return confidence
    
    def compute_optimal_link_positions(self, text_length: int, num_links: int = 5) -> List[int]:
        """Compute optimal positions for linking streams based on NLP features."""
        if not self.seed_positions:
            print("VERBOSE: No seed positions available, using uniform distribution")
            return [int(i * text_length / (num_links + 1)) for i in range(1, num_links + 1)]
        
        position_features = []
        candidate_positions = range(0, text_length, max(1, text_length // (num_links * 3)))
        
        for pos in candidate_positions:
            features = self._extract_position_features(pos, text_length)
            position_features.append(features)
        
        position_features = np.array(position_features)
        
        optimal_positions = self._select_optimal_positions(
            candidate_positions, position_features, num_links
        )
        
        self.optimal_link_positions = sorted(optimal_positions)
        print(f"VERBOSE: Computed {len(self.optimal_link_positions)} optimal link positions: {self.optimal_link_positions}")
        return self.optimal_link_positions
    
    def _extract_position_features(self, position: int, text_length: int) -> List[float]:
        """Extract NLP features for a given position."""
        features = []
        
        features.append(position / text_length)
        features.append(math.sin(2 * math.pi * position / text_length))
        features.append(math.cos(2 * math.pi * position / text_length))
        
        if self.seed_positions:
            min_seed_distance = min(abs(position - seed_pos[0]) for seed_pos in self.seed_positions)
            features.append(min_seed_distance / text_length)
            
            seed_influence = sum(
                seed_conf * math.exp(-abs(position - seed_pos) / (text_length * 0.1))
                for seed_pos, _, seed_conf in self.seed_positions
            )
            features.append(seed_influence)
        else:
            features.extend([1.0, 0.0])
        
        features.append(math.log(position + 1))
        features.append(1.0 if position % 50 == 0 else 0.0)
        
        return features
    
    def _select_optimal_positions(self, candidates: range, features: np.ndarray, num_links: int) -> List[int]:
        """Select optimal positions using feature-based scoring."""
        if len(candidates) <= num_links:
            return list(candidates)
        
        scores = []
        for i, pos in enumerate(candidates):
            feature_score = np.sum(features[i] ** 2)
            
            diversity_score = 0
            for j, other_pos in enumerate(candidates):
                if i != j:
                    distance = abs(pos - other_pos)
                    diversity_score += 1.0 / (1.0 + distance)
            
            total_score = feature_score - 0.1 * diversity_score
            scores.append((total_score, pos))
        
        scores.sort(reverse=True)
        return [pos for _, pos in scores[:num_links]]
    
    def create_multilinear_features(self, text: str) -> np.ndarray:
        """Create multilinear feature matrix linking early and later seed information."""
        words = self.predictor.preprocess_text(text)
        if not words:
            return np.array([])
        
        syntactic_stream = self._create_syntactic_stream(words)
        semantic_stream = self._create_semantic_stream(words)
        positional_stream = self._create_positional_stream(words)
        frequency_stream = self._create_frequency_stream(words)
        
        feature_streams = [syntactic_stream, semantic_stream, positional_stream, frequency_stream]
        self.multilinear_features = np.column_stack(feature_streams)
        
        print(f"VERBOSE: Created multilinear features matrix with shape {self.multilinear_features.shape}")
        return self.multilinear_features
    
    def _create_syntactic_stream(self, words: List[str]) -> np.ndarray:
        """Create syntactic feature stream."""
        features = []
        for word in words:
            feat = [
                float(len(word)),
                1.0 if word.endswith('ing') else 0.0,
                1.0 if word.endswith('ed') else 0.0,
                1.0 if word.startswith('un') else 0.0,
                float(word.count('e')),
            ]
            features.append(feat)
        return np.array(features)
    
    def _create_semantic_stream(self, words: List[str]) -> np.ndarray:
        """Create semantic feature stream using frequency-based embeddings."""
        features = []
        for i, word in enumerate(words):
            context_window = 30
            context_start = max(0, i - context_window)
            context_end = min(len(words), i + context_window + 1)
            context = words[context_start:context_end]
            
            feat = [
                float(self.predictor.unigram_counts.get(word, 1)),
                float(len(context)),
                float(sum(self.predictor.unigram_counts.get(w, 1) for w in context)),
                float(len(set(context))),
            ]
            features.append(feat)
        return np.array(features)
    
    def _create_positional_stream(self, words: List[str]) -> np.ndarray:
        """Create positional feature stream."""
        features = []
        text_length = len(words)
        for i, word in enumerate(words):
            feat = [
                float(i) / text_length,
                math.sin(2 * math.pi * i / text_length),
                math.cos(2 * math.pi * i / text_length),
                float(i),
            ]
            features.append(feat)
        return np.array(features)
    
    def _create_frequency_stream(self, words: List[str]) -> np.ndarray:
        """Create frequency-based feature stream."""
        features = []
        for i, word in enumerate(words):
            prev_word = words[i-1] if i > 0 else "<START>"
            next_word = words[i+1] if i < len(words)-1 else "<END>"
            
            bigram_freq_prev = self.predictor.bigram_frequencies.get((prev_word, word), 0)
            bigram_freq_next = self.predictor.bigram_frequencies.get((word, next_word), 0)
            
            feat = [
                float(bigram_freq_prev),
                float(bigram_freq_next),
                math.log(bigram_freq_prev + 1),
                math.log(bigram_freq_next + 1),
            ]
            features.append(feat)
        return np.array(features)

    def encode_neural_spikes(self, features: np.ndarray) -> np.ndarray:
        """Encode multilinear features as neural spike patterns."""
        print(f"VERBOSE: Encoding multilinear features to neural spikes")
        
        features_norm = (features - features.min()) / (features.max() - features.min() + 1e-8)
        
        spike_patterns = []
        for i in range(len(features_norm)):
            spike_train = np.random.poisson(features_norm[i] * 5, self.spike_encoding_window)
            spike_patterns.append(spike_train.flatten())
        
        return np.array(spike_patterns)


def enhanced_spiking_text_generation():
    """Enhanced text generation using spiking neural networks with save/load functionality."""
    print("VERBOSE: Starting spiking neural network text generation")
    
    # Initialize spiking components
    predictor = SpikingFrequencyPredictor()
    
    # Check for existing models
    saved_models = predictor.list_saved_models()
    if saved_models:
        print(f"\nVERBOSE: Found {len(saved_models)} saved models:")
        for i, model in enumerate(saved_models):
            print(f"  {i+1}. {model}")
        
        choice = input("\nLoad existing model? (y/n): ").lower()
        if choice == 'y':
            model_choice = input("Enter model filename (or number): ")
            try:
                if model_choice.isdigit():
                    model_idx = int(model_choice) - 1
                    if 0 <= model_idx < len(saved_models):
                        model_file = saved_models[model_idx]
                    else:
                        raise ValueError("Invalid model number")
                else:
                    model_file = model_choice if model_choice.endswith('.pth') else f"{model_choice}.pth"
                
                model_path = os.path.join(predictor.model_save_path, model_file)
                if predictor.load_model(model_path):
                    print("VERBOSE: Model loaded successfully!")
                else:
                    print("VERBOSE: Failed to load model, training new one...")
                    predictor = SpikingFrequencyPredictor()  # Reset
            except Exception as e:
                print(f"VERBOSE: Error loading model: {e}, training new one...")
                predictor = SpikingFrequencyPredictor()  # Reset
    
    # Train new model if not loaded
    if predictor.snn_model is None:
        # Load and process text
        text_content = predictor.load_text_file("test.txt")
        predictor.current_text = text_content  # Store for access in feature creation
        predictor.extract_bigram_frequencies(text_content)
        predictor.create_bigram_frequency_features()
        
        # Train spiking network
        predictor.train_spiking_predictor()
        
        # Auto-save the trained model
        save_choice = input("\nSave trained model? (y/n): ").lower()
        if save_choice == 'y':
            model_name = input("Enter model name (optional): ").strip()
            if not model_name:
                model_name = None
            saved_path = predictor.save_model(model_name)
            print(f"VERBOSE: Model saved to {saved_path}")
    
    print("\n" + "="*60)
    print("SPIKING NEURAL NETWORK TEXT GENERATOR READY")
    print("="*60)
    print("Commands:")
    print("  - Enter text prompts to generate responses")
    print("  - Type 'save [name]' to save current model")
    print("  - Type 'load [name]' to load a saved model")
    print("  - Type 'list' to list saved models")
    print("  - Type 'delete [name]' to delete a saved model")
    print("  - Type 'quit' to exit")
    print("="*60)
    
    while True:
        user_input = input("\nUSER: ").strip()
        
        if user_input.lower() == 'quit':
            print("Goodbye!")
            break
        
        elif user_input.lower().startswith('save'):
            parts = user_input.split(maxsplit=1)
            model_name = parts[1] if len(parts) > 1 else None
            saved_path = predictor.save_model(model_name)
            print(f"Model saved to: {saved_path}")
            continue
        
        elif user_input.lower().startswith('load'):
            parts = user_input.split(maxsplit=1)
            if len(parts) < 2:
                print("Please specify a model name to load")
                continue
            
            model_name = parts[1]
            if not model_name.endswith('.pth'):
                model_name += '.pth'
            
            model_path = os.path.join(predictor.model_save_path, model_name)
            if predictor.load_model(model_path):
                print(f"Model loaded successfully: {model_name}")
            else:
                print(f"Failed to load model: {model_name}")
            continue
        
        elif user_input.lower() == 'list':
            saved_models = predictor.list_saved_models()
            if saved_models:
                print(f"Saved models ({len(saved_models)}):")
                for i, model in enumerate(saved_models):
                    print(f"  {i+1}. {model}")
            else:
                print("No saved models found")
            continue
        
        elif user_input.lower().startswith('delete'):
            parts = user_input.split(maxsplit=1)
            if len(parts) < 2:
                print("Please specify a model name to delete")
                continue
            
            model_name = parts[1]
            if model_name.endswith('.pth'):
                model_name = model_name[:-4]  # Remove .pth extension
            
            if predictor.delete_model(model_name):
                print(f"Model deleted: {model_name}")
            else:
                print(f"Failed to delete model: {model_name}")
            continue
        
        # Generate text with spiking network
        if predictor.snn_model is None:
            print("No trained model available. Please train or load a model first.")
            continue
        
        spiking_frequencies = predictor.generate_spiking_predictions(num_variations=1)
        
        if spiking_frequencies:
            # Enhance frequencies based on linking results
            enhanced_frequencies = spiking_frequencies[0].copy()
            
            generated_text = predictor.expand_text_from_bigrams(
                enhanced_frequencies,
                text_length=250,
                seed_phrase=user_input
            )
            
            print("\n" + "="*50)
            print("SPIKING NEURAL NETWORK GENERATION")
            print("="*50)
            print(generated_text)
            print("="*50)
        else:
            print("VERBOSE: No spiking predictions generated")


if __name__ == "__main__":
    enhanced_spiking_text_generation()
