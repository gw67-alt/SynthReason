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
import numpy as np
from snntorch import spikegen

KB_LIMIT = 131072
# Check for optional dependencies

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
        self.num_steps = 5  # Number of time steps for SNN simulation
        self.beta = 0.5  # Neuron decay rate
        self.spike_grad = surrogate.fast_sigmoid()  # Surrogate gradient function
        self.current_text = ""  # Store current text for access

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
                words = content.lower().split()[:KB_LIMIT]
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

    def extract_bigram_frequencies(self, text: str) -> Dict[Tuple[str, str], int]:
        print("VERBOSE: Starting bigram frequency extraction.")
        words = self.preprocess_text(text)
        
        if len(words) < 2:
            print("VERBOSE: Insufficient words for bigram extraction.")
            return {}
        
        self.unigram_counts = Counter(words)
        bigram_counts = Counter()
        
        for i in range(len(words) - 1):
            bigram = (words[i], words[i + 1])
            bigram_counts[bigram] += 1
        
        self.bigram_frequencies = dict(bigram_counts)
        self.sorted_bigrams = [
            item[0] for item in sorted(
                self.bigram_frequencies.items(),
                key=lambda x: (x[1], x[0][0], x[0][1]),
                reverse=True
            )
        ]
        
        print(f"VERBOSE: Extracted {len(self.bigram_frequencies)} unique bigrams.")
        print(f"VERBOSE: Top 5 bigrams: {list(self.bigram_frequencies.items())[:5]}")
        return self.bigram_frequencies

    def preprocess_text(self, text: str) -> List[str]:
        print(f"VERBOSE: Preprocessing text. Initial length: {len(text)} characters.")
        words = text.split()
        valid_words = [word for word in words if word]
        print(f"VERBOSE: Preprocessing complete. Number of words: {len(valid_words)} (after cleaning and removing empty strings).")
        return valid_words

    def spread_bigrams_exponential_decay(self, target_size_multiplier=5, decay_rate=0.1, smoothing_alpha=0.01):
        """Spread bigrams using exponential decay for lower frequencies with probability smoothing"""
        original_bigrams = list(self.bigram_frequencies.keys())
        target_size = len(original_bigrams) * target_size_multiplier
        
        expanded_frequencies = {}
        for bigram in original_bigrams:
            duplicate_count = 2  # Duplicate each bigram once, so it appears twice
            for d in range(duplicate_count):
                # For each duplicate, create a unique key (if needed), or just use the bigram
                # Here, we'll use the bigram as the key (not recommended if you want unique entries)
                # For sampling, you might want to use a unique identifier, but for simple frequency halving:
                # If you want to sum the probabilities: (not recommended for halving)
                # expanded_frequencies[bigram] = expanded_frequencies.get(bigram, 0) + self.bigram_frequencies[bigram] * (0.5 ** d)
                # But if you want each duplicate to be separate (for sampling), use a unique key:
                key = (bigram[1], bigram[0], d)  # Unique key for each duplicate
                if key not in expanded_frequencies:  # Fixed condition
                    expanded_frequencies[key] = self.bigram_frequencies[bigram] * (1.5 ** d)

        # If you want to return the frequencies:
        return expanded_frequencies

    def _apply_smoothing(self, frequencies, alpha=0.01):
        """Apply Laplace smoothing and normalize to probabilities"""
        # Add smoothing constant to all frequencies
        smoothed_freqs = {bigram: freq + alpha for bigram, freq in frequencies.items()}
        
        # Normalize to create probability distribution
        total_mass = sum(smoothed_freqs.values())
        smoothed_probs = {bigram: freq / total_mass for bigram, freq in smoothed_freqs.items()}
        
        return smoothed_probs

    def _encode_features_to_spikes(self, features: np.ndarray) -> torch.Tensor:
        """Convert feature vectors to spike trains using rate coding, then unfold and fold the spikes."""
        print(f"VERBOSE: Encoding {features.shape} features to spike trains")
        
        # Normalize features to [0, 1]
        features_normalized = (features - features.min()) / (features.max() - features.min() + 1e-8)
        features_tensor = torch.FloatTensor(features_normalized)
        
        # Generate Poisson spike trains: (num_steps, num_samples, num_features)
        spike_data = spikegen.rate(features_tensor, num_steps=self.num_steps)
        
        # Process spike data through fold/unfold operations
        processed_steps = []
        for step in range(self.num_steps):
            single_spike = spike_data[step]  # shape: (num_samples, num_features)
            
            # Ensure we have the right number of features for reshaping to 4x4
            if single_spike.shape[1] == 16:
                # Reshape to (batch, 1, 4, 4) for 2D operations
                single_spike_reshaped = single_spike.view(-1, 1, 4, 4)
                
                # Apply unfold operation
                unfold = nn.Unfold(kernel_size=(2, 2), stride=2)
                patches = unfold(single_spike_reshaped)  # (batch, 4, 4)
                
                # Apply fold operation to reconstruct
                fold = nn.Fold(output_size=(4, 4), kernel_size=(2, 2), stride=2)
                reconstructed = fold(patches)  # (batch, 1, 4, 4)
                
                # Flatten back to original feature dimension for network input
                reconstructed_flat = reconstructed.view(-1, 16)  # (batch, 16)
                processed_steps.append(reconstructed_flat)
            else:
                # If not 16 features, skip fold/unfold and use original
                processed_steps.append(single_spike)
        
        # Stack processed steps back into (num_steps, num_samples, num_features)
        processed_spike_data = torch.stack(processed_steps)
        
        print(f"VERBOSE: Generated processed spike data with shape {processed_spike_data.shape}")
        return processed_spike_data

    def _create_spiking_network(self, input_size: int) -> nn.Module:
        """Create a spiking neural network architecture."""
        print(f"VERBOSE: Creating spiking network with input size {input_size}")
        
        snn_model = nn.Sequential(
            # First spiking layer
            nn.Linear(input_size, 512),
            snn.Leaky(beta=self.beta, init_hidden=True, spike_grad=self.spike_grad),
            # Second spiking layer with dropout
            nn.Linear(512, 256),  # Reduced size to avoid memory issues
            snn.Leaky(beta=self.beta, init_hidden=True, spike_grad=self.spike_grad),
            
            # Third spiking layer
            nn.Linear(256, 128),
            snn.Leaky(beta=self.beta, init_hidden=True, spike_grad=self.spike_grad),
            
            # Output layer
            nn.Linear(128, 1),
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
        
        # Get the text content
        text_content = self.current_text if self.current_text else self.get_sample_text()
        words = self.preprocess_text(text_content)
        
        neural_features = []
        
        for i in range(3, len(words) - 1):
            # Extract numerical neural features for both words
            neuron_w1 = np.array([float(i % 17), float((i * 12) % 5)], dtype=float)  # Fixed neural representation
            neuron_w2 = np.array([float((i + 1) % 17), float(((i + 1) * 2) % 15)], dtype=float)
            
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
        num_epochs = 5
        num_epochs = 5
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
            
            # Convert to frequency dictionary
            predictions_np = predictions.numpy().flatten()
            predictions_np = np.maximum(predictions_np, 0.01)
            
            new_freq_dict = {
                bigram: float(predictions_np[i]) 
                for i, bigram in enumerate(self.sorted_bigrams) 
                if i < len(predictions_np)
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



def enhanced_spiking_text_generation():
    """Enhanced text generation using spiking neural networks."""
    print("VERBOSE: Starting spiking neural network text generation")
    
    # Initialize spiking components
    predictor = SpikingFrequencyPredictor()
    
    # Load and process text
    text_content = predictor.load_text_file("test.txt")
    predictor.current_text = text_content  # Store for access in feature creation
    predictor.extract_bigram_frequencies(text_content)
    predictor.create_bigram_frequency_features()
    
    # Train spiking network
    predictor.train_spiking_predictor()
    
    print("\n" + "="*60)
    print("SPIKING NEURAL NETWORK TEXT GENERATOR READY")
    print("="*60)
    print("Enter text prompts to generate responses. Type 'quit' to exit.")
    print("="*60)
    
    while True:
        user_input = input("\nUSER: ")
        if user_input.lower() == 'quit':
            print("Goodbye!")
            break
            
        # Perform multilinear linking with spiking networks
        
        # Generate with spiking network
        spiking_frequencies = predictor.generate_spiking_predictions(num_variations=1)
        
        if spiking_frequencies:
            # Enhance frequencies based on linking results
            enhanced_frequencies = spiking_frequencies[0].copy()
            words = predictor.preprocess_text(text_content)
          
            generated_text = predictor.expand_text_from_bigrams(
                enhanced_frequencies,
                text_length=200,
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
