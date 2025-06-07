import numpy as np
from collections import Counter, defaultdict
import random
from typing import Dict, List, Tuple, Optional, Any
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm, trange
import pickle
import json
import os
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
import multiprocessing as mp
from functools import partial
import threading
import time
KB_LEN = -1
class InterstitialMarkovianPredictor:
    def __init__(self, n_threads: int = None):
        self.bigram_frequencies: Dict[Tuple[str, str], int] = {}
        self.transition_matrix: Dict[str, Dict[str, float]] = {}
        self.word_to_idx: Dict[str, int] = {}
        self.idx_to_word: Dict[int, str] = {}
        self.predictor_model: Optional[nn.Module] = None
        self.unigram_counts: Dict[str, int] = Counter()
        self.vocab_size = 0
        self.n_threads = n_threads or min(8, mp.cpu_count())
        self.model_features_shape = None
        self.feature_mean = None
        self.feature_std = None
        
    def save_model(self, filepath: str) -> None:
        """Save the entire model state to file"""
        print(f"Saving model to {filepath}...")
        
        # Prepare data to save
        save_data = {
            'bigram_frequencies': self.bigram_frequencies,
            'transition_matrix': dict(self.transition_matrix),  # Convert defaultdict
            'word_to_idx': self.word_to_idx,
            'idx_to_word': self.idx_to_word,
            'unigram_counts': dict(self.unigram_counts),  # Convert Counter
            'vocab_size': self.vocab_size,
            'n_threads': self.n_threads,
            'model_features_shape': self.model_features_shape,
            'feature_mean': self.feature_mean.tolist() if self.feature_mean is not None else None,
            'feature_std': self.feature_std.tolist() if self.feature_std is not None else None
        }
        
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(filepath) if os.path.dirname(filepath) else '.', exist_ok=True)
        
        # Save the data
        with open(filepath, 'wb') as f:
            pickle.dump(save_data, f, protocol=pickle.HIGHEST_PROTOCOL)
        
        # Save the neural network model if it exists
        if self.predictor_model is not None:
            model_path = filepath.replace('.pkl', '_model.pth')
            torch.save(self.predictor_model.state_dict(), model_path)
            print(f"Neural network weights saved to {model_path}")
        
        print("Model saved successfully!")
    
    def load_model(self, filepath: str) -> bool:
        """Load the entire model state from file"""
        print(f"Loading model from {filepath}...")
        
        try:
            # Load the main data
            with open(filepath, 'rb') as f:
                save_data = pickle.load(f)
            
            # Restore all attributes
            self.bigram_frequencies = save_data['bigram_frequencies']
            self.transition_matrix = defaultdict(lambda: defaultdict(float), save_data['transition_matrix'])
            self.word_to_idx = save_data['word_to_idx']
            self.idx_to_word = save_data['idx_to_word']
            self.unigram_counts = Counter(save_data['unigram_counts'])
            self.vocab_size = save_data['vocab_size']
            self.n_threads = save_data.get('n_threads', self.n_threads)
            self.model_features_shape = save_data.get('model_features_shape')
            self.feature_mean = np.array(save_data['feature_mean']) if save_data.get('feature_mean') else None
            self.feature_std = np.array(save_data['feature_std']) if save_data.get('feature_std') else None
            
            # Try to load the neural network model
            model_path = filepath.replace('.pkl', '_model.pth')
            if os.path.exists(model_path) and self.model_features_shape is not None:
                self._create_model(self.model_features_shape)
                self.predictor_model.load_state_dict(torch.load(model_path, map_location='cpu'))
                print(f"Neural network weights loaded from {model_path}")
            
            print("Model loaded successfully!")
            return True
            
        except Exception as e:
            print(f"Error loading model: {e}")
            return False
    
    def save_transitions(self, transitions: Dict[Tuple[str, str], float], filepath: str) -> None:
        """Save transition probabilities to file"""
        print(f"Saving transitions to {filepath}...")
        
        # Convert tuple keys to strings for JSON compatibility
        json_transitions = {f"{w1}|{w2}": prob for (w1, w2), prob in transitions.items()}
        
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(json_transitions, f, indent=2, ensure_ascii=False)
        
        print("Transitions saved successfully!")
    
    def load_transitions(self, filepath: str) -> Dict[Tuple[str, str], float]:
        """Load transition probabilities from file"""
        print(f"Loading transitions from {filepath}...")
        
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                json_transitions = json.load(f)
            
            # Convert string keys back to tuples
            transitions = {}
            for key, prob in json_transitions.items():
                w1, w2 = key.split('|', 1)
                transitions[(w1, w2)] = prob
            
            print("Transitions loaded successfully!")
            return transitions
            
        except Exception as e:
            print(f"Error loading transitions: {e}")
            return {}

    def extract_transition_probabilities(self, text: str) -> None:
        """Extract and normalize transition probabilities from text"""
        print("Extracting transition probabilities...")
        words = text.lower().split()
        if len(words) < 2:
            return
            
        # Build vocabulary
        unique_words = list(set(words))
        self.vocab_size = len(unique_words)
        self.word_to_idx = {word: idx for idx, word in enumerate(unique_words)}
        self.idx_to_word = {idx: word for word, idx in self.word_to_idx.items()}
        
        # Count bigrams and unigrams with progress bar
        print("Counting n-grams...")
        self.unigram_counts = Counter(words)
        
        # Use multithreading for bigram processing
        def process_bigrams_chunk(start_idx, end_idx):
            return [(words[i], words[i+1]) for i in range(start_idx, min(end_idx, len(words)-1))]
        
        chunk_size = max(1, (len(words) - 1) // self.n_threads)
        chunks = [(i, i + chunk_size) for i in range(0, len(words) - 1, chunk_size)]
        
        print(f"Processing bigrams with {self.n_threads} threads...")
        all_bigrams = []
        with ThreadPoolExecutor(max_workers=self.n_threads) as executor:
            futures = [executor.submit(process_bigrams_chunk, start, end) for start, end in chunks]
            for future in tqdm(futures, desc="Processing bigram chunks"):
                all_bigrams.extend(future.result())
        
        self.bigram_frequencies = dict(Counter(all_bigrams))
        
        # Build transition matrix with proper normalization
        print("Building transition matrix...")
        self.transition_matrix = defaultdict(lambda: defaultdict(float))
        word_totals = defaultdict(float)
        
        # Calculate totals for each starting word
        print("Calculating transition totals...")
        for (w1, w2), count in tqdm(self.bigram_frequencies.items(), desc="Computing totals"):
            word_totals[w1] += count
            
        # Normalize to get probabilities
        print("Normalizing probabilities...")
        for (w1, w2), count in tqdm(self.bigram_frequencies.items(), desc="Normalizing"):
            if word_totals[w1] > 0:
                self.transition_matrix[w1][w2] = count / word_totals[w1]

    def _calculate_interstitial_value_batch(self, words_batch: List[str]) -> List[Tuple[str, float]]:
        """Calculate interstitial values for a batch of words (thread-safe)"""
        results = []
        for word in words_batch:
            value = self._calculate_interstitial_value_single(word)
            results.append((word, value))
        return results
    
    def _calculate_interstitial_value_single(self, word: str) -> float:
        """Calculate the interstitial markovian value for a single word (thread-safe)"""
        if word not in self.transition_matrix or not self.transition_matrix[word]:
            return 0.0
            
        transitions = dict(self.transition_matrix[word])  # Create local copy
        
        # Combine multiple factors that represent "interstitial-ness"
        factors = []
        
        # 1. Transition entropy (higher = more interstitial/bridging)
        probs = list(transitions.values())
        entropy = -sum(p * np.log2(p + 1e-10) for p in probs if p > 0)
        factors.append(entropy)
        
        # 2. Bridging capacity (connects to words that themselves have many transitions)
        bridging_score = 0.0
        for target_word, prob in transitions.items():
            if target_word in self.transition_matrix:
                target_transitions = len(self.transition_matrix[target_word])
                bridging_score += prob * (target_transitions / self.vocab_size)
        factors.append(bridging_score)
        
        # 3. Frequency-adjusted connectivity
        word_freq = self.unigram_counts[word] / sum(self.unigram_counts.values())
        connectivity = len(transitions) / self.vocab_size
        freq_adjusted_connectivity = connectivity * (1 - word_freq)
        factors.append(freq_adjusted_connectivity)
        
        # 4. Transition uniformity (more uniform = more interstitial)
        if len(probs) > 1:
            expected_uniform = 1.0 / len(probs)
            uniformity = 1.0 - np.std(probs) / expected_uniform
            factors.append(max(0, uniformity))
        else:
            factors.append(0.0)
            
        # Combine factors with weights
        weights = [0.3, 0.3, 0.2, 0.2]
        interstitial_value = sum(w * f for w, f in zip(weights, factors))
        
        return interstitial_value
    
    def _calculate_interstitial_value(self, word: str) -> float:
        """Legacy method for backward compatibility"""
        return self._calculate_interstitial_value_single(word)

    def _process_word_features(self, word_data: Tuple[str, int]) -> Tuple[List[float], float]:
        """Process features for a single word (thread-safe)"""
        word1, w1_idx = word_data
        
        # Create feature vector for this word's transition context
        feature_vector = []
        
        # 1. Word's own frequency (normalized)
        word_freq = self.unigram_counts[word1] / sum(self.unigram_counts.values())
        feature_vector.append(word_freq)
        
        # 2. Number of possible transitions
        num_transitions = len(self.transition_matrix[word1])
        feature_vector.append(num_transitions / self.vocab_size)
        
        # 3. Entropy of transitions (uncertainty measure)
        probs = list(self.transition_matrix[word1].values())
        if probs:
            entropy = -sum(p * np.log2(p + 1e-10) for p in probs if p > 0)
            feature_vector.append(entropy / np.log2(len(probs) + 1))
        else:
            feature_vector.append(0.0)
            
        # 4. Max transition probability (how deterministic)
        max_prob = max(self.transition_matrix[word1].values()) if self.transition_matrix[word1] else 0
        feature_vector.append(max_prob)
        
        # 5. Transition distribution characteristics
        probs_array = np.array(list(self.transition_matrix[word1].values()))
        if len(probs_array) > 1:
            feature_vector.extend([
                np.mean(probs_array),
                np.std(probs_array),
                np.median(probs_array),
                len(probs_array) / self.vocab_size  # transition diversity
            ])
        else:
            feature_vector.extend([0.0, 0.0, 0.0, 0.0])
            
        # 6. Positional context features
        position_weights = []
        for word2 in self.transition_matrix[word1]:
            if word2 in self.transition_matrix:
                continuation_strength = len(self.transition_matrix[word2]) / self.vocab_size
                position_weights.append(continuation_strength * self.transition_matrix[word1][word2])
                
        if position_weights:
            feature_vector.extend([
                np.mean(position_weights),
                np.max(position_weights),
                np.sum(position_weights)
            ])
        else:
            feature_vector.extend([0.0, 0.0, 0.0])
            
        # 7. Semantic clustering features (simplified for thread safety)
        similar_transition_score = 0.0
        w1_targets = set(self.transition_matrix[word1].keys())
        if w1_targets:
            # Sample a subset for efficiency
            sample_words = list(self.transition_matrix.keys())[:min(100, len(self.transition_matrix))]
            for other_word in sample_words:
                if other_word != word1:
                    w2_targets = set(self.transition_matrix[other_word].keys())
                    if w2_targets:
                        jaccard = len(w1_targets & w2_targets) / len(w1_targets | w2_targets)
                        similar_transition_score += jaccard
                        
        feature_vector.append(similar_transition_score / max(1, len(sample_words) - 1))
        
        # Calculate target
        interstitial_value = self._calculate_interstitial_value_single(word1)
        
        return feature_vector, interstitial_value

    def create_interstitial_features(self) -> Tuple[np.ndarray, np.ndarray]:
        """Create features that capture interstitial markovian relationships using multithreading"""
        print("Creating interstitial features with multithreading...")
        
        words_to_process = [(word, self.word_to_idx[word]) 
                           for word in self.transition_matrix.keys() 
                           if word in self.word_to_idx]
        
        if not words_to_process:
            return np.array([]), np.array([])
        
        # Process features in parallel
        print(f"Processing {len(words_to_process)} words with {self.n_threads} threads...")
        features = []
        targets = []
        
        # Split work into chunks
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
            
            for future in tqdm(futures, desc="Processing feature chunks"):
                chunk_features, chunk_targets = future.result()
                features.extend(chunk_features)
                targets.extend(chunk_targets)
        
        return np.array(features), np.array(targets)
    
    def _create_model(self, input_size: int):
        """Create the neural network model"""
        class InterstitialNet(nn.Module):
            def __init__(self, input_size):
                super().__init__()
                self.layers = nn.Sequential(
                    nn.Linear(input_size, 64),
                    nn.ReLU(),
                    nn.Dropout(0.2),
                    nn.Linear(64, 32),
                    nn.ReLU(),
                    nn.Dropout(0.2),
                    nn.Linear(32, 16),
                    nn.ReLU(),
                    nn.Linear(16, 1),
                    nn.Sigmoid()
                )
                
            def forward(self, x):
                return self.layers(x)
        
        self.predictor_model = InterstitialNet(input_size)

    def train_interstitial_predictor(self, epochs: int = 200) -> None:
        """Train neural network to predict interstitial markovian values"""
        print("Training interstitial predictor...")
        features, targets = self.create_interstitial_features()
        
        if len(features) == 0:
            print("No features created for training")
            return
            
        print(f"Training on {len(features)} samples with {features.shape[1]} features")
        
        # Store feature normalization parameters
        print("Normalizing features...")
        self.feature_mean = np.mean(features, axis=0)
        self.feature_std = np.std(features, axis=0) + 1e-8
        features = (features - self.feature_mean) / self.feature_std
        self.model_features_shape = features.shape[1]
        
        # Convert to tensors
        X_tensor = torch.FloatTensor(features)
        y_tensor = torch.FloatTensor(targets).unsqueeze(1)
        
        # Create model
        self._create_model(features.shape[1])
        criterion = nn.MSELoss()
        optimizer = optim.Adam(self.predictor_model.parameters(), lr=0.001)
        
        # Training loop with progress bar
        print("Starting neural network training...")
        loss_history = []
        
        progress_bar = trange(epochs, desc="Training", unit="epoch")
        for epoch in progress_bar:
            optimizer.zero_grad()
            outputs = self.predictor_model(X_tensor)
            loss = criterion(outputs, y_tensor)
            loss.backward()
            optimizer.step()
            
            current_loss = loss.item()
            loss_history.append(current_loss)
            
            progress_bar.set_postfix({
                'Loss': f'{current_loss:.6f}',
                'Avg Loss (last 10)': f'{np.mean(loss_history[-10:]):.6f}' if len(loss_history) >= 10 else f'{np.mean(loss_history):.6f}'
            })
            
            if epoch % 50 == 0 or epoch == epochs - 1:
                tqdm.write(f"Epoch {epoch:3d}/{epochs} - Loss: {current_loss:.6f}")
        
        print(f"\nTraining completed! Final loss: {loss_history[-1]:.6f}")
        print(f"Loss improvement: {loss_history[0]:.6f} → {loss_history[-1]:.6f}")

    def _process_transition_chunk(self, words_chunk: List[str], features: np.ndarray, 
                                interpolation_factor: float) -> Dict[Tuple[str, str], float]:
        """Process a chunk of words for transition generation (thread-safe)"""
        chunk_transitions = {}
        
        for i, word1 in enumerate(words_chunk):
            if word1 not in self.word_to_idx:
                continue
                
            original_probs = dict(self.transition_matrix[word1])
            
            if i < len(features):
                base_features = features[i].copy()
                
                # Apply interpolation to features
                noise = np.random.normal(0, 0.1, size=base_features.shape)
                interpolated_features = base_features + interpolation_factor * noise
                
                # Normalize using stored parameters
                if self.feature_mean is not None and self.feature_std is not None:
                    interpolated_features = (interpolated_features - self.feature_mean) / self.feature_std
                
                # Get interstitial prediction
                with torch.no_grad():
                    feature_tensor = torch.FloatTensor(interpolated_features).unsqueeze(0)
                    interstitial_strength = self.predictor_model(feature_tensor).item()
                
                # Interpolate probabilities based on interstitial strength
                for word2, original_prob in original_probs.items():
                    uniform_prob = 1.0 / len(original_probs)
                    interpolated_prob = (1 - interstitial_strength) * original_prob + interstitial_strength * uniform_prob
                    
                    noise_factor = 0.1 * interstitial_strength
                    random_adjustment = np.random.normal(1.0, noise_factor)
                    final_prob = interpolated_prob * random_adjustment
                    
                    chunk_transitions[(word1, word2)] = max(0.001, final_prob)
        
        return chunk_transitions

    def generate_interstitial_transitions(self, interpolation_factor: float = 0.3) -> Dict[Tuple[str, str], float]:
        """Generate new transition probabilities using interstitial interpolation with multithreading"""
        if not self.predictor_model:
            print("Model not trained")
            return {}
            
        print("Generating interstitial transitions with multithreading...")
        
        # Get features for interpolation
        features, _ = self.create_interstitial_features()
        words_list = list(self.transition_matrix.keys())
        
        # Split work into chunks
        chunk_size = max(1, len(words_list) // self.n_threads)
        word_chunks = [words_list[i:i + chunk_size] for i in range(0, len(words_list), chunk_size)]
        feature_chunks = [features[i:i + chunk_size] for i in range(0, len(features), chunk_size)]
        
        print(f"Processing {len(words_list)} words in {len(word_chunks)} chunks with {self.n_threads} threads...")
        
        new_transitions = {}
        
        # Process chunks in parallel
        with ThreadPoolExecutor(max_workers=self.n_threads) as executor:
            futures = []
            for word_chunk, feature_chunk in zip(word_chunks, feature_chunks):
                future = executor.submit(self._process_transition_chunk, 
                                       word_chunk, feature_chunk, interpolation_factor)
                futures.append(future)
            
            for future in tqdm(futures, desc="Processing transition chunks"):
                chunk_result = future.result()
                new_transitions.update(chunk_result)
        
        print("Normalizing transition probabilities...")
        # Normalize to ensure proper probabilities
        word_totals = defaultdict(float)
        for (w1, w2), prob in new_transitions.items():
            word_totals[w1] += prob
            
        normalized_transitions = {}
        for (w1, w2), prob in tqdm(new_transitions.items(), desc="Normalizing"):
            if word_totals[w1] > 0:
                normalized_transitions[(w1, w2)] = prob / word_totals[w1]
            else:
                normalized_transitions[(w1, w2)] = 0.001
                
        print(f"Generated {len(normalized_transitions)} transition probabilities")
        return normalized_transitions

    def generate_text_with_interstitial_values(self, 
                                             transition_probs: Dict[Tuple[str, str], float],
                                             length: int = 100,
                                             seed: Optional[str] = None) -> str:
        """Generate text using interstitial markovian values"""
        if not transition_probs:
            return "No transition probabilities available"
            
        # Build transition lookup
        print("Building transition lookup...")
        transitions = defaultdict(list)
        for (w1, w2), prob in transition_probs.items():
            if prob > 0:
                transitions[w1].append((w2, prob))
        
        # Normalize transitions
        for w1 in transitions:
            total_prob = sum(prob for _, prob in transitions[w1])
            if total_prob > 0:
                transitions[w1] = [(w2, prob/total_prob) for w2, prob in transitions[w1]]
        
        if not transitions:
            return "No valid transitions found"
        
        # Generate text
        words = []
        
        # Choose starting word
        if seed:
            current_word = seed.lower().split()[-1] if seed else None
            if current_word not in transitions:
                current_word = None
        else:
            current_word = None
            
        if not current_word:
            valid_starters = [w for w in transitions.keys() if w in self.unigram_counts]
            if valid_starters:
                starter_probs = [self.unigram_counts[w] for w in valid_starters]
                total_prob = sum(starter_probs)
                starter_probs = [p/total_prob for p in starter_probs]
                current_word = np.random.choice(valid_starters, p=starter_probs)
            else:
                current_word = random.choice(list(transitions.keys()))
        
        if seed:
            words.extend(seed.lower().split())
        else:
            words.append(current_word)
        
        # Generate remaining words with progress bar
        print(f"Generating {length} words...")
        progress_bar = tqdm(total=length - len(words), desc="Generating text", unit="word")
        
        for _ in range(length - len(words)):
            if current_word not in transitions or not transitions[current_word]:
                valid_words = [w for w in transitions.keys() if transitions[w]]
                if not valid_words:
                    break
                current_word = random.choice(valid_words)
            
            next_words, probs = zip(*transitions[current_word])
            current_word = np.random.choice(next_words, p=probs)
            words.append(current_word)
            progress_bar.update(1)
        
        progress_bar.close()
        return ' '.join(words)


def demonstrate_interstitial_markovian():
    """Enhanced demo with save/load functionality"""
    # Sample text
    try:
        with open("test.txt", 'r', encoding='utf-8') as f:
            content = ' '.join(f.read().split()[:KB_LEN])
    except FileNotFoundError:
        print("test.txt not found, using sample text...")
        content = "The quick brown fox jumps over the lazy dog. The dog was sleeping under the tree. The tree provided shade on a sunny day. The day was perfect for a walk in the park."
    
    model_path = "interstitial_model.pkl"
    transitions_path = "interstitial_transitions.json"
    
    predictor = InterstitialMarkovianPredictor(n_threads=12)
    
    print("=" * 60)
    print("ENHANCED INTERSTITIAL MARKOVIAN PREDICTOR DEMO")
    print("=" * 60)
    
    # Try to load existing model
    if os.path.exists(model_path):
        print(f"Found existing model at {model_path}")
        choice = input("Load existing model? (y/n): ").lower().strip()
        if choice == 'y':
            if predictor.load_model(model_path):
                print("Model loaded successfully!")
                
                # Try to load existing transitions
                if os.path.exists(transitions_path):
                    interstitial_transitions = predictor.load_transitions(transitions_path)
                else:
                    print("Generating new transitions...")
                    interstitial_transitions = predictor.generate_interstitial_transitions(interpolation_factor=0.4)
                    predictor.save_transitions(interstitial_transitions, transitions_path)
            else:
                print("Failed to load model, training new one...")
                predictor.extract_transition_probabilities(content)
                predictor.train_interstitial_predictor(epochs=100)
                predictor.save_model(model_path)
                interstitial_transitions = predictor.generate_interstitial_transitions(interpolation_factor=0.4)
                predictor.save_transitions(interstitial_transitions, transitions_path)
        else:
            print("Training new model...")
            predictor.extract_transition_probabilities(content)
            predictor.train_interstitial_predictor(epochs=100)
            predictor.save_model(model_path)
            interstitial_transitions = predictor.generate_interstitial_transitions(interpolation_factor=0.4)
            predictor.save_transitions(interstitial_transitions, transitions_path)
    else:
        print("No existing model found, training new one...")
        predictor.extract_transition_probabilities(content)
        predictor.train_interstitial_predictor(epochs=100)
        predictor.save_model(model_path)
        interstitial_transitions = predictor.generate_interstitial_transitions(interpolation_factor=0.4)
        predictor.save_transitions(interstitial_transitions, transitions_path)
    
    print("\n" + "=" * 60)
    print("TEXT GENERATION")
    print("=" * 60)
    print("Commands: 'quit'/'exit'/'q' to quit, 'save' to save current state")
    
    while True:
        input_ = input("\nUSER: ")
        if input_.lower() in ['quit', 'exit', 'q']:
            break
        elif input_.lower() == 'save':
            predictor.save_model(model_path)
            predictor.save_transitions(interstitial_transitions, transitions_path)
            print("Model and transitions saved!")
            continue
            
        generated_text = predictor.generate_text_with_interstitial_values(
            interstitial_transitions, 
            length=250,
            seed=input_
        )
    
        print(f"\nGenerated text:\n{generated_text}")
    
    # Show some interstitial values
    print(f"\nSample interstitial values:")
    sample_words = list(predictor.transition_matrix.keys())[:5]
    for word in sample_words:
        interstitial_val = predictor._calculate_interstitial_value(word)
        print(f"'{word}': {interstitial_val:.3f}")
    
    # Performance statistics
    print(f"\nPerformance Statistics:")
    print(f"Vocabulary size: {predictor.vocab_size}")
    print(f"Bigram count: {len(predictor.bigram_frequencies)}")
    print(f"Threads used: {predictor.n_threads}")
    print(f"Model features: {predictor.model_features_shape}")


def batch_process_texts(text_files: List[str], output_dir: str = "models", n_threads: int = 4):
    """Process multiple text files and save individual models"""
    print(f"Batch processing {len(text_files)} files...")
    os.makedirs(output_dir, exist_ok=True)
    
    for i, file_path in enumerate(text_files):
        print(f"\n{'='*60}")
        print(f"Processing file {i+1}/{len(text_files)}: {file_path}")
        print(f"{'='*60}")
        
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = ' '.join(f.read().split())
            
            # Create predictor for this file
            predictor = InterstitialMarkovianPredictor(n_threads=n_threads)
            
            # Extract base filename for saving
            base_name = os.path.splitext(os.path.basename(file_path))[0]
            model_path = os.path.join(output_dir, f"{base_name}_model.pkl")
            transitions_path = os.path.join(output_dir, f"{base_name}_transitions.json")
            
            # Process the text
            predictor.extract_transition_probabilities(content)
            predictor.train_interstitial_predictor(epochs=100)
            
            # Generate transitions
            interstitial_transitions = predictor.generate_interstitial_transitions(interpolation_factor=0.4)
            
            # Save everything
            predictor.save_model(model_path)
            predictor.save_transitions(interstitial_transitions, transitions_path)
            
            print(f"Saved model to {model_path}")
            print(f"Saved transitions to {transitions_path}")
            
        except Exception as e:
            print(f"Error processing {file_path}: {e}")
            continue
    
    print(f"\nBatch processing completed! Models saved in {output_dir}")


def compare_models(model_paths: List[str], test_prompt: str = "the quick brown"):
    """Compare multiple trained models on the same prompt"""
    print(f"Comparing {len(model_paths)} models...")
    
    results = {}
    
    for model_path in model_paths:
        print(f"\nLoading model: {model_path}")
        predictor = InterstitialMarkovianPredictor()
        
        if predictor.load_model(model_path):
            # Try to load corresponding transitions
            transitions_path = model_path.replace('_model.pkl', '_transitions.json')
            if os.path.exists(transitions_path):
                transitions = predictor.load_transitions(transitions_path)
            else:
                print(f"No transitions file found for {model_path}, generating...")
                transitions = predictor.generate_interstitial_transitions(interpolation_factor=0.4)
            
            # Generate text
            generated = predictor.generate_text_with_interstitial_values(
                transitions, length=30, seed=test_prompt
            )
            
            model_name = os.path.basename(model_path).replace('_model.pkl', '')
            results[model_name] = {
                'text': generated,
                'vocab_size': predictor.vocab_size,
                'bigram_count': len(predictor.bigram_frequencies)
            }
        else:
            print(f"Failed to load {model_path}")
    
    # Display results
    print(f"\n{'='*80}")
    print(f"MODEL COMPARISON RESULTS")
    print(f"Test prompt: '{test_prompt}'")
    print(f"{'='*80}")
    
    for model_name, data in results.items():
        print(f"\nModel: {model_name}")
        print(f"Vocab size: {data['vocab_size']}, Bigrams: {data['bigram_count']}")
        print(f"Generated: {data['text']}")
        print("-" * 60)


def interactive_model_explorer():
    """Interactive tool to explore saved models"""
    print("Interactive Model Explorer")
    print("=" * 40)
    
    # Find available models
    model_files = []
    for file in os.listdir('.'):
        if file.endswith('_model.pkl'):
            model_files.append(file)
    
    if not model_files:
        print("No model files found in current directory.")
        return
    
    print("Available models:")
    for i, model_file in enumerate(model_files):
        print(f"{i+1}. {model_file}")
    
    try:
        choice = int(input("\nSelect model number: ")) - 1
        if 0 <= choice < len(model_files):
            model_path = model_files[choice]
            
            # Load the model
            predictor = InterstitialMarkovianPredictor()
            if predictor.load_model(model_path):
                print(f"Loaded model: {model_path}")
                
                # Load transitions
                transitions_path = model_path.replace('_model.pkl', '_transitions.json')
                if os.path.exists(transitions_path):
                    transitions = predictor.load_transitions(transitions_path)
                    print(f"Loaded transitions: {transitions_path}")
                else:
                    print("Generating transitions...")
                    transitions = predictor.generate_interstitial_transitions()
                
                # Interactive session
                print("\nInteractive session started. Type 'quit' to exit.")
                print("Commands: 'stats', 'top_words', 'interstitial <word>', 'generate <prompt>'")
                
                while True:
                    cmd = input("\n> ").strip()
                    
                    if cmd.lower() == 'quit':
                        break
                    elif cmd.lower() == 'stats':
                        print(f"Vocabulary size: {predictor.vocab_size}")
                        print(f"Bigram count: {len(predictor.bigram_frequencies)}")
                        print(f"Transition count: {len(transitions)}")
                        print(f"Model features: {predictor.model_features_shape}")
                    elif cmd.lower() == 'top_words':
                        top_words = predictor.unigram_counts.most_common(10)
                        print("Top 10 most frequent words:")
                        for word, count in top_words:
                            print(f"  {word}: {count}")
                    elif cmd.startswith('interstitial '):
                        word = cmd[13:].strip()
                        if word in predictor.transition_matrix:
                            value = predictor._calculate_interstitial_value(word)
                            print(f"Interstitial value for '{word}': {value:.4f}")
                            
                            # Show transitions
                            transitions_from_word = dict(predictor.transition_matrix[word])
                            sorted_transitions = sorted(transitions_from_word.items(), 
                                                      key=lambda x: x[1], reverse=True)[:5]
                            print(f"Top transitions from '{word}':")
                            for next_word, prob in sorted_transitions:
                                print(f"  {word} -> {next_word}: {prob:.4f}")
                        else:
                            print(f"Word '{word}' not found in vocabulary")
                    elif cmd.startswith('generate '):
                        prompt = cmd[9:].strip()
                        if prompt:
                            generated = predictor.generate_text_with_interstitial_values(
                                transitions, length=40, seed=prompt
                            )
                            print(f"Generated: {generated}")
                        else:
                            print("Please provide a prompt")
                    else:
                        print("Unknown command. Available: stats, top_words, interstitial <word>, generate <prompt>")
            else:
                print(f"Failed to load model: {model_path}")
        else:
            print("Invalid selection")
    except (ValueError, IndexError):
        print("Invalid input")


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1:
        command = sys.argv[1].lower()
        
        if command == 'demo':
            demonstrate_interstitial_markovian()
        elif command == 'batch' and len(sys.argv) > 2:
            text_files = sys.argv[2:]
            batch_process_texts(text_files)
        elif command == 'compare' and len(sys.argv) > 2:
            model_paths = [f for f in sys.argv[2:] if f.endswith('.pkl')]
            if model_paths:
                prompt = input("Enter test prompt (or press Enter for default): ").strip()
                if not prompt:
                    prompt = "the quick brown"
                compare_models(model_paths, prompt)
            else:
                print("No valid model files provided")
        elif command == 'explore':
            interactive_model_explorer()
        else:
            print("Usage:")
            print("  python script.py demo                    # Run interactive demo")
            print("  python script.py batch file1.txt file2.txt  # Batch process files")
            print("  python script.py compare model1.pkl model2.pkl  # Compare models")
            print("  python script.py explore                # Interactive model explorer")
    else:
        demonstrate_interstitial_markovian()
