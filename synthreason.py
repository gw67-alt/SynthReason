import numpy as np
from collections import Counter, defaultdict
import random
import os
from typing import Dict, List, Tuple, Optional, Any
import torch
import torch.nn as nn
import torch.optim as optim
from concurrent.futures import ThreadPoolExecutor
KB_LEN = 10000
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
        # Weights can be adjusted; here we use equal for simplicity
        weights = [0.4, 0.4, 0.1, 0.1]
        interstitial_value = sum(w * f for w, f in zip(weights, features))
        return interstitial_value

    def _process_word_features(self, word_data: Tuple[str, int]) -> Tuple[List[float], float]:
        """Process features for a single word (thread-safe)"""
        word1, w1_idx = word_data
        # For this example, use the last word in the state as the representative word
        # (in reality, you might want to use both words in a more sophisticated model)
        state = (self.idx_to_word.get(w1_idx-1, ""), word1) if w1_idx > 0 else ("", word1)
        feature_vector = [
            self._calculate_interstitial_value(state) if state[1] else 0.0,
            len(self.transition_matrix[word1]) / self.vocab_size if self.vocab_size > 0 else 0.0,
            self.unigram_counts[word1] / sum(self.unigram_counts.values()) if sum(self.unigram_counts.values()) > 0 else 0.0,
        ]
        # Add any other features you want here
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

    def train_interstitial_predictor(self, epochs: int = 100):
        """Train neural network to predict interstitial markovian values"""
        features, targets = self.create_interstitial_features()
        if len(features) == 0:
            print("No features created for training")
            return

        # Store feature normalization parameters
        self.feature_mean = np.mean(features, axis=0)
        self.feature_std = np.std(features, axis=0) + 1e-8
        features = (features - self.feature_mean) / self.feature_std
        self.model_features_shape = features.shape[1]

        # Convert to tensors
        features = np.array(features)  # Convert list to a single numpy array
        X_tensor = torch.FloatTensor(features)
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
            print(f"Epoch {epoch+1}/{epochs}, Loss: {loss.item():.6f}", end="\r")
        print("\nTraining completed!")

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
        if self.predictor_model:
            # Use model prediction if available
            features = []
            for w in next_words:
                # Create feature vector for (current_word, w)
                feature_vector = [
                    self._calculate_interstitial_value((current_word, w)),
                    len(self.transition_matrix[w]) / self.vocab_size if self.vocab_size > 0 else 0.0,
                    self.unigram_counts[w] / sum(self.unigram_counts.values()) if sum(self.unigram_counts.values()) > 0 else 0.0,
                ]
                feature_vector = (np.array(feature_vector) - self.feature_mean) / self.feature_std
                features.append(feature_vector)
            features = np.array(features)  # Convert list to a single numpy array
            features = torch.FloatTensor(features)
            with torch.no_grad():
                interstitial_values = self.predictor_model(features).squeeze().numpy()
        else:
            # Fallback to direct calculation
            interstitial_values = np.array([self._calculate_interstitial_value((current_word, w)) for w in next_words])

        transition_probs = np.array([transitions[w] for w in next_words])
        combined_probs = transition_probs * interstitial_values
        combined_probs = combined_probs / np.sum(combined_probs) if np.sum(combined_probs) > 0 else np.ones_like(combined_probs) / len(combined_probs)
        next_word = np.random.choice(next_words, p=combined_probs)
        return next_word

    def generate_text(self, length: int = 50, seed: Optional[str] = None) -> str:
        """Generate text using interstitial Markovian values and two-step transitions"""
        if not self.unigram_counts:
            return "No data to generate text"

        # Choose starting words
        if seed:
            seed_words = seed.lower().split()
            if len(seed_words) >= 2:
                prev_word, current_word = seed_words[-2], seed_words[-1]
            elif len(seed_words) == 1:
                prev_word, current_word = "", seed_words[0]
            else:
                prev_word, current_word = "", ""
        else:
            prev_word, current_word = "", ""

        if not current_word or current_word not in self.unigram_counts:
            valid_starters = [w for w in self.unigram_counts.keys() if w in self.transition_matrix]
            if valid_starters:
                starter_probs = [self.unigram_counts[w] for w in valid_starters]
                total_prob = sum(starter_probs)
                starter_probs = [p/total_prob for p in starter_probs]
                current_word = np.random.choice(valid_starters, p=starter_probs)
            else:
                current_word = random.choice(list(self.unigram_counts.keys()))

        words = []
        if seed and len(seed_words) >= 2:
            words.extend(seed_words[:-1])
            words.append(current_word)
        elif seed and len(seed_words) == 1:
            words.append(current_word)
        else:
            words.append(current_word)

        # Generate remaining words
        for _ in range(length - len(words)):
            next_word = self.generate_next_word(prev_word, current_word)
            words.append(next_word)
            prev_word, current_word = current_word, next_word

        return ' '.join(words)

# Example usage
if __name__ == "__main__":
    predictor = EnhancedInterstitialMarkovianPredictor()
    with open(input("Filename: "), 'r', encoding='utf-8') as f:
        content = ' '.join(f.read().split()[:KB_LEN])
    predictor.extract_transition_probabilities(content)
    predictor.train_interstitial_predictor(epochs=100)
    while True:
        generated_text = predictor.generate_text(length=250, seed=input("USER: "))
        print("Generated text:", generated_text)
