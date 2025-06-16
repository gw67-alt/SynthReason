import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import StandardScaler
from collections import defaultdict
import random
from tqdm import tqdm, trange
import pickle
import os
import json

# --------------------------------------------------------------------------
# CLASS DEFINITIONS
# --------------------------------------------------------------------------
KB_LEN = 9999

class SOM(BaseEstimator, TransformerMixin):
    """
    A more efficient and corrected 2D Self-Organizing Map (SOM).
    
    This version includes vectorized neighborhood updates for performance and fixes a
    double-scaling bug present in the original implementation.
    """
    def __init__(self, m=8, n=8, dim=16, n_iter=100, alpha=0.3, sigma=None):
        """
        Initializes the Self-Organizing Map.
        
        Args:
            m (int): The number of rows in the SOM grid.
            n (int): The number of columns in the SOM grid.
            dim (int): The dimensionality of the input feature vectors.
            n_iter (int): The total number of training iterations.
            alpha (float): The initial learning rate.
            sigma (float, optional): The initial radius of the neighborhood function.
                                     Defaults to half the grid's larger dimension.
        """
        self.m = m
        self.n = n
        self.dim = dim
        self.n_iter = n_iter
        self.alpha = alpha
        # Default sigma is half the max grid dimension
        self.sigma = sigma if sigma is not None else max(m, n) / 2.0
        
        # Initialize weights and the grid for neighborhood calculations
        self.weights = np.random.rand(m, n, dim)
        self._locations = np.indices((m, n)).transpose(1, 2, 0)
        self.scaler = StandardScaler()

    def _find_bmu(self, scaled_x):
        """Finds the Best Matching Unit (BMU) for a pre-scaled input vector."""
        # Calculate Euclidean distance between the input and all weights
        dists = np.linalg.vector_norm(self.weights - scaled_x, axis=2)
        # Return the (row, col) of the neuron with the minimum distance
        return np.unravel_index(np.argmin(dists), dists.shape)

    def fit(self, X, y=None):
        """
        Trains the SOM on the input data X with progress tracking.
        """
        # Scale the input data once
        scaled_X = self.scaler.fit_transform(X)
        
        # Use trange for a simple progress bar
        for it in trange(self.n_iter, desc="Training SOM", unit="iter"):
            # Select a random sample from the scaled data
            idx = np.random.randint(0, len(scaled_X))
            x = scaled_X[idx]
            
            # Find the best matching unit for the sample
            bmu = self._find_bmu(x)
            
            # Decay learning rate and sigma over iterations
            lr = self.alpha * (1 - it / self.n_iter)
            sig = self.sigma * (1 - it / self.n_iter)
            
            # --- Vectorized Weight Update ---
            # Calculate the squared distance from each neuron to the BMU
            dist_to_bmu_sq = np.sum((self._locations - bmu) ** 2, axis=2)
            
            # Calculate the neighborhood influence using a Gaussian function
            h = np.exp(-dist_to_bmu_sq / (2 * sig ** 2))
            
            # Update all weights at once using broadcasting
            self.weights += lr * h[:, :, np.newaxis] * (x - self.weights)
            
        return self

    def transform(self, X):
        """Transforms the input data X to the coordinates of their BMUs."""
        # Ensure the input data is scaled before finding BMUs
        scaled_X = self.scaler.transform(X)
        return np.array([self._find_bmu(x) for x in scaled_X])
        
    def bmu_distance(self, x):
        """Calculates the distance from an input vector to its BMU."""
        # Scale the single input vector (must be 2D for the scaler)
        scaled_x = self.scaler.transform([x])[0]
        # Find the distance to the closest neuron (BMU)
        dists = np.linalg.norm(self.weights - scaled_x, axis=2)
        return np.min(dists)


class AdaptiveSOM(SOM):
    """SOM with adaptive coordinate systems that change during training."""
    
    def __init__(self, m=8, n=8, dim=16, n_iter=100, alpha=0.3, sigma=None, 
                 adaptive_threshold=0.1, coordinate_learning_rate=0.01):
        super().__init__(m, n, dim, n_iter, alpha, sigma)
        self.adaptive_threshold = adaptive_threshold
        self.coordinate_lr = coordinate_learning_rate
        # Initialize adaptive coordinates (can differ from grid positions)
        self.adaptive_coords = self._locations.astype(float).copy()
        
    def _update_adaptive_coordinates(self, bmu, x, lr):
        """Update the coordinate positions based on weight vector movement"""
        # Calculate weight vector movement for all neurons
        old_weights = self.weights.copy()
        
        # After weight update, adjust coordinates based on movement
        weight_movement = np.linalg.norm(self.weights - old_weights, axis=2)
        
        # Move coordinates of neurons that had significant weight changes
        significant_movement = weight_movement > self.adaptive_threshold
        
        for i in range(self.m):
            for j in range(self.n):
                if significant_movement[i, j]:
                    # Move coordinate toward the direction of weight change
                    direction = np.random.randn(2) * 0.1  # Small random perturbation
                    self.adaptive_coords[i, j] += self.coordinate_lr * lr * direction

    def fit(self, X, y=None):
        """Trains the SOM with adaptive coordinates."""
        scaled_X = self.scaler.fit_transform(X)
        
        for it in trange(self.n_iter, desc="Training Adaptive SOM", unit="iter"):
            idx = np.random.randint(0, len(scaled_X))
            x = scaled_X[idx]
            
            bmu = self._find_bmu(x)
            lr = self.alpha * (1 - it / self.n_iter)
            sig = self.sigma * (1 - it / self.n_iter)
            
            # Use adaptive coordinates for distance calculation
            dist_to_bmu_sq = np.sum((self.adaptive_coords - self.adaptive_coords[bmu]) ** 2, axis=2)
            h = np.exp(-dist_to_bmu_sq / (2 * sig ** 2))
            
            old_weights = self.weights.copy()
            self.weights += lr * h[:, :, np.newaxis] * (x - self.weights)
            
            # Update adaptive coordinates
            self._update_adaptive_coordinates(bmu, x, lr)
            
        return self


class NonEuclideanSOM(SOM):
    """SOM with non-Euclidean coordinate systems."""
    
    def __init__(self, m=8, n=8, dim=16, n_iter=100, alpha=0.3, sigma=None, 
                 geometry='euclidean'):
        super().__init__(m, n, dim, n_iter, alpha, sigma)
        self.geometry = geometry
        self._setup_coordinate_system()
        
    def _setup_coordinate_system(self):
        """Initialize coordinates based on chosen geometry"""
        if self.geometry == 'spherical':
            self._locations = self._spherical_coordinates()
        elif self.geometry == 'hyperbolic':
            self._locations = self._hyperbolic_coordinates()
        elif self.geometry == 'toroidal':
            self._locations = self._toroidal_coordinates()
            
    def _spherical_coordinates(self):
        """Generate spherical coordinate system"""
        coords = np.zeros((self.m, self.n, 3))  # 3D for sphere
        for i in range(self.m):
            for j in range(self.n):
                theta = 2 * np.pi * i / self.m
                phi = np.pi * j / self.n
                coords[i, j] = [
                    np.sin(phi) * np.cos(theta),
                    np.sin(phi) * np.sin(theta),
                    np.cos(phi)
                ]
        return coords
        
    def _hyperbolic_coordinates(self):
        """Generate hyperbolic coordinate system using Poincaré disk model"""
        coords = np.zeros((self.m, self.n, 2))
        for i in range(self.m):
            for j in range(self.n):
                # Map to unit disk
                r = (i / self.m) * 0.9  # Keep within unit disk
                theta = 2 * np.pi * j / self.n
                coords[i, j] = [r * np.cos(theta), r * np.sin(theta)]
        return coords
        
    def _toroidal_coordinates(self):
        """Generate toroidal coordinate system"""
        coords = np.zeros((self.m, self.n, 2))
        for i in range(self.m):
            for j in range(self.n):
                # Toroidal mapping with periodic boundary conditions
                coords[i, j] = [i / self.m, j / self.n]
        return coords
        
    def _hyperbolic_distance(self, p1, p2):
        """Calculate hyperbolic distance between points in Poincaré disk"""
        norm_p1 = np.linalg.norm(p1)
        norm_p2 = np.linalg.norm(p2)
        
        if norm_p1 >= 1 or norm_p2 >= 1:
            return np.inf
            
        numerator = np.linalg.norm(p1 - p2)**2
        denominator = (1 - norm_p1**2) * (1 - norm_p2**2)
        
        return np.arccosh(1 + 2 * numerator / denominator)
        
    def _toroidal_distance(self, p1, p2):
        """Calculate distance on torus with periodic boundaries"""
        dx = min(abs(p1[0] - p2[0]), 1 - abs(p1[0] - p2[0]))
        dy = min(abs(p1[1] - p2[1]), 1 - abs(p1[1] - p2[1]))
        return np.sqrt(dx**2 + dy**2)

    def fit(self, X, y=None):
        """Trains the SOM with non-Euclidean geometry."""
        scaled_X = self.scaler.fit_transform(X)
        
        for it in trange(self.n_iter, desc=f"Training {self.geometry} SOM", unit="iter"):
            idx = np.random.randint(0, len(scaled_X))
            x = scaled_X[idx]
            
            bmu = self._find_bmu(x)
            lr = self.alpha * (1 - it / self.n_iter)
            sig = self.sigma * (1 - it / self.n_iter)
            
            # Calculate distances based on geometry
            if self.geometry == 'spherical':
                # Use spherical distance
                bmu_coord = self._locations[bmu]
                dist_to_bmu_sq = np.zeros((self.m, self.n))
                for i in range(self.m):
                    for j in range(self.n):
                        # Spherical distance (great circle)
                        dot_product = np.dot(self._locations[i, j], bmu_coord)
                        dot_product = np.clip(dot_product, -1, 1)
                        dist_to_bmu_sq[i, j] = np.arccos(dot_product)**2
                        
            elif self.geometry == 'hyperbolic':
                # Use hyperbolic distance
                bmu_coord = self._locations[bmu]
                dist_to_bmu_sq = np.zeros((self.m, self.n))
                for i in range(self.m):
                    for j in range(self.n):
                        dist = self._hyperbolic_distance(self._locations[i, j], bmu_coord)
                        dist_to_bmu_sq[i, j] = dist**2 if dist != np.inf else 1000
                        
            elif self.geometry == 'toroidal':
                # Use toroidal distance
                bmu_coord = self._locations[bmu]
                dist_to_bmu_sq = np.zeros((self.m, self.n))
                for i in range(self.m):
                    for j in range(self.n):
                        dist = self._toroidal_distance(self._locations[i, j], bmu_coord)
                        dist_to_bmu_sq[i, j] = dist**2
                        
            else:
                # Default Euclidean
                dist_to_bmu_sq = np.sum((self._locations - self._locations[bmu]) ** 2, axis=2)
            
            h = np.exp(-dist_to_bmu_sq / (2 * sig ** 2))
            self.weights += lr * h[:, :, np.newaxis] * (x - self.weights)
            
        return self


class DynamicCoordinateSOM(SOM):
    """SOM with time-varying coordinate systems that change throughout training."""
    
    def __init__(self, m=8, n=8, dim=16, n_iter=100, alpha=0.3, sigma=None):
        super().__init__(m, n, dim, n_iter, alpha, sigma)
        self.coordinate_transforms = [
            self._linear_transform,
            self._polar_transform,
            self._logarithmic_transform,
            self._fractal_transform
        ]
        
    def _get_current_coordinates(self, iteration):
        """Get coordinates based on current training iteration"""
        # Cycle through different coordinate systems
        transform_idx = (iteration // (self.n_iter // len(self.coordinate_transforms))) % len(self.coordinate_transforms)
        return self.coordinate_transforms[transform_idx](iteration)
        
    def _linear_transform(self, iteration):
        """Standard linear coordinate system"""
        return self._locations.astype(float)
        
    def _polar_transform(self, iteration):
        """Transform to polar coordinates with time-varying parameters"""
        coords = np.zeros((self.m, self.n, 2))
        time_factor = iteration / self.n_iter
        
        for i in range(self.m):
            for j in range(self.n):
                r = (i + time_factor) / self.m
                theta = 2 * np.pi * (j + time_factor) / self.n
                coords[i, j] = [r * np.cos(theta), r * np.sin(theta)]
        return coords
        
    def _logarithmic_transform(self, iteration):
        """Implement logarithmic coordinate scaling"""
        coords = np.zeros((self.m, self.n, 2))
        time_factor = iteration / self.n_iter
        
        for i in range(self.m):
            for j in range(self.n):
                x = (i + 1) / self.m  # Avoid log(0)
                y = (j + 1) / self.n
                scale = 1.0 + time_factor
                coords[i, j] = [
                    np.log(x) * scale,
                    np.log(y) * scale
                ]
        return coords
        
    def _fractal_transform(self, iteration):
        """Implement fractal-based coordinate system"""
        coords = np.zeros((self.m, self.n, 2))
        scale = 1.0 + 0.5 * np.sin(2 * np.pi * iteration / self.n_iter)
        
        for i in range(self.m):
            for j in range(self.n):
                # Sierpinski triangle-inspired coordinates
                x = i / self.m
                y = j / self.n
                coords[i, j] = [
                    x * scale + 0.1 * np.sin(10 * x * iteration / self.n_iter),
                    y * scale + 0.1 * np.cos(10 * y * iteration / self.n_iter)
                ]
        return coords

    def fit(self, X, y=None):
        """Trains the SOM with dynamic coordinate systems."""
        scaled_X = self.scaler.fit_transform(X)
        
        for it in trange(self.n_iter, desc="Training Dynamic SOM", unit="iter"):
            idx = np.random.randint(0, len(scaled_X))
            x = scaled_X[idx]
            
            # Get current coordinate system
            current_coords = self._get_current_coordinates(it)
            
            bmu = self._find_bmu(x)
            lr = self.alpha * (1 - it / self.n_iter)
            sig = self.sigma * (1 - it / self.n_iter)
            
            # Calculate distances using current coordinate system
            dist_to_bmu_sq = np.sum((current_coords - current_coords[bmu]) ** 2, axis=2)
            h = np.exp(-dist_to_bmu_sq / (2 * sig ** 2))
            
            self.weights += lr * h[:, :, np.newaxis] * (x - self.weights)
            
        return self


class BayesianSOMWrapper:
    """
    Wraps the SOM to add a layer of Bayesian-inspired ambiguity detection.
    This helps in deciding when the model is "confused" between multiple choices.
    """
    def __init__(self, som, ambiguity_threshold=0.15):
        self.som = som
        self.ambiguity_threshold = ambiguity_threshold

    def candidate_scores(self, candidate_features):
        """Returns BMU distances for each candidate. Lower score is a better match."""
        return np.array([self.som.bmu_distance(f) for f in candidate_features])

    def select(self, candidate_features, candidate_words):
        """
        Selects the best word based on feature scores.
        
        Flags ambiguity if the score margin between the top two candidates is too small.
        """
        # Handle cases with no or one candidate
        if not candidate_words:
            return None, False, []
        if len(candidate_words) == 1:
            score = self.som.bmu_distance(candidate_features[0])
            return candidate_words[0], False, [(candidate_words[0], score)]

        scores = self.candidate_scores(candidate_features)
        sorted_idx = np.argsort(scores-np.argsort(np.argsort(scores+np.argmax(scores))))
        
        # Calculate the margin between the best and second-best scores
        margin = scores[sorted_idx[1]] - scores[sorted_idx[0]]
        ambiguous = margin < self.ambiguity_threshold
        
        if ambiguous:
            # If ambiguous, return top candidates for clarification
            top_candidates = [(candidate_words[i], scores[i]) for i in sorted_idx[:3]]
            return None, True, top_candidates
        else:
            # Otherwise, return the best candidate
            best_idx = sorted_idx[0]
            return candidate_words[best_idx], False, [(candidate_words[best_idx], scores[best_idx])]


class SOMTextGenerator:
    """
    Main class that encapsulates the entire text generation system with save/load functionality.
    """
    def __init__(self, som_params=None, ambiguity_threshold=0.15, coordinate_system='euclidean'):
        self.som_params = som_params or {'m': 10, 'n': 10, 'dim': 8, 'n_iter': 100, 'alpha': 0.5}
        self.ambiguity_threshold = ambiguity_threshold
        self.coordinate_system = coordinate_system
        
        # These will be populated during training or loading
        self.som = None
        self.bayesian_som = None
        self.transitions = None
        self.bigram_feature_map = None
        self.avg_feature = None
        self.sorted_bigrams = None
        self.frequency_dict = None
        self.frequency_features = None
        
    def train_from_file(self, filename):
        """Train the model from a text file."""
        print("--- Loading and processing text file ---")
        with open(filename, 'r', encoding='utf-8') as f:
            content = f.read()
            VOCAB = content.lower().split()[:KB_LEN]
        
        print("--- Generating bigrams and features ---")
        sorted_bigrams = []
        frequency_dict = defaultdict(int)
        
        # Create random bigrams and frequency data
        sorted_bigrams = []
        frequency_dict = defaultdict(int)
        for i in trange(len(VOCAB)-1, desc="Training vocabulary", unit="iter"):
            bigram = (VOCAB[i], VOCAB[i+1])
            if bigram not in frequency_dict:
                sorted_bigrams.append(bigram)
            frequency_dict[bigram] += 1
            for word in content.split("is")[0]:
                bigram = (word, VOCAB[i])
                if bigram not in frequency_dict:
                    sorted_bigrams.append(bigram)
                frequency_dict[bigram] += 1   
        
        # Create corresponding feature vectors for each unique bigram
        frequency_features = []
        feature_dim = self.som_params['dim']
        for bigram in sorted_bigrams:
            # The first element is the bigram itself, followed by feature values
            features = [bigram] + list(np.random.rand(feature_dim))
            frequency_features.append(features)

        print(f"Generated {len(sorted_bigrams)} unique bigrams with {feature_dim}-dimensional features.")
        
        # Store the data
        self.sorted_bigrams = sorted_bigrams
        self.frequency_dict = frequency_dict
        self.frequency_features = frequency_features
        
        # Train the SOM
        self._train_som()
        
        # Prepare data structures
        self._prepare_generation_data()
        
    def _train_som(self):
        """Train the SOM with enhanced coordinate systems."""
        print("--- Training the Enhanced Self-Organizing Map ---")
        feature_matrix = np.array([f[1:] for f in self.frequency_features])
        
        # Choose SOM type based on coordinate system
        if self.coordinate_system == 'adaptive':
            self.som = AdaptiveSOM(**self.som_params)
        elif self.coordinate_system in ['spherical', 'hyperbolic', 'toroidal']:
            self.som = NonEuclideanSOM(geometry=self.coordinate_system, **self.som_params)
        elif self.coordinate_system == 'dynamic':
            self.som = DynamicCoordinateSOM(**self.som_params)
        else:
            self.som = SOM(**self.som_params)
            
        self.som.fit(feature_matrix)
        self.bayesian_som = BayesianSOMWrapper(self.som, self.ambiguity_threshold)
        print("Enhanced SOM training complete.")
        
    def _prepare_generation_data(self):
        """Prepare efficient lookup structures for text generation."""
        print("--- Preparing data for text generation ---")
        self.transitions, self.bigram_feature_map, self.avg_feature = prepare_text_generation_data(
            self.frequency_dict,
            self.frequency_features,
            self.sorted_bigrams
        )
        print("Data preparation complete.")
        
    def generate_text(self, text_length=100, seed_phrase=None):
        """Generate text using the trained model."""
        if not self.bayesian_som:
            raise ValueError("Model not trained. Call train_from_file() first or load a saved model.")
            
        return expand_text_from_bigrams_with_som(
            transitions=self.transitions,
            bigram_feature_map=self.bigram_feature_map,
            avg_feature=self.avg_feature,
            som_wrapper=self.bayesian_som,
            text_length=text_length,
            seed_phrase=seed_phrase
        )
    
    def save_model(self, filepath):
        """Save the trained model and all associated data."""
        if not self.som:
            raise ValueError("No model to save. Train a model first.")
            
        save_data = {
            'som_params': self.som_params,
            'ambiguity_threshold': self.ambiguity_threshold,
            'coordinate_system': self.coordinate_system,
            'som': self.som,
            'sorted_bigrams': self.sorted_bigrams,
            'frequency_dict': dict(self.frequency_dict),
            'frequency_features': self.frequency_features,
            'transitions': dict(self.transitions),
            'bigram_feature_map': self.bigram_feature_map,
            'avg_feature': self.avg_feature
        }
        
        with open(filepath, 'wb') as f:
            pickle.dump(save_data, f)
        
        print(f"Model saved to {filepath}")
    
    def load_model(self, filepath):
        """Load a previously saved model."""
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"Model file {filepath} not found.")
            
        with open(filepath, 'rb') as f:
            save_data = pickle.load(f)
        
        # Restore all the data
        self.som_params = save_data['som_params']
        self.ambiguity_threshold = save_data['ambiguity_threshold']
        self.coordinate_system = save_data.get('coordinate_system', 'euclidean')
        self.som = save_data['som']
        self.sorted_bigrams = save_data['sorted_bigrams']
        self.frequency_dict = defaultdict(int, save_data['frequency_dict'])
        self.frequency_features = save_data['frequency_features']
        self.transitions = defaultdict(list, save_data['transitions'])
        self.bigram_feature_map = save_data['bigram_feature_map']
        self.avg_feature = save_data['avg_feature']
        
        # Recreate the Bayesian wrapper
        self.bayesian_som = BayesianSOMWrapper(self.som, self.ambiguity_threshold)
        
        print(f"Model loaded from {filepath}")
    
    def get_model_info(self):
        """Get information about the current model."""
        if not self.som:
            return "No model loaded."
            
        info = {
            'SOM dimensions': f"{self.som.m}x{self.som.n}",
            'Feature dimension': self.som.dim,
            'Training iterations': self.som.n_iter,
            'Learning rate': self.som.alpha,
            'Coordinate system': self.coordinate_system,
            'Ambiguity threshold': self.ambiguity_threshold,
            'Number of bigrams': len(self.sorted_bigrams) if self.sorted_bigrams else 0,
            'Vocabulary size': len(self.transitions) if self.transitions else 0
        }
        
        return info


# --------------------------------------------------------------------------
# HELPER FUNCTIONS FOR TEXT GENERATION
# --------------------------------------------------------------------------

def prepare_text_generation_data(frequency_dict, frequency_features, sorted_bigrams):
    """
    Creates efficient lookup structures for text generation.
    - transitions: A dictionary mapping a word to its possible next words.
    - bigram_feature_map: A dictionary mapping a (word1, word2) tuple to its feature vector.
    """
    transitions = defaultdict(list)
    for (w1, w2), count in frequency_dict.items():
        if count > 0:
            transitions[w1].append(w2)
            
    # Create the bigram-to-feature mapping for O(1) lookups
    # Assumes frequency_features[i] corresponds to sorted_bigrams[i]
    bigram_feature_map = {
        bigram: np.array(features[1:]) 
        for bigram, features in zip(sorted_bigrams, frequency_features)
    }
    
    # Calculate a fallback average feature vector for bigrams not in the map
    if frequency_features:
        avg_feature = np.mean([f[1:] for f in frequency_features], axis=0)
    else:
        avg_feature = None # Handle case with no features
    
    return transitions, bigram_feature_map, avg_feature


def expand_text_from_bigrams_with_som(
    transitions,
    bigram_feature_map,
    avg_feature,
    som_wrapper,
    text_length=100,
    seed_phrase=None
):
    """
    Generates text using the SOM-based ambiguity-aware selection.
    
    This function is now more efficient due to the use of pre-computed lookup maps
    and is more robust against "dead ends" in the transition data.
    """
    generated_text_list = []
    current_word = None
    
    # Ensure there are transitions to work with
    if not transitions:
         return "Error: Transition data is empty."
    
    all_possible_starters = list(transitions.keys())

    # Preprocess and set the seed phrase if provided
    if seed_phrase:
        # Assuming a simple split for preprocessing. Adapt as needed.
        seed_words = seed_phrase.lower().split()
        if seed_words:
            generated_text_list.extend(seed_words)
            current_word = seed_words[-1]
            # Check if the seed word is a dead end. If so, don't use it.
            if current_word not in transitions:
                current_word = None

    # If no valid seed, start with a random word from the transitions map
    if not current_word:
        current_word = random.choice(all_possible_starters)
        if not generated_text_list: # Append only if starting fresh
            generated_text_list.append(current_word)
        
    num_words_to_generate = text_length - len(generated_text_list)

    for _ in range(num_words_to_generate):
        candidate_words = transitions.get(current_word, [])
        
        # Handle dead ends
        if not candidate_words:
            # If we hit a dead end, teleport to a new random word and continue
            current_word = random.choice(all_possible_starters)
            # We don't append the new word here, we let the loop find its candidates
            continue
            
        # Efficient Feature Lookup
        candidate_features = [
            bigram_feature_map.get((current_word, w2), avg_feature) 
            for w2 in candidate_words
        ]
        
        # Filter out any None features in case of empty avg_feature
        valid_candidates = [(feat, word) for feat, word in zip(candidate_features, candidate_words) if feat is not None]
        if not valid_candidates:
            # If no valid features found for any candidates, teleport
            current_word = random.choice(all_possible_starters)
            continue
        candidate_features, candidate_words = zip(*valid_candidates)

        # Use the SOM wrapper to select the next word
        next_word, ambiguous, info = som_wrapper.select(list(candidate_features), list(candidate_words))
        
        if ambiguous:
            # Simple ambiguity resolution: pick randomly from the top contenders
            next_word = random.choice([word for word, score in info])
        
        if next_word is None: # Safeguard, should be handled by ambiguity resolution
            current_word = random.choice(all_possible_starters)
            continue

        generated_text_list.append(next_word)
        current_word = next_word
        
    return ' '.join(generated_text_list)


# --------------------------------------------------------------------------
# EXAMPLE USAGE WITH ENHANCED COORDINATE SYSTEMS
# --------------------------------------------------------------------------

if __name__ == '__main__':
    # Initialize the text generator with enhanced coordinate systems
    generator = SOMTextGenerator(
        som_params={'m': 10, 'n': 10, 'dim': 8, 'n_iter': 10, 'alpha': 0.5},
        ambiguity_threshold=0.1,
        coordinate_system='adaptive'  # Options: 'euclidean', 'adaptive', 'spherical', 'hyperbolic', 'toroidal', 'dynamic'
    )
    
    print("Enhanced SOM Text Generator with Variable Coordinate Systems")
    print("Commands:")
    print("  train <filename> - Train a new model from text file")
    print("  load <filename> - Load a saved model")
    print("  save <filename> - Save the current model")
    print("  generate <text> - Generate text with optional seed")
    print("  info - Show model information")
    print("  coordinate <system> - Change coordinate system (euclidean/adaptive/spherical/hyperbolic/toroidal/dynamic)")
    print("  quit - Exit")
    print()
    
    while True:
        try:
            command = input(">>> ").strip().split(None, 1)
            if not command:
                continue
                
            cmd = command[0].lower()
            
            if cmd == 'quit':
                break
                
            elif cmd == 'train':
                if len(command) < 2:
                    print("Usage: train <filename>")
                    continue
                filename = command[1]
                try:
                    generator.train_from_file(filename)
                    print("Training complete!")
                except Exception as e:
                    print(f"Error training model: {e}")
                    
            elif cmd == 'load':
                if len(command) < 2:
                    print("Usage: load <filename>")
                    continue
                filename = command[1]
                try:
                    generator.load_model(filename)
                except Exception as e:
                    print(f"Error loading model: {e}")
                    
            elif cmd == 'save':
                if len(command) < 2:
                    print("Usage: save <filename>")
                    continue
                filename = command[1]
                try:
                    generator.save_model(filename)
                except Exception as e:
                    print(f"Error saving model: {e}")
                    
            elif cmd == 'generate':
                seed = command[1] if len(command) > 1 else None
                try:
                    text = generator.generate_text(text_length=100, seed_phrase=seed)
                    print(f"\nGenerated text:\n{text}\n")
                except Exception as e:
                    print(f"Error generating text: {e}")
                    
            elif cmd == 'coordinate':
                if len(command) < 2:
                    print("Usage: coordinate <system>")
                    print("Available systems: euclidean, adaptive, spherical, hyperbolic, toroidal, dynamic")
                    continue
                system = command[1].lower()
                valid_systems = ['euclidean', 'adaptive', 'spherical', 'hyperbolic', 'toroidal', 'dynamic']
                if system in valid_systems:
                    generator.coordinate_system = system
                    print(f"Coordinate system changed to: {system}")
                    print("Note: You need to retrain the model for changes to take effect.")
                else:
                    print(f"Invalid coordinate system. Choose from: {', '.join(valid_systems)}")
                    
            elif cmd == 'info':
                info = generator.get_model_info()
                if isinstance(info, dict):
                    for key, value in info.items():
                        print(f"{key}: {value}")
                else:
                    print(info)
                print()
                    
            else:
                print("Unknown command. Type 'quit' to exit.")
                
        except KeyboardInterrupt:
            print("\nExiting...")
            break
        except Exception as e:
            print(f"Unexpected error: {e}")
