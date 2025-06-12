import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import StandardScaler
from collections import defaultdict
import random
from tqdm import tqdm, trange

# --------------------------------------------------------------------------
# CLASS DEFINITIONS
# --------------------------------------------------------------------------
KB_LEN = -1
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
        dists = np.linalg.norm(self.weights - scaled_x, axis=2)
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
            dist_to_bmu_sq = np.sum((self._locations - bmu) ** 2, axis=-1)
            
            # Calculate the neighborhood influence using a Gaussian function
            h = np.exp(-dist_to_bmu_sq / (2 * sig ** 2))
            
            # Update all weights at once using broadcasting
            self.weights += lr * h[..., np.newaxis] * (x - self.weights)
            
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
        sorted_idx = np.argsort(scores)
        
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
            # **FIX**: Check if the seed word is a dead end. If so, don't use it.
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
        
        # --- ROBUSTNESS FIX: Handle dead ends ---
        if not candidate_words:
            # If we hit a dead end, teleport to a new random word and continue
            current_word = random.choice(all_possible_starters)
            # We don't append the new word here, we let the loop find its candidates
            continue
            
        # --- Efficient Feature Lookup ---
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
            #print(f"AMBIGUITY DETECTED -> Candidates: {[(w, round(s, 4)) for w, s in info]}")
            # Simple ambiguity resolution: pick randomly from the top contenders
            next_word = random.choice([word for word, score in info])
        
        if next_word is None: # Safeguard, should be handled by ambiguity resolution
            current_word = random.choice(all_possible_starters)
            continue

        generated_text_list.append(next_word)
        current_word = next_word
        
    return ' '.join(generated_text_list)


# --------------------------------------------------------------------------
# EXAMPLE USAGE
# --------------------------------------------------------------------------

if __name__ == '__main__':
    # --- 1. Generate Synthetic Data (as a stand-in for real data) ---
    print("--- Generating synthetic data for demonstration ---")
    FEATURE_DIM = 16
    with open("test.txt", 'r', encoding='utf-8') as f:
        content = f.read()
        VOCAB = content.lower().split()[:KB_LEN]
    
    # Create random bigrams and frequency data
    sorted_bigrams = []
    frequency_dict = defaultdict(int)
    for i in range(len(VOCAB)-1): # Increased count for a denser transition map
       
            bigram = (VOCAB[i], VOCAB[i+1])
            if bigram not in frequency_dict:
                sorted_bigrams.append(bigram)
            frequency_dict[bigram] += 1
            
    # Create corresponding feature vectors for each unique bigram
    frequency_features = []
    for bigram in sorted_bigrams:
        # The first element is the bigram itself, followed by feature values
        features = [bigram] + list(np.random.rand(FEATURE_DIM))
        frequency_features.append(features)

    print(f"Generated {len(sorted_bigrams)} unique bigrams with {FEATURE_DIM}-dimensional features.\n")

    # --- 2. Prepare the feature matrix for the SOM ---
    # We extract only the feature values (ignoring the first element, which is the bigram tuple)
    feature_matrix = np.array([f[1:] for f in frequency_features])

    # --- 3. Initialize and train the SOM ---
    print("--- Training the Self-Organizing Map ---")
    som = SOM(m=10, n=10, dim=FEATURE_DIM, n_iter=10000, alpha=0.5)
    som.fit(feature_matrix)
    print("SOM training complete.\n")
    
    # Wrap the trained SOM with our ambiguity detector
    bayesian_som = BayesianSOMWrapper(som, ambiguity_threshold=0.1)

    # --- 4. Prepare data structures for efficient text generation ---
    print("--- Preparing data for text generation ---")
    transitions, bigram_feature_map, avg_feature = prepare_text_generation_data(
        frequency_dict,
        frequency_features,
        sorted_bigrams
    )
    print("Data preparation complete.\n")

    # --- 5. Generate Text ---
    print("--- Generating Text ---")
    while True:
        user_input = input("USER: ")
        print(f"Seed phrase: '{user_input}'")
        
        generated_text = expand_text_from_bigrams_with_som(
            transitions=transitions,
            bigram_feature_map=bigram_feature_map,
            avg_feature=avg_feature,
            som_wrapper=bayesian_som,
            text_length=275,
            seed_phrase=user_input
        )

        print("\n--- Final Generated Text ---")
        print(generated_text)
