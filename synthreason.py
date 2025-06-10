import numpy as np
# import pandas as pd # No longer used
from collections import Counter, defaultdict, deque
import re
import random
import torch
# import matplotlib.pyplot as plt # No longer used for visualization in main flow
# from sklearn.linear_model import LinearRegression # Replaced by NN option
# from sklearn.ensemble import RandomForestRegressor # Replaced by NN option
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
# import json # No longer used for saving/loading in main flow
from typing import Dict, List, Tuple, Optional, Any, Callable

KB_LENGTH = 1000
# Attempt to import datasets
try:
    from datasets import load_dataset
    DATASETS_AVAILABLE = True
except ImportError:
    DATASETS_AVAILABLE = False
    def load_dataset(*args, **kwargs): # Placeholder
        print("VERBOSE: Warning: Hugging Face 'datasets' library not found. Cannot load from Hub.")
        raise ImportError("Hugging Face 'datasets' library is required for this feature but not found.")

# Attempt to import TensorFlow
try:
    import tensorflow as tf
    TENSORFLOW_AVAILABLE = True
except ImportError:
    TENSORFLOW_AVAILABLE = False
    # print("VERBOSE: Warning: TensorFlow library not found. Neural network model will not be available.")

# Helper functions for bigram key conversion
def bigram_to_key(bigram: Tuple[str, str]) -> str:
    return f"{bigram[0]}||{bigram[1]}"

def key_to_bigram(key: str) -> Tuple[str, str]:
    parts = key.split('||')
    if len(parts) == 2:
        return (parts[0], parts[1])
    return ("<malformed>", "<key>")


class FrequencyPredictor:
    def __init__(self):
        print("VERBOSE: FrequencyPredictor initialized.")
        self.bigram_frequencies: Dict[Tuple[str, str], int] = {}
        self.frequency_features: List[List[float]] = []
        self.predictor_model: Optional[Any] = None
        self.scaler = StandardScaler()
        self.sorted_bigrams: List[Tuple[str, str]] = []
        self.unigram_counts: Dict[str, int] = Counter()
        self.num_base_features: int = 80
        self.feature_operations: Optional[List[Optional[Callable[[np.ndarray], np.ndarray]]]] = None
        self.context_window = 99999
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
                    # print(f"VERBOSE: Applying operation to feature index {i}...") # Can be too verbose
                    try:
                        X_transformed[:, i] = operation(X_data[:, i].astype(float))
                    except Exception as e:
                        print(f"VERBOSE: Error applying operation to feature index {i}: {e}. Feature {i} remains as original.")
                        X_transformed[:, i] = X_data[:, i].astype(float)
        print(f"VERBOSE: Finished applying feature operations. Transformed X_data shape: {X_transformed.shape}")
        if X_data.shape[0] > 0 and X_transformed.shape[0] > 0:
             # Compare a small part of the first row before and after (if non-empty)
            comparison_limit = min(5, X_data.shape[1]) # Compare up to 5 features
            # print(f"VERBOSE: Sample original features (first row, first {comparison_limit}): {X_data[0, :comparison_limit]}")
            # print(f"VERBOSE: Sample transformed features (first row, first {comparison_limit}): {X_transformed[0, :comparison_limit]}")
        return X_transformed

    def load_text_from_hf_dataset(self,
                                  dataset_name: str,
                                  config_name: Optional[str] = None,
                                  split: str = 'train',
                                  text_column: str = 'text',
                                  max_total_words: int = 20000) -> Optional[str]:
        print(f"VERBOSE: Attempting to load text from Hugging Face dataset: {dataset_name} (Config: {config_name}, Split: {split}, Column: {text_column}).")
        if not DATASETS_AVAILABLE:
            print("VERBOSE: Hugging Face 'datasets' library not available. Cannot load from Hub.")
            return None
        try:
            dataset = load_dataset(dataset_name, name=config_name, split=split, streaming=True, trust_remote_code=True)
            print(f"VERBOSE: Accessed HF dataset stream: {dataset_name}, config: {config_name}, split: {split}")
            
            all_collected_words = []
            docs_processed = 0
            for doc in dataset:
                if len(all_collected_words) >= max_total_words:
                    print(f"VERBOSE: Reached max_total_words ({max_total_words}). Stopping collection.")
                    break
                text_content = doc.get(text_column)
                if isinstance(text_content, str):
                    current_doc_words = text_content.lower().split()
                    words_to_add_count = max_total_words - len(all_collected_words)
                    all_collected_words.extend(current_doc_words[:words_to_add_count])
                docs_processed +=1
            
            print(f"VERBOSE: Processed {docs_processed} documents from HF dataset. Collected {len(all_collected_words)} words.")
            if not all_collected_words:
                print("VERBOSE: No words collected from HF dataset.")
                return ""
            return ' '.join(all_collected_words)
        except Exception as e:
            print(f"VERBOSE: Error loading dataset '{dataset_name}': {e}")
            return None

    def load_text_file(self, file_path: str) -> str:
        print(f"VERBOSE: Attempting to load text from local file: {file_path}")
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
                words = content.lower().split()[:KB_LENGTH]
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
        #text_cleaned = re.sub(r'[^a-zA-Z\s]', '', text)
        words = text.split()
        valid_words = [word for word in words if word]
        print(f"VERBOSE: Preprocessing complete. Number of words: {len(valid_words)} (after cleaning and removing empty strings).")
        return valid_words

    def extract_bigram_frequencies(self, text: str) -> Dict[Tuple[str, str], int]:
        print("VERBOSE: Starting bigram frequency extraction.")
        words = self.preprocess_text(text)
        if len(words) < 2:
            print("VERBOSE: Not enough words to form bigrams. Extracted 0 bigrams.")
            self.bigram_frequencies = {}; self.sorted_bigrams = []
            self.unigram_counts = Counter(words); return {}

        self.unigram_counts = Counter(words)
        print(f"VERBOSE: Unigram counts calculated. Total unigrams: {len(self.unigram_counts)}, e.g., {list(self.unigram_counts.items())[:3]}")
        bigrams = [(words[i], words[i+1]) for i in range(len(words)-1)]
        self.bigram_frequencies = dict(Counter(bigrams))
        print(f"VERBOSE: Extracted {len(self.bigram_frequencies)} unique bigrams. Total bigram occurrences: {len(bigrams)}.")
        
        self.sorted_bigrams = [
            item[0] for item in sorted(self.bigram_frequencies.items(), key=lambda x: (x[1], x[0][0], x[0][1]), reverse=True)
        ]
        print(f"VERBOSE: Bigrams sorted by frequency. Top 3: {self.sorted_bigrams[:3] if len(self.sorted_bigrams) >=3 else self.sorted_bigrams}")
        return self.bigram_frequencies

    def create_bigram_frequency_features(self) -> List[List[float]]:
        print("VERBOSE: Starting creation of bigram frequency features.")
        features = []
        if not self.sorted_bigrams:
            print("VERBOSE: No sorted bigrams available to create features.")
            self.frequency_features = []; return []

        for bigram_idx, bigram in enumerate(self.sorted_bigrams):
            y, x = bigram
            x = len(x)
            y = len(y)
            freq = self.bigram_frequencies[bigram]
            bigram_features_vector = [
                # Original functions
        np.log1p(y),
        np.log1p(y),
        np.log1p(y),
        np.log1p(y),
        np.square(y),
        np.square(y),
        None, 
        np.sqrt(np.maximum(0, x)),
        np.log1p(y),
        np.log1p(y),
        None, 
        None, 
        None, 
        None, 
        x * 2.0, 
        None,

        None,
        
        # Additional math functions
        # Basic arithmetic
        x + 1.0,
        x - 1.0,
        x * 3.0,
        x / 2.0,
        x ** 3,
        x ** 0.5,
        x ** (1/3),
        1 / np.maximum(x, 1e-8),  # reciprocal with safety
        
        # Exponential and logarithmic
        np.exp(y),
        np.exp2(y),
        np.expm1(y),
        np.log(np.maximum(x, 1e-8)),
        np.log2(np.maximum(x, 1e-8)),
        np.log10(np.maximum(x, 1e-8)),
        
        # Trigonometric
        np.sin(y),
        np.cos(y),
        np.tan(y),
        np.arcsin(np.clip(x, -1, 1)),
        np.arccos(np.clip(x, -1, 1)),
        np.arctan(y),
        
        # Hyperbolic
        np.sinh(y),
        np.cosh(y),
        np.tanh(y),
        np.arcsinh(y),
        np.arccosh(np.maximum(x, 1)),
        np.arctanh(np.clip(x, -0.99, 0.99)),
        
        # Rounding and ceiling/floor
        np.round(y),
        np.floor(y),
        np.ceil(y),
        np.trunc(x),
        
        # Sign and absolute
        np.abs(x),
        np.sign(x),
        np.positive(x),
        np.negative(x),
        
        # Power and roots
        np.cbrt(x),  # cube root
        np.power(x, 4),
        np.power(x, 0.25),  # 4th root
        np.power(x, 1.5),
        
        # Special functions
        np.maximum(x, 0),  # ReLU
        np.minimum(x, 0),  # negative part
        np.maximum(x, 1),  # max with 1
        np.minimum(x, 1),  # min with 1
        
        # Statistical transforms
        (x - np.mean(x)) / (np.std(x) + 1e-8),  # standardize
        x / (np.max(np.abs(x)) + 1e-8),  # normalize by max
        np.clip(x, 0, 1),  # clip to [0,1]
        np.clip(x, -1, 1),  # clip to [-1,1]
        
        # Complex transformations
        x / (1 + np.abs(x)),  # soft sign
        np.where(x > 0, x, 0.01 * x),  # leaky ReLU
        np.log(1 + np.exp(-np.abs(x))) + np.maximum(x, 0),  # softplus
        x * (1 / (1 + np.exp(-x))),  # swish activation
        # Additional basic features
        x,  # identity
        np.sqrt(np.abs(x) + 1e-8),  # sqrt(abs(x))
        1 / (np.sqrt(np.abs(x) + 1e-8)),  # reciprocal sqrt
        1 / (x ** 2 + 1e-8),  # inverse square
        np.clip(x, 0, None),  # zero out negative values
        np.clip(x, None, 0),  # zero out positive values
        None , # binary indicator: positive
        None,  # binary indicator: negative
        None,  # binary indicator: zero
        x - np.mean(x),  # zero-mean
        (x - np.min(x)) / (np.max(x) - np.min(x) + 1e-8),  # min-max scaling
        np.exp(-x**2),  # Gaussian basis
        1 / (1 + np.exp(-x)),  # sigmoid
        np.heaviside(x, 0.0),  # Heaviside step
            ]
            features.append(bigram_features_vector)
            # if bigram_idx < 2: # Print features for first 2 bigrams
            #     print(f"VERBOSE: Features for bigram {bigram}: {bigram_features_vector}")

        self.frequency_features = features
        if features:
            print(f"VERBOSE: Created {len(features)} feature vectors, each with {len(features[0])} elements (1 target + {self.num_base_features} inputs).")
        else:
            print("VERBOSE: No features were created.")
        return features

    def train_predictor(self, model_type: str = 'neural_network') -> None:
        print(f"VERBOSE: Starting predictor training with model_type: {model_type}.")
        if not self.frequency_features:
            print("VERBOSE: No frequency features available. Skipping training.")
            self.predictor_model = None; return

        try:
            # Assuming self.frequency_features rows are [target, feat1, feat2, ...]
            X_raw = np.array([f[1:] for f in self.frequency_features], dtype=float) 
            y = np.array([f[0] for f in self.frequency_features], dtype=float)  
        except Exception as e:
            print(f"VERBOSE: Error converting features to NumPy array: {e}. Check feature creation. Skipping training.")
            self.predictor_model = None; return
            
        print(f"VERBOSE: Raw features X_raw shape: {X_raw.shape}, Target y shape: {y.shape}.")

        # This check is important. If it fails, _apply_feature_operations might skip,
        # or if it proceeds, it might operate on incorrectly shaped data.
        if X_raw.shape[0] <= 1 or X_raw.ndim != 2 or X_raw.shape[0] != y.shape[0] or X_raw.shape[1] != self.num_base_features:
            print(f"VERBOSE: Warning: Shape of X_raw ({X_raw.shape}) or y ({y.shape}) is unsuitable for training/transformation. Num base features: {self.num_base_features}. Skipping actual model training.")
            # If you commented out the 'return' here previously, ensure X_raw is what _apply_feature_operations expects
            # or that _apply_feature_operations handles the shape mismatch by returning X_raw untransformed.
            # For now, let's assume if this warning hits, X_transformed might just be X_raw.
            # self.predictor_model = None; return # Original line commented out by user.
        
        X_transformed = self._apply_feature_operations(X_raw) 
        print(f"VERBOSE: Transformed features X_transformed shape: {X_transformed.shape}.")
        
        # --- MODIFICATION TO "SKIP" (REPLACE) INFINITIES AND NANS ---
        # Check if X_transformed contains any NaN or Inf values before proceeding
        has_nan = np.isnan(X_transformed).any()
        has_inf = np.isinf(X_transformed).any()

        if has_nan or has_inf:
            if has_nan and has_inf:
                print("VERBOSE: X_transformed contains both NaNs and infinities.")
            elif has_nan:
                print("VERBOSE: X_transformed contains NaNs.")
            elif has_inf:
                print("VERBOSE: X_transformed contains infinities.")
            
            print("VERBOSE: Applying np.nan_to_num to replace NaNs/infinities.")
            # Replace NaN with 0.0
            # Replace positive infinity with the largest positive number for the dtype
            # Replace negative infinity with the smallest negative number for the dtype
            X_transformed = np.nan_to_num(X_transformed, 
                                          nan=0.0, 
                                          posinf=np.finfo(X_transformed.dtype).max, 
                                          neginf=np.finfo(X_transformed.dtype).min)
            print(f"VERBOSE: X_transformed after np.nan_to_num. Shape: {X_transformed.shape}.")
            # Verify again (optional)
            # if np.isnan(X_transformed).any() or np.isinf(X_transformed).any():
            #    print("VERBOSE: Warning: NaNs or infinities still present after np.nan_to_num.")
        # --- END MODIFICATION ---
        
        # Ensure X_transformed is not empty before proceeding
        if X_transformed.shape[0] == 0:
            print("VERBOSE: X_transformed is empty after operations and/or cleaning. Skipping training.")
            self.predictor_model = None; return

        test_size_for_split = 0.2 if X_transformed.shape[0] >= 10 else 0.0
        if X_transformed.shape[0] < 10:
            print(f"VERBOSE: Dataset too small ({X_transformed.shape[0]} samples), using all data for training and testing.")
            # Ensure y matches X_transformed's sample size if filtering happened (not in this nan_to_num)
            X_train, X_test, y_train, y_test = (X_transformed, X_transformed, y[:X_transformed.shape[0]], y[:X_transformed.shape[0]])
        else:
            X_train, X_test, y_train, y_test = train_test_split(X_transformed, y, test_size=test_size_for_split, random_state=42)
        
        print(f"VERBOSE: Data split. X_train shape: {X_train.shape}, X_test shape: {X_test.shape}, y_train shape: {y_train.shape}, y_test shape: {y_test.shape}.")
        
        if X_train.shape[0] == 0:
            print("VERBOSE: X_train is empty after split. Skipping training.")
            self.predictor_model = None; return

        print("VERBOSE: Fitting StandardScaler on X_train.")
        try:
            self.scaler.fit(X_train)
            X_train_scaled = self.scaler.transform(X_train)
            # X_test_scaled = self.scaler.transform(X_test) # Also scale X_test for evaluation
            print(f"VERBOSE: X_train scaled. Shape: {X_train_scaled.shape}. Scaler mean (first 3): {self.scaler.mean_[:3] if self.scaler.mean_.size >=3 else self.scaler.mean_}..., Scaler scale (first 3): {self.scaler.scale_[:3] if self.scaler.scale_.size >=3 else self.scaler.scale_}...")
        except ValueError as e:
            print(f"VERBOSE: ValueError during StandardScaler fit/transform: {e}. This might happen if data still contains NaNs/infs or all features are constant after nan_to_num.")
            self.predictor_model = None; return

        # ... (rest of the neural network training code) ...
        if model_type == 'neural_network':
            if not TENSORFLOW_AVAILABLE:
                print("VERBOSE: TensorFlow not available. Cannot train neural network model.")
                self.predictor_model = None; return
            
            print("VERBOSE: Defining Keras Sequential model.")
            # Ensure input_shape matches X_train_scaled.shape[1]
            if X_train_scaled.shape[1] == 0:
                 print("VERBOSE: X_train_scaled has 0 features. Cannot define model input layer.")
                 self.predictor_model = None; return

            nn_model = tf.keras.Sequential([
                tf.keras.layers.InputLayer(input_shape=(X_train_scaled.shape[1],)),
                tf.keras.layers.Dense(128, activation='relu'),
                tf.keras.layers.Dropout(0.2),
                tf.keras.layers.Dense(64, activation='relu'),
                tf.keras.layers.Dropout(0.2),
                tf.keras.layers.Dense(32, activation='relu'),
                tf.keras.layers.Dense(1) 
            ])
            nn_model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.001), loss='mean_squared_error')
            
            print(f"VERBOSE: Training Neural Network model on {X_train_scaled.shape[0]} samples for 100 epochs...")
            history = nn_model.fit(X_train_scaled, y_train, epochs=100, batch_size=32, verbose=0)
            self.predictor_model = nn_model
            print(f"VERBOSE: Neural Network predictor trained. Final training loss: {history.history['loss'][-1]:.4f}")
        else:
            print(f"VERBOSE: Unsupported model_type: {model_type} or required library missing. No model trained.")
            self.predictor_model = None


    def generate_new_bigram_frequencies(self, num_variations: int = 1) -> List[Dict[Tuple[str, str], float]]:
        print(f"VERBOSE: Starting generation of {num_variations} new bigram frequency set(s).")
        if self.predictor_model is None:
            print("VERBOSE: Predictor model not available. Cannot generate new frequencies.")
            return []
        if not self.frequency_features or not self.sorted_bigrams:
            print("VERBOSE: Missing frequency features or sorted bigrams. Cannot generate.")
            return []

        new_frequency_sets = []
        original_total_frequency_sum = sum(self.bigram_frequencies.values())
        if original_total_frequency_sum == 0 : original_total_frequency_sum = 1.0 # Avoid division by zero
        print(f"VERBOSE: Original total frequency sum for scaling: {original_total_frequency_sum}")

        try:
            base_X_for_prediction = np.array([f[1:] for f in self.frequency_features], dtype=float)
        except Exception as e:
            print(f"VERBOSE: Error preparing base_X_for_prediction from self.frequency_features: {e}. Cannot generate.")
            return []

        # This check is crucial and must pass for the rest of the function to be valid.
        # Assuming this passes based on the error occurring later at self.scaler.transform().
        if base_X_for_prediction.shape[1] != self.num_base_features:
            print(f"VERBOSE: Error: Mismatch in feature dimensions ({base_X_for_prediction.shape[1]} vs {self.num_base_features}) for generation. Cannot proceed.")
            return []
        print(f"VERBOSE: Base features for prediction (X_raw equivalent) shape: {base_X_for_prediction.shape}")

        for variation in range(num_variations):
            print(f"VERBOSE: Generating variation {variation + 1}/{num_variations}.")
            X_noised = base_X_for_prediction.astype(float).copy() # Ensure it's float for operations
            noise_factor = 0.1 + (variation * 0.02)
            
            for j in range(X_noised.shape[1]):
                # Adding noise based on column's std dev might be more stable than abs value if scales vary wildly
                col_std = np.std(X_noised[:, j])
                if col_std == 0: col_std = 0.1 # Add small noise even to constant columns to potentially break constancy
                
                # Or use the user's original noise logic if preferred, but be mindful of X_noised values
                # noise = np.random.normal(0, noise_factor * np.abs(X_noised[:, j] + 1e-8)) # Added epsilon to abs
                noise = np.random.normal(0, noise_factor * col_std, size=X_noised.shape[0])
                X_noised[:, j] = X_noised[:, j] + noise
                # Optional: If certain features should remain non-negative (e.g. counts, lengths after noise)
                # X_noised[:, j] = np.maximum(0, X_noised[:, j])
            
            X_transformed_noised = self._apply_feature_operations(X_noised)
            
            if X_transformed_noised.shape[0] == 0: # Should not happen if X_noised is not empty
                print("VERBOSE: Transformed noised features are unexpectedly empty. Skipping this variation.")
                continue
            
            # --- ADDED SECTION TO HANDLE INFINITIES/LARGE VALUES/NANS ---
            has_problematic_values = False
            if np.isnan(X_transformed_noised).any():
                print("VERBOSE: X_transformed_noised contains NaNs before scaling.")
                has_problematic_values = True
            if np.isinf(X_transformed_noised).any():
                print("VERBOSE: X_transformed_noised contains infinities before scaling.")
                has_problematic_values = True
            
            # Check for values too large for float64, though np.nan_to_num might not fix these if they are already finite but huge
            # Scikit-learn's check is more stringent. Replacing inf might create numbers that are still "too large".
            # However, this step primarily targets np.inf and np.nan.
            if has_problematic_values:
                print("VERBOSE: Applying np.nan_to_num to X_transformed_noised.")
                try:
                    # Replace NaN with 0, inf with large finite numbers
                    X_transformed_noised = np.nan_to_num(X_transformed_noised, 
                                                          nan=0.0, 
                                                          posinf=np.finfo(X_transformed_noised.dtype).max / 2, # Use a slightly smaller max to avoid edge issues
                                                          neginf=np.finfo(X_transformed_noised.dtype).min / 2) # Use a slightly smaller min
                    print(f"VERBOSE: X_transformed_noised after nan_to_num. Shape: {X_transformed_noised.shape}.")
                except Exception as e:
                    print(f"VERBOSE: Error during np.nan_to_num on X_transformed_noised: {e}. Skipping variation.")
                    continue
            # --- END ADDED SECTION ---

            # Line 480 where the error occurred:
            try:
                X_scaled = self.scaler.transform(X_transformed_noised)
            except ValueError as ve:
                print(f"VERBOSE: ValueError during scaler.transform: {ve}. This means X_transformed_noised still has problematic values (likely too large for float64 even if not strictly inf after nan_to_num, or all NaNs in a column). Skipping variation.")
                # You might want to print a sample of X_transformed_noised here for debugging
                # print(f"VERBOSE: Sample of X_transformed_noised that caused error: {X_transformed_noised[:5, :5]}")
                continue # Skip this variation if scaling fails
            
            print("VERBOSE: Predicting new counts with the model...")
            predicted_new_counts = self.predictor_model.predict(X_scaled)
            
            # ... (rest of the function for processing predicted_new_counts)
            if isinstance(predicted_new_counts, tf.Tensor):
                predicted_new_counts = predicted_new_counts.numpy()
            predicted_new_counts = predicted_new_counts.flatten()
            
            predicted_new_counts = np.maximum(predicted_new_counts, 0.01) 
            
            current_sum_predicted_counts = np.sum(predicted_new_counts)
            if current_sum_predicted_counts == 0:
                if len(predicted_new_counts) > 0:
                    predicted_new_counts = np.full_like(predicted_new_counts, 0.01)
                    print("VERBOSE: Sum of predicted counts was 0, filled with 0.01.")
                else:
                    new_frequency_sets.append({});
                    print("VERBOSE: Predicted counts array is empty for this variation.")
                    continue 
            current_sum_predicted_counts = np.sum(predicted_new_counts)
            
            scaled_predicted_counts = predicted_new_counts
            if original_total_frequency_sum > 0 and current_sum_predicted_counts > 0:
                scale_factor = original_total_frequency_sum / current_sum_predicted_counts
                scaled_predicted_counts = predicted_new_counts * scale_factor
            
            new_freq_dict: Dict[Tuple[str, str], float] = {
                bigram: float(scaled_predicted_counts[i]) 
                for i, bigram in enumerate(self.sorted_bigrams) 
                if i < len(scaled_predicted_counts)
            }
            new_frequency_sets.append(new_freq_dict)
            print(f"VERBOSE: Generated frequency dictionary for variation {variation + 1} with {len(new_freq_dict)} entries.")
        
        print(f"VERBOSE: Finished generating {len(new_frequency_sets)} new frequency sets.")
        return new_frequency_sets

    def expand_text_from_bigrams(self,frequency_dict: Dict[Tuple[str, str], float],text_length: int = 100,seed_phrase: Optional[str] = None) -> str:
        """Generates text by predicting subsequent words using bigram frequencies."""
        
        generated_words = []
        
        # Initialize with seed phrase or random starting word
        if seed_phrase:
            seed_words = seed_phrase.split()
            generated_words.extend(seed_words)
            current_word = seed_words[-1] if seed_words else None
        else:
            # Pick a random first word from available bigrams
            available_first_words = list(set(bigram[0] for bigram in frequency_dict.keys()))
            current_word = random.choice(available_first_words) if available_first_words else None
            if current_word:
                generated_words.append(current_word)
        
        if not current_word:
            return ""
        
        with torch.no_grad():
            for _ in range(text_length - len(generated_words)):
                # Find all bigrams that start with current_word
                possible_next_words = []
                probabilities = []
                
                for (first_word, second_word), freq in frequency_dict.items():
                    if first_word == current_word:
                        possible_next_words.append(second_word)
                        probabilities.append(freq)
                
                if not possible_next_words:
                    # No continuation found, break or pick random word
                    break
                
                # Convert to torch tensor for sampling
                prob_tensor = torch.tensor(probabilities, dtype=torch.float32)
                
                # Normalize probabilities
                prob_tensor = prob_tensor / prob_tensor.sum()
                
                # Sample next word based on probabilities
                next_word_idx = torch.multinomial(prob_tensor, 1).item()
                next_word = possible_next_words[next_word_idx]
                
                generated_words.append(next_word)
                current_word = next_word  # Update current word for next iteration
        
        return ' '.join(generated_words)


def core_text_generation_flow():
    print("VERBOSE: Starting core_text_generation_flow.")
    if TENSORFLOW_AVAILABLE:
        # tf.random.set_seed(42) 
        # np.random.seed(42)     
        print("VERBOSE: TensorFlow available.")
        pass

    predictor = FrequencyPredictor() # Init print is inside class

    print("VERBOSE: Defining custom feature operations...")


    custom_feature_operations: List[Optional[Callable[[np.ndarray], np.ndarray]]] = [
        # Original functions
        lambda x: np.log1p(x),
        lambda x: np.log1p(x),
        lambda x: np.log1p(x),
        lambda x: np.log1p(x),
        lambda x: np.square(x),
        lambda x: np.square(x),
        None, 
        lambda x: np.sqrt(np.maximum(0, x)),
        lambda x: np.log1p(x),
        lambda x: np.log1p(x),
        None, 
        None, 
        None, 
        None, 
        lambda x: x * 2.0, 
        None,
        
        
        # Additional math functions
        # Basic arithmetic
        lambda x: x + 1.0,
        lambda x: x - 1.0,
        lambda x: x * 3.0,
        lambda x: x / 2.0,
        lambda x: x ** 3,
        lambda x: x ** 0.5,
        lambda x: x ** (1/3),
        lambda x: 1 / np.maximum(x, 1e-8),  # reciprocal with safety
        
        # Exponential and logarithmic
        lambda x: np.exp(x),
        lambda x: np.exp2(x),
        lambda x: np.expm1(x),
        lambda x: np.log(np.maximum(x, 1e-8)),
        lambda x: np.log2(np.maximum(x, 1e-8)),
        lambda x: np.log10(np.maximum(x, 1e-8)),
        
        # Trigonometric
        lambda x: np.sin(x),
        lambda x: np.cos(x),
        lambda x: np.tan(x),
        lambda x: np.arcsin(np.clip(x, -1, 1)),
        lambda x: np.arccos(np.clip(x, -1, 1)),
        lambda x: np.arctan(x),
        
        # Hyperbolic
        lambda x: np.sinh(x),
        lambda x: np.cosh(x),
        lambda x: np.tanh(x),
        lambda x: np.arcsinh(x),
        lambda x: np.arccosh(np.maximum(x, 1)),
        lambda x: np.arctanh(np.clip(x, -0.99, 0.99)),
        
        # Rounding and ceiling/floor
        lambda x: np.round(x),
        lambda x: np.floor(x),
        lambda x: np.ceil(x),
        lambda x: np.trunc(x),
        
        # Sign and absolute
        lambda x: np.abs(x),
        lambda x: np.sign(x),
        lambda x: np.positive(x),
        lambda x: np.negative(x),
        
        # Power and roots
        lambda x: np.cbrt(x),  # cube root
        lambda x: np.power(x, 4),
        lambda x: np.power(x, 0.25),  # 4th root
        lambda x: np.power(x, 1.5),
        
        # Special functions
        lambda x: np.maximum(x, 0),  # ReLU
        lambda x: np.minimum(x, 0),  # negative part
        lambda x: np.maximum(x, 1),  # max with 1
        lambda x: np.minimum(x, 1),  # min with 1
        
        # Statistical transforms
        lambda x: (x - np.mean(x)) / (np.std(x) + 1e-8),  # standardize
        lambda x: x / (np.max(np.abs(x)) + 1e-8),  # normalize by max
        lambda x: np.clip(x, 0, 1),  # clip to [0,1]
        lambda x: np.clip(x, -1, 1),  # clip to [-1,1]
        
        # Complex transformations
        lambda x: x / (1 + np.abs(x)),  # soft sign
        lambda x: np.where(x > 0, x, 0.01 * x),  # leaky ReLU
        lambda x: np.log(1 + np.exp(-np.abs(x))) + np.maximum(x, 0),  # softplus
        lambda x: x * (1 / (1 + np.exp(-x))),  # swish activation
        # Additional basic features
        lambda x: x,  # identity
        lambda x: np.sqrt(np.abs(x) + 1e-8),  # sqrt(abs(x))
        lambda x: 1 / (np.sqrt(np.abs(x) + 1e-8)),  # reciprocal sqrt
        lambda x: 1 / (x ** 2 + 1e-8),  # inverse square
        lambda x: np.clip(x, 0, None),  # zero out negative values
        lambda x: np.clip(x, None, 0),  # zero out positive values
        lambda x: (x > 0).astype(float),  # binary indicator: positive
        lambda x: (x < 0).astype(float),  # binary indicator: negative
        lambda x: (x == 0).astype(float),  # binary indicator: zero
        lambda x: x - np.mean(x),  # zero-mean
        lambda x: (x - np.min(x)) / (np.max(x) - np.min(x) + 1e-8),  # min-max scaling
        lambda x: np.exp(-x**2),  # Gaussian basis
        lambda x: 1 / (1 + np.exp(-x)),  # sigmoid
        lambda x: np.heaviside(x, 0.0),  # Heaviside step
    ]
    predictor.set_feature_operations(custom_feature_operations) # add more as desired

    text_content = None
    hf_dataset_config = {
        "name": "wikitext", "config": "wikitext-2-raw-v1", 
        "split": "train", "text_column": "text" 
    }
    # hf_dataset_config = None # Uncomment to use local file fallback
    print(f"VERBOSE: Hugging Face dataset config: {'Active' if hf_dataset_config else 'Inactive'}")


    if not hf_dataset_config and DATASETS_AVAILABLE:
        print("VERBOSE: Attempting to load content from Hugging Face dataset.")
        text_content = predictor.load_text_from_hf_dataset(
            dataset_name=hf_dataset_config["name"], config_name=hf_dataset_config["config"],
            split=hf_dataset_config["split"], text_column=hf_dataset_config["text_column"],
            max_total_words=1000000 # Reduced for faster verbose testing
        )
    
    if text_content is None:
        print("VERBOSE: Falling back to local file/sample text for content.")
        input_file = "test.txt" # MODIFIED: Was "test.txt", made consistent with __main__
        text_content = predictor.load_text_file(input_file)
    
    if not text_content:
        print("VERBOSE: CRITICAL: No text content loaded. Cannot proceed."); return
    print(f"VERBOSE: Text content loaded. Approx length: {len(text_content)} chars.")

    print("VERBOSE: Extracting original bigram frequencies...")
    original_bigram_frequencies = predictor.extract_bigram_frequencies(text_content)
    if not original_bigram_frequencies:
        print("VERBOSE: CRITICAL: No bigrams extracted. Cannot proceed."); return
    print(f"VERBOSE: Original bigram frequencies extracted. Count: {len(original_bigram_frequencies)}.")

    print("VERBOSE: Creating bigram frequency features...")
    predictor.create_bigram_frequency_features()
    new_frequencies_set = None

    if not predictor.frequency_features:
        print("VERBOSE: Warning: No features created. Using original frequencies for text generation.")
        new_frequencies_set = {k: float(v) for k, v in original_bigram_frequencies.items()}
    else:
        print("VERBOSE: Features created. Proceeding with model training/prediction.")
        model_to_train = 'neural_network' 
        
        if model_to_train == 'neural_network' and not TENSORFLOW_AVAILABLE:
            print("VERBOSE: TensorFlow not found, cannot train neural network. Using original frequencies.")
            new_frequencies_set = {k: float(v) for k, v in original_bigram_frequencies.items()}
        else:
            print(f"VERBOSE: Attempting to train '{model_to_train}' model.")
            predictor.train_predictor(model_type=model_to_train) 
            if predictor.predictor_model:
                print("VERBOSE: Model trained successfully. Generating new bigram frequencies...")
                generated_sets = predictor.generate_new_bigram_frequencies(num_variations=1)
                if generated_sets:
                    new_frequencies_set = generated_sets[0]
                    print("VERBOSE: New bigram frequencies generated successfully.")
                else:
                    print("VERBOSE: Warning: Could not generate new freqs after model training. Using original.")
                    new_frequencies_set = {k: float(v) for k, v in original_bigram_frequencies.items()}
            else:
                print("VERBOSE: Warning: Predictor model not trained. Using original freqs.")
                new_frequencies_set = {k: float(v) for k, v in original_bigram_frequencies.items()}

    if not new_frequencies_set:
        print("VERBOSE: CRITICAL: Failed to obtain any frequency set for generation. Cannot generate text."); return
    print(f"VERBOSE: Frequency set for generation is ready (size: {len(new_frequencies_set)}). Top 3: {list(new_frequencies_set.items())[:3] if new_frequencies_set else 'N/A'}")
        
    while True:
        try:
            user_seed = input("Enter a seed phrase (or press Enter for default, Ctrl+C to exit): ").strip()
        except KeyboardInterrupt:
            print("\nVERBOSE: Exiting generation loop.")
            break
        effective_seed = user_seed
        if not user_seed:
            if predictor.unigram_counts:
                temp_transitions = defaultdict(list)
                for (w1, w2), count in new_frequencies_set.items():
                    if count > 0: temp_transitions[w1].append((w2, count))
                potential_seeds = [w for w,c in predictor.unigram_counts.most_common(25) if w in temp_transitions and temp_transitions[w]]
                if potential_seeds: effective_seed = random.choice(potential_seeds)
                else: effective_seed = "the" 
            else: effective_seed = "the"
            print(f"VERBOSE: No user seed. (Using default/random seed: '{effective_seed}')")
        
        text_len = 150 
        print(f"VERBOSE: Generating text with seed '{effective_seed}' and length {text_len}.")
        generated_text_new = predictor.expand_text_from_bigrams(new_frequencies_set, text_length=text_len, seed_phrase=effective_seed)
        
        output_prefix = ""
        # Use effective_seed for processing, user_seed for display if it was originally empty
        display_seed_for_prefix = user_seed if user_seed else effective_seed

        if display_seed_for_prefix:
            processed_seed_words = predictor.preprocess_text(display_seed_for_prefix) 
            processed_seed_phrase = " ".join(processed_seed_words)
            if generated_text_new.lower().startswith(processed_seed_phrase.lower()) and processed_seed_phrase:
                output_prefix = f"Continuing seed '{display_seed_for_prefix}':\n"
            elif processed_seed_phrase:
                output_prefix = f"Seed '{display_seed_for_prefix}' (new sequence shown below, may not directly continue seed):\n"
        
        print(f"\n{output_prefix}{generated_text_new}")
        print("-" * 30)


if __name__ == "__main__":
    print("VERBOSE: Script execution started (__main__).")
    if not DATASETS_AVAILABLE:
        print("---")
        print("NOTE: Hugging Face 'datasets' library not found (run: pip install datasets).")
        print("The script will rely on 'input_text.txt' or the internal sample text if HF loading fails.")
        print("---")
    if not TENSORFLOW_AVAILABLE:
        print("---")
        print("NOTE: TensorFlow library not found (run: pip install tensorflow).")
        print("The neural network model will not be available. The script might not function as intended for NN prediction.")
        print("---")

    input_file_main = "input_text.txt"
    try:
        with open(input_file_main, 'r', encoding='utf-8') as f: f.read(1) 
        print(f"VERBOSE: Fallback file '{input_file_main}' found.")
    except FileNotFoundError:
        print(f"VERBOSE: Creating sample file: {input_file_main} (as a fallback data source)")
        with open(input_file_main, 'w', encoding='utf-8') as f:
            f.write(FrequencyPredictor().get_sample_text()) # get_sample_text will also print
            f.write("\nThis is additional sample text for the fallback input_text.txt. ")
            
    core_text_generation_flow()
    print("VERBOSE: Script execution finished.")