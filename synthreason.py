import numpy as np
from collections import Counter, defaultdict
import re
import random
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from typing import Dict, List, Tuple, Optional, Any, Callable

KB_LENGTH = -1 # Keep as -1 to load all available, or set to a specific number (e.g., 500000) for testing large datasets

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
        # Adjusted num_base_features to match the number of active functions defined in set_feature_operations.
        # This should be updated if you change the length of custom_feature_operations list in core_text_generation_flow.
        self.num_base_features: int = 77 # Adjusted based on the provided list after removing problematic ones
        self.feature_operations: Optional[List[Optional[Callable[[np.ndarray], np.ndarray]]]] = None

    def set_feature_operations(self, operations: Optional[List[Optional[Callable[[np.ndarray], np.ndarray]]]]) -> None:
        print(f"VERBOSE: Setting feature operations. Received {len(operations) if operations else 'None'} operations.")
        if operations and len(operations) != self.num_base_features:
            # Dynamically adjust num_base_features if operations length changes.
            # Or raise error if strict match is required.
            print(f"VERBOSE: Warning: Number of operations ({len(operations)}) does not match expected num_base_features ({self.num_base_features}). Adjusting self.num_base_features.")
            self.num_base_features = len(operations)
        self.feature_operations = operations
        if operations:
            print(f"VERBOSE: {sum(1 for op in operations if op is not None)} active feature operations set for {self.num_base_features} base features.")

    def _apply_feature_operations(self, X_data: np.ndarray) -> np.ndarray:
        # print(f"VERBOSE: Attempting to apply feature operations on data with shape {X_data.shape}.")
        if self.feature_operations is None: # Changed from 'not self.feature_operations' to handle an empty list as well.
            print("VERBOSE: No feature operations defined. Returning data as is.")
            return X_data

        # Ensure X_data is float for calculations
        X_data_float = X_data.astype(float).copy()

        # Check if shape aligns with expected number of features *before* operations
        # If X_data.shape[1] is different from len(self.feature_operations), then applying element-wise
        # operations using indices from self.feature_operations will be problematic.
        # It's assumed X_data comes in with `self.num_base_features` columns *before* these ops.
        if X_data_float.ndim != 2 or X_data_float.shape[1] != self.num_base_features:
            print(f"VERBOSE: Warning: X_data shape ({X_data_float.shape}) does not match the number of expected base features ({self.num_base_features}) for operations. Operations might not apply correctly or will be skipped.")
            # We still try to apply if the first dimension is consistent.
            # The intention here is to apply operations to the *initial* raw features.
            # If the number of columns of X_data_float (raw features) does not match the number of operations,
            # this section will likely cause an index error or incorrect behavior.
            # Let's assume X_data_float already has the right number of columns.

        X_transformed = np.zeros_like(X_data_float) # Initialize with zeros, same shape as input
        # print(f"VERBOSE: Applying feature operations. Initial X_data shape: {X_data_float.shape}")

        for i in range(X_data_float.shape[1]): # Iterate through columns of the input X_data
            if i < len(self.feature_operations) and self.feature_operations[i] is not None:
                operation = self.feature_operations[i]
                try:
                    # Apply operation to the entire column.
                    # np.nan_to_num is applied here to handle potential NaNs/Infs *immediately after* the operation,
                    # before it can propagate or cause issues in subsequent operations or the scaler.
                    transformed_col = operation(X_data_float[:, i])
                    X_transformed[:, i] = np.nan_to_num(transformed_col,
                                                        nan=0.0,
                                                        posinf=np.finfo(transformed_col.dtype).max / 2,
                                                        neginf=np.finfo(transformed_col.dtype).min / 2)
                except Exception as e:
                    print(f"VERBOSE: Error applying operation to feature index {i}: {e}. Feature {i} remains as original (after nan_to_num).")
                    # If an error occurs, use the original data for that column, but still clean it.
                    X_transformed[:, i] = np.nan_to_num(X_data_float[:, i],
                                                        nan=0.0,
                                                        posinf=np.finfo(X_data_float.dtype).max / 2,
                                                        neginf=np.finfo(X_data_float.dtype).min / 2)
            else:
                # If no operation defined for this index, use the original data, but clean it.
                X_transformed[:, i] = np.nan_to_num(X_data_float[:, i],
                                                    nan=0.0,
                                                    posinf=np.finfo(X_data_float.dtype).max / 2,
                                                    neginf=np.finfo(X_data_float.dtype).min / 2)

        # print(f"VERBOSE: Finished applying feature operations. Transformed X_data shape: {X_transformed.shape}")
        return X_transformed


    def load_text_from_hf_dataset(self,
                                  dataset_name: str,
                                  config_name: Optional[str] = None,
                                  split: str = 'train',
                                  text_column: str = 'text',
                                  max_total_words: int = KB_LENGTH) -> Optional[str]:
        print(f"VERBOSE: Attempting to load text from Hugging Face dataset: {dataset_name} (Config: {config_name}, Split: {split}, Column: {text_column}).")
        if not DATASETS_AVAILABLE:
            print("VERBOSE: Hugging Face 'datasets' library not available. Cannot load from Hub.")
            return None
        try:
            # Use streaming=True for potentially very large datasets, max_total_words handles truncation.
            dataset = load_dataset(dataset_name, name=config_name, split=split, streaming=True, trust_remote_code=True)
            print(f"VERBOSE: Accessed HF dataset stream: {dataset_name}, config: {config_name}, split: {split}")

            all_collected_words = []
            docs_processed = 0
            for doc in dataset:
                if max_total_words != -1 and len(all_collected_words) >= max_total_words:
                    print(f"VERBOSE: Reached max_total_words ({max_total_words}). Stopping collection.")
                    break
                text_content = doc.get(text_column)
                if isinstance(text_content, str):
                    current_doc_words = self.preprocess_text(text_content) # Use internal preprocessor
                    words_to_add_count = max_total_words - len(all_collected_words) if max_total_words != -1 else len(current_doc_words)
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
            words = self.preprocess_text(content) # Use internal preprocessor
            if KB_LENGTH != -1:
                words = words[:KB_LENGTH]
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
        # print(f"VERBOSE: Preprocessing text. Initial length: {len(text)} characters.")
        # Remove non-alphabetic characters and split by spaces, then filter empty strings
        # Keep dots and commas for now if needed, but the prompt's preprocess removed them.
        # Let's clean as per original intent (only alpha and space)
        #text_cleaned = re.sub(r'[^a-z\s]', '', text) # only lowercase letters and spaces
        words = text.split()
        valid_words = [word for word in words if word] # remove any empty strings that might result from multiple spaces
        # print(f"VERBOSE: Preprocessing complete. Number of words: {len(valid_words)} (after cleaning and removing empty strings).")
        return valid_words

    def extract_bigram_frequencies(self, text: str) -> Dict[Tuple[str, str], int]:
        print("VERBOSE: Starting bigram frequency extraction.")
        words = self.preprocess_text(text)
        if len(words) < 2:
            print("VERBOSE: Not enough words to form bigrams. Extracted 0 bigrams.")
            self.bigram_frequencies = {}; self.sorted_bigrams = []
            self.unigram_counts = Counter(words); return {}

        self.unigram_counts = Counter(words)
        print(f"VERBOSE: Unigram counts calculated. Total unique unigrams: {len(self.unigram_counts)}, e.g., {list(self.unigram_counts.items())[:3]}")
        bigrams = [(words[i], words[i+1]) for i in range(len(words)-1)]
        self.bigram_frequencies = dict(Counter(bigrams))
        print(f"VERBOSE: Extracted {len(self.bigram_frequencies)} unique bigrams. Total bigram occurrences: {len(bigrams)}.")

        # Sort bigrams by frequency (descending), then alphabetically for tie-breaking
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
            # The features here are derived from the *lengths* of the words and the *frequency* of the bigram.
            # Make sure these values are numeric and sensible.
            first_word_len = len(bigram[0])
            second_word_len = len(bigram[1])
            freq = self.bigram_frequencies[bigram]

            # Assign x and y to distinct variables for clarity in feature definitions
            # x_val represents the second word's length, y_val represents the first word's length.
            # freq_val represents the bigram frequency itself.
            x_val = float(second_word_len)
            y_val = float(first_word_len)
            freq_val = float(freq) # This will be the target variable

            # It's important that each feature function receives a single scalar (x_val or y_val)
            # and returns a single scalar value. Statistical functions (mean, std, min, max)
            # applied to a single scalar will be problematic.
            # The problematic features from the original code (statistical operations on a single scalar)
            # have been removed or replaced with identity transforms.

            bigram_features_vector = [
                freq_val, # This is the target variable (freq), must be the first element
                # Features based on word lengths (x_val = len(second_word), y_val = len(first_word))

                # Original functions (some adjusted for scalar input or removed if problematic)
                np.log1p(y_val), # log(1 + len(first_word))
                np.square(y_val), # len(first_word)^2
                np.sqrt(np.maximum(0, x_val)), # sqrt(max(0, len(second_word)))
                x_val * 2.0, # 2 * len(second_word)

                # Additional math functions (all are now correctly applied to single scalars)
                # Basic arithmetic
                x_val + 1.0,
                x_val - 1.0,
                x_val * 3.0,
                x_val / (2.0 + 1e-8), # Add epsilon to denominator
                x_val ** 3,
                x_val ** 0.5, # sqrt
                x_val ** (1/3), # cbrt
                1 / np.maximum(x_val, 1e-8), # reciprocal with safety

                # Exponential and logarithmic
                np.exp(y_val),
                np.exp2(y_val),
                np.expm1(y_val),
                np.log(np.maximum(x_val, 1e-8)),
                np.log2(np.maximum(x_val, 1e-8)),
                np.log10(np.maximum(x_val, 1e-8)),

                # Trigonometric (input in radians, word lengths are integers)
                np.sin(y_val),
                np.cos(y_val),
                np.tan(y_val),
                np.arcsin(np.clip(x_val / (np.max([x_val, 1]) + 1e-8), -1, 1)), # scale x_val to [-1,1] for arcsin
                np.arccos(np.clip(x_val / (np.max([x_val, 1]) + 1e-8), -1, 1)), # scale x_val to [-1,1] for arccos
                np.arctan(y_val),

                # Hyperbolic
                np.sinh(y_val),
                np.cosh(y_val),
                np.tanh(y_val),
                np.arcsinh(y_val),
                np.arccosh(np.maximum(x_val, 1.0)), # Input must be >= 1
                np.arctanh(np.clip(x_val / (np.max([x_val, 1]) + 1e-8), -0.99, 0.99)), # Clip for arctanh domain

                # Rounding and ceiling/floor
                np.round(y_val),
                np.floor(y_val),
                np.ceil(y_val),
                np.trunc(x_val),

                # Sign and absolute
                np.abs(x_val),
                np.sign(x_val),
                np.positive(x_val), # Identity for positive numbers
                np.negative(x_val),

                # Power and roots
                np.cbrt(x_val), # cube root
                np.power(x_val, 4),
                np.power(x_val, 0.25), # 4th root
                np.power(x_val, 1.5),

                # Special functions
                np.maximum(x_val, 0), # ReLU
                np.minimum(x_val, 0), # negative part (will be 0 for word lengths)
                np.maximum(x_val, 1), # max with 1
                np.minimum(x_val, 1), # min with 1

                # Statistical transforms (these were the problematic ones, replaced with simpler, scalar-safe versions)
                x_val, # Identity
                x_val, # Identity
                np.clip(x_val, 0, 1), # clip to [0,1]
                np.clip(x_val, -1, 1), # clip to [-1,1]

                # Complex transformations
                x_val / (1 + np.abs(x_val)), # soft sign
                np.where(x_val > 0, x_val, 0.01 * x_val), # leaky ReLU (on scalar)
                np.log(1 + np.exp(-np.abs(x_val))) + np.maximum(x_val, 0), # softplus
                x_val * (1 / (1 + np.exp(-x_val))), # swish activation

                # Additional basic features
                x_val, # identity (redundant but kept for feature count)
                np.sqrt(np.abs(x_val) + 1e-8), # sqrt(abs(x))
                1 / (np.sqrt(np.abs(x_val) + 1e-8)), # reciprocal sqrt
                1 / (x_val ** 2 + 1e-8), # inverse square
                np.clip(x_val, 0, None), # zero out negative values (identity for word lengths)
                np.clip(x_val, None, 0), # zero out positive values (will be 0 for word lengths)

                # Binary indicators (must return float)
                float(x_val > 0), # binary indicator: positive
                float(x_val < 0), # binary indicator: negative (will be 0 for word lengths)
                float(x_val == 0), # binary indicator: zero

                x_val, # identity (to preserve count from original lambda list, but this one is correct for scalar x)
                x_val, # identity
                np.exp(-x_val**2), # Gaussian basis
                1 / (1 + np.exp(-x_val)), # sigmoid
                np.heaviside(x_val, 0.0), # Heaviside step
            ]
            features.append(bigram_features_vector)

        self.frequency_features = features
        if features:
            # Re-confirm num_base_features based on actual generated feature vector length minus 1 (for target)
            self.num_base_features = len(features[0]) - 1
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

        if X_raw.shape[0] == 0 or y.shape[0] == 0:
            print("VERBOSE: X_raw or y is empty. Skipping training.")
            self.predictor_model = None; return

        # Apply feature operations (this step includes `np.nan_to_num` internally now)
        X_transformed = self._apply_feature_operations(X_raw)
        print(f"VERBOSE: Transformed features X_transformed shape: {X_transformed.shape}.")

        # Check for and remove constant columns *before* StandardScaler fit
        if X_transformed.shape[0] > 1: # Only relevant if more than one sample
            stds = np.std(X_transformed, axis=0)
            # Identify columns where std dev is very close to zero
            constant_columns = np.where(stds < 1e-9)[0] # Using a small epsilon to catch near-constant features
            if len(constant_columns) > 0:
                print(f"VERBOSE: Detected {len(constant_columns)} constant feature columns (std < 1e-9). Removing them before scaling.")
                X_transformed = np.delete(X_transformed, constant_columns, axis=1)
                # Update num_base_features to reflect removed columns for model input shape
                self.num_base_features = X_transformed.shape[1]
                print(f"VERBOSE: num_base_features adjusted to {self.num_base_features} after removing constant columns.")

        # Ensure X_transformed is not empty after potential column removal
        if X_transformed.shape[0] == 0 or X_transformed.shape[1] == 0:
            print("VERBOSE: X_transformed is empty or has no features after cleaning. Skipping training.")
            self.predictor_model = None; return

        # Split data for training (only if enough samples)
        test_size_for_split = 0.2 if X_transformed.shape[0] >= 10 else 0.0
        if X_transformed.shape[0] < 10:
            print(f"VERBOSE: Dataset too small ({X_transformed.shape[0]} samples), using all data for training (no separate test set).")
            X_train, y_train = X_transformed, y[:X_transformed.shape[0]]
            X_test, y_test = X_transformed, y[:X_transformed.shape[0]] # Using training data for test if split isn't possible
        else:
            X_train, X_test, y_train, y_test = train_test_split(X_transformed, y, test_size=test_size_for_split, random_state=42)

        print(f"VERBOSE: Data split. X_train shape: {X_train.shape}, X_test shape: {X_test.shape}, y_train shape: {y_train.shape}, y_test shape: {y_test.shape}.")

        if X_train.shape[0] == 0 or X_train.shape[1] == 0:
            print("VERBOSE: X_train is empty or has no features after split. Skipping training.")
            self.predictor_model = None; return

        print("VERBOSE: Fitting StandardScaler on X_train.")
        try:
            # Refit scaler for new feature set if columns were removed
            self.scaler = StandardScaler() # Re-initialize to ensure it fits only on current columns
            self.scaler.fit(X_train)
            X_train_scaled = self.scaler.transform(X_train)
            X_test_scaled = self.scaler.transform(X_test) # Also scale X_test for evaluation
            print(f"VERBOSE: X_train scaled. Shape: {X_train_scaled.shape}. Scaler mean (first 3): {self.scaler.mean_[:min(3, self.scaler.mean_.size)]}..., Scaler scale (first 3): {self.scaler.scale_[:min(3, self.scaler.scale_.size)]}...")
        except ValueError as e:
            print(f"VERBOSE: ValueError during StandardScaler fit/transform: {e}. This might happen if data still contains NaNs/infs or all features are constant after nan_to_num. Skipping training.")
            self.predictor_model = None; return

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

            print(f"VERBOSE: Training Neural Network model on {X_train_scaled.shape[0]} samples for 5 epochs...")
            history = nn_model.fit(X_train_scaled, y_train, epochs=100, batch_size=512, verbose=0) # Set verbose to 0 to suppress per-epoch output
            self.predictor_model = nn_model
            print(f"VERBOSE: Neural Network predictor trained. Final training loss: {history.history['loss'][-1]:.4f}")

            # Optional: Evaluate on test set
            if X_test_scaled.shape[0] > 0:
                test_loss = nn_model.evaluate(X_test_scaled, y_test, verbose=0)
                print(f"VERBOSE: Neural Network model test loss: {test_loss:.4f}")

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
            # Get the base features (X_raw) that were used to train the model, excluding the target column
            base_X_for_prediction = np.array([f[1:] for f in self.frequency_features], dtype=float)
        except Exception as e:
            print(f"VERBOSE: Error preparing base_X_for_prediction from self.frequency_features: {e}. Cannot generate.")
            return []

        # It's crucial that `base_X_for_prediction` has the same number of columns that the model was trained on
        # (after constant column removal).
        # We need to re-apply the constant column removal logic from train_predictor to this base_X_for_prediction
        # so that its shape matches what `self.scaler.transform` expects.
        if base_X_for_prediction.shape[0] > 1:
            stds = np.std(np.array([f[1:] for f in self.frequency_features], dtype=float), axis=0) # Use original raw for identifying constant columns
            constant_columns = np.where(stds < 1e-9)[0]
            if len(constant_columns) > 0:
                print(f"VERBOSE: Re-applying constant column removal for prediction base features.")
                base_X_for_prediction = np.delete(base_X_for_prediction, constant_columns, axis=1)

        if base_X_for_prediction.shape[1] != self.num_base_features:
            print(f"VERBOSE: Error: Mismatch in feature dimensions for prediction. Expected {self.num_base_features} but got {base_X_for_prediction.shape[1]}. Cannot proceed.")
            return []
        print(f"VERBOSE: Base features for prediction (X_raw equivalent) shape after pre-processing: {base_X_for_prediction.shape}")


        for variation in range(num_variations):
            print(f"VERBOSE: Generating variation {variation + 1}/{num_variations}.")
            # Add noise to the raw (unscaled) features.
            X_noised_raw = base_X_for_prediction.astype(float).copy()
            noise_factor = 0.1 + (variation * 0.02) # Increase noise with variations

            for j in range(X_noised_raw.shape[1]):
                col_std = np.std(X_noised_raw[:, j])
                if col_std == 0:
                    col_std = 0.1 # Add a small non-zero std for constant columns to allow noise
                noise = np.random.normal(0, noise_factor * col_std, size=X_noised_raw.shape[0])
                X_noised_raw[:, j] = X_noised_raw[:, j] + noise
                # Optional: Ensure non-negativity for features like lengths/counts if noise makes them negative
                # X_noised_raw[:, j] = np.maximum(0, X_noised_raw[:, j])

            # Apply feature operations to the noised raw features.
            # `_apply_feature_operations` now handles NaNs/Infs directly.
            X_transformed_noised = self._apply_feature_operations(X_noised_raw)

            if X_transformed_noised.shape[0] == 0:
                print("VERBOSE: Transformed noised features are unexpectedly empty. Skipping this variation.")
                continue

            # Scale the transformed noised features using the *fitted* scaler from training.
            try:
                X_scaled = self.scaler.transform(X_transformed_noised)
            except ValueError as ve:
                print(f"VERBOSE: ValueError during scaler.transform: {ve}. This means X_transformed_noised still has problematic values (likely extreme values or all NaNs/infs in a column). Skipping variation.")
                continue # Skip this variation if scaling fails

            print("VERBOSE: Predicting new counts with the model...")
            predicted_new_counts = self.predictor_model.predict(X_scaled)

            if isinstance(predicted_new_counts, tf.Tensor):
                predicted_new_counts = predicted_new_counts.numpy()
            predicted_new_counts = predicted_new_counts.flatten()

            # Ensure predictions are non-negative, as frequencies cannot be negative.
            predicted_new_counts = np.maximum(predicted_new_counts, 0.01) # Small positive floor

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

            # Scale predicted counts to match the original total frequency sum
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
            # Only add if count is positive; ensures valid probabilities
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
                # Try to use the last word of the seed as the starting point
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
                else:
                    print(f"VERBOSE: Last word of seed ('{potential_start_node}') not found as a valid start in transitions or has no continuations. Will select a new start word.")
            else:
                print("VERBOSE: Seed phrase was empty after preprocessing. Will select a new start word.")


        if not start_word_selected_from_seed:
            print("VERBOSE: Selecting a starting word (seed not used or invalid).")
            # Select a starting word based on unigram counts that also have outgoing transitions
            valid_starting_unigrams = {w:c for w,c in self.unigram_counts.items() if w in transitions and transitions[w]}
            if valid_starting_unigrams:
                # Prioritize more common words that can actually start a sequence
                sorted_starters = sorted(valid_starting_unigrams.items(), key=lambda item: item[1], reverse=True)
                starters = [item[0] for item in sorted_starters]
                weights = [item[1] for item in sorted_starters]
                # Use random.choices for weighted selection
                current_word = random.choices(starters, weights=weights, k=1)[0]
                print(f"VERBOSE: Selected start word '{current_word}' based on weighted unigram counts.")
            elif any(transitions.values()): # Fallback to any word that has transitions
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
                num_words_to_generate = text_length - 1 # One word already added
            else: # Should be caught by earlier returns
                print("VERBOSE: Error: Failed to select a starting word.")
                return "Error: Failed to select a starting word."

        # Generate the rest of the text
        for i in range(max(0, num_words_to_generate)):
            if not current_word or current_word not in transitions or not transitions[current_word]:
                # If current word has no next transitions, try to restart the sequence.
                print(f"VERBOSE: Current word '{current_word}' has no further transitions. Attempting to restart sequence.")
                valid_restart_candidates = [w for w, trans_list in transitions.items() if trans_list]
                if not valid_restart_candidates:
                    print("VERBOSE: No valid restart candidates found. Ending generation.")
                    break # Cannot continue generating

                # Attempt to restart with a new word based on unigram frequency and ability to transition
                restart_options = {w:c for w,c in self.unigram_counts.items() if w in valid_restart_candidates}
                if restart_options:
                    sorted_restart_options = sorted(restart_options.items(), key=lambda item: item[1], reverse=True)
                    starters = [item[0] for item in sorted_restart_options]
                    weights = [item[1] for item in sorted_restart_options]
                    current_word = random.choices(starters, weights=weights, k=1)[0]
                    print(f"VERBOSE: Restarted with word '{current_word}' (weighted choice).")
                else: # Fallback to random if no unigram counts available for restart candidates
                    current_word = random.choice(valid_restart_candidates)
                    print(f"VERBOSE: Restarted with word '{current_word}' (random choice).")

                if not current_word: # Still no word, something is wrong
                    print("VERBOSE: Failed to select a restart word. Ending generation.")
                    break

            possible_next_words, weights = zip(*transitions[current_word])
            # Select the next word based on the predicted (or original) frequencies
            next_word = random.choices(possible_next_words, weights=weights, k=1)[0]
            generated_text_list.append(next_word)
            current_word = next_word
            # if i % 100 == 0: print(f"VERBOSE: Generated {len(generated_text_list)}/{text_length} words...")


        final_text = ' '.join(generated_text_list)
        print(f"VERBOSE: Text expansion complete. Generated {len(generated_text_list)} words. Preview: '{final_text[:70]}...'")
        return final_text


def core_text_generation_flow():
    print("VERBOSE: Starting core_text_generation_flow.")
    if TENSORFLOW_AVAILABLE:
        # tf.random.set_seed(42)
        # np.random.seed(42)
        print("VERBOSE: TensorFlow available.")
        pass

    predictor = FrequencyPredictor()

    print("VERBOSE: Defining custom feature operations...")

    # Define custom feature operations. The number of functions here defines `self.num_base_features`.
    # Removed statistical operations that operate on a single scalar, as they caused NaN/Inf.
    # Replaced some `None` with identity functions to maintain feature count explicitly if desired.
    custom_feature_operations: List[Optional[Callable[[np.ndarray], np.ndarray]]] = [
        # Original functions (adjusted for scalar input)
        lambda x: np.log1p(x),
        lambda x: np.log1p(x), # Duplicates are okay for exploring feature space
        lambda x: np.log1p(x),
        lambda x: np.log1p(x),
        lambda x: np.square(x),
        lambda x: np.square(x),
        lambda x: x, # Changed from None, identity
        lambda x: np.sqrt(np.maximum(0, x)),
        lambda x: np.log1p(x),
        lambda x: np.log1p(x),
        lambda x: x, # Changed from None, identity
        lambda x: x, # Changed from None, identity
        lambda x: x, # Changed from None, identity
        lambda x: x, # Changed from None, identity
        lambda x: x * 2.0,
        lambda x: x, # Changed from None, identity


        # Additional math functions
        # Basic arithmetic
        lambda x: x + 1.0,
        lambda x: x - 1.0,
        lambda x: x * 3.0,
        lambda x: x / (2.0 + 1e-8), # Safe division
        lambda x: x ** 3,
        lambda x: x ** 0.5,
        lambda x: x ** (1/3),
        lambda x: 1 / np.maximum(x, 1e-8), # reciprocal with safety

        # Exponential and logarithmic
        lambda x: np.exp(x),
        lambda x: np.exp2(x),
        lambda x: np.expm1(x),
        lambda x: np.log(np.maximum(x, 1e-8)),
        lambda x: np.log2(np.maximum(x, 1e-8)),
        lambda x: np.log10(np.maximum(x, 1e-8)),

        # Trigonometric (inputs are word lengths, results might be unexpected but valid features)
        lambda x: np.sin(x),
        lambda x: np.cos(x),
        lambda x: np.tan(x),
        lambda x: np.arcsin(np.clip(x / (np.max([x, 1]) + 1e-8), -1, 1)), # scale x for arcsin domain
        lambda x: np.arccos(np.clip(x / (np.max([x, 1]) + 1e-8), -1, 1)), # scale x for arccos domain
        lambda x: np.arctan(x),

        # Hyperbolic
        lambda x: np.sinh(x),
        lambda x: np.cosh(x),
        lambda x: np.tanh(x),
        lambda x: np.arcsinh(x),
        lambda x: np.arccosh(np.maximum(x, 1.0)), # ensure input >= 1
        lambda x: np.arctanh(np.clip(x / (np.max([x, 1]) + 1e-8), -0.99, 0.99)), # clip x for arctanh domain

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
        lambda x: np.cbrt(x),
        lambda x: np.power(x, 4),
        lambda x: np.power(x, 0.25),
        lambda x: np.power(x, 1.5),

        # Special functions
        lambda x: np.maximum(x, 0),
        lambda x: np.minimum(x, 0),
        lambda x: np.maximum(x, 1),
        lambda x: np.minimum(x, 1),

        # Statistical transforms (REMOVED problematic ones, replaced with identity/simpler)
        lambda x: x, # Placeholder for standardization (let StandardScaler handle)
        lambda x: x, # Placeholder for normalization (let StandardScaler handle)
        lambda x: np.clip(x, 0, 1),
        lambda x: np.clip(x, -1, 1),

        # Complex transformations
        lambda x: x / (1 + np.abs(x)),
        lambda x: np.where(x > 0, x, 0.01 * x),
        lambda x: np.log(1 + np.exp(-np.abs(x))) + np.maximum(x, 0),
        lambda x: x * (1 / (1 + np.exp(-x))),

        # Additional basic features (ensure float output for binary indicators)
        lambda x: x,
        lambda x: np.sqrt(np.abs(x) + 1e-8),
        lambda x: 1 / (np.sqrt(np.abs(x) + 1e-8)),
        lambda x: 1 / (x ** 2 + 1e-8),
        lambda x: np.clip(x, 0, None),
        lambda x: np.clip(x, None, 0),
        lambda x: (x > 0).astype(float),
        lambda x: (x < 0).astype(float),
        lambda x: (x == 0).astype(float),
        lambda x: x, # Placeholder for zero-mean
        lambda x: x, # Placeholder for min-max scaling
        lambda x: np.exp(-x**2),
        lambda x: 1 / (1 + np.exp(-x)),
        lambda x: np.heaviside(x, 0.0),
    ]
    predictor.set_feature_operations(custom_feature_operations)

    text_content = None
    hf_dataset_config = {
        "name": "wikitext", "config": "wikitext-2-raw-v1",
        "split": "train", "text_column": "text"
    }
    # hf_dataset_config = None # Uncomment to use local file fallback
    print(f"VERBOSE: Hugging Face dataset config: {'Active' if hf_dataset_config else 'Inactive'}")


    if not hf_dataset_config and DATASETS_AVAILABLE: # Corrected logic: if config is active AND datasets are available
        print("VERBOSE: Attempting to load content from Hugging Face dataset.")
        text_content = predictor.load_text_from_hf_dataset(
            dataset_name=hf_dataset_config["name"], config_name=hf_dataset_config["config"],
            split=hf_dataset_config["split"], text_column=hf_dataset_config["text_column"],
            max_total_words=100000 # Set a smaller value for faster testing, use KB_LENGTH for full
        )

    if text_content is None or not text_content: # Fallback if HF loading fails or returns empty
        print("VERBOSE: Falling back to local file/sample text for content.")
        input_file = "test.txt" # This is checked and created in __main__
        text_content = predictor.load_text_file(input_file)

    if not text_content:
        print("VERBOSE: CRITICAL: No text content loaded. Cannot proceed."); return
    print(f"VERBOSE: Text content loaded. Approx word count: {len(text_content.split())}.")

    print("VERBOSE: Extracting original bigram frequencies...")
    original_bigram_frequencies = predictor.extract_bigram_frequencies(text_content)
    if not original_bigram_frequencies:
        print("VERBOSE: CRITICAL: No bigrams extracted. Cannot proceed."); return
    print(f"VERBOSE: Original bigram frequencies extracted. Count: {len(original_bigram_frequencies)}.")

    print("VERBOSE: Creating bigram frequency features...")
    predictor.create_bigram_frequency_features() # This also sets self.num_base_features correctly

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
    print(f"VERBOSE: Frequency set for generation is ready (size: {len(new_frequencies_set)}). Top 3: {list(new_frequencies_set.items())[:min(3, len(new_frequencies_set))]} (sample, check for NaN).")
    # Verify no NaNs in the final frequency set before generating text
    for k, v in new_frequencies_set.items():
        if np.isnan(v) or np.isinf(v):
            print(f"VERBOSE: CRITICAL: Detected NaN/Inf in final frequency set for key {k}. This indicates a persistent problem.")
            # Fallback to original frequencies if NaNs/Infs are still present
            new_frequencies_set = {k_orig: float(v_orig) for k_orig, v_orig in original_bigram_frequencies.items()}
            print("VERBOSE: Falling back to original frequencies due to NaN/Inf in generated set.")
            break


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
                # Create a temporary transitions dict from the chosen frequency set
                for (w1, w2), count in new_frequencies_set.items():
                    if count > 0: temp_transitions[w1].append((w2, count))

                potential_seeds = [w for w,c in predictor.unigram_counts.most_common(25) if w in temp_transitions and temp_transitions[w]]
                if potential_seeds: effective_seed = random.choice(potential_seeds)
                else: effective_seed = "the"
            else: effective_seed = "the"
            print(f"VERBOSE: No user seed. (Using default/random seed: '{effective_seed}')")

        text_len = 500 # Set a reasonable default for generated text length
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

    input_file_main = "test.txt"
    try:
        with open(input_file_main, 'r', encoding='utf-8') as f: f.read(1)
        print(f"VERBOSE: Fallback file '{input_file_main}' found.")
    except FileNotFoundError:
        print(f"VERBOSE: Creating sample file: {input_file_main} (as a fallback data source)")
        with open(input_file_main, 'w', encoding='utf-8') as f:
            # Write a larger sample to the file
            sample_text = FrequencyPredictor().get_sample_text()
            f.write(sample_text * 5) # Write the sample text multiple times to make it larger
            f.write("\nThis is additional sample text for the fallback input_text.txt. It should provide more data for training the predictor.")
            f.write(" This text is repeated to create a larger corpus for frequency analysis and model training.")
            f.write(" More words means potentially better patterns for the neural network to learn.")

    core_text_generation_flow()
    print("VERBOSE: Script execution finished.")
