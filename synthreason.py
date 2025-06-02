import numpy as np
from collections import Counter, defaultdict
import re
import random
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from typing import Dict, List, Tuple, Optional, Any, Callable

# Import torch and define device
import torch
import torch.nn as nn
import torch.optim as optim

KB_LENGTH = 10000 # Keep as -1 to load all available, or set to a specific number (e.g., 500000) for testing large datasets

# Attempt to import datasets
try:
    from datasets import load_dataset
    DATASETS_AVAILABLE = True
except ImportError:
    DATASETS_AVAILABLE = False
    def load_dataset(*args, **kwargs):  # Placeholder
        print("VERBOSE: Warning: Hugging Face 'datasets' library not found. Cannot load from Hub.")
        raise ImportError("Hugging Face 'datasets' library is required for this feature but not found.")

# Determine if CUDA is available for PyTorch
TORCH_AVAILABLE = True
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"VERBOSE: PyTorch available. Using device: {DEVICE}")

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
        self.predictor_model: Optional[nn.Module] = None
        self.scaler = StandardScaler()
        self.sorted_bigrams: List[Tuple[str, str]] = []
        self.unigram_counts: Dict[str, int] = Counter()
        # Initialize num_base_features with a default. It will be updated dynamically.
        self.num_base_features: int = 0
        self.feature_operations: Optional[List[Optional[Callable[[np.ndarray], np.ndarray]]]] = None
        # Store indices of columns removed during training to apply same removal during prediction
        self.removed_feature_indices: Optional[np.ndarray] = None # Changed type hint to np.ndarray


    def set_feature_operations(self, operations: Optional[List[Optional[Callable[[np.ndarray], np.ndarray]]]]) -> None:
        print(f"VERBOSE: Setting feature operations. Received {len(operations) if operations else 'None'} operations.")
        # When operations are set, determine the initial expected number of base features
        if operations:
            self.num_base_features = len(operations)
            print(f"VERBOSE: Initial num_base_features set to {self.num_base_features} based on provided operations list.")
        else:
            self.num_base_features = 0 # No operations, so 0 base features from operations
        self.feature_operations = operations


    def _apply_feature_operations(self, X_data: np.ndarray) -> np.ndarray:
        # print(f"VERBOSE: Attempting to apply feature operations on data with shape {X_data.shape}.")
        if self.feature_operations is None:
            print("VERBOSE: No feature operations defined. Returning data as is (with NaN/Inf cleaning).")
            # Even if no operations, ensure the data is cleaned of NaNs/Infs
            return np.nan_to_num(X_data.astype(float),
                                 nan=0.0,
                                 posinf=np.finfo(X_data.dtype).max / 2,
                                 neginf=np.finfo(X_data.dtype).min / 2)


        # Ensure X_data is float for calculations
        X_data_float = X_data.astype(float).copy()
        X_transformed = np.zeros_like(X_data_float) # Initialize with zeros, same shape as input

        # Check if the number of columns in X_data matches the number of defined operations.
        if X_data_float.shape[1] != len(self.feature_operations):
            print(f"VERBOSE: Warning: Input data columns ({X_data_float.shape[1]}) do not match the number of feature operations ({len(self.feature_operations)}). Operations will apply only to the minimum of these counts or might miss features. This might indicate an issue with feature generation or setting feature operations.")
            num_cols_to_process = min(X_data_float.shape[1], len(self.feature_operations))
        else:
            num_cols_to_process = X_data_float.shape[1]


        for i in range(num_cols_to_process): # Iterate through columns of the input X_data
            operation = self.feature_operations[i]
            if operation is not None:
                try:
                    # Apply operation to the entire column.
                    transformed_col = operation(X_data_float[:, i])
                    X_transformed[:, i] = np.nan_to_num(transformed_col,
                                                        nan=0.0,
                                                        posinf=np.finfo(transformed_col.dtype).max / 2,
                                                        neginf=np.finfo(transformed_col.dtype).min / 2)
                except Exception as e:
                    print(f"VERBOSE: Error applying operation to feature index {i}: {e}. Feature {i} remains as original (after nan_to_num).")
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

        # If X_data_float had more columns than feature_operations, append the unprocessed (but cleaned) rest
        if X_data_float.shape[1] > num_cols_to_process:
            unprocessed_cols = X_data_float[:, num_cols_to_process:]
            unprocessed_cols_cleaned = np.nan_to_num(unprocessed_cols,
                                                     nan=0.0,
                                                     posinf=np.finfo(unprocessed_cols.dtype).max / 2,
                                                     neginf=np.finfo(unprocessed_cols.dtype).min / 2)
            X_transformed = np.hstack((X_transformed, unprocessed_cols_cleaned))
            print(f"VERBOSE: Appended {X_data_float.shape[1] - num_cols_to_process} unprocessed columns. New shape: {X_transformed.shape}")

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
            # X_raw contains the features as generated by create_bigram_frequency_features
            X_raw_initial = np.array([f[1:] for f in self.frequency_features], dtype=float)
            y = np.array([f[0] for f in self.frequency_features], dtype=float)
        except Exception as e:
            print(f"VERBOSE: Error converting features to NumPy array: {e}. Check feature creation. Skipping training.")
            self.predictor_model = None; return

        print(f"VERBOSE: Initial X_raw_initial shape: {X_raw_initial.shape}, Target y shape: {y.shape}.")

        if X_raw_initial.shape[0] == 0 or y.shape[0] == 0:
            print("VERBOSE: X_raw_initial or y is empty. Skipping training.")
            self.predictor_model = None; return

        # Apply feature operations to the initial raw features.
        X_transformed = self._apply_feature_operations(X_raw_initial)
        print(f"VERBOSE: Transformed features X_transformed shape: {X_transformed.shape}.")

        # Check for and remove constant columns *before* StandardScaler fit
        if X_transformed.shape[0] > 1: # Only relevant if more than one sample
            stds = np.std(X_transformed, axis=0)
            # Identify columns where std dev is very close to zero
            self.removed_feature_indices = np.where(stds < 1e-9)[0] # Store for prediction
            if self.removed_feature_indices.size > 0: # Corrected check
                print(f"VERBOSE: Detected {self.removed_feature_indices.size} constant feature columns (std < 1e-9). Removing them before scaling.")
                X_transformed = np.delete(X_transformed, self.removed_feature_indices, axis=1)
                # Update num_base_features to reflect the *actual* number of features the model will be trained on
                self.num_base_features = X_transformed.shape[1]
                print(f"VERBOSE: self.num_base_features adjusted to {self.num_base_features} after removing constant columns.")
            else:
                self.removed_feature_indices = np.array([], dtype=int) # Ensure it's an empty array if none were removed
        else:
            self.removed_feature_indices = np.array([], dtype=int) # No columns removed for single sample or empty data

        # Ensure X_transformed is not empty after potential column removal
        if X_transformed.shape[0] == 0 or X_transformed.shape[1] == 0:
            print("VERBOSE: X_transformed is empty or has no features after cleaning/constant column removal. Skipping training.")
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
            self.scaler = StandardScaler() # Re-initialize to ensure it fits only on current columns
            self.scaler.fit(X_train)
            X_train_scaled = self.scaler.transform(X_train)
            X_test_scaled = self.scaler.transform(X_test) # Also scale X_test for evaluation
            print(f"VERBOSE: X_train scaled. Shape: {X_train_scaled.shape}. Scaler mean (first 3): {self.scaler.mean_[:min(3, self.scaler.mean_.size)]}..., Scaler scale (first 3): {self.scaler.scale_[:min(3, self.scaler.scale_.size)]}...")
        except ValueError as e:
            print(f"VERBOSE: ValueError during StandardScaler fit/transform: {e}. This might happen if data still contains NaNs/infs or all features are constant in X_train. Skipping training.")
            self.predictor_model = None; return

        if model_type == 'neural_network':
            if not TORCH_AVAILABLE:
                print("VERBOSE: PyTorch not available. Cannot train neural network model.")
                self.predictor_model = None; return

            print("VERBOSE: Defining PyTorch Neural Network model.")
            if X_train_scaled.shape[1] == 0:
                print("VERBOSE: X_train_scaled has 0 features. Cannot define model input layer.")
                self.predictor_model = None; return

            # PyTorch Model Definition
            class NeuralNet(nn.Module):
                def __init__(self, input_size):
                    super(NeuralNet, self).__init__()
                    self.fc1 = nn.Linear(input_size, 128)
                    self.relu1 = nn.ReLU()
                    self.dropout1 = nn.Dropout(0.2)
                    self.fc2 = nn.Linear(128, 64)
                    self.relu2 = nn.ReLU()
                    self.dropout2 = nn.Dropout(0.2)
                    self.fc3 = nn.Linear(64, 32)
                    self.relu3 = nn.ReLU()
                    self.fc4 = nn.Linear(32, 1)

                def forward(self, x):
                    x = self.relu1(self.fc1(x))
                    x = self.dropout1(x)
                    x = self.relu2(self.fc2(x))
                    x = self.dropout2(x)
                    x = self.relu3(self.fc3(x))
                    x = self.fc4(x)
                    return x

            nn_model = NeuralNet(X_train_scaled.shape[1]).to(DEVICE)
            criterion = nn.MSELoss()
            optimizer = optim.Adam(nn_model.parameters(), lr=0.001)

            # Convert numpy arrays to torch tensors
            X_train_tensor = torch.from_numpy(X_train_scaled).float().to(DEVICE)
            y_train_tensor = torch.from_numpy(y_train).float().unsqueeze(1).to(DEVICE) # Unsqueeze for (N, 1) shape

            X_test_tensor = torch.from_numpy(X_test_scaled).float().to(DEVICE)
            y_test_tensor = torch.from_numpy(y_test).float().unsqueeze(1).to(DEVICE)

            print(f"VERBOSE: Training PyTorch Neural Network model on {X_train_scaled.shape[0]} samples for 100 epochs...")
            num_epochs = 100
            batch_size = 32
            num_batches = len(X_train_tensor) // batch_size + (1 if len(X_train_tensor) % batch_size > 0 else 0)

            for epoch in range(num_epochs):
                nn_model.train() # Set model to training mode
                epoch_loss = 0.0
                for i in range(num_batches):
                    start_idx = i * batch_size
                    end_idx = min((i + 1) * batch_size, len(X_train_tensor))
                    batch_X = X_train_tensor[start_idx:end_idx]
                    batch_y = y_train_tensor[start_idx:end_idx]

                    # Forward pass
                    outputs = nn_model(batch_X)
                    loss = criterion(outputs, batch_y)

                    # Backward and optimize
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()
                    epoch_loss += loss.item()

                if (epoch + 1) % 10 == 0 or epoch == 0:
                    print(f"VERBOSE: Epoch [{epoch+1}/{num_epochs}], Loss: {epoch_loss/num_batches:.4f}")

            self.predictor_model = nn_model
            print(f"VERBOSE: PyTorch Neural Network predictor trained. Final training loss: {epoch_loss/num_batches:.4f}")

            # Optional: Evaluate on test set
            if X_test_tensor.shape[0] > 0:
                nn_model.eval() # Set model to evaluation mode
                with torch.no_grad(): # Disable gradient calculation for inference
                    test_outputs = nn_model(X_test_tensor)
                    test_loss = criterion(test_outputs, y_test_tensor).item()
                    print(f"VERBOSE: PyTorch Neural Network model test loss: {test_loss:.4f}")

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
        # Ensure removed_feature_indices is initialized to an empty array if not set during training
        if self.removed_feature_indices is None:
            print("VERBOSE: `removed_feature_indices` not set during training (likely no constant features removed). Initializing to empty array.")
            self.removed_feature_indices = np.array([], dtype=int)


        new_frequency_sets = []
        original_total_frequency_sum = sum(self.bigram_frequencies.values())
        if original_total_frequency_sum == 0 : original_total_frequency_sum = 1.0 # Avoid division by zero
        print(f"VERBOSE: Original total frequency sum for scaling: {original_total_frequency_sum}")

        try:
            # Get the base features (X_raw) from the original feature generation, excluding the target column
            base_X_raw_full = np.array([f[1:] for f in self.frequency_features], dtype=float)
        except Exception as e:
            print(f"VERBOSE: Error preparing base_X_raw_full from self.frequency_features: {e}. Cannot generate.")
            return []

        if base_X_raw_full.shape[0] == 0:
            print("VERBOSE: base_X_raw_full is empty. Cannot generate frequencies.")
            return []

        # Apply feature operations to this full raw set.
        X_transformed_full = self._apply_feature_operations(base_X_raw_full)

        # Apply the *same* constant column removal that was performed during training.
        if self.removed_feature_indices.size > 0: # Corrected check
            print(f"VERBOSE: Removing {self.removed_feature_indices.size} constant feature columns from prediction data based on training.")
            valid_indices_to_remove = [idx for idx in self.removed_feature_indices if idx < X_transformed_full.shape[1]]
            X_transformed_for_prediction = np.delete(X_transformed_full, valid_indices_to_remove, axis=1)
        else:
            X_transformed_for_prediction = X_transformed_full.copy() # No columns to remove

        # After removing columns, ensure the shape matches the model's input expectations.
        if X_transformed_for_prediction.shape[1] != self.num_base_features:
            print(f"VERBOSE: CRITICAL Error: Mismatch in feature dimensions for prediction after removal. Expected {self.num_base_features} but got {X_transformed_for_prediction.shape[1]}. Cannot proceed.")
            return []
        print(f"VERBOSE: Features prepared for prediction (after operations and consistent removal) shape: {X_transformed_for_prediction.shape}")


        for variation in range(num_variations):
            print(f"VERBOSE: Generating variation {variation + 1}/{num_variations}.")
            # Add noise to the consistently preprocessed (transformed and column-removed) features.
            X_noised_for_prediction = X_transformed_for_prediction.astype(float).copy()
            noise_factor = 0.1 + (variation * 0.02)

            for j in range(X_noised_for_prediction.shape[1]):
                col_std = np.std(X_noised_for_prediction[:, j])
                if col_std == 0:
                    col_std = 0.1
                noise = np.random.normal(0, noise_factor * col_std, size=X_noised_for_prediction.shape[0])
                X_noised_for_prediction[:, j] = X_noised_for_prediction[:, j] + noise
                X_noised_for_prediction[:, j] = np.maximum(0, X_noised_for_prediction[:, j])


            try:
                X_scaled = self.scaler.transform(X_noised_for_prediction)
            except ValueError as ve:
                print(f"VERBOSE: ValueError during scaler.transform on noised data: {ve}. Skipping variation.")
                continue

            print("VERBOSE: Predicting new counts with the model...")
            # Convert to tensor, move to device, make prediction, convert back to numpy
            X_scaled_tensor = torch.from_numpy(X_scaled).float().to(DEVICE)
            self.predictor_model.eval() # Set model to evaluation mode
            with torch.no_grad(): # Disable gradient calculations
                predicted_new_counts_tensor = self.predictor_model(X_scaled_tensor)
            predicted_new_counts = predicted_new_counts_tensor.cpu().numpy().flatten()


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
                else:
                    print(f"VERBOSE: Last word of seed ('{potential_start_node}') not found as a valid start in transitions or has no continuations. Will select a new start word.")
            else:
                print("VERBOSE: Seed phrase was empty after preprocessing. Will select a new start word.")


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
                print(f"VERBOSE: Current word '{current_word}' has no further transitions. Attempting to restart sequence.")
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


def core_text_generation_flow():
    print("VERBOSE: Starting core_text_generation_flow.")
    # No direct PyTorch seed for random numbers in numpy, so keep numpy/random seed calls if needed
    # torch.manual_seed(42) # For PyTorch operations
    # np.random.seed(42)
    # random.seed(42)

    predictor = FrequencyPredictor()

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

    predictor.set_feature_operations(custom_feature_operations)

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
            max_total_words=20000 # Keep at 20k for faster testing, use KB_LENGTH for full
        )

    if text_content is None or not text_content:
        print("VERBOSE: Falling back to local file/sample text for content.")
        input_file = "test.txt"
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
    predictor.create_bigram_frequency_features()

    new_frequencies_set = None

    if not predictor.frequency_features:
        print("VERBOSE: Warning: No features created. Using original frequencies for text generation.")
        new_frequencies_set = {k: float(v) for k, v in original_bigram_frequencies.items()}
    else:
        print("VERBOSE: Features created. Proceeding with model training/prediction.")
        model_to_train = 'neural_network'

        if model_to_train == 'neural_network' and not TORCH_AVAILABLE: # Changed from TENSORFLOW_AVAILABLE
            print("VERBOSE: PyTorch not found, cannot train neural network. Using original frequencies.")
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
    print(f"VERBOSE: Frequency set for generation is ready (size: {len(new_frequencies_set)}). Top 3: {list(new_frequencies_set.items())[:min(3, len(new_frequencies_set))]} (sample).")

    # Final check for NaNs/Infs in the generated frequencies before text expansion
    for k, v in new_frequencies_set.items():
        if np.isnan(v) or np.isinf(v):
            print(f"VERBOSE: CRITICAL: Detected NaN/Inf in final frequency set for key {k} (value: {v}). This indicates a persistent problem. Falling back to original frequencies.")
            new_frequencies_set = {k_orig: float(v_orig) for k_orig, v_orig in original_bigram_frequencies.items()}
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
    if not TORCH_AVAILABLE: # Changed from TENSORFLOW_AVAILABLE
        print("---")
        print("NOTE: PyTorch library not found (run: pip install torch).")
        print("The neural network model will not be available. The script might not function as intended for NN prediction.")
        print("---")

    input_file_main = "input_text.txt"
    try:
        with open(input_file_main, 'r', encoding='utf-8') as f: f.read(1)
        print(f"VERBOSE: Fallback file '{input_file_main}' found.")
    except FileNotFoundError:
        print(f"VERBOSE: Creating sample file: {input_file_main} (as a fallback data source)")
        with open(input_file_main, 'w', encoding='utf-8') as f:
            sample_text = FrequencyPredictor().get_sample_text()
            f.write(sample_text * 5)
            f.write("\nThis is additional sample text for the fallback input_text.txt. It should provide more data for training the predictor.")
            f.write(" This text is repeated to create a larger corpus for frequency analysis and model training.")
            f.write(" More words means potentially better patterns for the neural network to learn.")

    core_text_generation_flow()
    print("VERBOSE: Script execution finished.")
