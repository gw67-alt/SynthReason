import numpy as np
# import pandas as pd # No longer used
from collections import Counter, defaultdict
import re
import random
# import matplotlib.pyplot as plt # No longer used for visualization in main flow
# from sklearn.linear_model import LinearRegression # Replaced by NN option
# from sklearn.ensemble import RandomForestRegressor # Replaced by NN option
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
# import json # No longer used for saving/loading in main flow
from typing import Dict, List, Tuple, Optional, Any, Callable

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
        self.num_base_features: int = 66
        self.feature_operations: Optional[List[Optional[Callable[[np.ndarray], np.ndarray]]]] = None

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
                words = content.lower().split()
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
            word1, word2 = bigram
            freq = self.bigram_frequencies[bigram]
            bigram_features_vector = [
                float(freq), 
                float(len(word1)), float(len(word2)), float(len(set(word1))), float(len(set(word2))),
                float(word1.count('e')), float(word2.count('e')), float(word1.count('a')), float(word2.count('a')),
                float(self.unigram_counts.get(word1, 0)), float(self.unigram_counts.get(word2, 0)),
                1.0 if word1.endswith('ing') else 0.0, 1.0 if word2.endswith('ing') else 0.0,
                1.0 if word1.endswith('ed') else 0.0, 1.0 if word2.endswith('ed') else 0.0,
                1.0 if word1.startswith('un') else 0.0, 1.0 if word2.startswith('un') else 0.0,
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

        X_raw = np.array([f[1:] for f in self.frequency_features]) 
        y = np.array([f[0] for f in self.frequency_features])  
        print(f"VERBOSE: Raw features X_raw shape: {X_raw.shape}, Target y shape: {y.shape}.")

        if X_raw.shape[0] <= 1 or X_raw.ndim != 2 or X_raw.shape[0] != y.shape[0] or X_raw.shape[1] != self.num_base_features:
            print(f"VERBOSE: Warning: Shape of X_raw ({X_raw.shape}) or y ({y.shape}) is unsuitable for training. Num base features: {self.num_base_features}. Skipping training.")
            self.predictor_model = None; return
        
        X_transformed = self._apply_feature_operations(X_raw) # This method now has its own verbose prints
        print(f"VERBOSE: Transformed features X_transformed shape: {X_transformed.shape}.")
        
        test_size_for_split = 0.2 if X_transformed.shape[0] >= 10 else 0.0 # adjust if too small
        if X_transformed.shape[0] < 10:
             print(f"VERBOSE: Dataset too small ({X_transformed.shape[0]} samples), using all data for training and testing.")
             X_train, X_test, y_train, y_test = (X_transformed, X_transformed, y, y)
        else:
             X_train, X_test, y_train, y_test = train_test_split(X_transformed, y, test_size=test_size_for_split, random_state=42)
        
        print(f"VERBOSE: Data split. X_train shape: {X_train.shape}, X_test shape: {X_test.shape}, y_train shape: {y_train.shape}, y_test shape: {y_test.shape}.")
        
        if X_train.shape[0] == 0:
            print("VERBOSE: X_train is empty after split. Skipping training.")
            self.predictor_model = None; return

        print("VERBOSE: Fitting StandardScaler on X_train.")
        self.scaler.fit(X_train)
        X_train_scaled = self.scaler.transform(X_train)
        print(f"VERBOSE: X_train scaled. Shape: {X_train_scaled.shape}. Scaler mean: {self.scaler.mean_[:3]}..., Scaler scale: {self.scaler.scale_[:3]}...")
        
        if model_type == 'neural_network':
            if not TENSORFLOW_AVAILABLE:
                print("VERBOSE: TensorFlow not available. Cannot train neural network model.")
                self.predictor_model = None; return
            
            print("VERBOSE: Defining Keras Sequential model.")
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
            # nn_model.summary(print_fn=lambda x: print(f"VERBOSE: NN Model Summary: {x}")) # For very detailed model structure
            
            print(f"VERBOSE: Training Neural Network model on {X_train_scaled.shape[0]} samples for 100 epochs...")
            history = nn_model.fit(X_train_scaled, y_train, epochs=100, batch_size=32, verbose=0) # verbose=0 for TF's own prints
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
        print(f"VERBOSE: Original total frequency sum for scaling: {original_total_frequency_sum}")

        base_X_for_prediction = np.array([f[1:] for f in self.frequency_features])
        if base_X_for_prediction.shape[1] != self.num_base_features:
            print(f"VERBOSE: Error: Mismatch in feature dimensions ({base_X_for_prediction.shape[1]} vs {self.num_base_features}) for generation. Cannot proceed.")
            return []
        print(f"VERBOSE: Base features for prediction (X_raw equivalent) shape: {base_X_for_prediction.shape}")

        for variation in range(num_variations):
            print(f"VERBOSE: Generating variation {variation + 1}/{num_variations}.")
            X_noised = base_X_for_prediction.astype(float).copy()
            noise_factor = 0.1 + (variation * 0.02)
            # print(f"VERBOSE: Applying noise with factor {noise_factor:.3f}.")
            for j in range(X_noised.shape[1]):
                noise = np.random.normal(0, noise_factor * np.abs(X_noised[:, j] + 0.01))
                X_noised[:, j] = np.maximum(0, X_noised[:, j] + noise)
            # print(f"VERBOSE: Noised features X_noised shape: {X_noised.shape}. Sample (1st row, 1st 3 vals): {X_noised[0, :3] if X_noised.size > 0 else 'N/A'}")
            
            X_transformed_noised = self._apply_feature_operations(X_noised)
            # print(f"VERBOSE: Transformed noised features X_transformed_noised shape: {X_transformed_noised.shape}. Sample: {X_transformed_noised[0, :3] if X_transformed_noised.size > 0 else 'N/A'}")
            
            if X_transformed_noised.shape[0] == 0:
                print("VERBOSE: Transformed noised features are empty. Skipping this variation.")
                continue

            X_scaled = self.scaler.transform(X_transformed_noised)
            # print(f"VERBOSE: Scaled features for prediction shape: {X_scaled.shape}. Sample: {X_scaled[0, :3] if X_scaled.size > 0 else 'N/A'}")
            
            print("VERBOSE: Predicting new counts with the model...")
            predicted_new_counts = self.predictor_model.predict(X_scaled)
            
            if isinstance(predicted_new_counts, tf.Tensor):
                predicted_new_counts = predicted_new_counts.numpy()
            predicted_new_counts = predicted_new_counts.flatten()
            # print(f"VERBOSE: Raw predicted counts (first 5): {predicted_new_counts[:5] if predicted_new_counts.size > 0 else 'N/A'}")
            
            predicted_new_counts = np.maximum(predicted_new_counts, 0.01) 
            # print(f"VERBOSE: Predicted counts after np.maximum(0.01) (first 5): {predicted_new_counts[:5] if predicted_new_counts.size > 0 else 'N/A'}")
            
            current_sum_predicted_counts = np.sum(predicted_new_counts)
            if current_sum_predicted_counts == 0:
                if len(predicted_new_counts) > 0:
                    predicted_new_counts = np.full(len(predicted_new_counts), 0.01)
                    print("VERBOSE: Sum of predicted counts was 0, filled with 0.01.")
                else:
                    new_frequency_sets.append({});
                    print("VERBOSE: Predicted counts array is empty for this variation.")
                    continue 
            current_sum_predicted_counts = np.sum(predicted_new_counts) # Recalculate
            
            scaled_predicted_counts = predicted_new_counts
            if original_total_frequency_sum > 0 and current_sum_predicted_counts > 0:
                scaled_predicted_counts = (predicted_new_counts / current_sum_predicted_counts) * original_total_frequency_sum
                # print(f"VERBOSE: Scaled predicted counts to match original sum. Current sum: {current_sum_predicted_counts:.2f}, Target sum: {original_total_frequency_sum:.2f}. First 5 scaled: {scaled_predicted_counts[:5] if scaled_predicted_counts.size > 0 else 'N/A'}")
            
            new_freq_dict: Dict[Tuple[str, str], float] = {
                bigram: float(scaled_predicted_counts[i]) for i, bigram in enumerate(self.sorted_bigrams) if i < len(scaled_predicted_counts)
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
            if count > 0: transitions[w1].append((w2, count))
        
        if not transitions:
            print("VERBOSE: Error: Frequency data has no usable transitions.")
            return "Error: Frequency data has no usable transitions."
        # print(f"VERBOSE: Built transitions model with {len(transitions)} starting words. E.g., transitions for 'the': {transitions.get('the', [])[:3]}")


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
            else: # Should be caught by earlier returns
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
            # print(f"VERBOSE: From '{current_word}', next options: {list(zip(possible_next_words, weights))[:3]}")
            next_word = random.choices(possible_next_words, weights=weights, k=1)[0]
            generated_text_list.append(next_word)
            current_word = next_word
            # if i % 25 == 0: print(f"VERBOSE: Generated {len(generated_text_list)}/{text_length} words...")


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
            max_total_words=10000 # Reduced for faster verbose testing
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