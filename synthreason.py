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
from typing import Dict, List, Tuple, Optional, Any

# Attempt to import datasets
try:
    from datasets import load_dataset
    DATASETS_AVAILABLE = True
except ImportError:
    DATASETS_AVAILABLE = False
    def load_dataset(*args, **kwargs): # Placeholder
        print("Warning: Hugging Face 'datasets' library not found. Cannot load from Hub.")
        raise ImportError("Hugging Face 'datasets' library is required for this feature but not found.")

# Attempt to import TensorFlow
try:
    import tensorflow as tf
    TENSORFLOW_AVAILABLE = True
except ImportError:
    TENSORFLOW_AVAILABLE = False
    # print("Warning: TensorFlow library not found. Neural network model will not be available.")

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
        self.bigram_frequencies: Dict[Tuple[str, str], int] = {}
        self.frequency_features: List[List[float]] = []
        self.predictor_model: Optional[Any] = None
        self.scaler = StandardScaler()
        self.sorted_bigrams: List[Tuple[str, str]] = []
        self.unigram_counts: Dict[str, int] = Counter()

    def load_text_from_hf_dataset(self,
                                  dataset_name: str,
                                  config_name: Optional[str] = None,
                                  split: str = 'train',
                                  text_column: str = 'text',
                                  max_total_words: int = 2000000) -> Optional[str]:
        if not DATASETS_AVAILABLE:
            print("Hugging Face 'datasets' library not available.")
            return None
        try:
            # For most standard datasets from Hugging Face, trust_remote_code is not needed.
            dataset = load_dataset(dataset_name, name=config_name, split=split, streaming=True)
            print(f"Accessing Hugging Face dataset: {dataset_name}, config: {config_name}, split: {split}")

            all_collected_words = []
            for doc in dataset:
                if len(all_collected_words) >= max_total_words: break
                text_content = doc.get(text_column)
                if isinstance(text_content, str):
                    current_doc_words = text_content.lower().split()
                    words_to_add_count = max_total_words - len(all_collected_words)
                    all_collected_words.extend(current_doc_words[:words_to_add_count])

            if not all_collected_words: return ""
            return ' '.join(all_collected_words)
        except Exception as e:
            print(f"Error loading dataset '{dataset_name}': {e}")
            return None

    def load_text_file(self, file_path: str) -> str:
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                return ' '.join(f.read().lower().split()[:20000])
        except FileNotFoundError:
            print(f"File {file_path} not found. Using internal sample text.")
            return self.get_sample_text()

    def get_sample_text(self) -> str:
        return """
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

    def preprocess_text(self, text: str) -> List[str]:
        #text_cleaned = re.sub(r'[^a-zA-Z\s]', '', text) # Remove punctuation
        words = text.split()
        return [word for word in words if word]

    def extract_bigram_frequencies(self, text: str) -> Dict[Tuple[str, str], int]:
        words = self.preprocess_text(text)
        if len(words) < 2:
            self.bigram_frequencies = {}; self.sorted_bigrams = []
            self.unigram_counts = Counter(words); return {}

        self.unigram_counts = Counter(words)
        bigrams = [(words[i], words[i+1]) for i in range(len(words)-1)]
        self.bigram_frequencies = dict(Counter(bigrams))

        self.sorted_bigrams = [
            item[0] for item in sorted(self.bigram_frequencies.items(), key=lambda x: (x[1], x[0][0], x[0][1]), reverse=True)
        ]
        return self.bigram_frequencies

    def create_bigram_frequency_features(self) -> List[List[float]]:
        features = []
        if not self.sorted_bigrams: self.frequency_features = []; return []

        for bigram in self.sorted_bigrams:
            word1, word2 = bigram
            freq = self.bigram_frequencies[bigram]
            bigram_features = [
                freq, len(word1), len(word2), len(set(word1)), len(set(word2)),
                word1.count('e'), word2.count('e'), word1.count('a'), word2.count('a'),
                self.unigram_counts.get(word1, 0), self.unigram_counts.get(word2, 0),
                1 if word1.endswith('ing') else 0, 1 if word2.endswith('ing') else 0,
                1 if word1.endswith('ed') else 0, 1 if word2.endswith('ed') else 0,
                1 if word1.startswith('un') else 0, 1 if word2.startswith('un') else 0,
            ]
            features.append(bigram_features)
        self.frequency_features = features
        return features

    def train_predictor(self, model_type: str = 'neural_network') -> None:
        if not self.frequency_features: self.predictor_model = None; return

        X = np.array([f[1:] for f in self.frequency_features])
        y = np.array([f[0] for f in self.frequency_features])

        if X.shape[0] <= 1 or X.ndim != 2 or X.shape[0] != y.shape[0]:
            self.predictor_model = None; return

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42) if X.shape[0] >= 10 else (X, X, y, y)

        if X_train.shape[0] == 0: self.predictor_model = None; return

        self.scaler.fit(X_train)
        X_train_scaled = self.scaler.transform(X_train)

        if model_type == 'neural_network':
            if not TENSORFLOW_AVAILABLE:
                print("TensorFlow not available. Cannot train neural network model.")
                self.predictor_model = None; return

            nn_model = tf.keras.Sequential([
                tf.keras.layers.InputLayer(input_shape=(X_train_scaled.shape[1],)),
                tf.keras.layers.Dense(256, activation='relu'),
                tf.keras.layers.Dropout(0.2),
                tf.keras.layers.Dense(128, activation='relu'),
                tf.keras.layers.Dropout(0.2),
                tf.keras.layers.Dense(64, activation='relu'),
                tf.keras.layers.Dense(1)
            ])
            nn_model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.001), loss='mean_squared_error')
            nn_model.fit(X_train_scaled, y_train, epochs=100, batch_size=32, verbose=0)
            self.predictor_model = nn_model
            print("Neural Network predictor trained.")
        else:
            print(f"Unsupported model_type: {model_type} or required library missing. No model trained.")
            self.predictor_model = None


    def generate_new_bigram_frequencies(self, num_variations: int = 1) -> List[Dict[Tuple[str, str], float]]:
        if self.predictor_model is None or not self.frequency_features or not self.sorted_bigrams: return []

        new_frequency_sets = []
        original_total_frequency_sum = sum(self.bigram_frequencies.values())

        for variation in range(num_variations):
            modified_features_for_prediction = []
            for i in range(len(self.sorted_bigrams)):
                original_pred_features = self.frequency_features[i][1:].copy()
                noise_factor = 0.1 + (variation * 0.02)
                for j in range(len(original_pred_features)):
                    if isinstance(original_pred_features[j], (int, float)):
                        noise = np.random.normal(0, noise_factor * abs(original_pred_features[j] + 0.01))
                        original_pred_features[j] = max(0, original_pred_features[j] + noise)
                modified_features_for_prediction.append(original_pred_features)

            if not modified_features_for_prediction: continue

            X_modified_np = np.array(modified_features_for_prediction)
            if X_modified_np.ndim == 1:
                 X_modified_np = X_modified_np.reshape(1, -1)

            X_modified_scaled = self.scaler.transform(X_modified_np)
            predicted_new_counts = self.predictor_model.predict(X_modified_scaled)
            if hasattr(predicted_new_counts, 'numpy'): # Check if it's a TensorFlow EagerTensor
                predicted_new_counts = predicted_new_counts.numpy()
            predicted_new_counts = predicted_new_counts.flatten()

            predicted_new_counts = np.maximum(predicted_new_counts, 0.01)

            current_sum_predicted_counts = np.sum(predicted_new_counts)
            if current_sum_predicted_counts == 0:
                if len(predicted_new_counts) > 0:
                    predicted_new_counts = np.full(len(predicted_new_counts), 0.01)
                else: new_frequency_sets.append({}); continue
            current_sum_predicted_counts = np.sum(predicted_new_counts)

            scaled_predicted_counts = predicted_new_counts
            if original_total_frequency_sum > 0 and current_sum_predicted_counts > 0:
                 scaled_predicted_counts = (predicted_new_counts / current_sum_predicted_counts) * original_total_frequency_sum

            new_freq_dict: Dict[Tuple[str, str], float] = {
                bigram: float(scaled_predicted_counts[i]) for i, bigram in enumerate(self.sorted_bigrams) if i < len(scaled_predicted_counts)
            }
            new_frequency_sets.append(new_freq_dict)
        return new_frequency_sets

    def expand_text_from_bigrams(self,
                                 frequency_dict: Dict[Tuple[str, str], float],
                                 text_length: int = 100,
                                 seed_phrase: Optional[str] = None) -> str:
        if not frequency_dict: return "Error: No frequency data provided."

        transitions = defaultdict(list)
        for (w1, w2), count in frequency_dict.items():
            if count > 0: transitions[w1].append((w2, count))
        if not transitions: return "Error: Frequency data has no usable transitions."

        generated_text_list = []
        current_word: Optional[str] = None
        num_words_to_generate = text_length
        start_word_selected_from_seed = False

        if seed_phrase:
            seed_words = self.preprocess_text(seed_phrase)
            if seed_words:
                potential_start_node = seed_words[-1]
                if potential_start_node in transitions and transitions[potential_start_node]:
                    generated_text_list.extend(seed_words)
                    current_word = potential_start_node
                    start_word_selected_from_seed = True
                    num_words_to_generate = text_length - len(generated_text_list)
                    if num_words_to_generate <= 0: return ' '.join(generated_text_list[:text_length])

        if not start_word_selected_from_seed:
            valid_starting_unigrams = {w:c for w,c in self.unigram_counts.items() if w in transitions and transitions[w]}
            if valid_starting_unigrams:
                sorted_starters = sorted(valid_starting_unigrams.items(), key=lambda item: item[1], reverse=True)
                starters = [item[0] for item in sorted_starters]
                weights = [item[1] for item in sorted_starters]
                if not starters: # Should not happen if valid_starting_unigrams is true
                    return "Error: No valid starters found despite unigram counts."
                current_word = random.choices(starters, weights=weights, k=1)[0]
            elif any(transitions.values()):
                possible_starters = [w1 for w1, w2_list in transitions.items() if w2_list]
                if possible_starters: current_word = random.choice(possible_starters)
                else: return "Error: Cannot determine any valid starting word."
            else: return "Error: Cannot determine a starting word (no valid transitions)."

            if current_word:
                 generated_text_list.append(current_word)
                 num_words_to_generate = text_length - 1
            else: return "Error: Failed to select a starting word."

        for _ in range(max(0, num_words_to_generate)):
            if not current_word or current_word not in transitions or not transitions[current_word]:
                valid_restart_candidates = [w for w, trans_list in transitions.items() if trans_list]
                if not valid_restart_candidates: break

                restart_options = {w:c for w,c in self.unigram_counts.items() if w in valid_restart_candidates}
                if restart_options:
                    sorted_restart_options = sorted(restart_options.items(), key=lambda item: item[1], reverse=True)
                    starters = [item[0] for item in sorted_restart_options]
                    weights = [item[1] for item in sorted_restart_options]
                    if not starters: # Should not happen if restart_options is true
                        current_word = random.choice(valid_restart_candidates) # Fallback
                    else:
                        current_word = random.choices(starters, weights=weights, k=1)[0]
                else: current_word = random.choice(valid_restart_candidates)
                if not current_word: break

            possible_next_words, weights = zip(*transitions[current_word])
            current_word = random.choices(possible_next_words, weights=weights, k=1)[0]
            generated_text_list.append(current_word)

        return ' '.join(generated_text_list)


def core_text_generation_flow():
    """Core flow: load, train, predict frequencies, generate text."""
    if TENSORFLOW_AVAILABLE:
        # Optional: For reproducible results with TensorFlow
        # tf.random.set_seed(42)
        # np.random.seed(42)
        pass

    predictor = FrequencyPredictor()
    text_content = None

    # Using openwebtext as a contemporary dataset.
    # Other options: 'wikitext', 'pg19', 'bookcorpusopen', or specific news datasets.
    # If using a dataset that requires specific handling or configs, adjust below.
    hf_dataset_config = {
        "name": "openwebtext",  # Contemporary general-purpose text dataset
        "config": None,         # openwebtext usually doesn't have specific configurations
        "split": "train",
        "text_column": "text"
    }
    # To use a local file instead (e.g., "input_text.txt"), uncomment the next line:
    # hf_dataset_config = None

    if hf_dataset_config and DATASETS_AVAILABLE:
        text_content = predictor.load_text_from_hf_dataset(
            dataset_name=hf_dataset_config["name"], config_name=hf_dataset_config["config"],
            split=hf_dataset_config["split"], text_column=hf_dataset_config["text_column"]
        )

    if text_content is None: # Fallback if HF dataset loading fails or not configured
        print("Falling back to local file 'input_text.txt' or internal sample text for content.")
        input_file = "input_text.txt"
        text_content = predictor.load_text_file(input_file)

    if not text_content:
        print("CRITICAL: No text content loaded. Cannot proceed."); return

    original_bigram_frequencies = predictor.extract_bigram_frequencies(text_content)
    if not original_bigram_frequencies:
        print("CRITICAL: No bigrams extracted. Cannot proceed."); return

    predictor.create_bigram_frequency_features()
    new_frequencies_set = None

    if not predictor.frequency_features:
        print("Warning: No features created. Using original frequencies for text generation.")
        new_frequencies_set = {k: float(v) for k, v in original_bigram_frequencies.items()}
    else:
        model_to_train = 'neural_network' # Defaulting to neural network

        if model_to_train == 'neural_network' and not TENSORFLOW_AVAILABLE:
            print("TensorFlow not found, cannot train neural network. Using original frequencies.")
            new_frequencies_set = {k: float(v) for k, v in original_bigram_frequencies.items()}
        else:
            predictor.train_predictor(model_type=model_to_train)
            if predictor.predictor_model:
                generated_sets = predictor.generate_new_bigram_frequencies(num_variations=1)
                if generated_sets: new_frequencies_set = generated_sets[0]
                else:
                    print("Warning: Could not generate new freqs with model. Using original.")
                    new_frequencies_set = {k: float(v) for k, v in original_bigram_frequencies.items()}
            else:
                print("Warning: Predictor model not trained. Using original freqs.")
                new_frequencies_set = {k: float(v) for k, v in original_bigram_frequencies.items()}

    if not new_frequencies_set:
        print("CRITICAL: Failed to obtain any frequency set. Cannot generate text."); return

    # Loop for multiple seed inputs
    while True:
        user_seed = input("Enter a seed phrase (or type 'quit' to exit, Enter for default seed): ").strip()
        if user_seed.lower() == 'quit':
            break
        
        effective_seed = user_seed
        if not user_seed: # If user just pressed Enter
            if predictor.unigram_counts:
                temp_transitions = defaultdict(list)
                # Ensure new_frequencies_set is not None before iterating
                if new_frequencies_set:
                    for (w1, w2), count in new_frequencies_set.items():
                        if count > 0: temp_transitions[w1].append((w2, count))
                
                potential_seeds = [w for w,c in predictor.unigram_counts.most_common(25) 
                                   if w in temp_transitions and temp_transitions[w]]
                if potential_seeds: effective_seed = random.choice(potential_seeds)
                else: effective_seed = "the" # Absolute fallback
            else: effective_seed = "the"
            print(f"(Using default seed: '{effective_seed}')")
        
        text_len = 150
        generated_text_new = predictor.expand_text_from_bigrams(new_frequencies_set, 
                                                                text_length=text_len, 
                                                                seed_phrase=effective_seed if effective_seed else None) # Pass None if effective_seed is empty

        output_prefix = ""
        if effective_seed: # Only add prefix if a seed was actually used
            processed_seed_words = predictor.preprocess_text(effective_seed)
            processed_seed_phrase = " ".join(processed_seed_words)
            if generated_text_new.lower().startswith(processed_seed_phrase.lower()) and processed_seed_phrase:
                 output_prefix = f"Continuing seed '{effective_seed}':\n"
            elif processed_seed_phrase: # Seed was given but not directly continued
                 output_prefix = f"Seed '{effective_seed}' (new sequence shown, may not directly continue seed):\n"
        
        print(f"\n{output_prefix}{generated_text_new}\n")


if __name__ == "__main__":
    if not DATASETS_AVAILABLE:
        print("---")
        print("NOTE: Hugging Face 'datasets' library not found (run: pip install datasets).")
        print("The script will rely on 'input_text.txt' or the internal sample text if HF loading fails.")
        print("---")
    if not TENSORFLOW_AVAILABLE:
        print("---")
        print("NOTE: TensorFlow library not found (run: pip install tensorflow).")
        print("The neural network model will not be available.")
        print("---")

    # Fallback sample file creation
    try:
        with open("input_text.txt", 'r', encoding='utf-8') as f: f.read(1)
    except FileNotFoundError:
        print(f"Creating sample file: input_text.txt (as a fallback data source)")
        with open("input_text.txt", 'w', encoding='utf-8') as f:
            f.write(FrequencyPredictor().get_sample_text())
            f.write("\nThis is additional sample text for the fallback input_text.txt. ")

    core_text_generation_flow()
