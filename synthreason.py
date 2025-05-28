import numpy as np
# import pandas as pd # No longer used
from collections import Counter, defaultdict
import re
import random
# import matplotlib.pyplot as plt # No longer used for visualization in main flow
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
# import json # No longer used for saving/loading in main flow
from typing import Dict, List, Tuple, Optional, Any

# Helper functions for bigram key conversion (still needed if saving/loading were to be re-added)
# For this minimal version, they are not directly used by the core generation path if JSON methods are removed.
# However, if any internal mechanism might hypothetically use them (though not apparent now),
# they are harmless. Let's assume they might be useful for potential extensions and keep them for now.
def bigram_to_key(bigram: Tuple[str, str]) -> str:
    return f"{bigram[0]}||{bigram[1]}"

def key_to_bigram(key: str) -> Tuple[str, str]:
    parts = key.split('||')
    if len(parts) == 2:
        return (parts[0], parts[1])
    # Minimal print for critical issues
    # print(f"Warning: Malformed key '{key}' encountered.")
    return ("<malformed>", "<key>")


class FrequencyPredictor:
    def __init__(self):
        self.bigram_frequencies: Dict[Tuple[str, str], int] = {}
        self.frequency_features: List[List[float]] = []
        self.predictor_model: Optional[Any] = None
        self.scaler = StandardScaler()
        self.sorted_bigrams: List[Tuple[str, str]] = []
        self.unigram_counts: Dict[str, int] = Counter()

    def load_text_file(self, file_path: str) -> str:
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                return ' '.join(f.read().lower().split()[:15000])
        except FileNotFoundError:
            # Critical fallback, print is useful here
            print(f"File {file_path} not found. Using internal sample text.")
            return self.get_sample_text()

    def get_sample_text(self) -> str:
        return """
        The quick brown fox jumps over the lazy dog. The fox is very agile and quick.
        Machine learning is revolutionizing technology. Artificial intelligence learns from data patterns.
        Natural language processing enables computers to understand human communication effectively.
        Data science combines statistics, programming, and domain expertise to extract insights.
        Python programming language offers powerful libraries for machine learning applications.
        Deep learning neural networks can model complex relationships in large datasets.
        The quick brown fox returned. The lazy dog slept. The quick fox ran again.
        Machine learning models improve with more data. Data patterns are key.
        """

    def preprocess_text(self, text: str) -> List[str]:

        words = text.split()
        return [word for word in words if word]

    def extract_bigram_frequencies(self, text: str) -> Dict[Tuple[str, str], int]:
        words = self.preprocess_text(text)
        if len(words) < 2:
            self.bigram_frequencies = {}
            self.sorted_bigrams = []
            self.unigram_counts = Counter(words)
            return {}

        self.unigram_counts = Counter(words)
        bigrams = [(words[i], words[i+1]) for i in range(len(words)-1)]
        self.bigram_frequencies = dict(Counter(bigrams))
        
        self.sorted_bigrams = [
            item[0] for item in sorted(self.bigram_frequencies.items(), key=lambda x: (x[1], x[0][0], x[0][1]), reverse=True)
        ]
        # Minimal print: print(f"Extracted {len(self.bigram_frequencies)} unique bigrams.")
        return self.bigram_frequencies

    def create_bigram_frequency_features(self) -> List[List[float]]:
        features = []
        if not self.sorted_bigrams:
            self.frequency_features = []
            return []

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

    def train_predictor(self, model_type: str = 'linear') -> None:
        if not self.frequency_features:
            self.predictor_model = None
            return

        X = np.array([f[1:] for f in self.frequency_features]) 
        y = np.array([f[0] for f in self.frequency_features])  

        if X.shape[0] <= 1 or X.ndim != 2 or X.shape[0] != y.shape[0]:
            self.predictor_model = None
            return
        
        if X.shape[0] >= 5 :
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        else: 
            X_train, X_test, y_train, y_test = X, X, y, y
        
        if X_train.shape[0] == 0: 
            self.predictor_model = None
            return

        self.scaler.fit(X_train)
        X_train_scaled = self.scaler.transform(X_train)
        # X_test_scaled = self.scaler.transform(X_test) if X_test.shape[0] > 0 else np.array([]).reshape(0, X_train.shape[1]) # Test score not printed
        
        if model_type == 'linear':
            self.predictor_model = LinearRegression()
        else:
            self.predictor_model = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
        
        self.predictor_model.fit(X_train_scaled, y_train)
        # train_score = self.predictor_model.score(X_train_scaled, y_train) # Score not printed
        # if X_test_scaled.shape[0] > 0:
        # test_score = self.predictor_model.score(X_test_scaled, y_test) # Score not printed

    def generate_new_bigram_frequencies(self, num_variations: int = 1) -> List[Dict[Tuple[str, str], float]]:
        if self.predictor_model is None or not self.frequency_features or not self.sorted_bigrams:
            return []

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

            X_modified_scaled = self.scaler.transform(np.array(modified_features_for_prediction))
            predicted_new_counts = self.predictor_model.predict(X_modified_scaled)
            predicted_new_counts = np.maximum(predicted_new_counts, 0.01) 
            
            current_sum_predicted_counts = np.sum(predicted_new_counts)
            if current_sum_predicted_counts == 0:
                if len(predicted_new_counts) > 0:
                    predicted_new_counts = np.full(len(predicted_new_counts), 0.01)
                    current_sum_predicted_counts = np.sum(predicted_new_counts)
                else:
                    new_frequency_sets.append({}); continue
            
            scaled_predicted_counts = predicted_new_counts
            if original_total_frequency_sum > 0 and current_sum_predicted_counts > 0:
                 scaled_predicted_counts = (predicted_new_counts / current_sum_predicted_counts) * original_total_frequency_sum
            
            new_freq_dict: Dict[Tuple[str, str], float] = {
                bigram: float(scaled_predicted_counts[i]) for i, bigram in enumerate(self.sorted_bigrams)
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
            if count > 0:
                transitions[w1].append((w2, count))
        
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
                    if num_words_to_generate <= 0:
                        return ' '.join(generated_text_list[:text_length])
                # else: # Minimal prints: Avoid warning if seed cannot be continued
                    # print(f"Warning: Seed phrase '{seed_phrase}' cannot be directly continued.")
                    # Fall through to default start mechanism

        if not start_word_selected_from_seed:
            valid_starting_unigrams = {w:c for w,c in self.unigram_counts.items() if w in transitions and transitions[w]}
            if valid_starting_unigrams:
                starters = list(valid_starting_unigrams.keys())
                weights = [valid_starting_unigrams[s] for s in starters]
                current_word = random.choices(starters, weights=weights, k=1)[0]
            elif any(transitions.values()):
                possible_starters = [w1 for w1, w2_list in transitions.items() if w2_list]
                if possible_starters: current_word = random.choice(possible_starters)
                else: return "Error: Cannot determine any valid starting word."
            else: return "Error: Cannot determine a starting word (no valid transitions)."
            
            if current_word:
                 generated_text_list.append(current_word)
                 num_words_to_generate = text_length -1
            else: # Should be caught by earlier checks
                 return "Error: Failed to select a starting word."


        for _ in range(max(0, num_words_to_generate)): # Ensure non-negative loop count
            if not current_word or current_word not in transitions or not transitions[current_word]:
                valid_restart_candidates = [w for w, trans_list in transitions.items() if trans_list]
                if not valid_restart_candidates: break 
                
                restart_options = {w:c for w,c in self.unigram_counts.items() if w in valid_restart_candidates}
                if restart_options:
                    starters = list(restart_options.keys())
                    weights = [restart_options[s] for s in starters]
                    current_word = random.choices(starters, weights=weights, k=1)[0]
                else: 
                    current_word = random.choice(valid_restart_candidates)
                if not current_word: break 

            possible_next_words, weights = zip(*transitions[current_word])
            current_word = random.choices(possible_next_words, weights=weights, k=1)[0]
            generated_text_list.append(current_word)

        return ' '.join(generated_text_list)


def core_text_generation_flow():
    """Core flow: load, train, predict frequencies, generate text."""
    predictor = FrequencyPredictor()
    
    input_file = "test.txt" 
    text_content = predictor.load_text_file(input_file)
    
    original_bigram_frequencies = predictor.extract_bigram_frequencies(text_content)
    if not original_bigram_frequencies:
        print("CRITICAL: No bigrams extracted. Text generation cannot proceed.")
        return

    predictor.create_bigram_frequency_features()
    if not predictor.frequency_features:
        print("CRITICAL: No features created. Text generation will use original frequencies only (if any).")
        # Fallback: use original frequencies if model cannot be trained
        new_frequencies_set = {k: float(v) for k, v in original_bigram_frequencies.items()}

    else:
        predictor.train_predictor() 
        if predictor.predictor_model:
            generated_sets = predictor.generate_new_bigram_frequencies(num_variations=1)
            if generated_sets:
                new_frequencies_set = generated_sets[0]
            else:
                print("Warning: Could not generate new frequency sets. Using original frequencies for text generation.")
                new_frequencies_set = {k: float(v) for k, v in original_bigram_frequencies.items()}
        else:
            print("Warning: Predictor model not trained. Using original frequencies for text generation.")
            new_frequencies_set = {k: float(v) for k, v in original_bigram_frequencies.items()}

    while True:
        user_seed = input("Enter a seed phrase (or press Enter for default): ").strip()
        if not user_seed:
            if predictor.unigram_counts:
                potential_seeds = [w for w,c in predictor.unigram_counts.most_common(10) 
                                   if w in (bg[0] for bg in new_frequencies_set)] # Check against keys of new_frequencies_set
                if potential_seeds: user_seed = random.choice(potential_seeds)
                else: user_seed = "the" 
            else: user_seed = "the"
            print(f"(Using default seed: '{user_seed}')")
        
        text_len = 250 # Generate a decent length of text

        if new_frequencies_set:

            # print(f"\n--- Generating text from PREDICTED frequencies (seed: '{user_seed}') ---")
            generated_text_new = predictor.expand_text_from_bigrams(new_frequencies_set, text_length=text_len, seed_phrase=user_seed)
            
            # Determine if seed was actually used as a prefix in the output
            output_prefix = ""
            if user_seed:
                # Check if the generated text starts with the processed seed words
                processed_seed_words = predictor.preprocess_text(user_seed)
                processed_seed_phrase = " ".join(processed_seed_words)
                if generated_text_new.lower().startswith(processed_seed_phrase.lower()):
                     output_prefix = f"Continuing seed '{user_seed}':\n"
                else:
                     output_prefix = f"Seed '{user_seed}' (could not be directly continued by model, new sequence shown):\n"
            
            print(f"\n{output_prefix}{generated_text_new}")
        else:
            print("No frequency data available to generate text.")


if __name__ == "__main__":

            
    core_text_generation_flow()
