import numpy as np
import pandas as pd
from collections import defaultdict, Counter, deque
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import PCA
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
from typing import Dict, List, Tuple, Optional, Set, Generator
import re
import random
from scipy.stats import entropy
from scipy.special import softmax
import time
import threading
import queue

# Install required packages if not available
try:
    import serial
    import serial.tools.list_ports
except ImportError:
    print("Please install pyserial: pip install pyserial")
    exit()

    def _process_segment(self, signal: np.ndarray) -> Tuple[np.ndarray, float]:
        """Process a segment of data"""
        # Basic preprocessing
        signal = signal - np.mean(signal)  # Remove DC component
        signal = signal / (np.std(signal) + 1e-10)  # Normalize
        
        # Simple denoising (moving average)
        if len(signal) > 3:
            kernel = np.ones(3) / 3
            signal = np.convolve(signal, kernel, mode='same')
        
        # Estimate confidence
        signal_power = np.var(signal)
        confidence = min(1.0, signal_power / (signal_power + 0.1))
        
        return signal, confidence
    
    

class BigramProcessor:
    """Handles bigram extraction and semantic weighting using TF-IDF with set-based operations"""
    
    def __init__(self, min_freq: int = 1):
        self.min_freq = min_freq
        self.bigram_vocab = {}
        self.bigram_set = set()
        self.tfidf_vectorizer = None
        self.vocab_size = 0
        
    def extract_bigrams(self, texts: List[str]) -> List[Set[str]]:
        all_bigrams = []
        
        for text in texts:
            words = text.lower().split()
            if len(words) < 2:
                all_bigrams.append(set())
                continue
            
            bigrams_set = {f"{words[i]} {words[i+1]}" for i in range(len(words)-1)}
            all_bigrams.append(bigrams_set)
            self.bigram_set.update(bigrams_set)
            
        return all_bigrams
    
    def build_vocabulary(self, texts: List[str]):
        bigram_sets = self.extract_bigrams(texts)
        
        bigram_texts = []
        for bigram_set in bigram_sets:
            if bigram_set:
                bigram_texts.append(" ".join(sorted(bigram_set)))
            else:
                bigram_texts.append("")
        
        non_empty_texts = [text for text in bigram_texts if text.strip()]
        
        if not non_empty_texts:
            raise ValueError("No valid bigrams found in training texts")
        
        self.tfidf_vectorizer = TfidfVectorizer(ngram_range=(2, 2), min_df=self.min_freq)
        self.tfidf_vectorizer.fit(non_empty_texts)
        
        feature_names = self.tfidf_vectorizer.get_feature_names_out()
        self.bigram_set = set(feature_names)
        
        ordered_bigrams = sorted(list(self.bigram_set))
        for idx, bigram in enumerate(ordered_bigrams):
            self.bigram_vocab[bigram] = idx
        
        self.vocab_size = len(self.bigram_vocab)
        print(f"Built vocabulary with {self.vocab_size} unique bigrams")
    
    def get_bigram_weights(self, text: str) -> np.ndarray:
        """Get TF-IDF weighted bigram representation with consistent dimensions"""
        if self.tfidf_vectorizer is None:
            raise ValueError("Vocabulary not built yet. Call build_vocabulary first.")
        
        words = text.lower().split()
        if len(words) < 2:
            return np.zeros(self.vocab_size)
        
        # Create set of bigrams from text
        text_bigrams_set = {f"{words[i]} {words[i+1]}" for i in range(len(words)-1)}
        
        # Only use bigrams that exist in our vocabulary set
        valid_bigrams_set = text_bigrams_set.intersection(self.bigram_set)
        
        if not valid_bigrams_set:
            return np.zeros(self.vocab_size)
        
        # Convert to space-separated string for TF-IDF
        bigram_text = " ".join(sorted(valid_bigrams_set))
        
        # Get TF-IDF vector and ensure consistent size
        tfidf_vector = self.tfidf_vectorizer.transform([bigram_text]).toarray()[0]
        
        # Create output vector with correct dimensions
        output_vector = np.zeros(self.vocab_size)
        
        # Map TF-IDF features to our vocabulary indices
        feature_names = self.tfidf_vectorizer.get_feature_names_out()
        for i, feature in enumerate(feature_names):
            if feature in self.bigram_vocab:
                output_vector[self.bigram_vocab[feature]] = tfidf_vector[i]
        
        return output_vector

class ProbabilisticTextGenerator:
    """Generates probabilistic text using set-based bigram operations"""
    
    def __init__(self, bigram_processor: BigramProcessor):
        self.bigram_processor = bigram_processor
        self.bigram_set = bigram_processor.bigram_set.copy()
        self.bigram_to_words = {bigram: bigram.split() for bigram in self.bigram_set}
    
    def generate_probabilistic_text(self, bigram_vector: np.ndarray,
                                  bigram_uncertainty: np.ndarray,
                                  max_length: int = 50,
                                  temperature: float = 1.0,
                                  seed_text: str = None,
                                  seed_weight: float = 0.3) -> Tuple[str, List[float], float]:
        """Generate probabilistic text by sampling bigrams from a set without replacement."""
        
        # Create ordered list of bigrams from the vocabulary
        vocab_list = sorted(list(self.bigram_set))
        
        scaled_vector = bigram_vector / temperature
        bigram_probs = softmax(scaled_vector)
        
        uncertainty_penalty = 1 - (bigram_uncertainty / (np.max(bigram_uncertainty) + 1e-10))
        adjusted_probs = bigram_probs * uncertainty_penalty
        adjusted_probs = adjusted_probs / np.sum(adjusted_probs)
        
        if seed_text and seed_text.strip():
            seed_bigram_vector = self.bigram_processor.get_bigram_weights(seed_text)
            seed_probs = softmax(seed_bigram_vector / temperature)
            adjusted_probs = (1 - seed_weight) * adjusted_probs + seed_weight * seed_probs
            adjusted_probs = adjusted_probs / np.sum(adjusted_probs)
        
        # Create a mapping from bigram to probability for efficient lookup
        bigram_prob_map = {bigram: prob for bigram, prob in zip(vocab_list, adjusted_probs)}

        # Use a set of available bigrams to sample from without replacement
        available_bigrams = set(vocab_list)
        
        words = []
        word_confidences = []
        current_word = None
        
        if seed_text and seed_text.strip():
            seed_words = seed_text.lower().split()
            if seed_words:
                words = seed_words[-2:] if len(seed_words) >= 2 else seed_words
                current_word = words[-1] if words else None
                word_confidences.extend([0.9] * len(words))
                print(f"Starting probabilistic generation with seed: {' '.join(words)}")
        
        for step in range(max_length):


            if current_word:
                valid_candidates = {bg for bg in available_bigrams if bg.split()[0] == current_word}
                if not valid_candidates:
                    valid_candidates = available_bigrams.copy()
            else:
                valid_candidates = available_bigrams.copy()

            valid_bigrams_list = list(valid_candidates)
            probs = np.array([bigram_prob_map[bg] for bg in valid_bigrams_list])
            
          
            
            probs = probs / np.sum(probs)
            
            selected_idx = np.random.choice(len(valid_bigrams_list), p=probs)
            selected_bigram = valid_bigrams_list[selected_idx]
            selected_prob = probs[selected_idx]
            
            # Remove the chosen bigram so it can't be used again
            available_bigrams.remove(selected_bigram)
            
            bigram_words = selected_bigram.split()
            if not words:
                words.extend(bigram_words)
                word_confidences.extend([selected_prob] * 2)
                current_word = bigram_words[1]
            else:
                words.append(bigram_words[1])
                word_confidences.append(selected_prob)
                current_word = bigram_words[1]
            
            
        overall_confidence = np.mean(word_confidences) if word_confidences else 0.0
        generated_text = " ".join(words) if words else "Unable to generate text"
        return generated_text, word_confidences, overall_confidence

class USBSymbolicNeuralAI:
    """Complete symbolic AI system with USB serial input"""
    
    def __init__(self, port: str = None, baudrate: int = 115200, window_size: int = 100):
        self.bigram_processor = BigramProcessor()
        self.text_generator = None
        
        # Feature processing
        self.feature_history = deque(maxlen=10)
        self.is_trained = False
        
    def train_vocabulary(self, training_texts: List[str]):
        """Train the bigram vocabulary"""
        self.bigram_processor.build_vocabulary(training_texts)
        self.text_generator = ProbabilisticTextGenerator(self.bigram_processor)
        self.is_trained = True
        print("Vocabulary training completed")
        
    def create_multiple_arrays(self):
        segments = []
        
        for i in range(52):
            # Generate a more robust signal
            t = np.linspace(0, 3, 1024)
            signal_array = (np.sqrt(i * np.pi * 10 * t) + 
                           0.5 * np.sqrt(2 * np.pi * 25 * t) + 
                           0.2 * np.random.randn(len(t)))
            
            confidence = 0.7 + i * 0.05
            
            # Debug: Print signal statistics
            print(f"Signal {i}: mean={np.mean(signal_array):.3f}, "
                  f"std={np.std(signal_array):.3f}, "
                  f"min={np.min(signal_array):.3f}, "
                  f"max={np.max(signal_array):.3f}")
            
            segments.append((signal_array, confidence))
        
        return segments
    def extract_features_from_segment(self, signal: np.ndarray, confidence: float, user_input: str = "") -> Tuple[np.ndarray, np.ndarray]:
        """Extract features from processed signal segment with user input iteration"""
        # Temporal features
        mean_activity = np.mean(signal)
        variance = np.var(signal)
        peak_amplitude = np.max(np.abs(signal))
        
        # Frequency domain analysis
        power_spectrum = signal / (np.sum(signal) + 1e-10)
        freqs = np.fft.fftfreq(len(signal))
        
        # Convert user input to numerical values for frequency band adjustment
        user_chars = [ord(char) for char in user_input.lower()] if user_input else [100]
        char_sum = sum(user_chars) % 1000  # Normalize to reasonable range
        
        # Neural frequency bands with user input iteration
        base_offset = char_sum / 10000.0  # Small offset based on user input
        
        delta_start = 0.101 + (char_sum % 10) / 1000.0
        delta_end = 0.12 + (char_sum % 15) / 1000.0
        
        theta_start = 0.22 + (char_sum % 20) / 1000.0
        theta_end = 0.24 + (char_sum % 25) / 1000.0
        
        alpha_start = 0.34 + (char_sum % 30) / 1000.0
        alpha_end = 0.36 + (char_sum % 35) / 1000.0
        
        beta_start = 0.46 + (char_sum % 40) / 1000.0
        beta_end = 0.48 + (char_sum % 45) / 1000.0  # Fixed: was 0.45
        
        gamma_start = 0.55 + (char_sum % 50) / 1000.0
        gamma_end = 0.60 + (char_sum % 55) / 1000.0  # Fixed: was 0.5
        
        # Create indices based on user input
        delta_indices = np.where((freqs >= delta_start) & (freqs <= delta_end))[0]
        theta_indices = np.where((freqs >= theta_start) & (freqs <= theta_end))[0]
        alpha_indices = np.where((freqs >= alpha_start) & (freqs <= alpha_end))[0]
        beta_indices = np.where((freqs >= beta_start) & (freqs <= beta_end))[0]
        gamma_indices = np.where((freqs >= gamma_start) & (freqs <= gamma_end))[0]
        
        # Calculate band powers
        delta_band = np.mean(power_spectrum[delta_indices]) if len(delta_indices) > 0 else 0
        theta_band = np.mean(power_spectrum[theta_indices]) if len(theta_indices) > 0 else 0
        alpha_band = np.mean(power_spectrum[alpha_indices]) if len(alpha_indices) > 0 else 0
        beta_band = np.mean(power_spectrum[beta_indices]) if len(beta_indices) > 0 else 0
        gamma_band = np.mean(power_spectrum[gamma_indices]) if len(gamma_indices) > 0 else 0
        
        signal_entropy = -np.sum(power_spectrum * np.log(power_spectrum + 1e-10))
        
        features = np.array([
            mean_activity, variance, peak_amplitude,
            delta_band, theta_band, alpha_band, beta_band, gamma_band,
            signal_entropy, confidence
        ])
        
        base_uncertainty = 1 - confidence
        feature_uncertainties = np.array([
            base_uncertainty * 0.1, base_uncertainty * 0.2, base_uncertainty * 0.15,
            base_uncertainty * 0.25, base_uncertainty * 0.3, base_uncertainty * 0.3,
            base_uncertainty * 0.3, base_uncertainty * 0.3, base_uncertainty * 0.25,
            base_uncertainty * 0.1
        ])
        
        return features, feature_uncertainties

    def run_realtime_processing(self):
        """Run complete real-time processing from USB to text"""
        segment_count = 0
        
        # Get user input once at the start
        user_input = input("Enter text to influence frequency bands: ")
        
        try:
            # Process real-time stream
            for processed_signal, confidence in self.create_multiple_arrays():
                segment_count += 1
                
                # Extract features with user input
                features, uncertainties = self.extract_features_from_segment(
                    processed_signal, confidence, user_input
                )
                
                self.feature_history.append(features)
                
                print(f"\nSegment {segment_count}:")
                print(f"Signal length: {len(processed_signal)}")
                print(f"Confidence: {confidence:.3f}")
                print(f"User input influence: {user_input}")
                print(f"Features: {features[:5]}...")  # Show first 5 features
                
                # Generate text based on features
                generated_text = self._generate_complete_text(features, uncertainties, confidence)
                print(f"Generated: {generated_text}")
                print("-" * 60)
                
        except KeyboardInterrupt:
            print("\nStopping processing...")
    def _generate_complete_text(self, features: np.ndarray, uncertainties: np.ndarray, confidence: float, user_input: str = "") -> str:
        """Complete text generation based on features using the trained vocabulary"""
        
        if not self.is_trained or self.text_generator is None:
            return "System not trained - no text generator available"
        
        try:
            # Convert features to a pseudo-bigram vector for text generation
            bigram_vector = self._features_to_bigram_vector(features)
            bigram_uncertainty = uncertainties[:len(bigram_vector)] if len(uncertainties) >= len(bigram_vector) else np.ones(len(bigram_vector)) * 0.1
            
            # Use the user_input that was already provided instead of asking for new input
            seed_text = user_input if user_input.strip() else "the quick brown"
            
            # Generate text using the probabilistic text generator (ONLY ONCE)
            generated_text, word_confidences, overall_confidence = self.text_generator.generate_probabilistic_text(
                bigram_vector=bigram_vector,
                bigram_uncertainty=bigram_uncertainty,
                max_length=250,  # Reduced for better output
                temperature=0.8,  # Increased for more variety
                seed_text=seed_text,
                seed_weight=0.4
            )
            
            return f"{generated_text} (conf: {overall_confidence:.2f})"
            
        except Exception as e:
            print(f"Text generation failed: {e}")
            return f"Generation error: {str(e)}"



    def run_realtime_processing(self):
        """Run complete real-time processing from USB to text"""
        segment_count = 0
        
        # Get user input once at the start
        user_input = input("Enter text to influence frequency bands: ")
        print(f"Using input: '{user_input}' to influence processing\n")
        
        try:
            # Process real-time stream
            for processed_signal, confidence in self.create_multiple_arrays():
                segment_count += 1
                
                # Extract features with user input
                features, uncertainties = self.extract_features_from_segment(
                    processed_signal, confidence, user_input
                )
                
                self.feature_history.append(features)
                
                print(f"\nSegment {segment_count}:")
                print(f"Signal length: {len(processed_signal)}")
                print(f"Confidence: {confidence:.3f}")
                print(f"User input influence: {user_input}")
                print(f"Features: {features[:5]}...")  # Show first 5 features
                
                # Generate text based on features - PASS user_input parameter
                generated_text = self._generate_complete_text(features, uncertainties, confidence, user_input)
                print(f"Generated: {generated_text}")
                print("-" * 60)
                
                # Add small delay to make output readable
                time.sleep(0.5)
                
        except KeyboardInterrupt:
            print("\nStopping processing...")
    
    def _features_to_bigram_vector(self, features: np.ndarray) -> np.ndarray:
        """Convert features to a bigram-space vector with better numerical stability"""
        vocab_size = self.bigram_processor.vocab_size
        
        if vocab_size == 0:
            print("Warning: No vocabulary built, using small random vector")
            return np.random.normal(0, 0.1, 10)
        
        # Clean input features
        features = np.nan_to_num(features, nan=0.0, posinf=1.0, neginf=-1.0)
        
        if len(features) == 0:
            return np.random.normal(0, 0.1, vocab_size)
        
        # Normalize features to prevent extreme values
        feature_std = np.std(features)
        if feature_std > 0:
            features = features / feature_std
        
        # Create expanded vector
        expanded_features = np.tile(features, (vocab_size // len(features)) + 1)[:vocab_size]
        
        # Add controlled noise
        noise_scale = min(0.5, max(0.01, np.mean(np.abs(features)) * 0.1))
        noise = np.random.normal(0, noise_scale, vocab_size)
        
        result = expanded_features + noise
        
        # Clip to reasonable range for softmax stability
        result = np.clip(result, -5, 5)
        
        return result

    
def main():
    """Main function to demonstrate neural processing"""
    
    # Initialize system
    usb_ai = USBSymbolicNeuralAI(baudrate=115200, window_size=50)
    
    # Train vocabulary
    try:
        with open(input("Filename: "), 'r', encoding='utf-8') as f:
            content = f.read()
        training_texts = [text.strip() for text in content.split(".") if text.strip()]
    except FileNotFoundError:
        print("File not found. Using default training data.")
        training_texts = [
            "the quick brown fox jumps over the lazy dog today",
            "artificial intelligence learns from neural signals in the brain",
            "brain activity patterns reveal complex cognitive processes in humans", 
            "symbolic reasoning combines with neural networks for better performance",
            "language generation requires deep semantic understanding of context",
            "time signals encode temporal information patterns in neural data",
            "machine learning algorithms process complex data from multiple sources",
            "cognitive science studies mental processes and brain functions",
            "neural networks learn hierarchical representations from training data",
            "natural language processing enables human computer communication systems"
        ]
    
    usb_ai.train_vocabulary(training_texts)
    usb_ai.run_realtime_processing()

if __name__ == "__main__":
    main()
