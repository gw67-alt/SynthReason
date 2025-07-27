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

class USBSerialMRIReader:
    """Reads MRI data from USB serial port in real-time"""
    
    def __init__(self, port: str = None, baudrate: int = 115200, timeout: float = 0.1):
        self.port = port
        self.baudrate = baudrate
        self.timeout = timeout
        self.serial_connection = None
        self.is_connected = False
        self.data_queue = queue.Queue()
        self.read_thread = None
        self.stop_reading = False
        
    def list_available_ports(self) -> List[str]:
        """List all available USB serial ports"""
        ports = serial.tools.list_ports.comports()
        available_ports = []
        
        print("Available USB Serial Ports:")
        for port in ports:
            print(f"  {port.device}: {port.description}")
            available_ports.append(port.device)
            
        return available_ports
    
    def connect(self, port: str = None) -> bool:
        """Connect to USB serial port"""
        if port:
            self.port = port
        
        if not self.port:
            available_ports = self.list_available_ports()
            if not available_ports:
                print("No USB serial ports found!")
                return False
            self.port = available_ports[0]  # Use first available port
            print(f"Auto-selecting port: {self.port}")
        
        try:
            self.serial_connection = serial.Serial(
                port=self.port,
                baudrate=self.baudrate,
                timeout=self.timeout,
                parity=serial.PARITY_NONE,
                stopbits=serial.STOPBITS_ONE,
                bytesize=serial.EIGHTBITS
            )
            
            # Clear any existing data in buffers
            self.serial_connection.flushInput()
            self.serial_connection.flushOutput()
            
            self.is_connected = True
            print(f"Connected to {self.port} at {self.baudrate} baud")
            return True
            
        except serial.SerialException as e:
            print(f"Failed to connect to {self.port}: {e}")
            return False
    
    def start_reading(self):
        """Start background thread to read serial data"""
        if not self.is_connected:
            print("Not connected to serial port!")
            return
        
        self.stop_reading = False
        self.read_thread = threading.Thread(target=self._read_serial_data, daemon=True)
        self.read_thread.start()
        print("Started reading serial data in background")
    
    def _read_serial_data(self):
        """Background thread function to continuously read serial data"""
        buffer = ""
        
        while not self.stop_reading and self.is_connected:
            try:
                if self.serial_connection.in_waiting > 0:
                    # Read available data
                    data = self.serial_connection.read(self.serial_connection.in_waiting)
                    buffer += data.decode('utf-8', errors='ignore')
                    
                    # Process complete lines
                    while '\n' in buffer:
                        line, buffer = buffer.split('\n', 1)
                        line = line.strip()
                        
                        if line:
                            try:
                                # Convert to float (assuming numeric MRI data)
                                value = float(line)
                                self.data_queue.put(value)
                            except ValueError:
                                # Skip non-numeric lines
                                continue
                
                time.sleep(0.001)  # Small delay to prevent CPU overload
                
            except Exception as e:
                print(f"Error reading serial data: {e}")
                break
    
    def get_sample(self, timeout: float = 0.1) -> Optional[float]:
        """Get next MRI sample from queue"""
        try:
            return self.data_queue.get(timeout=timeout)
        except queue.Empty:
            return None
    
    def get_queue_size(self) -> int:
        """Get number of samples waiting in queue"""
        return self.data_queue.qsize()
    
    def disconnect(self):
        """Disconnect from serial port"""
        self.stop_reading = True
        
        if self.read_thread and self.read_thread.is_alive():
            self.read_thread.join(timeout=1.0)
        
        if self.serial_connection and self.serial_connection.is_open:
            self.serial_connection.close()
            
        self.is_connected = False
        print("Disconnected from serial port")

class USBSerialMRIProcessor:
    """Processes MRI data from USB serial in real-time"""
    
    def __init__(self, port: str = None, baudrate: int = 115200, window_size: int = 100):
        self.usb_reader = USBSerialMRIReader(port, baudrate)
        self.window_size = window_size
        self.data_buffer = deque(maxlen=window_size)
        self.processed_segments = []
        self.total_samples_processed = 0
        
    def connect_and_start(self, port: str = None) -> bool:
        """Connect to USB port and start reading"""
        if not self.usb_reader.connect(port):
            return False
        
        self.usb_reader.start_reading()
        return True
    
    def process_realtime_stream(self) -> Generator[Tuple[np.ndarray, float], None, None]:
        """Generator that yields processed MRI segments as they become available"""
        
        print("Starting real-time MRI processing from USB serial...")
        print("Waiting for data... (Press Ctrl+C to stop)")
        
        try:
            while True:
                # Get next sample from USB
                sample = self.usb_reader.get_sample(timeout=1.0)
                
                if sample is not None:
                    self.data_buffer.append(sample)
                    self.total_samples_processed += 1
                    
                    # Process when buffer is full
                    if len(self.data_buffer) == self.window_size:
                        segment = np.array(list(self.data_buffer))
                        processed_segment, confidence = self._process_segment(segment)
                        
                        yield processed_segment, confidence
                        
                        # Optional: Show progress
                        if self.total_samples_processed % 100 == 0:
                            queue_size = self.usb_reader.get_queue_size()
                            print(f"Processed {self.total_samples_processed} samples, "
                                  f"Queue: {queue_size}, Confidence: {confidence:.3f}")
                
                elif not self.usb_reader.is_connected:
                    print("USB connection lost")
                    break
                    
        except KeyboardInterrupt:
            print("\nStopping real-time processing...")
        finally:
            self.usb_reader.disconnect()
    
    def _process_segment(self, signal: np.ndarray) -> Tuple[np.ndarray, float]:
        """Process a segment of MRI data"""
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
    
    def extract_features_from_segment(self, signal: np.ndarray, confidence: float) -> Tuple[np.ndarray, np.ndarray]:
        """Extract features from processed signal segment"""
        # Temporal features
        mean_activity = np.mean(signal)
        variance = np.var(signal)
        peak_amplitude = np.max(np.abs(signal))
        
        # Frequency domain analysis
        fft = np.fft.fft(signal)
        power_spectrum = np.abs(fft) ** 2
        power_spectrum = power_spectrum / (np.sum(power_spectrum) + 1e-10)
        
        freqs = np.fft.fftfreq(len(signal))
        
        # Neural frequency bands (normalized frequencies)
        delta_indices = np.where((freqs >= 0.001) & (freqs <= 0.02))[0]
        theta_indices = np.where((freqs >= 0.02) & (freqs <= 0.04))[0]
        alpha_indices = np.where((freqs >= 0.04) & (freqs <= 0.06))[0]
        beta_indices = np.where((freqs >= 0.06) & (freqs <= 0.15))[0]
        gamma_indices = np.where((freqs >= 0.15) & (freqs <= 0.4))[0]
        
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
    """Complete symbolic AI system with USB serial MRI input"""
    
    def __init__(self, port: str = None, baudrate: int = 115200, window_size: int = 100):
        self.mri_processor = USBSerialMRIProcessor(port, baudrate, window_size)
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
    
    def run_realtime_processing(self, port: str = None):
        """Run complete real-time processing from USB to text"""
        
        # Connect to USB port
        if not self.mri_processor.connect_and_start(port):
            return
        
        print("\n=== Real-time USB MRI to Text Processing ===")
        
        segment_count = 0
        
        try:
            # Process real-time stream
            for processed_signal, confidence in self.mri_processor.process_realtime_stream():
                segment_count += 1
                
                # Extract features
                features, uncertainties = self.mri_processor.extract_features_from_segment(
                    processed_signal, confidence
                )
                
                self.feature_history.append(features)
                
                print(f"\nSegment {segment_count}:")
                print(f"Signal length: {len(processed_signal)}")
                print(f"Confidence: {confidence:.3f}")
                print(f"Features: {features[:5]}...")  # Show first 5 features
                
                # Generate text based on features
                generated_text = self._generate_complete_text(features, uncertainties, confidence)
                print(f"Generated: {generated_text}")
                print("-" * 60)
                
        except KeyboardInterrupt:
            print("\nStopping USB processing...")
        finally:
            self.mri_processor.usb_reader.disconnect()
    
    def _generate_complete_text(self, features: np.ndarray, uncertainties: np.ndarray, confidence: float) -> str:
        """Complete text generation based on MRI features using the trained vocabulary"""
        
        if not self.is_trained or not self.text_generator:
            return self._generate_simple_text(features, confidence)
        
        try:
            # Convert features to a pseudo-bigram vector for text generation
            # This is a simplified approach - in a full implementation, you'd use the internal entity
            bigram_vector = self._features_to_bigram_vector(features)
            bigram_uncertainty = uncertainties[:len(bigram_vector)] if len(uncertainties) >= len(bigram_vector) else np.ones(len(bigram_vector)) * 0.1
            
            # Generate text using the probabilistic text generator
            generated_text, word_confidences, overall_confidence = self.text_generator.generate_probabilistic_text(
                bigram_vector=bigram_vector,
                bigram_uncertainty=bigram_uncertainty,
                max_length=250,
                temperature=1.2,
                seed_text=input("USER: "),
                seed_weight=0.3
            )
            
            return f"{generated_text} (conf: {overall_confidence:.2f})"
            
        except Exception as e:
            print(f"Advanced generation failed: {e}")
            return self._generate_simple_text(features, confidence)
    
    def _features_to_bigram_vector(self, features: np.ndarray) -> np.ndarray:
        """Convert MRI features to a bigram-space vector"""
        # This is a simplified mapping - expand the feature vector to match bigram vocabulary size
        vocab_size = self.bigram_processor.vocab_size
        
        if vocab_size == 0:
            return np.random.random(10)
        
        # Create a pseudo-bigram vector by repeating and transforming features
        expanded_features = np.tile(features, (vocab_size // len(features)) + 1)[:vocab_size]
        
        # Add some controlled randomness based on features
        noise_scale = np.mean(np.abs(features)) * 0.1
        noise = np.random.normal(0, noise_scale, vocab_size)
        
        return expanded_features + noise
    
    def _generate_simple_text(self, features: np.ndarray, confidence: float) -> str:
        """Fallback simple text generation based on MRI features"""
        
        # Basic feature-to-text mapping for demonstration
        mean_activity = features[0]
        variance = features[1]
        peak_amplitude = features[2]
        
        if confidence > 0.7:
            if mean_activity > 0.2:
                base_text = "high neural activity detected"
            elif mean_activity < -0.2:
                base_text = "low neural activity observed"
            else:
                base_text = "moderate brain activity"
        else:
            base_text = "uncertain signal quality"
        
        if variance > 0.5:
            base_text += " with high variability"
        elif variance < 0.2:
            base_text += " with stable patterns"
        
        # Add frequency band information if available
        if len(features) > 6:
            alpha_band = features[5]
            beta_band = features[6]
            
            if alpha_band > 0.1:
                base_text += " alpha dominant"
            elif beta_band > 0.1:
                base_text += " beta prominent"
        
        return base_text

def main():
    """Main function to demonstrate USB serial MRI processing"""
    
    print("=== USB Serial MRI Data Processing ===\n")
    
    # Initialize system
    usb_ai = USBSymbolicNeuralAI(baudrate=115200, window_size=50)
    
    # Train vocabulary
    try:
        with open("test.txt", 'r', encoding='utf-8') as f:
            content = f.read()
        training_texts = [text.strip() for text in content.split(".") if text.strip()]
    except FileNotFoundError:
        print("test.txt file not found. Using default training data.")
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
    
    # List available ports
    print("Scanning for USB serial ports...")
    available_ports = usb_ai.mri_processor.usb_reader.list_available_ports()
    
    if not available_ports:
        print("No USB serial ports found!")
        print("Make sure your device is connected and drivers are installed.")
        return
    
    # Get port selection from user
    if len(available_ports) == 1:
        selected_port = available_ports[0]
        print(f"Using port: {selected_port}")
    else:
        print("\nSelect a port:")
        for i, port in enumerate(available_ports):
            print(f"{i+1}: {port}")
        
        try:
            choice = int(input("Enter port number: ")) - 1
            selected_port = available_ports[choice]
        except (ValueError, IndexError):
            selected_port = available_ports[0]
            print(f"Invalid selection, using: {selected_port}")
    
    print(f"\nExpecting numeric MRI data on {selected_port}")
    print("Data format: one number per line")
    print("Press Ctrl+C to stop\n")
    
    # Start real-time processing
    usb_ai.run_realtime_processing(selected_port)

if __name__ == "__main__":
    main()
