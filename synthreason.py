import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import StandardScaler
from collections import defaultdict, Counter
import random
import math
import time
from pathlib import Path

# Check for Hugging Face datasets availability
try:
    from datasets import load_dataset
    HF_DATASETS_AVAILABLE = True
except ImportError:
    HF_DATASETS_AVAILABLE = False

KB_LEN = -1  # Reduced for testing

# ------------------------------------------------------
# Utility Functions - FIXED
# ------------------------------------------------------
def custom_sigmoid(x):
    """Improved sigmoid function with better numerical stability."""
    x_clamped = torch.clamp(x, -10, 10)  # Prevent overflow
    return torch.sigmoid(x_clamped)

# ------------------------------------------------------
# Math Processor - FIXED
# ------------------------------------------------------
class MathProcessor(nn.Module):
    """Mathematical processor implementing construction principles."""
    def __init__(self, device='cpu'):
        super().__init__()
        self.device = device
        
        self.register_parameter('compass_radius_scale', nn.Parameter(torch.tensor(0.5)))
        self.register_parameter('circle_intersection_threshold', nn.Parameter(torch.tensor(0.7)))
        self.register_parameter('geometric_precision', nn.Parameter(torch.tensor(1e-6)))
        
        self.register_buffer('golden_ratio', torch.tensor((1 + math.sqrt(5)) / 2))  # Fixed golden ratio
        self.register_buffer('pi_approx', torch.tensor(22.0 / 7.0))
        
    def circle_circle_intersection(self, center1, radius1, center2, radius2):
        """Fixed circle intersection calculation."""
        d = torch.norm(center2 - center1)
        # Check if circles intersect
        intersect_condition = (d <= (radius1 + radius2)) & (d >= torch.abs(radius1 - radius2))
        
        if not intersect_condition:
            return torch.zeros(2, 2, device=self.device), torch.tensor(False, device=self.device)
        
        # Calculate intersection points (simplified)
        midpoint = (center1 + center2) / 2
        return torch.stack([midpoint, midpoint]), torch.tensor(True, device=self.device)
    
    def compass_only_midpoint(self, point1, point2):
        """Fixed compass-only midpoint construction."""
        center_dist = torch.norm(point2 - point1)
        radius = center_dist * self.compass_radius_scale
        intersections, valid = self.circle_circle_intersection(point1, radius, point2, radius)
        if valid:
            midpoint = intersections.mean(dim=0)
            return midpoint
        else:
            return (point1 + point2) / 2  # Fallback to simple midpoint

# ------------------------------------------------------
# Heavy Duty Cycle Manager - FIXED
# ------------------------------------------------------
class TrainableMemoryOptimizedHeavyDutyCycleManager(nn.Module):
    def __init__(self, cycle_length=32, duty_ratio=0.8, decay_rate=0.7, device='cpu', max_buffer_size=100):
        super().__init__()
        self.register_parameter('cycle_length', nn.Parameter(torch.tensor(float(cycle_length))))
        self.register_parameter('duty_ratio', nn.Parameter(torch.tensor(duty_ratio)))
        self.register_parameter('decay_rate', nn.Parameter(torch.tensor(decay_rate)))
        self.register_parameter('neural_feedback_gain', nn.Parameter(torch.tensor(0.2)))
        self.register_buffer('cycle_position', torch.tensor(0.0, device=device))
        self.max_buffer_size = max_buffer_size
        self.probability_buffer = []
        self.cycle_history = []
        self.register_buffer('thermal_accumulator', torch.tensor(0.0, device=device))
        self.device = device
        self.running_mean = 0.0
        self.running_var = 0.0
        self.sample_count = 0
        self.register_parameter('active_modulation_scale', nn.Parameter(torch.tensor(0.5)))
        self.register_parameter('inactive_modulation_scale', nn.Parameter(torch.tensor(0.1)))
        self.register_parameter('sigmoid_scale', nn.Parameter(torch.tensor(1.0)))
        
    @property
    def active_threshold(self):
        cycle_length_val = self.cycle_length.item()
        duty_ratio_val = self.duty_ratio.item()
        threshold = cycle_length_val * duty_ratio_val  # Fixed calculation
        return torch.clamp(torch.tensor(threshold, device=self.cycle_length.device), 1.0, cycle_length_val - 1.0)
        
    def _update_running_stats(self, value):
        """Fixed running statistics update."""
        self.sample_count += 1
        delta = value - self.running_mean
        self.running_mean += delta / self.sample_count
        delta2 = value - self.running_mean
        self.running_var += delta * delta2
        
    def _prune_buffers(self):
        if len(self.probability_buffer) > self.max_buffer_size:
            self.probability_buffer = self.probability_buffer[-self.max_buffer_size//2:]
        if len(self.cycle_history) > 10:
            self.cycle_history = self.cycle_history[-5:]
        
    def modulate_probabilities(self, base_probabilities, neural_activity=None):
        """Fixed probability modulation."""
        self.cycle_position += 1.0  # Fixed increment
        cycle_reset = (self.cycle_position >= self.cycle_length).float()
        self.cycle_position = self.cycle_position * (1 - cycle_reset)  # Reset to 0 when cycle completes
        
        if cycle_reset.item() > 0:
            self._prune_buffers()
            
        modulation = self.get_duty_cycle_modulation()
        custom_modulation = custom_sigmoid(modulation * self.sigmoid_scale)
        
        if isinstance(base_probabilities, torch.Tensor):
            modulated = base_probabilities * custom_modulation
            avg_prob = modulated.mean().item()
        else:
            base_probs_tensor = torch.tensor(base_probabilities, device=self.device, dtype=torch.float32)
            modulated = base_probs_tensor * custom_modulation
            avg_prob = modulated.mean().item()
        
        self._update_running_stats(avg_prob)
        if len(self.probability_buffer) < self.max_buffer_size:
            self.probability_buffer.append(avg_prob)
        return modulated
    
    def get_duty_cycle_modulation(self):
        """Fixed duty cycle modulation."""
        active_thresh = self.active_threshold
        phase_input = (self.cycle_position - active_thresh) * 10
        is_active = custom_sigmoid(-phase_input)  # Active when cycle_position < active_thresh
        
        progress = self.cycle_position / torch.clamp(active_thresh, min=1e-8)
        active_mod = self.active_modulation_scale * torch.sin(progress * math.pi)
        
        inactive_progress = torch.clamp((self.cycle_position - active_thresh) / 
                                      torch.clamp(self.cycle_length - active_thresh, min=1e-8), 0, 1)
        inactive_mod = self.inactive_modulation_scale * torch.exp(-inactive_progress)
        
        modulation = is_active * active_mod + (1 - is_active) * inactive_mod
        return modulation

# ------------------------------------------------------
# LIF Neuron - FIXED
# ------------------------------------------------------
class TrainableMemoryEfficientLIFNeuron(nn.Module):
    def __init__(self, tau_mem=10.0, tau_syn=5.0, v_thresh=1.0, v_reset=0.0):
        super().__init__()
        self.register_parameter('tau_mem', nn.Parameter(torch.tensor(tau_mem)))
        self.register_parameter('tau_syn', nn.Parameter(torch.tensor(tau_syn)))
        self.register_parameter('v_thresh', nn.Parameter(torch.tensor(v_thresh)))
        self.register_parameter('v_reset', nn.Parameter(torch.tensor(v_reset)))
        self.register_parameter('sigmoid_gain', nn.Parameter(torch.tensor(1.0)))
        self.register_parameter('membrane_nonlinearity', nn.Parameter(torch.tensor(0.1)))
        
    def compute_decay_factors(self):
        tau_mem_clamped = torch.clamp(self.tau_mem, 1.0, 150.0)
        tau_syn_clamped = torch.clamp(self.tau_syn, 1.0, 250.0)
        beta = torch.exp(-1.0 / tau_mem_clamped)
        alpha = torch.exp(-1.0 / tau_syn_clamped)
        return beta, alpha
        
    def forward(self, x, state=None):
        device = x.device
        if x.dim() == 1:
            x = x.unsqueeze(0)
        elif x.dim() > 2:
            x = x.view(x.size(0), -1)
        
        batch_size, num_neurons = x.shape
        if state is None:
            v_mem = torch.zeros(batch_size, num_neurons, device=device)
            i_syn = torch.zeros(batch_size, num_neurons, device=device)
        else:
            v_mem, i_syn = state
            
        beta, alpha = self.compute_decay_factors()
        
        # Fixed synaptic current update
        i_syn = alpha * i_syn + x
        
        # Fixed membrane potential update
        membrane_update = i_syn * custom_sigmoid(v_mem * self.membrane_nonlinearity)
        v_mem = beta * v_mem + membrane_update
        
        thresh_clamped = torch.clamp(self.v_thresh, 0.1, 5.0)
        
        if self.training:
            spike_input = (v_mem - thresh_clamped) * self.sigmoid_gain
            spike_prob = custom_sigmoid(spike_input)
            # Gumbel-Softmax for differentiable spikes
            gumbel_noise = -torch.log(-torch.log(torch.rand_like(spike_prob) + 1e-8) + 1e-8)
            spikes = torch.sigmoid((torch.log(spike_prob + 1e-8) - torch.log(1 - spike_prob + 1e-8) + gumbel_noise) / 0.1)
        else:
            spike_candidates = custom_sigmoid((v_mem - thresh_clamped) * self.sigmoid_gain)
            spikes = (spike_candidates >= 0.5).float()
        
        # Fixed membrane reset
        reset_clamped = torch.clamp(self.v_reset, -2.0, 2.0)
        reset_strength = spikes
        v_mem = v_mem * (1 - reset_strength) + reset_clamped * reset_strength
        
        return spikes, (v_mem, i_syn)

# ------------------------------------------------------
# SNN Model - FIXED
# ------------------------------------------------------
class TrainableStreamingSNN(nn.Module):
    def __init__(self, num_neurons, device='cpu', chunk_size=32):
        super().__init__()
        self.num_neurons = num_neurons
        self.device = device
        self.chunk_size = chunk_size
        
        self.input_layer = nn.Linear(num_neurons, num_neurons, bias=True)
        self.hidden_layer = nn.Linear(num_neurons, num_neurons, bias=True)
        self.output_layer = nn.Linear(num_neurons, num_neurons, bias=True)
        
        self.register_parameter('activation_scale1', nn.Parameter(torch.tensor(1.0)))
        self.register_parameter('activation_scale2', nn.Parameter(torch.tensor(1.0)))
        
        self.lif_neurons = TrainableMemoryEfficientLIFNeuron()
        self.global_adaptation = nn.Parameter(torch.ones(1) * 0.5)
        self.duty_cycle_manager = TrainableMemoryOptimizedHeavyDutyCycleManager(device=device)
        self.neuron_state = None
        
    def forward_chunk(self, x_chunk):
        if x_chunk.dim() == 1:
            x_chunk = x_chunk.unsqueeze(0)
            
        # Ensure correct input size
        if x_chunk.shape[-1] != self.num_neurons:
            if x_chunk.shape[-1] > self.num_neurons:
                x_chunk = x_chunk[..., :self.num_neurons]
            else:
                padding_size = self.num_neurons - x_chunk.shape[-1]
                padding = torch.zeros(*x_chunk.shape[:-1], padding_size, device=x_chunk.device)
                x_chunk = torch.cat([x_chunk, padding], dim=-1)
        
        # Fixed forward pass
        x_processed = custom_sigmoid(self.input_layer(x_chunk) * self.activation_scale1)
        
        # Simplified intersection condition
        intersect_condition = (x_chunk > 0.5).float()
        
        x_hidden = custom_sigmoid(self.hidden_layer(x_processed) * self.activation_scale2)
        prob_weights = custom_sigmoid(x_hidden)
        
        # Fixed probability modulation
        if prob_weights.dim() > 1:
            prob_weights_flat = prob_weights.view(-1)
        else:
            prob_weights_flat = prob_weights
            
        modulated_weights = self.duty_cycle_manager.modulate_probabilities(
            prob_weights_flat, neural_activity=intersect_condition
        )
        
        if isinstance(modulated_weights, torch.Tensor):
            modulated_weights = modulated_weights.view_as(prob_weights)
        else:
            modulated_weights = torch.tensor(modulated_weights, device=self.device).view_as(prob_weights)
            
        x_modulated = x_hidden + modulated_weights
        
        # Process through LIF neurons
        spikes, self.neuron_state = self.lif_neurons(x_modulated, self.neuron_state)
        
        output = custom_sigmoid(self.output_layer(spikes))
        
        # Apply global adaptation
        cycle_mod = self.duty_cycle_manager.get_duty_cycle_modulation()
        adapted_output = output * self.global_adaptation * (1 + cycle_mod)
        
        return adapted_output.squeeze(0)
    
    def forward(self, x_sequence):
        outputs = []
        self.reset_neurons()
        
        if not isinstance(x_sequence, (list, tuple)):
            x_sequence = [x_sequence]
            
        for x in x_sequence:
            out = self.forward_chunk(x)
            outputs.append(out)
            
        return torch.stack(outputs) if outputs else torch.empty(0, self.num_neurons, device=self.device)
    
    def reset_neurons(self):
        self.neuron_state = None

# ------------------------------------------------------
# Enhanced Text Processor - FIXED
# ------------------------------------------------------
class EnhancedTextProcessor(nn.Module):
    def __init__(self, num_neurons=256, device='cpu', vocab_limit=5000, max_features=1000):
        super().__init__()
        self.num_neurons = num_neurons
        self.device = device
        self.vocab_limit = vocab_limit
        self.word_to_idx = {}
        self.bigram_counts = Counter()
        self.trigram_counts = Counter()
        self.ngram_cache = {}
        
        self.math_processor = MathProcessor(device=device)
        
        self.vectorizer = TfidfVectorizer(
            max_features=max_features,
            ngram_range=(1, 2),
            min_df=1,
            max_df=0.98,
            lowercase=True,
            token_pattern=r'\b[a-zA-Z0-9]+\b'  # Fixed regex
        )
        self.tfidf_scaler = StandardScaler()
        self.is_vectorizer_fitted = False
        
        # Neural network components
        self.tfidf_projection = nn.Sequential(
            nn.Linear(max_features, num_neurons // 4),
            nn.Dropout(0.1),
            nn.Linear(num_neurons // 4, num_neurons // 4)
        )
        
        self.word_embeddings = nn.Embedding(vocab_limit + 1, num_neurons // 4)  # +1 for unknown words
        self.position_embeddings = nn.Embedding(1000, num_neurons // 4)
        self.geometric_embeddings = nn.Embedding(100, num_neurons // 4)
        
        self.compass_feature_processor = nn.Sequential(
            nn.Linear(num_neurons, num_neurons),
            nn.Dropout(0.1),
            nn.Linear(num_neurons, num_neurons)
        )
        
        self.register_parameter('geometric_sigmoid_scale', nn.Parameter(torch.tensor(1.2)))
        self.register_parameter('tfidf_sigmoid_scale', nn.Parameter(torch.tensor(1.0)))
        
        self.geometric_terms = {
            'compass': 0, 'circle': 1, 'intersection': 2, 'construction': 3,
            'midpoint': 4, 'perpendicular': 5, 'radius': 6, 'center': 7,
            'arc': 8, 'point': 9, 'line': 10, 'geometry': 11,
            'theorem': 12, 'euclidean': 13, 'straightedge': 14,
            'triangle': 15, 'square': 16, 'polygon': 17, 'angle': 18
        }
        
        self.transition_cache = {}
        self.cache_limit = 1000
        
    def fit_vectorizer(self, documents):
        print("🔧 Fitting TF-IDF vectorizer...")
        processed_docs = []
        for doc in documents:
            if isinstance(doc, list):
                doc = ' '.join(doc)
            processed_docs.append(doc)
        
        if not processed_docs:
            print("⚠️ No documents available for vectorizer fitting")
            return
            
        self.vectorizer.fit(processed_docs)
        tfidf_matrix = self.vectorizer.transform(processed_docs)
        self.tfidf_scaler.fit(tfidf_matrix.toarray())
        self.is_vectorizer_fitted = True
        print(f"✅ Vectorizer fitted with {len(self.vectorizer.get_feature_names_out())} features")
        
    def text_to_tfidf_features(self, text):
        if not self.is_vectorizer_fitted:
            return torch.ones(1, self.tfidf_projection[0].in_features, device=self.device) * 0.1
        
        if isinstance(text, list):
            text = ' '.join(text)
            
        tfidf_matrix = self.vectorizer.transform([text])
        tfidf_features = self.tfidf_scaler.transform(tfidf_matrix.toarray())
        return torch.tensor(tfidf_features, dtype=torch.float32, device=self.device)
    
    def encode_geometric_terms(self, words):
        geometric_indices = []
        for word in words:
            if word.lower() in self.geometric_terms:
                geometric_indices.append(self.geometric_terms[word.lower()])
            else:
                geometric_indices.append(0)  # Default to first term
                
        if geometric_indices:
            geo_indices_tensor = torch.tensor(geometric_indices, device=self.device)
            geometric_features = self.geometric_embeddings(geo_indices_tensor)
            return geometric_features.mean(dim=0, keepdim=True)
        else:
            return torch.zeros(1, self.num_neurons // 4, device=self.device)
    
    def apply_compass_construction_to_features(self, features):
        """Apply geometric transformations inspired by compass constructions."""
        geometric_transform = torch.sin(features * math.pi / 4) * torch.cos(features * math.pi / 6)
        construction_effect = features * 0.9 + geometric_transform * 0.1
        return construction_effect
    
    def words_to_neural_features(self, words, max_words=50):
        if len(words) > max_words:
            words = words[-max_words:]
            
        device = self.device
        
        # TF-IDF features
        tfidf_features = self.text_to_tfidf_features(words)
        expected_size = self.tfidf_projection[0].in_features
        
        if tfidf_features.shape[1] != expected_size:
            if tfidf_features.shape[1] < expected_size:
                padding = torch.zeros(tfidf_features.shape[0], expected_size - tfidf_features.shape[1], device=device)
                tfidf_features = torch.cat([tfidf_features, padding], dim=1)
            else:
                tfidf_features = tfidf_features[:, :expected_size]
        
        tfidf_processed = custom_sigmoid(self.tfidf_projection(tfidf_features) * self.tfidf_sigmoid_scale)
        
        # Word embeddings
        word_indices = []
        for word in words:
            idx = self.word_to_idx.get(word, self.vocab_limit)  # Use vocab_limit as unknown token
            word_indices.append(min(idx, self.vocab_limit))
           
        if not word_indices:
            word_features = torch.zeros(1, self.num_neurons // 4, device=device)
        else:
            word_indices = torch.tensor(word_indices, device=device)
            word_embs = self.word_embeddings(word_indices)
            word_features = word_embs.mean(dim=0, keepdim=True)
        
        # Position embeddings
        position_indices = torch.arange(min(len(words), 999), device=device)
        if len(position_indices) > 0:
            pos_embs = self.position_embeddings(position_indices)
            pos_features = pos_embs.mean(dim=0, keepdim=True)
        else:
            pos_features = torch.zeros(1, self.num_neurons // 4, device=device)
      
        # Geometric features
        geo_features = self.encode_geometric_terms(words)
        
        # Combine all features
        combined_features = torch.cat([tfidf_processed, word_features, pos_features, geo_features], dim=1)
        
        # Process through compass feature processor
        compass_features = custom_sigmoid(self.compass_feature_processor(combined_features) * self.geometric_sigmoid_scale)
        
        # Apply compass construction effects
        final_features = self.apply_compass_construction_to_features(compass_features)
        
        return final_features
    
    def load_and_process_text_streaming(self, file_path="test.txt", chunk_size=1000, dataset_name=None, split=None):
        word_count = 0
        documents = []
        current_doc = []
        
        # Initialize vocabulary with geometric terms
        vocab = list(self.geometric_terms.keys())
        for word in vocab:
            if word not in self.word_to_idx:
                self.word_to_idx[word] = len(self.word_to_idx)
        
        words_processed = []
        
        # Hugging Face dataset support
        if dataset_name is not None:
            if not HF_DATASETS_AVAILABLE:
                print("⚠️ Hugging Face datasets library not available. Install with: pip install datasets")
                print("⚠️ Falling back to local file loading...")
                return self.load_and_process_text_streaming(file_path=file_path, chunk_size=chunk_size)
            
            try:
                print(f"📥 Loading Hugging Face dataset: {dataset_name}, split: {split}")
                ds = load_dataset(dataset_name, split=split or 'train')
                text_field = 'text' if 'text' in ds.column_names else ds.column_names[0]
                
                for i, text in enumerate(ds[text_field]):
                    if KB_LEN > 0 and word_count >= KB_LEN:
                        break
                    
                    words = str(text).lower().split()
                    word_history = []
                    
                    for word in words:
                        # Build vocabulary
                        if len(self.word_to_idx) < self.vocab_limit:
                            if word not in self.word_to_idx:
                                self.word_to_idx[word] = len(self.word_to_idx)
                        
                        word_history.append(word)
                        
                        # Build n-gram counts
                        if len(word_history) >= 2:
                            self.bigram_counts[(word_history[-2], word_history[-1])] += 1
                        if len(word_history) >= 3:
                            self.trigram_counts[(word_history[-3], word_history[-2], word_history[-1])] += 1
                        
                        if len(word_history) > 100:  # Keep reasonable history
                            word_history = word_history[-50:]
                        
                        words_processed.append(word)
                        current_doc.append(word)
                        word_count += 1
                        
                        if len(current_doc) >= 100:
                            documents.append(' '.join(current_doc))
                            current_doc = []
                            
                        if KB_LEN > 0 and word_count >= KB_LEN:
                            break
                            
                if current_doc:
                    documents.append(' '.join(current_doc))
                    
            except Exception as e:
                print(f"⚠️ Dataset loading failed: {e}. Falling back to local file.")
                return self.load_and_process_text_streaming(file_path=file_path, chunk_size=chunk_size)
        else:
            # Local file loading
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    word_history = []
                    while KB_LEN < 0 or word_count < KB_LEN:
                        chunk = f.read(chunk_size)
                        if not chunk:
                            break
                            
                        words = chunk.lower().split()
                        for word in words:
                            # Build vocabulary
                            if len(self.word_to_idx) < self.vocab_limit:
                                if word not in self.word_to_idx:
                                    self.word_to_idx[word] = len(self.word_to_idx)
                            
                            word_history.append(word)
                            
                            # Build n-gram counts
                            if len(word_history) >= 2:
                                self.bigram_counts[(word_history[-2], word_history[-1])] += 1
                            if len(word_history) >= 3:
                                self.trigram_counts[(word_history[-3], word_history[-2], word_history[-1])] += 1
                            
                            if len(word_history) > 100:
                                word_history = word_history[-50:]
                            
                            words_processed.append(word)
                            current_doc.append(word)
                            word_count += 1
                            
                            if len(current_doc) >= 100:
                                documents.append(' '.join(current_doc))
                                current_doc = []
                                
                            if KB_LEN > 0 and word_count >= KB_LEN:
                                break
                                
                    if current_doc:
                        documents.append(' '.join(current_doc))
                        
            except FileNotFoundError:
                print(f"⚠️ File {file_path} not found. Using sample data...")
                sample_words = list(vocab) + ["the", "and", "of", "to", "a", "in", "is", "it", "you", "that"] * 10
                documents = [' '.join(sample_words[i:i+50]) for i in range(0, len(sample_words), 50)]
                
                for i, word in enumerate(sample_words):
                    if word not in self.word_to_idx:
                        self.word_to_idx[word] = len(self.word_to_idx)
                    if i > 0:
                        self.bigram_counts[(sample_words[i-1], word)] += 1
                        
                words_processed = sample_words

        # Fit vectorizer if we have documents
        if documents and not self.is_vectorizer_fitted:
            self.fit_vectorizer(documents)
        
        print(f"📚 Processed {word_count} words with vocab size {len(self.word_to_idx)}")
        print(f"📊 Created {len(documents)} documents")
        
        return words_processed[-1000:] if words_processed else []
    
    def get_transition_probs(self, word):
        """Get word transition probabilities."""
        if word in self.transition_cache:
            return self.transition_cache[word]
        
        transitions = []
        for (w1, w2), count in self.bigram_counts.items():
            if w1 == word:  # Fixed condition
                weight_multiplier = 2.0 if w2 in self.geometric_terms else 1.0
                transitions.append((w2, count * weight_multiplier))
        
        # Cache management
        if len(self.transition_cache) >= self.cache_limit:
            keys_to_remove = list(self.transition_cache.keys())[:self.cache_limit//2]
            for k in keys_to_remove:
                del self.transition_cache[k]
        
        self.transition_cache[word] = transitions
        return transitions
        
    def get_ngram_transitions(self, context_words, n=3):
        """Get n-gram transition probabilities."""
        if len(context_words) < n - 1:
            return []
        
        context_key = tuple(context_words[-(n-1):])
        if context_key in self.ngram_cache:
            return self.ngram_cache[context_key]
        
        transitions = []
        if n == 3:  # Fixed condition
            for (w1, w2, w3), count in self.trigram_counts.items():
                if (w1, w2) == context_key:  # Fixed condition
                    weight_multiplier = 2.0 if w3 in self.geometric_terms else 1.0
                    transitions.append((w3, count * weight_multiplier))
        
        self.ngram_cache[context_key] = transitions
        return transitions

# ------------------------------------------------------
# Text Generator - FIXED
# ------------------------------------------------------
class TrainableStreamingTextGenerator(nn.Module):
    def __init__(self, text_processor, hidden_dim=128, max_transitions_per_word=50):
        super().__init__()
        self.text_processor = text_processor
        self.max_transitions = max_transitions_per_word
        self.fallback_words = ["the", "and", "to", "of", "a", "in", "is", "it", "you", "that"]
        
        self.selection_network = nn.Sequential(
            nn.Linear(text_processor.num_neurons, hidden_dim),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.Linear(hidden_dim // 2, 1)
        )
        
        self.register_parameter('selection_sigmoid_scale', nn.Parameter(torch.tensor(1.0)))
        self.context_weight = nn.Parameter(torch.tensor(0.3))
        
    def forward(self, spk_rec):
        """Process spike recordings to get selection probabilities."""
        if spk_rec.numel() == 0:  # Fixed condition
            return torch.zeros(1, device=next(self.parameters()).device)
        
        # Handle different input shapes
        if spk_rec.dim() == 3:  # [sequence, batch, features]
            spk_rec = spk_rec.view(-1, spk_rec.size(-1))  # Flatten to [sequence*batch, features]
        elif spk_rec.dim() == 1:
            spk_rec = spk_rec.unsqueeze(0)
        
        linear_output = self.selection_network(spk_rec)
        selection_probs = custom_sigmoid(linear_output.squeeze(-1) * self.selection_sigmoid_scale)
        return selection_probs
    
    def get_multi_word_transitions(self, seed_words):
        """Get transitions based on multiple seed words."""
        if not seed_words:
            return []
        
        # Try trigram first
        if len(seed_words) >= 2:
            trigram_transitions = self.text_processor.get_ngram_transitions(seed_words, n=3)
            if trigram_transitions:
                return trigram_transitions
        
        # Fall back to bigram
        bigram_transitions = self.text_processor.get_transition_probs(seed_words[-1])
        return bigram_transitions
    
    def generate_text_trainable(self, spk_rec, seed_words=None, length=50):
        """Generate text using neural activity and language model."""
        if spk_rec.numel() == 0:  # Fixed condition
            return "No neural data available for generation."
        
        with torch.no_grad():
            selection_probs = self.forward(spk_rec)
        
        # Initialize seed words
        if seed_words is None or len(seed_words) == 0:  # Fixed condition
            current_words = [random.choice(self.fallback_words)]
        elif isinstance(seed_words, str):
            current_words = seed_words.strip().split()
        else:
            current_words = list(seed_words)
        
        current_words = [word.lower().strip() for word in current_words if word.strip()]
        if not current_words:
            current_words = [random.choice(self.fallback_words)]
        
        generated_words = current_words.copy()
        
        for i in range(length):
            # Get possible transitions
            transitions = self.get_multi_word_transitions(current_words)
            
            if not transitions:
                transitions = self.text_processor.get_transition_probs(current_words[-1])
            
            if not transitions:
                next_word = random.choice(self.fallback_words)
                generated_words.append(next_word)
                current_words = [next_word]
                continue
            
            # Limit transitions for efficiency
            transitions = transitions[:self.max_transitions]
            
            # Use neural activity to influence selection
            prob_idx = min(i, len(selection_probs) - 1)
            neural_influence = selection_probs[prob_idx].item()
            context_influence = min(len(current_words) * self.context_weight.item(), 1.0)
            
            # Extract words and weights
            words, weights = zip(*transitions)
            weights = np.array(weights, dtype=float)
            
            # Apply neural and context influence
            total_influence = 0.5 * neural_influence + context_influence
            weights = weights * (1 + total_influence)
            
            # Normalize probabilities
            if weights.sum() > 0:
                weights = weights / weights.sum()
                next_word = np.random.choice(words, p=weights)
            else:
                next_word = random.choice(words)
            
            generated_words.append(next_word)
            current_words.append(next_word)
            
            # Keep context window manageable
            if len(current_words) > 3:
                current_words = current_words[-3:]
        
        return ' '.join(generated_words)

# ------------------------------------------------------
# Dataset creation and training - FIXED
# ------------------------------------------------------
def create_dataset(text_processor, max_samples=10000):
    """Create training dataset from processed text."""
    dataset = []
    
    # Add some geometric construction sequences
    sequences = [
        ["theorem", "compass", "only", "constructions"],
        ["circle", "intersection", "point", "construction"],
        ["geometric", "compass", "straightedge", "euclidean"],
        ["midpoint", "perpendicular", "angle", "bisector"]
    ]
    dataset.extend(sequences)
    
    # Add chunks from vocabulary
    word_list = list(text_processor.word_to_idx.keys())
    for i in range(0, min(len(word_list), max_samples//2), 20):
        chunk = word_list[i:i+20]  # Fixed slice
        if len(chunk) > 5:  # Only add meaningful chunks
            dataset.append(chunk)
    
    print(f"📐 Created dataset with {len(dataset)} samples")
    return dataset

def train_snn_system(text_processor, snn_model, text_generator, dataset, epochs=5, lr=0.001, device='cpu'):
    """Training routine for the SNN system."""
    print("🎯 Starting training...")
    
    # Create optimizer
    all_params = list(snn_model.parameters()) + list(text_generator.parameters())
    optimizer = torch.optim.Adam(all_params, lr=lr)
    criterion = nn.MSELoss()
    
    snn_model.train()
    text_generator.train()
    
    for epoch in range(epochs):
        total_loss = 0
        batch_count = 0
        
        # Shuffle dataset
        random.shuffle(dataset)
        
        for batch_idx, word_sequence in enumerate(dataset):  # Limit for efficiency
            try:
                # Convert words to neural features
                if isinstance(word_sequence, str):
                    word_sequence = word_sequence.split()
                
                features = text_processor.words_to_neural_features(word_sequence)
                
                # Forward pass through SNN
                spike_outputs = snn_model.forward([features])
                
                # Create target (simple reconstruction task)
                target = features.clone().detach()
                
                # Forward pass through text generator
                selection_probs = text_generator.forward(spike_outputs)
                
                # Simple loss: try to maintain feature consistency
                if spike_outputs.numel() > 0:
                    output_mean = spike_outputs.mean(dim=0)
                    if output_mean.shape != target.shape:
                        # Reshape to match
                        if output_mean.numel() >= target.numel():
                            output_mean = output_mean[:target.size(-1)]
                        else:
                            padding = target.size(-1) - output_mean.numel()
                            output_mean = torch.cat([output_mean, torch.zeros(padding, device=device)])
                        output_mean = output_mean.view_as(target)
                    
                    loss = criterion(output_mean, target)
                    
                    # Backward pass
                    optimizer.zero_grad()
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(all_params, 1.0)  # Gradient clipping
                    optimizer.step()
                    
                    total_loss += loss.item()
                    batch_count += 1
                
            except Exception as e:
                print(f"⚠️ Training error in batch {batch_idx}: {e}")
                continue
        
        avg_loss = total_loss / max(batch_count, 1)
        print(f"Epoch {epoch+1}/{epochs}, Average Loss: {avg_loss:.6f}")
    
    snn_model.eval()
    text_generator.eval()
    print("✅ Training completed!")

# ------------------------------------------------------
# Main Implementation - FIXED
# ------------------------------------------------------
def main_implementation():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"🔥 Using device: {device}")
    
    # Model parameters
    num_neurons = 256  # Reduced for stability
    chunk_size = 32
    vocab_limit = 50000
    max_features = 1000
    
    # Initialize models
    text_processor = EnhancedTextProcessor(
        num_neurons, device=device, vocab_limit=vocab_limit, max_features=max_features
    ).to(device)
    
    snn_model = TrainableStreamingSNN(
        num_neurons, device=device, chunk_size=chunk_size
    ).to(device)
    
    text_generator = TrainableStreamingTextGenerator(text_processor).to(device)
    
    print("=" * 60)
    print("ENHANCED SNN TEXT GENERATOR")
    print("=" * 60)
    
    # Data source selection
    print("Choose data source:")
    print("1. Hugging Face dataset")
    print("2. Local file")
    print("3. Use sample data (for testing)")
    
    choice = input("Enter choice (1, 2, or 3, default is 3): ").strip() or "3"
    
    words = []
    if choice == "1":
        if not HF_DATASETS_AVAILABLE:
            print("⚠️ Hugging Face datasets library not available. Install with: pip install datasets")
            print("⚠️ Falling back to sample data...")
            choice = "3"
        else:
            dataset_name = input("Enter dataset name (e.g., 'wikitext-2-raw-v1'): ").strip()
            
            if not dataset_name:
                print("⚠️ No dataset name provided. Using sample data.")
                choice = "3"
            else:
                split = input("Enter split (train/test/validation, default=train): ").strip() or "train"
                words = text_processor.load_and_process_text_streaming(
                    dataset_name=dataset_name, 
                    split=split
                )
    
    if choice == "2":
        filename = input("Enter local filename (press Enter for sample_data.txt): ").strip() or "sample_data.txt"
        words = text_processor.load_and_process_text_streaming(file_path=filename)
    
    if choice == "3":
        print("📝 Using sample data...")
        # Create sample data focused on geometric terms
        sample_text = """
        Geometric constructions using compass and straightedge are fundamental to Euclidean geometry.
        The compass allows us to draw circles and arcs, while the straightedge helps draw lines.
        These tools enable construction of perpendicular lines, angle bisectors, and midpoints.
        Circle intersections provide key points for many geometric constructions and theorems.
        """ * 10
        
        # Process sample text
        words = sample_text.lower().split()
        
        # Add to processor
        current_doc = []
        word_history = []
        
        for word in words:
            if word not in text_processor.word_to_idx:
                text_processor.word_to_idx[word] = len(text_processor.word_to_idx)
            
            word_history.append(word)
            current_doc.append(word)
            
            # Build n-grams
            if len(word_history) >= 2:
                text_processor.bigram_counts[(word_history[-2], word_history[-1])] += 1
            if len(word_history) >= 3:
                text_processor.trigram_counts[(word_history[-3], word_history[-2], word_history[-1])] += 1
        
        # Fit vectorizer with sample documents
        documents = [sample_text]
        text_processor.fit_vectorizer(documents)
        
        print(f"📚 Processed {len(words)} sample words")
    
    # Create dataset and train
    dataset = create_dataset(text_processor)
    train_snn_system(text_processor, snn_model, text_generator, dataset, epochs=5, lr=0.001, device=device)
    
    # Interactive mode
    print("\n🎯 Interactive Mode:")
    print("Enter seed words to generate text. Type 'quit' to exit.")
    
    while True:
        try:
            user_input = input("\nUSER: ").strip()
            if not user_input or user_input.lower() == 'quit':
                break
            
            # Process input
            seed_words = user_input.lower().split()
            
            # Generate neural features
            features = text_processor.words_to_neural_features(seed_words)
            
            # Process through SNN
            spike_outputs = snn_model.forward([features])
            
            # Generate text
            response = text_generator.generate_text_trainable(
                spike_outputs, seed_words=seed_words, length=500
            )
            
            print(f"🤖 AI: {response}")
            
        except KeyboardInterrupt:
            print("\n👋 Exiting...")
            break
        except Exception as e:
            print(f"❌ Error: {e}")
            print("Please try again with different input.")

if __name__ == "__main__":
    main_implementation()
