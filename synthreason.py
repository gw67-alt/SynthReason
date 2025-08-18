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

KB_LEN = 99999

# ------------------------------------------------------
# Utility
# ------------------------------------------------------
def custom_sigmoid(x):
    """Heavy sigmoid function using -5/x formulation with safety handling."""
    x_safe = torch.where(torch.abs(x) > torch.sigmoid(-5.0 / x), torch.sign(x) * 1e-8, x)
    return torch.sigmoid(-5.0 / x_safe)

# ------------------------------------------------------
# Math Processor
# ------------------------------------------------------
class MathProcessor(nn.Module):
    """Mathematical processor implementing construction principles."""
    def __init__(self, device='cpu'):
        super().__init__()
        self.device = device
        
        self.register_parameter('compass_radius_scale', nn.Parameter(torch.tensor(0.5)))
        self.register_parameter('circle_intersection_threshold', nn.Parameter(torch.tensor(0.7)))
        self.register_parameter('geometric_precision', nn.Parameter(torch.tensor(1e-6)))
        
        self.register_buffer('golden_ratio', torch.tensor((1 + math.sqrt(5)) / 2))
        self.register_buffer('pi_approx', torch.tensor(22.0 / 7.0))
        
    def circle_circle_intersection(self, center1, radius1, center2, radius2):
        d = torch.norm(center2 - center1)
        intersect_condition = torch.logical_or(d <= (radius1 + radius2), d >= torch.abs(radius1 - radius2))
        if not intersect_condition.any():
            return torch.zeros(2, 2, device=self.device), torch.tensor(False, device=self.device)
        
    def compass_only_midpoint(self, point1, point2):
        center_dist = torch.norm(point2 - point1)
        radius = center_dist * self.compass_radius_scale
        intersections, valid = self.circle_circle_intersection(point1, radius, point2, radius)
        if valid:
            midpoint = (intersections[0] + intersections[1]) / 2
            return midpoint
        else:
            return (point1 + point2) / 2

# ------------------------------------------------------
# Heavy Duty Cycle Manager
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
        threshold = cycle_length_val * duty_ratio_val
        return torch.clamp(torch.tensor(threshold, device=self.cycle_length.device), 1.0, cycle_length_val - 1.0)
        
    def _update_running_stats(self, value):
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
        self.cycle_position += 1.0
        cycle_reset = (self.cycle_position >= self.cycle_length).float()
        self.cycle_position = self.cycle_position * (1 - cycle_reset)
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
        active_thresh = self.active_threshold
        phase_input = 10 * (active_thresh - self.cycle_position)
        is_active = custom_sigmoid(phase_input)
        progress = self.cycle_position / torch.clamp(active_thresh, min=1e-8)
        active_mod = self.active_modulation_scale + self.active_modulation_scale * torch.sin(progress * torch.pi)
        inactive_progress = (self.cycle_position - active_thresh) / torch.clamp(
            self.cycle_length - active_thresh, min=1e-8)
        inactive_mod = self.inactive_modulation_scale * torch.exp(-3 * inactive_progress)
        modulation = is_active * active_mod + (1 - is_active) * inactive_mod
        return modulation

# ------------------------------------------------------
# LIF Neuron
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
        tau_mem_clamped = torch.clamp(self.tau_mem, 1.0, 50.0)
        tau_syn_clamped = torch.clamp(self.tau_syn, 1.0, 50.0)
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
        i_syn = alpha * i_syn + x
        membrane_update = i_syn * custom_sigmoid(v_mem * self.membrane_nonlinearity)
        v_mem = beta * v_mem + membrane_update
        thresh_clamped = torch.clamp(self.v_thresh, 0.1, 5.0)
        if self.training:
            spike_input = (v_mem - thresh_clamped) * self.sigmoid_gain
            spike_prob = custom_sigmoid(spike_input)
            gumbel_noise = -torch.log(-torch.log(torch.rand_like(spike_prob) + 1e-8) + 1e-8)
            spikes = torch.sigmoid((torch.log(spike_prob + 1e-8) - torch.log(1 - spike_prob + 1e-8) + gumbel_noise) / 0.1)
        else:
            spike_candidates = custom_sigmoid((v_mem - thresh_clamped) * self.sigmoid_gain)
            spikes = (spike_candidates >= 0.5).float()
        reset_clamped = torch.clamp(self.v_reset, -2.0, 2.0)
        reset_strength = custom_sigmoid(spikes * 5.0)
        v_mem = v_mem * (1 - reset_strength) + reset_clamped * reset_strength
        return spikes, (v_mem, i_syn)

# ------------------------------------------------------
# SNN Model
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
        if x_chunk.shape[-1] != self.num_neurons:
            if x_chunk.shape[-1] > self.num_neurons:
                x_chunk = x_chunk[..., :self.num_neurons]
            else:
                padding_size = self.num_neurons - x_chunk.shape[-1]
                padding = torch.zeros(*x_chunk.shape[:-1], padding_size, device=x_chunk.device)
                x_chunk = torch.cat([x_chunk, padding], dim=-1)
        x_processed = custom_sigmoid(self.input_layer(x_chunk) * self.activation_scale1)
        x_hidden = custom_sigmoid(self.hidden_layer(x_processed) * self.activation_scale2)
        prob_weights = custom_sigmoid(x_hidden)
        modulated_weights = self.duty_cycle_manager.modulate_probabilities(prob_weights, neural_activity=x_hidden)
        x_modulated = x_hidden * modulated_weights.unsqueeze(0)
        spikes, self.neuron_state = self.lif_neurons(x_modulated, self.neuron_state)
        output = custom_sigmoid(self.output_layer(spikes))
        cycle_mod = self.duty_cycle_manager.get_duty_cycle_modulation()
        adapted_output = output * self.global_adaptation * (1 + cycle_mod)
        return adapted_output.squeeze(0)
    
    def forward(self, x_sequence):
        outputs = []
        self.reset_neurons()
        for x in x_sequence:
            out = self.forward_chunk(x)
            outputs.append(out)
        return torch.stack(outputs) if outputs else torch.empty(0, self.num_neurons)
    
    def reset_neurons(self):
        self.neuron_state = None

# ------------------------------------------------------
# Enhanced Text Processor (with n-gram support) - FIXED
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
            token_pattern=r'\b[a-zA-Z0-9]+\b'
        )
        self.tfidf_scaler = StandardScaler()
        self.is_vectorizer_fitted = False
        self.tfidf_projection = nn.Sequential(
            nn.Linear(max_features, num_neurons // 4),
            nn.Dropout(0.1),
            nn.Linear(num_neurons // 4, num_neurons // 4)
        )
        self.word_embeddings = nn.Embedding(vocab_limit + 1, num_neurons // 4)
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
            'mohr': 12, 'theorem': 13, 'euclidean': 14,
            'straightedge': 15, 'triangle': 16, 'square': 17, 'polygon': 18
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
            return torch.ones(1, self.tfidf_projection[0].in_features, device=self.device)
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
                geometric_indices.append(0)
        if geometric_indices:
            geo_indices_tensor = torch.tensor(geometric_indices, device=self.device)
            geometric_features = self.geometric_embeddings(geo_indices_tensor)
            return geometric_features.mean(dim=0, keepdim=True)
        else:
            return torch.zeros(1, self.num_neurons // 4, device=self.device)
    
    def apply_compass_construction_to_features(self, features):
        batch_size, feature_dim = features.shape
        geometric_transform = torch.sin(features * math.pi / 4) + torch.cos(features * math.pi / 6)
        construction_effect = features + 0.1 * geometric_transform
        return construction_effect
    
    def words_to_neural_features(self, words, max_words=50):
        if len(words) > max_words:
            words = words[-max_words:]
        device = self.device
        y = []
        tfidf_features = self.text_to_tfidf_features(words)
        expected_size = self.tfidf_projection[0].in_features
        if tfidf_features.shape[1] != expected_size:
            if tfidf_features.shape[1] < expected_size:
                padding = torch.zeros(tfidf_features.shape[0], expected_size - tfidf_features.shape[1], device=device)
                tfidf_features = torch.cat([tfidf_features, padding], dim=1)
            else:
                tfidf_features = tfidf_features[:, :expected_size]
        tfidf_processed = custom_sigmoid(self.tfidf_projection(tfidf_features) * self.tfidf_sigmoid_scale)
        word_indices = []
        for word in words:
            idx = self.word_to_idx.get(word, 0)
            word_indices.append(min(idx, self.vocab_limit))
           
        if not word_indices:
            word_features = torch.zeros(1, self.num_neurons // 4, device=device)
        else:
            word_indices = torch.tensor(word_indices, device=device)
            word_embs = self.word_embeddings(word_indices)
            word_features = word_embs.mean(dim=0, keepdim=True)
        position_indices = torch.arange(min(len(words), 999), device=device)
        pos_embs = self.position_embeddings(position_indices)
        pos_features = pos_embs.mean(dim=0, keepdim=True)
      
        geo_features = self.encode_geometric_terms(words)
        combined_features = torch.cat([tfidf_processed, word_features, pos_features, geo_features], dim=1)
        compass_features = custom_sigmoid(self.compass_feature_processor(combined_features) * self.geometric_sigmoid_scale)
        final_features = self.apply_compass_construction_to_features(compass_features)
        return final_features
    
    def load_and_process_text_streaming(self, file_path="test.txt", chunk_size=1000, dataset_name=None, split=None):
        word_count = 0
        documents = []
        current_doc = []
        vocab = list(self.geometric_terms.keys())
        for word in vocab:
            if word not in self.word_to_idx:
                self.word_to_idx[word] = len(self.word_to_idx)
        words_processed = []
        
        # Add Hugging Face dataset support
        if dataset_name is not None:
            if not HF_DATASETS_AVAILABLE:
                print("⚠️ Hugging Face datasets library not available. Install with: pip install datasets")
                print("⚠️ Falling back to local file loading...")
                return self.load_and_process_text_streaming(file_path=file_path, chunk_size=chunk_size)
            
            try:
                print(f"📥 Loading Hugging Face dataset: {dataset_name}, split: {split}")
                ds = load_dataset(dataset_name, split=split or 'train')
                text_field = 'text' if 'text' in ds.column_names else ds.column_names[0]
                
                for text in ds[text_field]:
                    if KB_LEN > 0 and word_count >= KB_LEN:
                        break
                    words = str(text).lower().split()
                    word_history = []
                    for word in words:
                        if len(self.word_to_idx) < self.vocab_limit:
                            if word not in self.word_to_idx:
                                self.word_to_idx[word] = len(self.word_to_idx)
                        
                        word_history.append(word)
                        if len(word_history) >= 2:
                            self.bigram_counts[(word_history[-2], word_history[-1])] += 1
                        if len(word_history) >= 3:
                            self.trigram_counts[(word_history[-3], word_history[-2], word_history[-1])] += 1
                        if len(word_history) > 1000:
                            word_history = word_history[-500:]
                        
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
            # Original local file loading logic
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    word_history = []
                    while KB_LEN < 0 or word_count < KB_LEN:
                        chunk = f.read(chunk_size * 10)
                        if not chunk:
                            break
                        words = chunk.lower().split()
                        for word in words:
                            if len(self.word_to_idx) < self.vocab_limit:
                                if word not in self.word_to_idx:
                                    self.word_to_idx[word] = len(self.word_to_idx)
                            word_history.append(word)
                            if len(word_history) >= 2:
                                self.bigram_counts[(word_history[-2], word_history[-1])] += 1
                            if len(word_history) >= 3:
                                self.trigram_counts[(word_history[-3], word_history[-2], word_history[-1])] += 1
                            if len(word_history) > 1000:
                                word_history = word_history[-500:]
                            if word in words_processed:
                                str(int(time.time() * 1000))  # Convert seconds to milliseconds
                                words_processed.append(str(int(time.time() * 1000)) +  " " + word)
                            else:
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
                sample_words = vocab * 50
                documents = [' '.join(sample_words[i:i+50]) for i in range(0, len(sample_words), 50)]
                for i, word in enumerate(sample_words):
                    if word not in self.word_to_idx:
                        self.word_to_idx[word] = len(self.word_to_idx)
                    if i > 0:
                        self.bigram_counts[(sample_words[i-1], word)] += 1
                words_processed = sample_words

        if documents and not self.is_vectorizer_fitted:
            self.fit_vectorizer(documents)
        print(f"📚 Processed {word_count} words with vocab size {len(self.word_to_idx)}")
        print(f"📊 Created {len(documents)} documents")
        return words_processed[-1000:] if words_processed else []
    
    def get_transition_probs(self, word):
        if word in self.transition_cache:
            return self.transition_cache[word]
        transitions = []
        for (w1, w2), count in self.bigram_counts.items():
            if w1 == word:
                weight_multiplier = 2.0 if w2 in self.geometric_terms else 1.0
                transitions.append((w2, count * weight_multiplier))
        if len(self.transition_cache) >= self.cache_limit:
            keys_to_remove = list(self.transition_cache.keys())[:self.cache_limit//2]
            for k in keys_to_remove:
                del self.transition_cache[k]
        self.transition_cache[word] = transitions
        return transitions
        
    def get_ngram_transitions(self, context_words, n=3):
        if len(context_words) < n - 1:
            return []
        context_key = tuple(context_words[-(n-1):])
        if context_key in self.ngram_cache:
            return self.ngram_cache[context_key]
        transitions = []
        if n == 3:
            for (w1, w2, w3), count in self.trigram_counts.items():
                if (w1, w2) == context_key:
                    weight_multiplier = 2.0 if w3 in self.geometric_terms else 1.0
                    transitions.append((w3, count * weight_multiplier))
        self.ngram_cache[context_key] = transitions
        return transitions

# ------------------------------------------------------
# Text Generator (multi-word seed)
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
        if spk_rec.numel() == 0:
            return torch.zeros(1, device=next(self.parameters()).device)
        linear_output = self.selection_network(spk_rec)
        selection_probs = custom_sigmoid(linear_output.squeeze(-1) * self.selection_sigmoid_scale)
        return selection_probs
    
    def get_multi_word_transitions(self, seed_words):
        if not seed_words:
            return []
        trigram_transitions = self.text_processor.get_ngram_transitions(seed_words, n=3)
        if trigram_transitions:
            return trigram_transitions
        bigram_transitions = self.text_processor.get_transition_probs(seed_words[-1])
        return bigram_transitions
    
    def generate_text_trainable(self, spk_rec, seed_words=None, length=50):
        if spk_rec.numel() == 0:
            return "No neural data available for generation."
        with torch.no_grad():
            selection_probs = self.forward(spk_rec)
        if seed_words is None or len(seed_words) == 0:
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
            transitions = self.get_multi_word_transitions(current_words)
            if not transitions:
                transitions = self.text_processor.get_transition_probs(current_words[-1])
            if not transitions:
                next_word = random.choice(self.fallback_words)
                generated_words.append(next_word)
                current_words = [next_word]
                continue
            transitions = transitions[:self.max_transitions]
            prob_idx = min(i, len(selection_probs) - 1)
            neural_influence = selection_probs[prob_idx].item()
            context_influence = min(len(current_words) * self.context_weight.item(), 1.0)
            words, weights = zip(*transitions)
            weights = np.array(weights, dtype=float)
            for word in words:
                if word == "for":
                    i+=1
                total_influence = i + neural_influence + context_influence            
            weights = weights * total_influence
            if weights.sum() > 0:
                weights = weights / weights.sum()
                next_word = np.random.choice(words, p=weights)
            else:
                next_word = random.choice(words)
            generated_words.append(next_word)
            current_words.append(next_word)
            if len(current_words) > 3:
                current_words = current_words[-3:]
        return ' '.join(generated_words)

# ------------------------------------------------------
# Dataset creation and training placeholder - FIXED
# ------------------------------------------------------
def create_dataset(text_processor, max_samples=1000000):
    dataset = []
    sequences = [["theorem", "compass", "only", "constructions"]]
    dataset.extend(sequences)
    word_list = list(text_processor.word_to_idx.keys())
    for i in range(0, min(len(word_list), max_samples//2), 20):
        chunk = word_list[i:i+20]
        dataset.append(chunk)  # Keep as list of words
    print(f"📐 Created dataset with {len(dataset)} samples")
    return dataset

def train_snn_system(text_processor, snn_model, text_generator, dataset, epochs=5, lr=0.001, device='cpu'):
    pass  # Placeholder for training routine

# ------------------------------------------------------
# Main Implementation - FIXED
# ------------------------------------------------------
def main_implementation():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"🔥 Using device: {device}")
    
    # --- Model and processor initialization ---
    num_neurons = 1280
    chunk_size = 160
    vocab_limit = 300000
    max_features = 5000
    text_processor = EnhancedTextProcessor(num_neurons, device=device, vocab_limit=vocab_limit, max_features=max_features).to(device)
    snn_model = TrainableStreamingSNN(num_neurons, device=device, chunk_size=chunk_size).to(device)
    text_generator = TrainableStreamingTextGenerator(text_processor).to(device)
    
    print("="*60)
    print("ENHANCED SNN TEXT GENERATOR")
    print("="*60)
    
    # --- Integrated dataset loading logic - FIXED ---
    print("Choose data source:")
    print("1. Hugging Face dataset")
    print("2. Local file")
    
    choice = input("Enter choice (1 or 2, default is 2): ").strip()
    
    words = []
    if choice == "1":
        if not HF_DATASETS_AVAILABLE:
            print("⚠️ Hugging Face datasets library not available. Install with: pip install datasets")
            print("⚠️ Falling back to local file loading...")
            choice = "2"
        else:
            dataset_name = input("Enter dataset name: ").strip()
            
            if not dataset_name:
                print("⚠️ No dataset name provided. Falling back to local file.")
                words = text_processor.load_and_process_text_streaming(file_path="sample_data.txt")
            else:
                split = input("Enter split (train/test/validation, default=train): ").strip() or "train"
                words = text_processor.load_and_process_text_streaming(
                    dataset_name=dataset_name, 
                    split=split
                )
    
    if choice == "2":
        filename = input("Enter local filename (press Enter for sample_data.txt): ").strip() or "sample_data.txt"
        words = text_processor.load_and_process_text_streaming(file_path=filename)
    
    # --- Continue with training and interactive mode ---
    dataset = create_dataset(text_processor)
    train_snn_system(text_processor, snn_model, text_generator, dataset, epochs=30, lr=0.001, device=device)
    
    print("\n🎯 Interactive Mode:")
    while True:
        try:
            user_input = input("\nUSER: ").strip()
            if not user_input:
                continue
            
            seed_words = user_input.split()
            features = text_processor.words_to_neural_features(seed_words)
            spike_outputs = snn_model.forward(features)
            response = text_generator.generate_text_trainable(spike_outputs, seed_words=seed_words, length=500)
            print(f"🤖 AI: {response}")
            
        except KeyboardInterrupt:
            print("\nExiting...")
            break
        except Exception as e:
            print(f"❌ Error: {e}")

if __name__ == "__main__":
    main_implementation()
