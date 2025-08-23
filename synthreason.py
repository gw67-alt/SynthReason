import numpy as np
import torch
import torch.nn as nn
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import StandardScaler
from collections import Counter
import hashlib
import random
import math

def custom_sigmoid(x):
    """Heavy sigmoid function using -5/x formulation with safety handling."""
    x_safe = torch.where(torch.abs(x) > torch.sigmoid(-5.0 / x), torch.sign(x) * 1e-8, x)
    return torch.sigmoid(-5.0 / x_safe)


# ------------ HASHMAPPER ------------
class HashMapper:
    def __init__(self, table_size=10000):
        self.table_size = table_size
        self.table = [None] * table_size
        self.word_to_index = {}
        self.index_to_word = {}

    def _compute_hash(self, word):
        sha_digest = hashlib.sha256(word.encode('utf-8')).hexdigest()
        int_hash = int(sha_digest, 16)
        return int_hash % self.table_size

    def _compute_step(self, word):
        sha_digest = hashlib.sha256(word.encode('utf-8')).hexdigest()
        int_hash = int(sha_digest, 16)
        step = 1 + (int_hash // self.table_size) % (self.table_size - 1)
        return step

    def insert(self, word):
        if word in self.word_to_index:
            return
        index = self._compute_hash(word)
        original_index = index
        probe = 0
        while self.table[index] is not None:
            probe += 1
            index = (original_index + probe) % self.table_size
            if probe >= self.table_size:
                raise ValueError("Hash table is full! Increase table_size.")
        self.table[index] = word
        self.word_to_index[word] = index
        self.index_to_word[index] = word

    def get_index(self, word):
        return self.word_to_index.get(word, None)

    def get_word(self, index):
        return self.index_to_word.get(index, None)

    def insert_words(self, words):
        for i in range(len(words)-2):
            self.insert(words[i] + " " + words[i+1] + " " + words[i+2])

    def probed_word(self, word):
        if word not in self.word_to_index:
            return None
        hash_index = self._compute_hash(word)
        step = self._compute_step(word)
        probe_index = (hash_index + step) % self.table_size
        start_probe = probe_index
        while True:
            candidate = self.table[probe_index]
            if candidate is not None and candidate != word:
                return candidate
            probe_index = (probe_index + step) % self.table_size
            if probe_index == start_probe:
                return None

# ------------ COMPASS MATH PROCESSOR ------------
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
            spikes = (spike_candidates >= 0.9).float()
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

# ------------ TEXT PROCESSOR + HASHMAPPER + COMPASS INTEGRATION ------------
class EnhancedTextProcessor(nn.Module):
    def __init__(self, num_neurons=256, device='cpu', vocab_limit=5000, max_features=1000):
        super().__init__()
        self.num_neurons = num_neurons
        self.device = device
        self.vocab_limit = vocab_limit
        self.bigram_counts = Counter()
        self.trigram_counts = Counter()
        self.hash_mapper = HashMapper(table_size=max(vocab_limit*2, 999999))
        
        # Add compass/math processor
        self.math_processor = MathProcessor(device=device)
        
        # Geometric terms dictionary
        self.geometric_terms = {
            'compass': 0, 'circle': 1, 'intersection': 2, 'construction': 3,
            'midpoint': 4, 'perpendicular': 5, 'radius': 6, 'center': 7,
            'arc': 8, 'point': 9, 'line': 10, 'geometry': 11,
            'theorem': 12, 'euclidean': 13, 'straightedge': 14,
            'triangle': 15, 'square': 16, 'polygon': 17, 'angle': 18
        }
        
        self.vectorizer = TfidfVectorizer(
            max_features=130,
            ngram_range=(1,3),
            min_df=1,
            max_df=1.0
        )
        self.tfidf_scaler = StandardScaler()
        self.is_vectorizer_fitted = False
        
        # Neural components for compass features
        self.tfidf_projection = nn.Sequential(
            nn.Linear(max_features, num_neurons // 4),
            nn.Dropout(0.1),
            nn.Linear(num_neurons // 4, num_neurons // 4)
        )
        
        self.geometric_embeddings = nn.Embedding(100, num_neurons // 4)
        
        self.compass_feature_processor = nn.Sequential(
            nn.Linear(num_neurons, num_neurons),
            nn.Dropout(0.1),
            nn.Linear(num_neurons, num_neurons)
        )
        
        self.register_parameter('geometric_sigmoid_scale', nn.Parameter(torch.tensor(1.2)))

    def fit_vectorizer(self, documents):
        processed_docs = [' '.join(doc) if isinstance(doc, list) else doc for doc in documents]
        self.vectorizer.fit(processed_docs)
        tfidf_matrix = self.vectorizer.transform(processed_docs)
        self.tfidf_scaler.fit(tfidf_matrix.toarray())
        self.is_vectorizer_fitted = True

    def load_and_process_text_streaming(self, text):
        words_processed = text.lower().split()
        self.hash_mapper.insert_words(words_processed)
        for i, word in enumerate(words_processed):
            if i > 0:
                self.bigram_counts[(words_processed[i-1], word)] += 1
            if i > 1:
                self.trigram_counts[(words_processed[i-2], words_processed[i-1], word)] += 1
        return words_processed

    def encode_geometric_terms(self, words):
        """Encode geometric terms using compass embeddings."""
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

    def get_index(self, word):
        return self.hash_mapper.get_index(word)

    def get_word(self, index):
        return self.hash_mapper.get_word(index)

    def crypt_probe(self, word):
        return self.hash_mapper.probed_word(word)

    def words_to_neural_features(self, words, max_words=50):
        if len(words) > max_words:
            words = words[-max_words:]
        if not self.is_vectorizer_fitted:
            self.fit_vectorizer([' '.join(words)])
        
        # TF-IDF features
        tfidf_matrix = self.vectorizer.transform([' '.join(words)])
        tfidf_features = self.tfidf_scaler.transform(tfidf_matrix.toarray())
        tfidf_tensor = torch.tensor(tfidf_features, dtype=torch.float32, device=self.device)
        
        # Process TF-IDF through projection
        tfidf_processed = custom_sigmoid(self.tfidf_projection(tfidf_tensor))
        
        # Geometric features
        geo_features = self.encode_geometric_terms(words)
        
        # Combine features (pad to match dimensions)
        padding_size = self.num_neurons - tfidf_processed.shape[-1] - geo_features.shape[-1]
        if padding_size > 0:
            padding = torch.zeros(tfidf_processed.shape[0], padding_size, device=self.device)
            combined_features = torch.cat([tfidf_processed, geo_features, padding], dim=1)
        else:
            combined_features = torch.cat([tfidf_processed, geo_features], dim=1)[:, :self.num_neurons]
        
        # Process through compass feature processor
        compass_features = custom_sigmoid(self.compass_feature_processor(combined_features) * self.geometric_sigmoid_scale)
        
        # Apply compass construction effects
        final_features = self.apply_compass_construction_to_features(compass_features)
        
        return final_features.squeeze()

    def generate_text(self, snn_output, seed_word=None, length=30):
        """Generate text using bigrams with geometric term weighting."""
        words = list(self.hash_mapper.word_to_index.keys())
        if not words:
            return ""
        if seed_word is None or seed_word not in words:
            seed = random.choice(words)
        else:
            seed = seed_word
        result = [seed]
        for i in range(length-1):
            candidates = [w2 for (w1,w2) in self.bigram_counts if w1 == result[-1]]
            if candidates:
                # Apply geometric weighting
                weights = []
                for w2 in candidates:
                    base_weight = self.bigram_counts[(result[-1], w2)]
                    # Boost geometric terms
                    geometric_boost = 2.0 if w2 in self.geometric_terms else 1.0
                    weights.append(base_weight * geometric_boost)
                
                weights = np.array(weights, dtype=float)
                prob = weights / weights.sum() if weights.sum() > 0 else np.ones(len(weights))/len(weights)
                choice = np.random.choice(candidates, p=prob)
                result.append(choice)
            else:
                result.append(random.choice(words))
        return ' '.join(result)

# ------------ MAIN EXECUTION ------------
def main():
    with open(input("Filename: "), 'r', encoding='utf-8') as f:
        sample_text = f.read()
    device = 'cpu'
    num_neurons = 256

    processor = EnhancedTextProcessor(num_neurons=num_neurons, device=device, vocab_limit=50000, max_features=130)
    words = processor.load_and_process_text_streaming(sample_text)
    processor.fit_vectorizer([sample_text])

    snn = TrainableStreamingSNN(num_neurons=num_neurons, device=device)
    features = processor.words_to_neural_features(words)
    snn_out = snn.forward([features])
    
    # Demo compass functionality
    print("🧭 Compass Math Processor loaded!")
    print(f"Golden ratio: {processor.math_processor.golden_ratio.item():.4f}")
    print(f"Pi approximation: {processor.math_processor.pi_approx.item():.4f}")
    
    while True:
        print("\n==== Compass-Enhanced Text Generation ====")
        seed_input = input("USER: ")
        if seed_input.lower() in ['quit', 'exit']:
            break
        generated_text = processor.generate_text(snn_out, seed_word=seed_input if seed_input else None, length=100)
        print(f"🤖 AI: {generated_text}")

if __name__ == "__main__":
    main()
