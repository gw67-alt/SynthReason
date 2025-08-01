import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import snntorch as snn
from snntorch import surrogate, utils, spikegen
from sklearn.preprocessing import StandardScaler
from torch_geometric.data import Data
from torch_geometric.nn import GCNConv
import matplotlib.pyplot as plt
from collections import defaultdict, Counter
import random

KB_LEN = -1

class PolymorphicNeuron(nn.Module):
    """A neuron that can switch between different behavioral modes."""
    def __init__(self, input_dim, num_modes=3):
        super().__init__()
        self.num_modes = num_modes
        self.input_dim = input_dim
        self.neuron_modes = nn.ModuleList([
            snn.Leaky(beta=0.3, threshold=1.0, spike_grad=surrogate.fast_sigmoid()),
            snn.Leaky(beta=0.7, threshold=0.8, spike_grad=surrogate.fast_sigmoid()),
            snn.Leaky(beta=0.5, threshold=1.2, spike_grad=surrogate.fast_sigmoid()),
        ])
        self.mode_selector = nn.Sequential(
            nn.Linear(input_dim, 16),
            nn.ReLU(),
            nn.Linear(16, num_modes),
            nn.Softmax(dim=-1)
        )
        self.adaptation_rate = 0.1
        self.mode_history = torch.zeros(num_modes)
        
    def forward(self, x, mem_states=None):
        mode_probs = self.mode_selector(x)
        if mem_states is None:
            mem_states = [None] * self.num_modes
        mode_outputs = []
        new_mem_states = []
        for i, neuron_mode in enumerate(self.neuron_modes):
            if mem_states[i] is None:
                spk, mem = neuron_mode(x)
            else:
                spk, mem = neuron_mode(x, mem_states[i])
            mode_outputs.append(spk)
            new_mem_states.append(mem)
        mode_outputs = torch.stack(mode_outputs, dim=0)
        if len(mode_probs.shape) == 1:
            mode_probs = mode_probs.unsqueeze(-1)
        final_spikes = torch.sum(mode_outputs * mode_probs, dim=0)
        self.mode_history = self.mode_history * (1 - self.adaptation_rate) + mode_probs.squeeze() * self.adaptation_rate
        return final_spikes, new_mem_states, mode_probs

class PolymorphicSNN(nn.Module):
    """Enhanced SNN with polymorphic capabilities."""
    def __init__(self, num_neurons, num_polymorphic=None):
        super().__init__()
        self.num_neurons = num_neurons
        self.num_polymorphic = num_polymorphic or num_neurons // 4
        self.input_layer = nn.Linear(num_neurons, num_neurons)
        self.regular_neurons = snn.Leaky(
            beta=0.5, 
            init_hidden=False, 
            spike_grad=surrogate.fast_sigmoid()
        )
        self.polymorphic_neurons = nn.ModuleList([
            PolymorphicNeuron(input_dim=1, num_modes=3) 
            for _ in range(self.num_polymorphic)
        ])
        self.poly_connectivity = nn.Parameter(
            torch.randn(self.num_polymorphic, num_neurons) * 0.9
        )
        self.global_adaptation = nn.Parameter(torch.ones(1) * 0.5)
        
    def forward(self, x, mem=None, poly_mem_states=None):
        x_processed = self.input_layer(x)
        if mem is None:
            reg_spk, reg_mem = self.regular_neurons(x_processed)
        else:
            reg_spk, reg_mem = self.regular_neurons(x_processed, mem)
        poly_spikes = []
        new_poly_mem_states = []
        mode_distributions = []
        if poly_mem_states is None:
            poly_mem_states = [None] * self.num_polymorphic
        for i, poly_neuron in enumerate(self.polymorphic_neurons):
            neuron_input = torch.sum(reg_spk * self.poly_connectivity[i], dim=-1, keepdim=True)
            poly_spk, poly_mem, mode_prob = poly_neuron(neuron_input, poly_mem_states[i])
            poly_spikes.append(poly_spk)
            new_poly_mem_states.append(poly_mem)
            mode_distributions.append(mode_prob)
        if poly_spikes:
            poly_spikes_tensor = torch.stack(poly_spikes, dim=0)
            if len(poly_spikes_tensor.shape) > 1:
                poly_spikes_flat = poly_spikes_tensor.squeeze()
                if len(poly_spikes_flat.shape) == 0:
                    poly_spikes_flat = poly_spikes_flat.unsqueeze(0)
            else:
                poly_spikes_flat = poly_spikes_tensor
            combined_spikes = torch.cat([reg_spk, poly_spikes_flat], dim=-1)
        else:
            combined_spikes = reg_spk
        combined_spikes = combined_spikes * self.global_adaptation
        return combined_spikes, reg_mem, new_poly_mem_states, mode_distributions

def run_polymorphic_snn(spike_data, snn_model):
    spk_rec, mem_rec, poly_mem_rec, mode_rec = [], [], [], []
    utils.reset(snn_model)
    mem = None
    poly_mem_states = None
    for step in range(spike_data.shape[0]):
        if spike_data.shape[1] > 0:
            input_data = spike_data[step][0]
            spk, mem, poly_mem_states, mode_dist = snn_model(input_data, mem, poly_mem_states)
            spk_rec.append(spk)
            mem_rec.append(mem)
            poly_mem_rec.append(poly_mem_states)
            mode_rec.append(mode_dist)
    spk_rec = torch.stack(spk_rec)
    mem_rec = torch.stack(mem_rec)
    return spk_rec, mem_rec, poly_mem_rec, mode_rec

def create_neuron_aligned_graph(spk_rec, mem_rec):
    print(f"Creating graph from spk_rec: {spk_rec.shape}, mem_rec: {mem_rec.shape}")
    min_steps = min(spk_rec.shape[0], mem_rec.shape[0])
    spk_rec_aligned = spk_rec[:min_steps]
    mem_rec_aligned = mem_rec[:min_steps]
    min_neurons = min(spk_rec_aligned.shape[1], mem_rec_aligned.shape[1])
    spk_rec_aligned = spk_rec_aligned[:, :min_neurons]
    mem_rec_aligned = mem_rec_aligned[:, :min_neurons]
    print(f"Aligned shapes - spk_rec: {spk_rec_aligned.shape}, mem_rec: {mem_rec_aligned.shape}")
    node_features = spk_rec_aligned.T
    node_features = torch.cat([node_features, mem_rec_aligned.T], dim=1)
    print(f"Node features shape: {node_features.shape}")
    num_nodes = node_features.shape[0]
    print(f"Number of graph nodes: {num_nodes}")
    edge_index = []
    if num_nodes > 1:
        for i in range(num_nodes):
            for j in range(i+1, min(i+4, num_nodes)):
                edge_index.extend([[j, i], [i, j]])
    if edge_index:
        edge_index = torch.tensor(edge_index, dtype=torch.long).t().contiguous()
    else:
        edge_index = torch.empty((2, 0), dtype=torch.long)
    print(f"Edge index shape: {edge_index.shape}")
    data = Data(x=node_features, edge_index=edge_index)
    return data

class DataAwareFGCN(nn.Module):
    def __init__(self, in_dim, hidden_dim=64, out_dim=32):
        super().__init__()
        self.gcn1 = GCNConv(in_dim, hidden_dim)
        self.gcn2 = GCNConv(hidden_dim, out_dim)
        self.attn = nn.Linear(out_dim, 1)
    
    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        print(f"GCN input shape: {x.shape}")
        print(f"Edge index shape: {edge_index.shape}")
        x = F.relu(self.gcn1(x, edge_index))
        x = F.relu(self.gcn2(x, edge_index))
        attn_weights = torch.sigmoid(self.attn(x))
        x = x * attn_weights
        print(f"GCN output shape: {x.shape}")
        return x

def neuron_to_image_mapping(node_features, target_size=8):
    num_nodes, feat_dim = node_features.shape
    target_pixels = target_size * target_size
    print(f"Mapping {num_nodes} neurons to {target_pixels} pixels")
    if num_nodes < target_pixels:
        repeat_factor = target_pixels // num_nodes + 1
        node_features = node_features.repeat(repeat_factor, 1)[:target_pixels]
    else:
        node_features = node_features[:target_pixels]
    img_data = node_features.mean(dim=1).reshape(target_size, target_size)
    img_data = img_data.detach().cpu().numpy()
    img_data = (img_data - img_data.min()) / (img_data.max() - img_data.min() + 1e-8)
    print(f"Final image shape: {img_data.shape}")
    print(f"Image value range: [{img_data.min():.3f}, {img_data.max():.3f}]")
    return img_data

class NeuronAwareTextProcessor:
    def __init__(self, num_neurons=16):
        self.num_neurons = num_neurons
        self.vocab = {}
        self.word_to_idx = {}
        self.idx_to_word = {}
        self.bigram_counts = Counter()
        self.transition_matrix = None
        self.transition_probs = None
    
    def load_and_process_text(self, file_path="test.txt"):
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = ' '.join(f.read().split()[:KB_LEN])
                print(f"Loaded {len(content)} characters from {file_path}")
        except FileNotFoundError:
            content = "The neural network processes information through spiking patterns. Each neuron contributes to the overall computation. Machine learning algorithms use artificial neural networks to simulate biological processes. Deep learning models can generate text by learning patterns from large datasets. Spiking neural networks offer a more biologically plausible approach to artificial intelligence."
            print("Using sample text")
        words = content.lower().split()
        words = [w for w in words if w]
        unique_words = list(set(words))
        self.word_to_idx = {word: idx for idx, word in enumerate(unique_words)}
        self.idx_to_word = {idx: word for word, idx in self.word_to_idx.items()}
        for i in range(len(words) - 1):
            self.bigram_counts[(words[i], words[i+1])] += 1
        # Initialize transition matrix features
        self.create_transition_matrix_features()
        return words
    
    def create_transition_matrix_features(self):
        """Create a transition matrix and extract statistical features"""
        vocab_size = len(self.word_to_idx)
        self.transition_matrix = np.zeros((vocab_size, vocab_size))
        
        # Fill transition matrix
        for (w1, w2), count in self.bigram_counts.items():
            if w1 in self.word_to_idx and w2 in self.word_to_idx:
                i, j = self.word_to_idx[w1], self.word_to_idx[w2]
                self.transition_matrix[i, j] = count
        
        # Normalize rows to get probabilities
        row_sums = self.transition_matrix.sum(axis=1, keepdims=True)
        self.transition_probs = np.divide(self.transition_matrix, row_sums, 
                                        out=np.zeros_like(self.transition_matrix), 
                                        where=row_sums!=0)
    
    def get_transition_features(self, word):
        """Extract transition-based features for a word"""
        features = []
        word_idx = self.word_to_idx.get(word, 0)
        
        if self.transition_probs is not None:
            # Outgoing transition features
            out_transitions = self.transition_probs[word_idx]
            
            # Number of possible next words
            transition_diversity = np.sum(out_transitions > 0)
            features.append(transition_diversity)
            
            # Maximum transition probability
            max_prob = np.max(out_transitions)
            features.append(max_prob)
            
            # Entropy of transitions
            probs = out_transitions[out_transitions > 0]
            if len(probs) > 0:
                entropy = -np.sum(probs * np.log2(probs + 1e-8))
            else:
                entropy = 0
            features.append(entropy)
            
            # Incoming transitions (how often other words lead to this word)
            in_transitions = self.transition_probs[:, word_idx]
            in_degree = np.sum(in_transitions > 0)
            features.append(in_degree)
            
            # Centrality measure (sum of incoming probabilities)
            centrality = np.sum(in_transitions)
            features.append(centrality)
        else:
            # Default features if transition matrix not available
            features.extend([0, 0, 0, 0, 0])
        
        return features
    
    def get_semantic_similarity(self, word, user_input):
        """Simple semantic similarity based on shared characters and context"""
        if not user_input:
            return 0.5
        
        user_words = user_input.lower().split()
        max_similarity = 0.0
        
        for user_word in user_words:
            # Character overlap similarity
            common_chars = set(word.lower()) & set(user_word.lower())
            similarity = len(common_chars) / max(len(word), len(user_word)) if max(len(word), len(user_word)) > 0 else 0
            max_similarity = max(max_similarity, similarity)
        
        return max_similarity
    
    def get_context_features(self, word, prev_word=None, next_word=None):
        """Get contextual features based on bigrams"""
        features = []
        
        # Previous word context
        if prev_word:
            prev_bigram_count = self.bigram_counts.get((prev_word, word), 0)
            features.append(prev_bigram_count)
        else:
            features.append(0)
        
        # Next word context
        if next_word:
            next_bigram_count = self.bigram_counts.get((word, next_word), 0)
            features.append(next_bigram_count)
        else:
            features.append(0)
        
        return features
    
    def words_to_neural_features(self, words, user_input=None, max_words=50):
        features = []
        
        # If user input is provided, prioritize it
        if user_input:
            user_words = user_input.lower().split()
            combined_words = user_words + words[:max(0, max_words-len(user_words))]
        else:
            combined_words = words[:max_words]
        
        for i, word in enumerate(combined_words):
            word_idx = self.word_to_idx.get(word, 0)
            
            # Start with transition-based features
            feature_vector = self.get_transition_features(word)
            
            # Add context features
            prev_word = combined_words[i-1] if i > 0 else None
            next_word = combined_words[i+1] if i < len(combined_words)-1 else None
            context_features = self.get_context_features(word, prev_word, next_word)
            feature_vector.extend(context_features)
            
            # Create context-aware features based on user input
            if user_input and i < len(user_input.split()):
                # Higher weight for user input words
                context_weight = 2.0
                position_weight = 1.0 - (i / len(user_input.split())) if len(user_input.split()) > 0 else 1.0
            else:
                context_weight = 1.0
                position_weight = 0.5
            
            # Apply weights to existing features
            feature_vector = [f * context_weight * position_weight for f in feature_vector]
            
            # Add word embedding-like features
            feature_vector.append(word_idx / len(self.word_to_idx))  # Normalized word index
            feature_vector.append(len(word) / 20.0)  # Normalized word length
            
            # Add semantic similarity to user input
            if user_input:
                similarity = self.get_semantic_similarity(word, user_input)
                feature_vector.append(similarity)
            else:
                feature_vector.append(0.0)
            
            # Pad or truncate to match neuron count
            while len(feature_vector) < self.num_neurons:
                feature_vector.append(np.sin(len(feature_vector) * word_idx / 10.0))
            feature_vector = feature_vector[:self.num_neurons]
            features.append(feature_vector)
        
        return np.array(features)

class TextGenerator:
    def __init__(self, text_processor: NeuronAwareTextProcessor):
        self.text_processor = text_processor
        self.transitions = defaultdict(list)
        self.seed_transitions = defaultdict(list)
        self.build_transitions()
    
    def build_transitions(self):
        """Build both regular and seed-based transition probabilities."""
        for (w1, w2), count in self.text_processor.bigram_counts.items():
            self.transitions[w1].append((w2, count))
            self.seed_transitions[w1].append((w2, count))
    
    def get_seed_candidates(self, seed_words):
        """Use * operator to unpack seed words and find candidates."""
        if not seed_words:
            return []
        candidates = []
        for word in seed_words:
            if word in self.transitions:
                candidates.extend(self.transitions[word])
        return candidates if candidates else []
    
    def extract_graph_features(self, spk_rec, mem_rec):
        """Placeholder for graph features extraction."""
        return None
    
    def generate_text_from_neural_output(self, spk_rec, mem_rec, seed_word: str = None, length: int = 50) -> str:
        """Generate text with FGCN-based moderation."""
        if not self.transitions:
            return "No training data available for text generation."
        
        graph_features = self.extract_graph_features(spk_rec, mem_rec)
        neural_influence = spk_rec.mean(dim=1).detach().cpu().numpy()
        
        if seed_word:
            seed_words = seed_word.split()
            current_word = seed_words[-1] if seed_words else random.choice(list(self.transitions.keys()))
        else:
            seed_words = []
            current_word = random.choice(list(self.transitions.keys()))
        
        generated_words = [current_word]
        
        for i in range(length - 1):
            # Fix: Use modulo indexing to cycle through neural_influence
            neural_idx = i % len(neural_influence)
            neural_gate = neural_influence[neural_idx]
            
            if neural_gate > 0.8 and seed_words:
                candidates = self.get_seed_candidates(seed_words)
                if not candidates:
                    candidates = self.transitions.get(current_word, [])
            else:
                candidates = self.transitions.get(current_word, [])
            
            if not candidates:
                current_word = random.choice(list(self.transitions.keys()))
                generated_words.append(current_word)
                continue
            
            moderated_candidates = self.moderate_candidate_selection(
                candidates, graph_features, neural_idx
            )
            
            words, weights = zip(*moderated_candidates)
            weights = np.array(weights, dtype=float)
            
            if graph_features is not None:
                # Fix: Use modulo indexing for coherence_boost as well
                coherence_idx = i % len(graph_features['coherence'])
                coherence_boost = graph_features['coherence'][coherence_idx]
                neural_weight = max(0.1, neural_influence[neural_idx] * (1 + coherence_boost))
            else:
                neural_weight = max(0.1, neural_influence[neural_idx])
                
            weights = weights * (1 + neural_weight)
            weights = weights / weights.sum() if weights.sum() > 0 else np.ones_like(weights) / len(weights)
            
            next_word = np.random.choice(words, p=weights)
            generated_words.append(next_word)
            current_word = next_word
        
        return ' '.join(generated_words)
    
    def moderate_candidate_selection(self, candidates, graph_features, position):
        """Default moderation - can be overridden by subclasses."""
        return candidates

class FGCNModeratedTextGenerator(TextGenerator):
    def __init__(self, text_processor: NeuronAwareTextProcessor, fgcn_model=None):
        super().__init__(text_processor)
        self.fgcn_model = fgcn_model
        self.moderation_threshold = 0.5
        self.coherence_weight = 0.3
        
    def extract_graph_features(self, spk_rec, mem_rec):
        """Extract graph-based features using FGCN for text moderation."""
        if self.fgcn_model is None:
            return None
            
        data = create_neuron_aligned_graph(spk_rec, mem_rec)
        
        with torch.no_grad():
            graph_features = self.fgcn_model(data)
            
        coherence_signal = torch.mean(graph_features, dim=1)
        stability_signal = torch.std(graph_features, dim=1)
        
        return {
            'coherence': coherence_signal.detach().cpu().numpy(),
            'stability': stability_signal.detach().cpu().numpy(),
            'raw_features': graph_features.detach().cpu().numpy()
        }
    
    def compute_word_quality_score(self, word, graph_features, position):
        """Compute quality score for a word based on graph features."""
        if graph_features is None:
            return 1.0
            
        coherence = graph_features['coherence']
        stability = graph_features['stability']
        
        pos_idx = min(position, len(coherence) - 1)
        
        quality_score = (
            coherence[pos_idx % len(coherence)] * self.coherence_weight +
            (1 - stability[pos_idx % len(stability)]) * (1 - self.coherence_weight)
        )
        
        word_length_factor = min(len(word) / 10.0, 1.0)
        
        return float(quality_score * word_length_factor)
    
    def moderate_candidate_selection(self, candidates, graph_features, position):
        """Use FGCN features to moderate candidate word selection."""
        if not candidates or graph_features is None:
            return candidates
            
        moderated_candidates = []
        
        for word, weight in candidates:
            quality_score = self.compute_word_quality_score(word, graph_features, position)
            
            if quality_score > self.moderation_threshold:
                moderated_weight = weight * quality_score
                moderated_candidates.append((word, moderated_weight))
            else:
                moderated_weight = weight * 0.1
                moderated_candidates.append((word, moderated_weight))
                
        return moderated_candidates if moderated_candidates else candidates

class UserContextAwareTextGenerator(FGCNModeratedTextGenerator):
    def __init__(self, text_processor: NeuronAwareTextProcessor, fgcn_model=None):
        super().__init__(text_processor, fgcn_model)
        self.user_context_weight = 0.7
        
    def find_best_starting_word(self, user_words):
        """Find the best starting word based on user input."""
        if not user_words:
            return random.choice(list(self.transitions.keys()))
        
        # Try to find user words in transitions
        for word in reversed(user_words):  # Start with last word
            if word in self.transitions:
                return word
        
        # If no exact match, find similar words
        best_word = None
        best_similarity = 0
        
        for word in user_words:
            for transition_word in self.transitions.keys():
                similarity = self.text_processor.get_semantic_similarity(word, transition_word)
                if similarity > best_similarity:
                    best_similarity = similarity
                    best_word = transition_word
        
        return best_word if best_word else random.choice(list(self.transitions.keys()))
    
    def get_contextual_candidates(self, current_word, user_words, context_strength):
        """Get candidates with user context bias."""
        candidates = self.transitions.get(current_word, [])
        
        if not user_words or context_strength < 0.1:
            return candidates
        
        # Boost candidates that are similar to user words
        contextual_candidates = []
        for word, weight in candidates:
            context_boost = 1.0
            
            # Check similarity to user words
            for user_word in user_words:
                similarity = self.text_processor.get_semantic_similarity(word, user_word)
                if similarity > 0.3:  # Threshold for similarity
                    context_boost += similarity * context_strength
            
            contextual_candidates.append((word, weight * context_boost))
        
        return contextual_candidates if contextual_candidates else candidates
    
    def generate_contextual_text(self, user_input, spk_rec, mem_rec, length=50):
        """Generate text that's contextually aware of user input"""
        
        if not user_input.strip():
            return self.generate_text_from_neural_output(spk_rec, mem_rec, length=length)
        
        user_words = user_input.lower().split()
        graph_features = self.extract_graph_features(spk_rec, mem_rec)
        
        # Start with user's last word or a contextually relevant word
        current_word = self.find_best_starting_word(user_words)
        generated_words = [current_word]
        
        # Track user context throughout generation
        user_context_strength = 1.0
        context_decay = 0.95
        
        for i in range(length - 1):
            neural_idx = i % len(spk_rec.mean(dim=1))
            neural_influence = spk_rec.mean(dim=1)[neural_idx].item()
            
            # Get candidates with user context bias
            candidates = self.get_contextual_candidates(
                current_word, user_words, user_context_strength
            )
            
            if graph_features:
                candidates = self.moderate_candidate_selection(
                    candidates, graph_features, i
                )
            
            if not candidates:
                current_word = random.choice(list(self.transitions.keys()))
                generated_words.append(current_word)
                continue
            
            words, weights = zip(*candidates)
            weights = np.array(weights, dtype=float)
            
            # Apply neural influence and user context
            neural_weight = max(0.1, neural_influence * (1 + user_context_strength))
            weights = weights * neural_weight
            weights = weights / weights.sum() if weights.sum() > 0 else np.ones_like(weights) / len(weights)
            
            next_word = np.random.choice(words, p=weights)
            generated_words.append(next_word)
            current_word = next_word
            
            # Decay user context over time
            user_context_strength *= context_decay
        
        return ' '.join(generated_words)

def process_user_input_through_snn(filename, user_input, text_processor, snn_model, num_steps=10):
    """Process user input through the SNN to generate contextual neural states"""
    
    # Load base text data
    words = text_processor.load_and_process_text(filename)
    
    # Generate features specifically for user input
    user_features = text_processor.words_to_neural_features(
        words, 
        user_input=user_input
    )
    
    # Scale features
    scaler = StandardScaler()
    features_scaled = scaler.fit_transform(user_features)
    features_tensor = torch.FloatTensor(features_scaled)
    
    # Generate spikes with user input bias
    spike_data = spikegen.rate(features_tensor, num_steps=num_steps)
    
    # Process through SNN
    spk_rec, mem_rec, poly_mem_rec, mode_rec = run_polymorphic_snn(spike_data, snn_model)
    
    return spk_rec, mem_rec, poly_mem_rec, mode_rec

def analyze_moderation_impact(regular_text, moderated_text):
    """Analyze the impact of FGCN moderation on text quality."""
    regular_words = regular_text.split()
    moderated_words = moderated_text.split()
    
    analysis = {
        'length_difference': len(moderated_words) - len(regular_words),
        'unique_words_regular': len(set(regular_words)),
        'unique_words_moderated': len(set(moderated_words)),
        'vocabulary_diversity_regular': len(set(regular_words)) / len(regular_words) if regular_words else 0,
        'vocabulary_diversity_moderated': len(set(moderated_words)) / len(moderated_words) if moderated_words else 0,
        'common_words': len(set(regular_words) & set(moderated_words)),
        'word_overlap_ratio': len(set(regular_words) & set(moderated_words)) / len(set(regular_words) | set(moderated_words)) if (regular_words or moderated_words) else 0
    }
    
    return analysis

def visualize_moderation_effects(spk_rec, mem_rec, fgcn_model):
    """Visualize how FGCN features affect text generation."""
    data = create_neuron_aligned_graph(spk_rec, mem_rec)
    
    with torch.no_grad():
        graph_features = fgcn_model(data)
    
    coherence_signal = torch.mean(graph_features, dim=0).detach().cpu().numpy()
    stability_signal = torch.std(graph_features, dim=0).detach().cpu().numpy()
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    
    ax1.plot(coherence_signal)
    ax1.set_title('FGCN Coherence Signal')
    ax1.set_xlabel('Feature Dimension')
    ax1.set_ylabel('Coherence Value')
    
    ax2.plot(stability_signal)
    ax2.set_title('FGCN Stability Signal')
    ax2.set_xlabel('Feature Dimension')
    ax2.set_ylabel('Stability Value')
    
    plt.tight_layout()
    plt.show()
    
    return coherence_signal, stability_signal

def main_with_user_context_awareness():
    num_neurons = 256
    num_polymorphic = 64
    num_steps = 10
    img_size = 8
    
    print(f"Initializing User-Context-Aware Polymorphic SNN with {num_neurons} neurons ({num_polymorphic} polymorphic)")
    
    # Initialize components
    text_processor = NeuronAwareTextProcessor(num_neurons)
    snn_model = PolymorphicSNN(num_neurons, num_polymorphic)
    
    print("="*60)
    print("USER-CONTEXT-AWARE POLYMORPHIC SNN TEXT GENERATOR")
    print("="*60)
    print("This system processes your input through a spiking neural network")
    print("and generates contextually relevant text based on neural patterns.")
    print("Enter text to generate responses.")
    print("="*60)
    filename = input("Enter dataset filename: ")
    while True:
        user_input = input("\nUSER: ").strip()
        if not user_input:
            print("Please enter some text.")
            continue
        
        print(f"\nProcessing input: '{user_input}'")
        print("="*40)
        
        # Process user input through SNN
        spk_rec, mem_rec, poly_mem_rec, mode_rec = process_user_input_through_snn(filename,
            user_input, text_processor, snn_model, num_steps
        )
        
        # Initialize FGCN model
        data = create_neuron_aligned_graph(spk_rec, mem_rec)
        fgcn_model = DataAwareFGCN(data.x.shape[1])
        
        # Create user-context-aware text generator
        context_generator = UserContextAwareTextGenerator(text_processor, fgcn_model)
        
        # Generate contextual response
        contextual_text = context_generator.generate_contextual_text(
            user_input, spk_rec, mem_rec, length=500
        )
        print()
        print("AI:", contextual_text)
        

if __name__ == "__main__":
    main_with_user_context_awareness()
