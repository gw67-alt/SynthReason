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

def analyze_polymorphic_behavior(mode_rec, poly_mem_rec):
    if not mode_rec or not mode_rec[0]:
        return {}
    mode_arrays = []
    for step_modes in mode_rec:
        if step_modes:
            step_tensors = []
            for mode_tensor in step_modes:
                if len(mode_tensor.shape) == 2:
                    step_tensors.append(mode_tensor.squeeze())
                else:
                    step_tensors.append(mode_tensor)
            if step_tensors:
                step_array = torch.stack(step_tensors).detach().cpu().numpy()
                mode_arrays.append(step_array)
    if not mode_arrays:
        return {}
    mode_evolution = np.array(mode_arrays)
    analysis = {
        'mode_stability': np.std(mode_evolution, axis=0),
        'dominant_modes': np.argmax(np.mean(mode_evolution, axis=0), axis=1),
        'mode_switching_frequency': np.sum(np.diff(np.argmax(mode_evolution, axis=2), axis=0) != 0, axis=0),
        'average_mode_distribution': np.mean(mode_evolution, axis=0)
    }
    return analysis

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

def generate_polymorphic_visualization(spk_rec, mem_rec, mode_rec, poly_analysis, img_size=8):
    print("="*50)
    print("GENERATING POLYMORPHIC SNN VISUALIZATION")
    print("="*50)
    data = create_neuron_aligned_graph(spk_rec, mem_rec)
    model = DataAwareFGCN(data.x.shape[1])
    with torch.no_grad():
        node_feats = model(data)
    img = neuron_to_image_mapping(node_feats, target_size=img_size)
    fig, axes = plt.subplots(3, 2, figsize=(15, 18))
    im1 = axes[0, 0].imshow(img, cmap='viridis', interpolation='nearest')
    axes[0, 0].set_title('Neural Activity Heatmap')
    axes[0, 0].axis('off')
    plt.colorbar(im1, ax=axes[0, 0])
    spike_activity = spk_rec.sum(dim=1).detach().cpu().numpy()
    axes[0, 1].plot(spike_activity)
    axes[0, 1].set_title('Total Spike Activity Over Time')
    axes[0, 1].set_xlabel('Time Step')
    axes[0, 1].set_ylabel('Total Spikes')
    if poly_analysis and 'average_mode_distribution' in poly_analysis:
        mode_dist = poly_analysis['average_mode_distribution']
        im2 = axes[1, 0].imshow(mode_dist.T, cmap='plasma', aspect='auto')
        axes[1, 0].set_title('Polymorphic Mode Distribution')
        axes[1, 0].set_xlabel('Neuron Index')
        axes[1, 0].set_ylabel('Mode Type')
        plt.colorbar(im2, ax=axes[1, 0])
    else:
        axes[1, 0].text(0.5, 0.5, 'No Polymorphic Data', ha='center', va='center')
        axes[1, 0].set_title('Polymorphic Mode Distribution')
    if poly_analysis and 'mode_switching_frequency' in poly_analysis:
        switching_freq = poly_analysis['mode_switching_frequency']
        axes[1, 1].bar(range(len(switching_freq)), switching_freq)
        axes[1, 1].set_title('Mode Switching Frequency')
        axes[1, 1].set_xlabel('Polymorphic Neuron Index')
        axes[1, 1].set_ylabel('Switches')
    else:
        axes[1, 1].text(0.5, 0.5, 'No Switching Data', ha='center', va='center')
        axes[1, 1].set_title('Mode Switching Frequency')
    neuron_activities = spk_rec.sum(dim=0).detach().cpu().numpy()
    axes[2, 0].bar(range(len(neuron_activities)), neuron_activities)
    axes[2, 0].set_title('Individual Neuron Activity')
    axes[2, 0].set_xlabel('Neuron Index')
    axes[2, 0].set_ylabel('Total Spikes')
    if poly_analysis and 'mode_stability' in poly_analysis:
        stability = poly_analysis['mode_stability']
        im3 = axes[2, 1].imshow(stability.T, cmap='coolwarm', aspect='auto')
        axes[2, 1].set_title('Mode Stability (Lower = More Stable)')
        axes[2, 1].set_xlabel('Neuron Index')
        axes[2, 1].set_ylabel('Mode Type')
        plt.colorbar(im3, ax=axes[2, 1])
    else:
        axes[2, 1].text(0.5, 0.5, 'No Stability Data', ha='center', va='center')
        axes[2, 1].set_title('Mode Stability')
    plt.tight_layout()
    plt.show()
    return img

class NeuronAwareTextProcessor:
    def __init__(self, num_neurons=16):
        self.num_neurons = num_neurons
        self.vocab = {}
        self.word_to_idx = {}
        self.idx_to_word = {}
        self.bigram_counts = Counter()
    
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
        return words
    
    def words_to_neural_features(self, words, max_words=50):
        features = []
        for i, word in enumerate(words[:max_words]):
            word_idx = self.word_to_idx.get(word, 0)
            feature_vector = [
                word_idx / len(self.word_to_idx),
                len(word) / 20.0,
                i / len(words),
                len(set(word)) / max(len(word), 1),
            ]
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
                candidates.extend(*[self.transitions[word]])
        return candidates if candidates else []
    
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
                candidates, graph_features, i
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
            
        coherence_signal = torch.mean(graph_features, dim=0)
        stability_signal = torch.std(graph_features, dim=0)
        
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
    
    def generate_moderated_text(self, spk_rec, mem_rec, seed_word: str = None, length: int = 50) -> str:
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
            neural_gate = neural_influence[i] if i < len(neural_influence) else 0.5
            
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
                candidates, graph_features, i
            )
            
            words, weights = zip(*moderated_candidates)
            weights = np.array(weights, dtype=float)
            
            if graph_features is not None:
                coherence_boost = graph_features['coherence'][i % len(graph_features['coherence'])]
                neural_weight = max(0.1, neural_influence[i] * (1 + coherence_boost))
            else:
                neural_weight = max(0.1, neural_influence[i])
                
            weights = weights * (1 + neural_weight)
            weights = weights / weights.sum() if weights.sum() > 0 else np.ones_like(weights) / len(weights)
            
            next_word = np.random.choice(words, p=weights)
            generated_words.append(next_word)
            current_word = next_word
        
        return ' '.join(generated_words)

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

def main_with_fgcn_moderation():
    num_neurons = 256
    num_polymorphic = 64
    num_steps = 10
    img_size = 8
    
    print(f"Initializing with {num_neurons} neurons ({num_polymorphic} polymorphic)")
    
    # Initialize components
    text_processor = NeuronAwareTextProcessor(num_neurons)
    words = text_processor.load_and_process_text()
    features = text_processor.words_to_neural_features(words)
    
    print(f"Feature matrix shape: {features.shape}")
    assert features.shape[1] == num_neurons, f"Feature count {features.shape[1]} != neuron count {num_neurons}"
    
    # Process neural data
    scaler = StandardScaler()
    features_scaled = scaler.fit_transform(features)
    features_tensor = torch.FloatTensor(features_scaled)
    spike_data = spikegen.rate(features_tensor, num_steps=num_steps)
    
    print(f"Spike data shape: {spike_data.shape}")
    
    # Run SNN
    snn_model = PolymorphicSNN(num_neurons, num_polymorphic)
    spk_rec, mem_rec, poly_mem_rec, mode_rec = run_polymorphic_snn(spike_data, snn_model)
    
    # Initialize FGCN model
    data = create_neuron_aligned_graph(spk_rec, mem_rec)
    fgcn_model = DataAwareFGCN(data.x.shape[1])
    
    # Create moderated text generator
    moderated_generator = FGCNModeratedTextGenerator(text_processor, fgcn_model)
    
    # Analysis and visualization
    poly_analysis = analyze_polymorphic_behavior(mode_rec, poly_mem_rec)
    img = generate_polymorphic_visualization(spk_rec, mem_rec, mode_rec, poly_analysis, img_size=img_size)
    
    print("="*50)
    print("POLYMORPHIC SNN ANALYSIS COMPLETE")
    print("="*50)
    print(f"Successfully processed {num_neurons} neurons ({num_polymorphic} polymorphic)")
    print(f"Spike activity range: {spk_rec.min().item():.3f} to {spk_rec.max().item():.3f}")
    if poly_analysis:
        print(f"Average mode switches per neuron: {np.mean(poly_analysis.get('mode_switching_frequency', [0])):.2f}")
        print(f"Most stable mode: {np.argmin(np.mean(poly_analysis.get('mode_stability', [[1]]), axis=0))}")
    
    # Visualize moderation effects
    coherence_signal, stability_signal = visualize_moderation_effects(spk_rec, mem_rec, fgcn_model)
    
    print("\n" + "="*60)
    print("FGCN-MODERATED TEXT GENERATOR READY")
    print("="*60)
    print("The system now uses graph neural networks to moderate text generation.")
    print("Enter a seed word(s) to generate text, or 'quit' to exit.")
    print("="*60)
    
    while True:
        user_input = input("\nEnter seed word(s) (or 'quit'): ").strip()
        if user_input.lower() == 'quit':
            print("Goodbye!")
            break
            
        seed_word = user_input if user_input else None
        
        moderated_text = moderated_generator.generate_text_from_neural_output(
    spk_rec, mem_rec, seed_word=seed_word, length=250)
        
        print(f"\n{'='*60}")
        print("FGCN-MODERATED GENERATION:")
        print(f"{'='*60}")
        print(moderated_text)
        print(f"{'='*60}")

if __name__ == "__main__":
    main_with_fgcn_moderation()
