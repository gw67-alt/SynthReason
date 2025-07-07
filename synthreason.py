import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import snntorch as snn
from snntorch import surrogate, utils, spikegen
from sklearn.preprocessing import StandardScaler
from torch_geometric.data import Data
from torch_geometric.nn import GCNConv, global_mean_pool
import matplotlib.pyplot as plt
from collections import defaultdict, Counter
import random
import re


class NeuronManagedSNN(nn.Module):
    def __init__(self, num_neurons):
        super().__init__()
        # Preserve neuron count throughout network
        self.input_layer = nn.Linear(num_neurons, num_neurons)
        self.snn_layer = snn.Leaky(beta=0.5, init_hidden=False, spike_grad=surrogate.fast_sigmoid())
        self.num_neurons = num_neurons
        
    def forward(self, x, mem=None):
        x = self.input_layer(x)
        if mem is None:
            spk, mem = self.snn_layer(x)
        else:
            spk, mem = self.snn_layer(x, mem)
        return spk, mem

def run_snn_with_proper_tracking(spike_data, snn_model):
    """Run SNN while properly tracking all neuron states"""
    spk_rec = []
    mem_rec = []
    utils.reset(snn_model)
    mem = None
    
    #print(f"Input spike_data shape: {spike_data.shape}")
    
    for step in range(spike_data.shape[0]):  # num_steps
        if spike_data.shape[1] > 0:  # Check samples exist
            # Use first sample for processing
            input_data = spike_data[step][0]  # Shape: [num_features]
            #print(f"Step {step}, input shape: {input_data.shape}")
            
            spk, mem = snn_model(input_data, mem)
            spk_rec.append(spk)
            mem_rec.append(mem)
            
            #print(f"Step {step}, spike shape: {spk.shape}, mem shape: {mem.shape}")
    
    spk_rec = torch.stack(spk_rec)  # [num_steps, num_neurons]
    mem_rec = torch.stack(mem_rec)  # [num_steps, num_neurons]
    
    #print(f"Final spk_rec shape: {spk_rec.shape}")
    #print(f"Final mem_rec shape: {mem_rec.shape}")
    
    return spk_rec, mem_rec

def create_neuron_aligned_graph(spk_rec, mem_rec):
    """Create graph ensuring each neuron becomes a node"""
    print(f"Creating graph from spk_rec: {spk_rec.shape}, mem_rec: {mem_rec.shape}")
    
    # Transpose to get [num_neurons, num_steps]
    node_features = spk_rec.T  
    node_features = torch.cat([node_features, mem_rec.T], dim=1)  # [num_neurons, 2*num_steps]
    
    print(f"Node features shape: {node_features.shape}")
    
    # Ensure we have the right number of nodes
    num_nodes = node_features.shape[0]
    print(f"Number of graph nodes: {num_nodes}")
    
    # Create edges between neurons
    edge_index = []
    if num_nodes > 1:
        # Create a more structured connectivity pattern
        for i in range(num_nodes):
            for j in range(i+1, min(i+4, num_nodes)):  # Connect to next 3 neurons
                edge_index.extend([[i, j], [j, i]])  # Bidirectional edges
    
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
        
        # Apply graph convolutions
        x = F.relu(self.gcn1(x, edge_index))
        x = F.relu(self.gcn2(x, edge_index))
        
        # Spatial attention
        attn_weights = torch.sigmoid(self.attn(x))
        x = x * attn_weights
        
        print(f"GCN output shape: {x.shape}")
        
        return x

def neuron_to_image_mapping(node_features, target_size=8):
    """Map neurons to image pixels with proper scaling"""
    num_nodes, feat_dim = node_features.shape
    target_pixels = target_size * target_size
    
    print(f"Mapping {num_nodes} neurons to {target_pixels} pixels")
    
    # If we have fewer neurons than pixels, replicate
    if num_nodes < target_pixels:
        repeat_factor = target_pixels // num_nodes + 1
        node_features = node_features.repeat(repeat_factor, 1)[:target_pixels]
    else:
        # If we have more neurons, take the first target_pixels
        node_features = node_features[:target_pixels]
    
    # Convert to image
    img_data = node_features.mean(dim=1).reshape(target_size, target_size)
    img_data = img_data.detach().cpu().numpy()
    
    # Normalize to [0, 1]
    img_data = (img_data - img_data.min()) / (img_data.max() - img_data.min() + 1e-8)
    
    print(f"Final image shape: {img_data.shape}")
    print(f"Image value range: [{img_data.min():.3f}, {img_data.max():.3f}]")
    
    return img_data

def generate_data_aware_visualization(spk_rec, mem_rec, img_size=8):
    """Generate visualization that properly shows neural data"""
    print("="*50)
    print("GENERATING DATA-AWARE VISUALIZATION")
    print("="*50)
    
    # Create graph from neural data
    data = create_neuron_aligned_graph(spk_rec, mem_rec)
    
    # Process through FGCN
    model = DataAwareFGCN(data.x.shape[1])
    with torch.no_grad():
        node_feats = model(data)
    
    # Map to image
    img = neuron_to_image_mapping(node_feats, target_size=img_size)
    
    # Create comprehensive visualization
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    # Main neural activity image
    im1 = axes[0, 0].imshow(img, cmap='viridis', interpolation='nearest')
    axes[0, 0].set_title('Neural Activity Heatmap')
    axes[0, 0].axis('off')
    plt.colorbar(im1, ax=axes[0, 0])
    
    # Spike activity over time
    spike_activity = spk_rec.sum(dim=1).detach().cpu().numpy()
    axes[0, 1].plot(spike_activity)
    axes[0, 1].set_title('Total Spike Activity Over Time')
    axes[0, 1].set_xlabel('Time Step')
    axes[0, 1].set_ylabel('Total Spikes')
    
    # Individual neuron activities
    neuron_activities = spk_rec.sum(dim=0).detach().cpu().numpy()
    axes[1, 0].bar(range(len(neuron_activities)), neuron_activities)
    axes[1, 0].set_title('Individual Neuron Activity')
    axes[1, 0].set_xlabel('Neuron Index')
    axes[1, 0].set_ylabel('Total Spikes')
    
    # Neural connectivity visualization
    if data.edge_index.shape[1] > 0:
        # Simple connectivity matrix
        num_neurons = spk_rec.shape[1]
        conn_matrix = torch.zeros(num_neurons, num_neurons)
        edges = data.edge_index.t()
        for edge in edges:
            conn_matrix[edge[0], edge[1]] = 1
        
        im4 = axes[1, 1].imshow(conn_matrix.numpy(), cmap='Blues')
        axes[1, 1].set_title('Neural Connectivity')
        axes[1, 1].set_xlabel('Target Neuron')
        axes[1, 1].set_ylabel('Source Neuron')
        plt.colorbar(im4, ax=axes[1, 1])
    else:
        axes[1, 1].text(0.5, 0.5, 'No Connections', ha='center', va='center')
        axes[1, 1].set_title('Neural Connectivity')
    
    plt.tight_layout()
    plt.show()
    
    return img

# Updated text processing with proper neuron management
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
                content = f.read()
                print(f"Loaded {len(content)} characters from {file_path}")
        except FileNotFoundError:
            content = "The neural network processes information through spiking patterns. Each neuron contributes to the overall computation."
            print("Using sample text")
        
        # Clean and tokenize
        words = content.lower().split()
        words = [w for w in words if w]
        
        # Build vocabulary
        unique_words = list(set(words))
        self.word_to_idx = {word: idx for idx, word in enumerate(unique_words)}
        self.idx_to_word = {idx: word for word, idx in self.word_to_idx.items()}
        
        # Build bigrams
        for i in range(len(words) - 1):
            self.bigram_counts[(words[i], words[i+1])] += 1
        
        return words
    
    def words_to_neural_features(self, words, max_words=50):
        """Convert words to features ensuring proper neuron count"""
        features = []
        
        for i, word in enumerate(words[:max_words]):
            # Create exactly num_neurons features
            word_idx = self.word_to_idx.get(word, 0)
            
            feature_vector = [
                word_idx / len(self.word_to_idx),  # Normalized word index
                len(word) / 20.0,                  # Word length
                i / len(words),                    # Position
                len(set(word)) / max(len(word), 1), # Character diversity
            ]
            
            # Extend to exactly num_neurons features
            while len(feature_vector) < self.num_neurons:
                # Add derived features
                feature_vector.append(np.sin(len(feature_vector) * word_idx / 10.0))
            
            # Ensure exactly num_neurons features
            feature_vector = feature_vector[:self.num_neurons]
            features.append(feature_vector)
        
        return np.array(features)

class TextGenerator:
    def __init__(self, text_processor: NeuronAwareTextProcessor):
        self.text_processor = text_processor
        self.transitions = defaultdict(list)
        self.build_transitions()
    
    def build_transitions(self):
        """Build transition probabilities from bigram counts"""
        for (w1, w2), count in self.text_processor.bigram_counts.items():
            self.transitions[w1].append((w2, count))
    
    def generate_text_from_neural_output(self, spk_rec, seed_word: str = None, length: int = 50) -> str:
        """Generate text using neural network output to influence word selection"""
        if not self.transitions:
            return "No training data available for text generation."
        
        # Use neural output to influence generation
        neural_influence = spk_rec.mean(dim=0).detach().cpu().numpy()
        
        # Start with seed word or random word
        if seed_word and seed_word in self.transitions:
            current_word = seed_word
        else:
            current_word = random.choice(list(self.transitions.keys()))
        
        generated_words = [current_word]
        
        for i in range(length - 1):
            if current_word not in self.transitions:
                # Restart with a random word
                current_word = random.choice(list(self.transitions.keys()))
            
            candidates = self.transitions[current_word]
            if not candidates:
                break
            
            # Use neural influence to modify probabilities
            words, weights = zip(*candidates)
            weights = np.array(weights, dtype=float)
            
            # Apply neural influence (use position in neural output)
            neural_idx = i % len(neural_influence)
            neural_weight = max(0.1, neural_influence[neural_idx])
            weights = weights * (1 + neural_weight)
            
            # Normalize weights
            weights = weights / weights.sum()
            
            # Select next word
            next_word = np.random.choice(words, p=weights)
            generated_words.append(next_word)
            current_word = next_word
        
        return ' '.join(generated_words)
# Main execution with proper neuron management
if __name__ == "__main__":
    # Parameters
    num_neurons = 2560
    num_steps = 10
    img_size = 8
    
    print(f"Initializing with {num_neurons} neurons")
    
    # Initialize components
    text_processor = NeuronAwareTextProcessor(num_neurons)
    words = text_processor.load_and_process_text()
    
    # Convert to features
    features = text_processor.words_to_neural_features(words)
    print(f"Feature matrix shape: {features.shape}")
    
    # Ensure we have the right number of features
    assert features.shape[1] == num_neurons, f"Feature count {features.shape[1]} != neuron count {num_neurons}"
    
    # Normalize and create spikes
    scaler = StandardScaler()
    features_scaled = scaler.fit_transform(features)
    features_tensor = torch.FloatTensor(features_scaled)
    spike_data = spikegen.rate(features_tensor, num_steps=num_steps)
    
    print(f"Spike data shape: {spike_data.shape}")
    
    # Initialize SNN with proper neuron management
    snn_model = NeuronManagedSNN(num_neurons)
    
    # Run SNN with tracking
    spk_rec, mem_rec = run_snn_with_proper_tracking(spike_data, snn_model)
    
    # Generate visualization
    img = generate_data_aware_visualization(spk_rec, mem_rec, img_size=img_size)
    
    print("="*50)
    print("NEURAL DATA VISUALIZATION COMPLETE")
    print("="*50)
    print(f"Successfully visualized {num_neurons} neurons")
    print(f"Spike activity range: {spk_rec.min().item():.3f} to {spk_rec.max().item():.3f}")
    print(f"Membrane activity range: {mem_rec.min().item():.3f} to {mem_rec.max().item():.3f}")

    # Initialize text generator and generate text
    text_generator = TextGenerator(text_processor)
    
    # Interactive text generation
    print("\n" + "="*60)
    print("NEURAL TEXT GENERATOR READY")
    print("="*60)
    print("Enter a seed word to generate text, or 'quit' to exit.")
    print("Leave empty for random generation.")
    print("="*60)
    
    while True:
        user_input = input("\nEnter seed word (or 'quit'): ").strip()
        
        if user_input.lower() == 'quit':
            print("Goodbye!")
            break
        
        # Generate text
        seed_word = user_input if user_input else None
        generated_text = text_generator.generate_text_from_neural_output(
            spk_rec, seed_word=seed_word, length=230
        )
        
        print(f"\nGenerated text: {generated_text}")
        
        # Optionally regenerate visualization with new neural state
        if user_input:
            # Process the seed word through the network
            seed_features = text_processor.words_to_neural_features([user_input])
            if seed_features.shape[0] > 0:
                seed_scaled = scaler.transform(seed_features)
                seed_tensor = torch.FloatTensor(seed_scaled)
                seed_spikes = spikegen.rate(seed_tensor, num_steps=num_steps)
                new_spk_rec, new_mem_rec = run_snn_with_proper_tracking(seed_spikes, snn_model)
                
