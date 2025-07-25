import torch
import torch.nn as nn
import snntorch as snn
import numpy as np
from torch_geometric.data import Data
from torch_geometric.nn import GCNConv
from sklearn.preprocessing import StandardScaler
import random
import time
from collections import defaultdict

class PolymorphicNeuron(nn.Module):
    def __init__(self, input_dim, num_modes=256):
        super().__init__()
        self.modes = nn.ModuleList([
            snn.Leaky(beta=max(0.01, i/(num_modes+1))) for i in range(1, num_modes+1)
        ])
        self.mode_selector = nn.Linear(input_dim, num_modes)
        self.softmax = nn.Softmax(dim=-1)
        
        # Force dynamics parameters
        self.force_weights = nn.Parameter(torch.randn(num_modes, input_dim) * 0.1)
        self.damping_factor = nn.Parameter(torch.tensor(0.3))
        self.spring_constant = nn.Parameter(torch.tensor(0.8))
        self.previous_x = None
        self.velocity = None
        
        # Microsleep parameters
        self.microsleep_threshold = nn.Parameter(torch.tensor(0.7))
        self.microsleep_duration = 3  # timesteps
        self.microsleep_counter = torch.zeros(num_modes)
        self.in_microsleep = torch.zeros(num_modes, dtype=torch.bool)

    def apply_force_dynamics(self, x, mode_scores, dt=0.01):
        """Apply spring-damper force equations to neuron dynamics"""
        batch_size = x.shape[0]
        
        # Initialize velocity if needed
        if self.velocity is None:
            self.velocity = torch.zeros_like(x)
            self.previous_x = x.clone()
        
        # Calculate displacement from equilibrium
        equilibrium = torch.matmul(mode_scores, self.force_weights)
        displacement = x - equilibrium
        
        # Update velocity (simple finite difference)
        self.velocity = (x - self.previous_x) / dt if self.previous_x is not None else torch.zeros_like(x)
        
        # Apply spring-damper forces: F = -kx - cv
        spring_force = -self.spring_constant * displacement
        damping_force = -self.damping_factor * self.velocity
        total_force = spring_force + damping_force
        
        # Update position using Euler integration
        acceleration = total_force / 1.0  # assuming unit mass
        new_x = x + self.velocity * dt + 0.5 * acceleration * dt**2
        
        # Store current state for next iteration
        self.previous_x = x.clone()
        
        return new_x, total_force

    def manage_microsleeps(self, mode_scores):
        """Implement microsleep dynamics for overactive modes"""
        batch_size = mode_scores.shape[0]
        
        # Check for microsleep triggers (average across batch)
        avg_scores = mode_scores.mean(dim=0)
        high_activity = avg_scores > self.microsleep_threshold
        
        # Start microsleeps for highly active modes
        new_microsleeps = high_activity & ~self.in_microsleep
        self.in_microsleep = self.in_microsleep | new_microsleeps
        self.microsleep_counter[new_microsleeps] = self.microsleep_duration
        
        # Update microsleep counters
        self.microsleep_counter[self.in_microsleep] -= 1
        
        # Wake up modes that have completed microsleep
        wake_up = (self.microsleep_counter <= 0) & self.in_microsleep
        self.in_microsleep[wake_up] = False
        
        # Apply microsleep suppression
        suppressed_scores = mode_scores.clone()
        for i, in_sleep in enumerate(self.in_microsleep):
            if in_sleep:
                suppressed_scores[:, i] *= 0.1  # Reduce activity during microsleep
                
        return suppressed_scores

    def forward(self, x, mem_states=None):
        # Get mode selection scores
        mode_scores = self.softmax(self.mode_selector(x))
        
        # Apply force dynamics
        x_modified, forces = self.apply_force_dynamics(x, mode_scores)
        
        # Manage microsleeps
        mode_scores = self.manage_microsleeps(mode_scores)
        
        mode_outs, new_states = [], []
        for i, mode in enumerate(self.modes):
            # Introduce microsleep delays
            if self.in_microsleep[i]:
                time.sleep(0.0001)  # 0.1ms microsleep
                
            out, state = mode(x_modified) if mem_states is None else mode(x_modified, mem_states[i])
            mode_outs.append(out)
            new_states.append(state)
        
        stacked = torch.stack(mode_outs, dim=0)
        mixed = (stacked.permute(1,0,2) * mode_scores.unsqueeze(-1)).sum(1)
        
        return mixed, new_states, mode_scores, forces

class PolymorphicSNN(nn.Module):
    def __init__(self, num_neurons, num_polymorphic):
        super().__init__()
        self.num_neurons = num_neurons
        self.num_polymorphic = num_polymorphic
        
        self.lin = nn.Linear(num_neurons, num_neurons)
        self.regular = snn.Leaky(beta=0.5)
        self.poly = nn.ModuleList([
            PolymorphicNeuron(num_neurons, 8) for _ in range(num_polymorphic)
        ])
        
        # Network-level force dynamics
        self.network_coupling = nn.Parameter(torch.randn(num_polymorphic, num_polymorphic) * 0.1)
        self.global_inhibition = nn.Parameter(torch.tensor(0.2))
        
        # Network microsleep management
        self.network_fatigue = torch.zeros(1)
        self.fatigue_threshold = 5.0
        self.recovery_rate = 0.1

    def apply_network_forces(self, poly_outputs):
        """Apply inter-neuron coupling forces"""
        if len(poly_outputs) < 2:
            return poly_outputs
            
        stacked_outputs = torch.stack([out.mean(dim=-1) for out in poly_outputs], dim=0)
        
        # Apply coupling matrix
        coupling_forces = torch.matmul(self.network_coupling, stacked_outputs)
        
        # Apply global inhibition
        mean_activity = torch.stack(poly_outputs, dim=0).mean()
        inhibition = self.global_inhibition * mean_activity
        
        # Modify outputs based on forces
        modified_outputs = []
        for i, output in enumerate(poly_outputs):
            force_modulation = coupling_forces[i].unsqueeze(-1).expand_as(output)
            modified_output = output + force_modulation - inhibition
            modified_outputs.append(modified_output)
            
        return modified_outputs

    def network_microsleep(self, total_activity):
        """Implement network-wide microsleep when fatigue is high"""
        self.network_fatigue += total_activity.item()
        
        if self.network_fatigue > self.fatigue_threshold:
            print("Network microsleep initiated...")
            time.sleep(0.001)  # 1ms network-wide microsleep
            self.network_fatigue *= 0.5  # Partial recovery
            return True
        
        # Gradual recovery
        self.network_fatigue = max(0, self.network_fatigue - self.recovery_rate)
        return False

    def forward(self, x, mem=None, pmem=None):
        if x.dim() == 1:
            x = x.unsqueeze(0)
        
        x = self.lin(x)
        
        # Regular neuron processing
        reg_out, reg_mem = self.regular(x) if mem is None else self.regular(x, mem)
        
        # Polymorphic neuron processing
        poly_outputs = []
        poly_mems = []
        total_forces = []
        
        for i, poly_neuron in enumerate(self.poly):
            p_mem = None if pmem is None else pmem[i]
            p_out, p_mem_new, mode_scores, forces = poly_neuron(x, p_mem)
            poly_outputs.append(p_out)
            poly_mems.append(p_mem_new)
            total_forces.append(forces)
        
        # Apply network-level force dynamics
        if poly_outputs:
            poly_outputs = self.apply_network_forces(poly_outputs)
            
            # Calculate total network activity
            total_activity = sum(out.abs().sum() for out in poly_outputs)
            
            # Check for network microsleep
            self.network_microsleep(total_activity)
            
            # Combine polymorphic outputs
            combined_poly = torch.stack(poly_outputs, dim=0).mean(dim=0)
        else:
            combined_poly = torch.zeros_like(reg_out)
            poly_mems = []
        
        # Combine regular and polymorphic outputs with force influence
        final_output = reg_out + combined_poly
        
        return final_output, reg_mem, poly_mems

def create_neuron_graph(spk_rec):
    """Create a graph from spike recordings with proper edge handling"""
    # spk_rec shape is [timesteps, neurons]
    # We want node_features to be [nodes, features]
    # Each neuron becomes a node, each timestep becomes a feature
    node_features = spk_rec.T  # Shape: [neurons, timesteps]
    num_nodes = node_features.shape[0]
    
    print(f"Creating graph with {num_nodes} nodes, features shape: {node_features.shape}")
    print(f"Original spk_rec shape: {spk_rec.shape}")
    
    if num_nodes == 0:
        edge_index = torch.empty((2, 0), dtype=torch.long)
    elif num_nodes == 1:
        # Single node - no edges, empty edge_index
        edge_index = torch.empty((2, 0), dtype=torch.long)
        print(f"Single node - using empty edge_index shape: {edge_index.shape}")
    elif num_nodes == 2:
        # Two nodes - connect them bidirectionally
        edge_index = torch.tensor([[0, 1], [1, 0]], dtype=torch.long).T
        print(f"Two nodes edge_index shape: {edge_index.shape}")
    else:
        # Multiple nodes - create ring topology with bidirectional edges
        src_nodes, dst_nodes = [], []
        for i in range(num_nodes):
            # Connect to next node (with wraparound)
            next_node = (i + 1) % num_nodes
            src_nodes.extend([i, next_node])
            dst_nodes.extend([next_node, i])
        edge_index = torch.tensor([src_nodes, dst_nodes], dtype=torch.long)
        print(f"Multi-node edge_index shape: {edge_index.shape}")
    
    print(f"Final edge_index shape: {edge_index.shape}")
    if edge_index.numel() > 0:
        print(f"Edge index content: {edge_index}")
    
    return Data(x=node_features, edge_index=edge_index)

class DataAwareGCN(nn.Module):
    def __init__(self, input_dim, hidden=64, out=32):
        super().__init__()
        # Remove GCN layers entirely and use only linear layers
        self.input_dim = input_dim
        self.out_dim = out
        
        # Use a more sophisticated linear network instead of GCN
        self.network = nn.Sequential(
            nn.Linear(input_dim, hidden),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden, out),
            nn.ReLU()
        )

    def forward(self, data):
        print(f"DEBUG Linear Network: data.x.shape = {data.x.shape}")
        print(f"DEBUG Linear Network: data.num_nodes = {data.num_nodes}")
        
        # Handle empty graph
        if data.num_nodes == 0:
            return torch.empty((0, self.out_dim))
        
        # Process all nodes with linear network
        try:
            result = self.network(data.x)
            print(f"Linear network result shape: {result.shape}")
            return result
        except Exception as e:
            print(f"Linear Network Error: {e}")
            # Final fallback - return zeros
            return torch.zeros((data.num_nodes, self.out_dim))

class TextGenerator:
    def __init__(self, corpus):
        self.transitions = defaultdict(list)
        self.build_transitions(corpus)

    def build_transitions(self, words):
        if len(words) < 2:
            return
        for i in range(len(words) - 1):
            self.transitions[words[i]].append(words[i + 1])

    def generate_text(self, start_word=None, length=50):
        if not self.transitions:
            return "No data for generation."
        
        if start_word not in self.transitions:
            current_word = random.choice(list(self.transitions.keys()))
        else:
            current_word = start_word

        words = [current_word]
        for _ in range(length - 1):
            next_words = self.transitions.get(current_word)
            if not next_words:
                if not self.transitions:
                    break
                current_word = random.choice(list(self.transitions.keys()))
            else:
                current_word = random.choice(next_words)
            words.append(current_word)
        return ' '.join(words)

def max_psychological_overlap(generator, generator_instruction, psychological_words, n=1000):
    """Find text generation with maximum psychological word overlap"""
    max_intersection = 0
    max_text = ""
    max_seed = None
    psychological_set = set(w.lower() for w in psychological_words)
    
    if not psychological_words:
        print("No psychological words provided.")
        return
    
    for i in range(min(n, len(psychological_set) * 10)):  # Prevent infinite loops
        seed = list(psychological_set)[i % len(psychological_set)]
        generated = generator.generate_text(start_word=seed, length=230)
        generated_set = set(generated.lower().split())
        intersection = psychological_set & generated_set
        intersect_size = len(intersection)
        
        if intersect_size > max_intersection:
            max_intersection = intersect_size
            max_text = generated
            max_seed = seed

    print(f"\nBest instruction word: {max_seed}")
    print(f"Overlap count: {max_intersection}")
    print("Generated text sample:")
    print(max_text)

def main():
    print("Initializing Polymorphic SNN with Force Dynamics and Microsleeps...")
    
    # Network parameters
    num_neurons = 64
    num_poly = 4
    steps = 20
    
    # Initialize network
    snn = PolymorphicSNN(num_neurons, num_poly)
    features = torch.rand((steps, num_neurons)) * 0.5  # Reduced amplitude for stability
    
    print(f"Processing {steps} timesteps...")
    
    # Run simulation
    spk_list = []
    mem = None
    pmem = None
    
    for t in range(steps):
        print(f"Timestep {t+1}/{steps}", end='\r')
        
        spk, mem, pmem = snn(features[t], mem, pmem)
        spk_list.append(spk)
        
        # Add small delay between timesteps
        time.sleep(0.001)
    
    print("\nSimulation complete!")
    
    # Combine spike data
    if spk_list:
        spk_rec = torch.cat([s.unsqueeze(0) for s in spk_list], dim=0)
        
        # Process graph data
        gdata = create_neuron_graph(spk_rec)
        print(f"Graph created with {gdata.num_nodes} nodes and {gdata.num_node_features} features")
        print(f"Graph edge_index: {gdata.edge_index}")
        print(f"Graph x shape: {gdata.x.shape}")

        # Skip GCN entirely if problematic
        if gdata.num_nodes <= 1 or gdata.num_node_features == 0:
            print(f"Skipping GCN processing - insufficient nodes or features")
            node_feats = gdata.x if gdata.num_nodes > 0 else None
        else:
            try:
                gcn = DataAwareGCN(gdata.num_node_features)
                node_feats = gcn(gdata)
                print(f"Node features from GCN: {node_feats.shape}")
            except Exception as e:
                print(f"GCN processing failed: {e}")
                print("Using raw node features instead")
                node_feats = gdata.x
    
    # Text generation component
    try:
        filename = input("\nEnter filename for text corpus (or press Enter for default): ").strip()
        if filename:
            with open(filename, 'r', encoding='utf-8') as f:
                text_corpus = f.read()
        else:
            raise FileNotFoundError
    except FileNotFoundError:
        print("Using default corpus.")
        text_corpus = """
        attention memory learning neural network brain cognitive science
        psychology neuroscience artificial intelligence machine learning
        consciousness perception emotion behavior thought reasoning logic
        the quick brown fox jumps over the lazy dog hello world
        neural networks process information through interconnected nodes
        cognitive systems exhibit emergent properties and complex behaviors
        """
   
    # Clean and prepare corpus
    corpus_words = text_corpus.lower().split()
    if len(corpus_words) < 2:
        corpus_words = ["hello", "world", "neural", "network", "brain", "mind"]
    
    generator = TextGenerator(corpus_words)
    
    # Interactive loop
    print("\nEntering interactive mode. Type 'quit' to exit.")
    instructions = []
    
    while True:
        user_input = input("\nUSER: ").strip()
        
        if user_input.lower() in ['quit', 'exit', 'q']:
            print("Goodbye!")
            break
        
        if not user_input:
            continue
        
        # Process user input
        user_words = user_input.lower().split()
        if not user_words:
            print("Please provide some words for analysis.")
            continue
            
        # Build instruction generator from user input and corpus
        combined_words = corpus_words + user_words
        generator_instruction = TextGenerator(combined_words)
        
        # Generate text with psychological overlap
        max_psychological_overlap(generator, generator_instruction, user_words, n=500)

if __name__ == "__main__":
    main()
