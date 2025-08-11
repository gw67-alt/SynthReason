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
import math

KB_LEN = 99999
class HeavyDutyCycleProbabilityManager:
    """Heavy duty cycle manager for probability distributions with neural feedback."""
    def __init__(self, cycle_length=100000000, duty_ratio=100000000000, decay_rate=100000000000):
        self.cycle_length = cycle_length
        self.duty_ratio = duty_ratio  # Active portion of the cycle
        self.decay_rate = decay_rate
        self.cycle_position = 0
        self.active_threshold = cycle_length * duty_ratio
        self.probability_buffer = []
        self.cycle_history = []
        self.thermal_accumulator = 0.0
        self.neural_feedback_gain = 0.2
        
        print(f"🔧 Heavy Duty Cycle Manager initialized:")
        print(f"   Cycle Length: {cycle_length}")
        print(f"   Duty Ratio: {duty_ratio:.2f} ({duty_ratio*100:.1f}%)")
        print(f"   Active Threshold: {self.active_threshold:.1f}")
        
    def is_active_phase(self):
        """Check if we're in the active phase of the duty cycle."""
        return self.cycle_position < self.active_threshold
        
    def get_duty_cycle_modulation(self):
        """Get current duty cycle modulation factor."""
        if self.is_active_phase():
            # Active phase: sine wave modulation for smooth transitions
            progress = self.cycle_position / self.active_threshold
            modulation = 0.5 + 0.5 * math.sin(progress * math.pi)
        else:
            # Inactive phase: exponential decay
            inactive_progress = (self.cycle_position - self.active_threshold) / (self.cycle_length - self.active_threshold)
            modulation = 0.1 * math.exp(-3 * inactive_progress)
            
        return modulation
        
    def apply_thermal_feedback(self, neural_activity):
        """Apply thermal feedback based on neural activity."""
        activity_level = np.mean(neural_activity) if isinstance(neural_activity, np.ndarray) else neural_activity.mean().item()
        
        # Accumulate thermal energy
        self.thermal_accumulator += activity_level * 0.1
        self.thermal_accumulator *= self.decay_rate  # Natural cooling
        
        # Thermal modulation affects cycle timing
        thermal_speedup = 1.0 + self.thermal_accumulator * 0.2
        return thermal_speedup
        
    def update_cycle(self, neural_activity=None):
        """Update the duty cycle position with neural feedback."""
        speedup = 1.0
        if neural_activity is not None:
            speedup = self.apply_thermal_feedback(neural_activity)
            
        self.cycle_position += speedup
        
        if self.cycle_position >= self.cycle_length:
            # Complete cycle - record and reset
            self.cycle_history.append({
                'thermal_peak': self.thermal_accumulator,
                'avg_activity': np.mean(self.probability_buffer) if self.probability_buffer else 0.0
            })
            self.cycle_position = 0
            self.probability_buffer.clear()
            
            # Keep only last 10 cycles
            if len(self.cycle_history) > 10:
                self.cycle_history.pop(0)
                
    def modulate_probabilities(self, base_probabilities, neural_activity=None):
        """Apply heavy duty cycle modulation to probabilities."""
        self.update_cycle(neural_activity)
        
        modulation = self.get_duty_cycle_modulation()
        phase_indicator = "ACTIVE" if self.is_active_phase() else "INACTIVE"
        
        # Store for history tracking
        if isinstance(base_probabilities, torch.Tensor):
            self.probability_buffer.append(base_probabilities.mean().item())
            modulated = base_probabilities * modulation
        else:
            avg_prob = np.mean(base_probabilities)
            self.probability_buffer.append(avg_prob)
            modulated = base_probabilities * modulation
            
        # Neural feedback amplification during active phase
        if self.is_active_phase() and neural_activity is not None:
            if isinstance(neural_activity, torch.Tensor):
                feedback_boost = 1.0 + (neural_activity.mean().item() * self.neural_feedback_gain * modulation)
            else:
                feedback_boost = 1.0 + (np.mean(neural_activity) * self.neural_feedback_gain * modulation)
            modulated = modulated * feedback_boost
            
       
        return modulated
        
    def get_cycle_statistics(self):
        """Get current cycle statistics."""
        return {
            'current_position': self.cycle_position,
            'cycle_progress': self.cycle_position / self.cycle_length,
            'is_active': self.is_active_phase(),
            'modulation_factor': self.get_duty_cycle_modulation(),
            'thermal_level': self.thermal_accumulator,
            'completed_cycles': len(self.cycle_history)
        }

class SimpleSNN(nn.Module):
    """Simplified SNN with heavy duty cycle probability integration."""
    def __init__(self, num_neurons):
        super().__init__()
        self.num_neurons = num_neurons
        self.input_layer = nn.Linear(num_neurons, num_neurons)
        self.neurons = snn.Leaky(
            beta=0.5, 
            init_hidden=False, 
            spike_grad=surrogate.fast_sigmoid()
        )
        self.global_adaptation = nn.Parameter(torch.ones(1) * 0.5)
        
        # Heavy duty cycle integration
        self.duty_cycle_manager = HeavyDutyCycleProbabilityManager()
        self.probability_gate = nn.Linear(num_neurons, num_neurons)
        
    def forward(self, x, mem=None):
        # Ensure input has correct shape and type
        if x.dim() == 1:
            x = x.unsqueeze(0)  # Add batch dimension if missing
        
        x_processed = self.input_layer(x)
        
        # Apply probability gating with duty cycle modulation
        prob_weights = torch.sigmoid(self.probability_gate(x_processed))
        modulated_weights = self.duty_cycle_manager.modulate_probabilities(
            prob_weights, 
            neural_activity=x_processed
        )
        
        # Apply modulated probabilities to input
        x_modulated = x_processed * modulated_weights
        
        if mem is None:
            spk, mem = self.neurons(x_modulated)
        else:
            spk, mem = self.neurons(x_modulated, mem)
        
        # Apply global adaptation with duty cycle influence
        cycle_stats = self.duty_cycle_manager.get_cycle_statistics()
        adaptive_gain = self.global_adaptation * (1 + cycle_stats['modulation_factor'])
        spk = spk * adaptive_gain
        
        return spk.squeeze(0), mem.squeeze(0)  # Remove batch dim for consistency
    
    def get_duty_cycle_status(self):
        """Get current duty cycle status for monitoring."""
        return self.duty_cycle_manager.get_cycle_statistics()

def run_simple_snn(spike_data, snn_model):
    """Run the simplified SNN with heavy duty cycle tracking."""
    spk_rec, mem_rec = [], []
    duty_cycle_states = []
    utils.reset(snn_model)
    mem = None
    
    print("🚀 Starting SNN processing with heavy duty cycle...")
    
    for step in range(spike_data.shape[0]):
        if spike_data.shape[1] > 0:
            # Fix: Access the correct dimension based on spike_data shape
            if spike_data.dim() == 3:  # (time_steps, batch, features)
                input_data = spike_data[step, 0, :]  # Get first batch
            else:  # (time_steps, features)
                input_data = spike_data[step, :]
                
            spk, mem = snn_model(input_data, mem)
            spk_rec.append(spk)
            mem_rec.append(mem)
            
            # Record duty cycle state
            duty_state = snn_model.get_duty_cycle_status()
            duty_cycle_states.append(duty_state)
            
            if step % 20 == 0:
                phase = "ACTIVE" if duty_state['is_active'] else "INACTIVE"
                print(f"   Step {step}: Phase={phase}, Modulation={duty_state['modulation_factor']:.3f}")
    
    if spk_rec:
        spk_rec = torch.stack(spk_rec)
        mem_rec = torch.stack(mem_rec)
    else:
        # Handle empty case
        spk_rec = torch.zeros(1, snn_model.num_neurons)
        mem_rec = torch.zeros(1, snn_model.num_neurons)
    
    print(f"✅ SNN processing complete. Output shapes: spk={spk_rec.shape}, mem={mem_rec.shape}")
    return spk_rec, mem_rec, duty_cycle_states

class NeuronAwareTextProcessor:
    """Text processor with heavy duty cycle probability enhancement."""
    def __init__(self, num_neurons=16):
        self.num_neurons = num_neurons
        self.vocab = {}
        self.word_to_idx = {}
        self.idx_to_word = {}
        self.bigram_counts = Counter()
        self.transition_matrix = None
        self.transition_probs = None
        self.duty_cycle_manager = HeavyDutyCycleProbabilityManager(
            cycle_length=32, 
            duty_ratio=0.8, 
            decay_rate=0.92
        )
    
    def load_and_process_text(self, file_path="test.txt"):
        with open(file_path, 'r', encoding='utf-8') as f:
            content = ' '.join(f.read().split()[:KB_LEN])
            print(f"📚 Loaded {len(content)} characters from {file_path}")
   
        words = content.lower().split()
        words = [w for w in words if w]
        unique_words = list(set(words))
        self.word_to_idx = {word: idx for idx, word in enumerate(unique_words)}
        self.idx_to_word = {idx: word for word, idx in self.word_to_idx.items()}
        
        for i in range(len(words) - 1):
            self.bigram_counts[(words[i], words[i+1])] += 1
        
        self.create_transition_matrix_features()
        return words
    
    def create_transition_matrix_features(self):
        """Create transition matrix with heavy duty cycle probability enhancement."""
        vocab_size = len(self.word_to_idx)
        self.transition_matrix = np.zeros((vocab_size, vocab_size))
        
        # Fill transition matrix
        for (w1, w2), count in self.bigram_counts.items():
            if w1 in self.word_to_idx and w2 in self.word_to_idx:
                i, j = self.word_to_idx[w1], self.word_to_idx[w2]
                self.transition_matrix[i, j] += count
        
        # Normalize rows to get initial probabilities
        row_sums = self.transition_matrix.sum(axis=1, keepdims=True)
        initial_probs = np.divide(self.transition_matrix, row_sums, 
                                out=np.zeros_like(self.transition_matrix), 
                                where=row_sums!=0)
        
        # Apply heavy duty cycle modulation to transition probabilities
        modulated_probs = self.duty_cycle_manager.modulate_probabilities(
            initial_probs, 
            neural_activity=initial_probs
        )
        
        # Apply double negative logic with duty cycle enhancement
        first_negation = 1 - modulated_probs
        epsilon = 1e-8
        first_negation = np.maximum(first_negation, epsilon)
        
        # Second negation with duty cycle boost
        self.transition_probs = 1 - first_negation
        duty_boost = 1.0 + self.duty_cycle_manager.get_duty_cycle_modulation() * 0.1
        self.transition_probs = np.where(self.transition_probs > epsilon, 
                                        self.transition_probs * duty_boost, 
                                        self.transition_probs)
        
        # Re-normalize
        final_sums = self.transition_probs.sum(axis=1, keepdims=True)
        self.transition_probs = np.divide(self.transition_probs, final_sums, 
                                        out=np.zeros_like(self.transition_probs), 
                                        where=final_sums!=0)
    
    def get_transition_features(self, word):
        """Extract transition-based features with duty cycle modulation."""
        features = []
        word_idx = self.word_to_idx.get(word, 0)
        
        if self.transition_probs is not None:
            # Get base transition features
            out_transitions = self.transition_probs[word_idx]
            
            # Apply duty cycle modulation to features
            modulated_transitions = self.duty_cycle_manager.modulate_probabilities(
                out_transitions,
                neural_activity=out_transitions
            )
            
            # Number of possible next words
            transition_diversity = np.sum(modulated_transitions > 0)
            features.append(transition_diversity)
            
            # Maximum transition probability (duty cycle enhanced)
            max_prob = np.max(modulated_transitions)
            features.append(max_prob)
            
            # Entropy of transitions (with duty cycle weighting)
            probs = modulated_transitions[modulated_transitions > 0]
            if len(probs) > 0:
                entropy = -np.sum(probs * np.log2(probs + 1e-8))
            else:
                entropy = 0
            features.append(entropy)
            
            # Incoming transitions
            in_transitions = self.transition_probs[:, word_idx]
            in_degree = np.sum(in_transitions > 0)
            features.append(in_degree)
            
            # Centrality measure with duty cycle boost
            centrality = np.sum(in_transitions) * (1 + self.duty_cycle_manager.get_duty_cycle_modulation())
            features.append(centrality)
        else:
            features.extend([0, 0, 0, 0, 0])
        
        return features
    
    def get_semantic_similarity(self, word, user_input):
        """Semantic similarity with heavy duty cycle probability weighting."""
        if not user_input:
            return 0.5
        
        user_words = user_input.lower().split()
        max_similarity = 0.0
        common = 0
        
        for user_word in user_words:
            word_weight = self.word_to_idx.get(user_word, 0)
            common += word_weight
        
        # Apply duty cycle modulation to similarity
        base_similarity = min(1.0, common / 100.0)  # Normalize
        duty_modulated = self.duty_cycle_manager.modulate_probabilities(
            np.array([base_similarity]),
            neural_activity=np.array([common])
        )
        
        return float(duty_modulated[0])
    
    def get_context_features(self, word, prev_word=None, next_word=None):
        """Get contextual features with duty cycle enhancement."""
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
        
        # Apply duty cycle modulation to context features
        if features:
            modulated_features = self.duty_cycle_manager.modulate_probabilities(
                np.array(features),
                neural_activity=np.array(features)
            )
            return modulated_features.tolist()
        
        return features
    
    def words_to_neural_features(self, words, user_input=None, max_words=50):
        """Convert words to neural-compatible features with heavy duty cycle integration."""
        features = []
        
        # If user input is provided, prioritize it
        if user_input:
            user_words = user_input.lower().split()
            combined_words = user_words + words[:max(0, max_words-len(user_words))]
        else:
            combined_words = words[:max_words]
        
        for i, word in enumerate(combined_words):
            word_idx = self.word_to_idx.get(word, 0)
            
            # Start with transition-based features (duty cycle enhanced)
            feature_vector = self.get_transition_features(word)
            
            # Add context features (duty cycle enhanced)
            prev_word = combined_words[i-1] if i > 0 else None
            next_word = combined_words[i+1] if i < len(combined_words)-1 else None
            context_features = self.get_context_features(word, prev_word, next_word)
            feature_vector.extend(context_features)
            
            # Apply heavy duty cycle weighting to all features
            if user_input and i < len(user_input.split()):
                context_weight = 2.0
                position_weight = 1.0 - (i / len(user_input.split())) if len(user_input.split()) > 0 else 1.0
            else:
                context_weight = 1.0
                position_weight = 0.5
            
            # Get duty cycle modulation for this word
            duty_modulation = self.duty_cycle_manager.get_duty_cycle_modulation()
            total_weight = context_weight * position_weight * (1 + duty_modulation)
            
            # Apply weights to existing features
            feature_vector = [f * total_weight * word_idx for f in feature_vector]
            
            # Add word embedding-like features
            feature_vector.append(word_idx / len(self.word_to_idx))
            feature_vector.append(len(word) / 20.0)
            
            # Add semantic similarity to user input (duty cycle enhanced)
            if user_input:
                similarity = self.get_semantic_similarity(word, user_input)
                feature_vector.append(similarity)
            else:
                feature_vector.append(0.0)
            
            # Pad or truncate to match neuron count
            while len(feature_vector) < self.num_neurons:
                feature_vector.append(np.sin(len(feature_vector) * word_idx / 10.0) * duty_modulation)
            feature_vector = feature_vector[:self.num_neurons]
            features.append(feature_vector)
        
        return np.array(features)

# [Rest of the classes remain the same: create_neuron_aligned_graph, DataAwareFGCN, neuron_to_image_mapping, TextGenerator, FGCNModeratedTextGenerator, SubjectiveOntologyProcessor, UserContextAwareTextGenerator]

def process_user_input_through_snn(filename, user_input, text_processor, snn_model, num_steps=10):
    """Process user input through the SNN with heavy duty cycle tracking."""
    # Load base text data
    words = text_processor.load_and_process_text(filename)
    
    print("🔧 Processing through Heavy Duty Cycle SNN...")
    
    # Calculate subjective ontology components
    user_words = user_input.lower().split()
    last_word = user_words[-1] if user_words else ''
    p = text_processor.get_semantic_similarity(last_word, user_input)
    transition_features = text_processor.get_transition_features(last_word)
    c = np.mean(transition_features) if transition_features else 0.0
    e = 0.0  # Environmental factor
    
    print(f"📊 Subjective Ontology - P: {p:.3f}, C: {c:.3f}, E: {e:.3f}")
    
    # Generate features for user input (with duty cycle enhancement)
    user_features = text_processor.words_to_neural_features(
        words, 
        user_input=user_input
    )
    
    # Scale features
    scaler = StandardScaler()
    features_scaled = scaler.fit_transform(user_features)
    
    # Convert to tensor and ensure correct shape
    features_tensor = torch.FloatTensor(features_scaled)
    
    # Ensure we have proper dimensions for pooling
    if features_tensor.dim() == 2:
        features_tensor = features_tensor.unsqueeze(1)  # Add channel dimension
    
    # Handle case where feature dimension might be odd
    if features_tensor.shape[2] % 2 != 0:
        # Pad to make it even for pooling
        padding = torch.zeros(features_tensor.shape[0], features_tensor.shape[1], 1)
        features_tensor = torch.cat([features_tensor, padding], dim=2)
    
    # Create pooling layers
    pool = nn.MaxPool1d(kernel_size=2, stride=2, return_indices=True)
    unpool = nn.MaxUnpool1d(kernel_size=2, stride=2)
    
    # Apply pooling and unpooling
    pooled_output, indices = pool(features_tensor)
    unpooled_features = unpool(pooled_output, indices, output_size=features_tensor.size())
    
    # Remove channel dimension if added
    if unpooled_features.dim() == 3 and unpooled_features.size(1) == 1:
        unpooled_features = unpooled_features.squeeze(1)
    
    # Ensure correct feature dimension matches num_neurons
    if unpooled_features.shape[1] != snn_model.num_neurons:
        # Resize to match expected dimensions
        if unpooled_features.shape[1] > snn_model.num_neurons:
            unpooled_features = unpooled_features[:, :snn_model.num_neurons]
        else:
            # Pad with zeros
            padding_size = snn_model.num_neurons - unpooled_features.shape[1]
            padding = torch.zeros(unpooled_features.shape[0], padding_size)
            unpooled_features = torch.cat([unpooled_features, padding], dim=1)
    
    # Generate spikes
    spike_data = spikegen.rate(unpooled_features, num_steps=num_steps)
    
    # Process through SNN with heavy duty cycle tracking - FIXED
    spk_rec, mem_rec, duty_cycle_states = run_simple_snn(spike_data, snn_model)
    
    print("✅ Heavy duty cycle SNN processing complete!")
    print(f"📈 Final duty cycle stats: {snn_model.get_duty_cycle_status()}")
    
    return spk_rec, mem_rec, duty_cycle_states

# [Include all other unchanged classes here: create_neuron_aligned_graph, DataAwareFGCN, etc.]

def main_with_user_context_awareness():
    """Main function with heavy duty cycle SNN implementation."""
    num_neurons = 256
    num_steps = 10
    img_size = 8
    
    print("="*70)
    print("🔧 HEAVY DUTY CYCLE SNN TEXT GENERATOR")
    print("="*70)
    print("Enhanced with heavy duty cycle probability modulation")
    print("for robust neural pattern processing and text generation.")
    print("="*70)
    
    print(f"⚙️  Initializing Heavy Duty Cycle SNN with {num_neurons} neurons")
    
    # Initialize components with heavy duty cycle SNN
    text_processor = NeuronAwareTextProcessor(num_neurons)
    snn_model = SimpleSNN(num_neurons)
    
    filename = input("📁 Enter dataset filename: ")
    
    while True:
        user_input = input("\n👤 USER: ").strip()
        if not user_input:
            print("Please enter some text.")
            continue
        
        print(f"\n🔄 Processing input: '{user_input}'")
        print("="*50)
        
        # Process user input through heavy duty cycle SNN
        spk_rec, mem_rec, duty_cycle_states = process_user_input_through_snn(
            filename, user_input, text_processor, snn_model, num_steps
        )
        
        # Print duty cycle statistics
        final_stats = snn_model.get_duty_cycle_status()
        print(f"\n📊 Heavy Duty Cycle Final Stats:")
        print(f"   Completed Cycles: {final_stats['completed_cycles']}")
        print(f"   Current Progress: {final_stats['cycle_progress']*100:.1f}%")
        print(f"   Active Phase: {'YES' if final_stats['is_active'] else 'NO'}")
        print(f"   Thermal Level: {final_stats['thermal_level']:.3f}")
        
        # Initialize FGCN model
        data = create_neuron_aligned_graph(spk_rec, mem_rec)
        fgcn_model = DataAwareFGCN(data.x.shape[1])
        
        # Create user-context-aware text generator
        context_generator = UserContextAwareTextGenerator(text_processor, fgcn_model)
        
        # Generate contextual response
        contextual_text = context_generator.generate_contextual_text(
            user_input, spk_rec, mem_rec, length=50
        )
        
        print(f"\n🤖 AI: {contextual_text}")
        print(f"\n📈 Duty cycle enhanced neural patterns successfully applied!")



class SimpleSNN(nn.Module):
    """Simplified SNN with only regular leaky neurons."""
    def __init__(self, num_neurons):
        super().__init__()
        self.num_neurons = num_neurons
        self.input_layer = nn.Linear(num_neurons, num_neurons)
        self.neurons = snn.Leaky(
            beta=0.5, 
            init_hidden=False, 
            spike_grad=surrogate.fast_sigmoid()
        )
        self.global_adaptation = nn.Parameter(torch.ones(1) * 0.5)
        
    def forward(self, x, mem=None):
        # Ensure input has correct shape and type
        if x.dim() == 1:
            x = x.unsqueeze(0)  # Add batch dimension if missing
        
        x_processed = self.input_layer(x)
        if mem is None:
            spk, mem = self.neurons(x_processed)
        else:
            spk, mem = self.neurons(x_processed, mem)
        
        # Apply global adaptation
        spk = spk * self.global_adaptation
        return spk.squeeze(0), mem.squeeze(0)  # Remove batch dim for consistency

def run_simple_snn(spike_data, snn_model):
    """Run the simplified SNN without polymorphic components."""
    spk_rec, mem_rec = [], []
    utils.reset(snn_model)
    mem = None
    
    for step in range(spike_data.shape[0]):
        if spike_data.shape[1] > 0:
            # Fix: Access the correct dimension based on spike_data shape
            if spike_data.dim() == 3:  # (time_steps, batch, features)
                input_data = spike_data[step, 0, :]  # Get first batch
            else:  # (time_steps, features)
                input_data = spike_data[step, :]
                
            spk, mem = snn_model(input_data, mem)
            spk_rec.append(spk)
            mem_rec.append(mem)
    
    if spk_rec:
        spk_rec = torch.stack(spk_rec)
        mem_rec = torch.stack(mem_rec)
    else:
        # Handle empty case
        spk_rec = torch.zeros(1, snn_model.num_neurons)
        mem_rec = torch.zeros(1, snn_model.num_neurons)
    
    return spk_rec, mem_rec

def create_neuron_aligned_graph(spk_rec, mem_rec):
    """Create graph from spike and membrane recordings."""
    print(f"Creating graph from spk_rec: {spk_rec.shape}, mem_rec: {mem_rec.shape}")
    min_steps = min(spk_rec.shape[0], mem_rec.shape[0])
    spk_rec_aligned = spk_rec[:min_steps]
    mem_rec_aligned = mem_rec[:min_steps]
    min_neurons = min(spk_rec_aligned.shape[1], mem_rec_aligned.shape[1])
    spk_rec_aligned = spk_rec_aligned[:, :min_neurons]
    mem_rec_aligned = mem_rec_aligned[:, :min_neurons]
    
    print(f"Aligned shapes - spk_rec: {spk_rec_aligned.shape}, mem_rec: {mem_rec_aligned.shape}")
    
    # Create node features by concatenating spike and membrane data
    node_features = spk_rec_aligned.T
    node_features = torch.cat([node_features, mem_rec_aligned.T], dim=1)
    print(f"Node features shape: {node_features.shape}")
    
    num_nodes = node_features.shape[0]
    print(f"Number of graph nodes: {num_nodes}")
    
    # Create edges (connect each node to its neighbors)
    edge_index = []
    if num_nodes > 1:
        for i in range(num_nodes):
            for j in range(i+1, min(i+4, num_nodes)):
                edge_index.extend([[j, i], [i, j]])
                for k in range(i+1, min(i+4, num_nodes)):
                    edge_index.extend([[i, k], [i, j]])

    
    if edge_index:
        edge_index = torch.tensor(edge_index, dtype=torch.long).t().contiguous()
    else:
        edge_index = torch.empty((2, 0), dtype=torch.long)
    
    print(f"Edge index shape: {edge_index.shape}")
    data = Data(x=node_features, edge_index=edge_index)
    return data

class DataAwareFGCN(nn.Module):
    """Graph Convolutional Network for neural data processing."""
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
    """Map neural features to image representation."""
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

class TextGenerator:
    """Base text generator using neural outputs."""
    def __init__(self, text_processor: NeuronAwareTextProcessor):
        self.text_processor = text_processor
        self.transitions = defaultdict(list)
        self.seed_transitions = defaultdict(list)
        self.build_transitions()
    
    def build_transitions(self):
        """Build transition probabilities."""
        for (w1, w2), count in self.text_processor.bigram_counts.items():
            self.transitions[w1].append((w2, count))
            self.seed_transitions[w1].append((w2, count))
    
    def get_seed_candidates(self, seed_words):
        """Find candidates based on seed words."""
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
        """Generate text based on neural output."""
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
                coherence_idx = i % len(graph_features['coherence'])
                coherence_boost = graph_features['coherence'][coherence_idx]
                neural_weight = max(0.1, neural_influence[neural_idx] * (1 + coherence_boost))
            else:
                neural_weight = max(0.1, neural_influence[neural_idx])
                
            weights = weights * (1 + neural_weight)
            weights = weights / weights.sum() if weights.sum() > 0 else np.ones_like(weights) / len(weights)
            
            next_word = np.random.choice(words, p=weights)
            if next_word != generated_words[-1]:
                generated_words.append(next_word)
            current_word = next_word
        
        return ' '.join(generated_words)
    
    def moderate_candidate_selection(self, candidates, graph_features, position):
        """Default moderation - can be overridden by subclasses."""
        return candidates

class FGCNModeratedTextGenerator(TextGenerator):
    """Text generator with FGCN moderation."""
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
    """User-context-aware text generator."""
    def __init__(self, text_processor: NeuronAwareTextProcessor, fgcn_model=None):
        super().__init__(text_processor, fgcn_model)
        self.user_context_weight = 0.7
        self.so_processor = SubjectiveOntologyProcessor(text_processor)
        
    def find_best_starting_word(self, user_words):
        """Find the best starting word based on user input."""
        if not user_words:
            return random.choice(list(self.transitions.keys()))
        
        # Try to find user words in transitions
        for word in reversed(user_words):
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
            
            for user_word in user_words:
                similarity = self.text_processor.get_semantic_similarity(word, user_word)
                if similarity > 0.1:
                    context_boost += similarity * weight
                    candidates = self.transitions.get(current_word, [])
                    
                    if not user_words or context_strength < 0.1:
                        return candidates
                    
                    # Boost candidates that are similar to user words
                    contextual_candidates = []
                    for word, weight in self.transitions.get(user_word, []):
                        context_boost = 1.0
                        
                        for user_word in user_words:
                            similarity = self.text_processor.get_semantic_similarity(word, user_word)
                            if similarity > 0.1:
                                context_boost += similarity * weight
            
            contextual_candidates.append((word, weight * context_boost))
        
        return contextual_candidates if contextual_candidates else candidates
    
    def generate_contextual_text(self, user_input, spk_rec, mem_rec, length=50):
        """Generate text that's contextually aware of user input."""
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
                if current_word != generated_words[-1]:
                    generated_words.append(current_word)
                continue
            
            # Apply subjective ontology weighting
            so_weighted_candidates = []
            for word, weight in candidates:
                so_score = self.so_processor.compute_so_score(word, user_input, spk_rec)
                adjusted_weight = weight * so_score
                so_weighted_candidates.append((word, adjusted_weight))
            
            words, weights = zip(*so_weighted_candidates)
            weights = np.array(weights, dtype=float)
            
            # Apply neural influence and user context
            neural_weight = max(0.1, neural_influence * (1 + user_context_strength))
            weights = weights * neural_weight
            weights = weights / weights.sum() if weights.sum() > 0 else np.ones_like(weights) / len(weights)
            
            next_word = np.random.choice(words, p=weights)
            if next_word != generated_words[-1]:
                generated_words.append(next_word)
            current_word = next_word
            
            # Decay user context over time
            user_context_strength *= context_decay
        
        return ' '.join(generated_words)
        
class SubjectiveOntologyProcessor:
    """Implements the subjective ontology equation: S_O = f(P, C, E)"""
    def __init__(self, text_processor):
        self.text_processor = text_processor
    
    def compute_so_score(self, word, user_input, spk_rec, weights=(0.4, 0.3, 0.3)):
        """Computes a subjective ontology score for a word."""
        # P: Personal Perception
        p = self.text_processor.get_semantic_similarity(word, user_input)
        
        # C: Cultural Context
        transition_features = self.text_processor.get_transition_features(word)
        c = np.mean(transition_features) if transition_features else 0.0
        
        # E: Environmental Factors
        e = spk_rec.mean().item() if spk_rec.numel() > 0 else 0.0
        
        # f: Linear combination
        s_o = weights[0] * p + weights[1] * c + weights[2] * e
        return max(0.1, s_o)


def process_user_input_through_snn(filename, user_input, text_processor, snn_model, num_steps=10):
    """Process user input through the simplified SNN."""
    # Load base text data
    words = text_processor.load_and_process_text(filename)
    
    # Calculate subjective ontology components
    user_words = user_input.lower().split()
    last_word = user_words[-1] if user_words else ''
    p = text_processor.get_semantic_similarity(last_word, user_input)
    transition_features = text_processor.get_transition_features(last_word)
    c = np.mean(transition_features) if transition_features else 0.0
    e = 0.0  # Environmental factor
    
    # Generate features for user input
    user_features = text_processor.words_to_neural_features(
        words, 
        user_input=user_input
    )
    
    # Scale features
    scaler = StandardScaler()
    features_scaled = scaler.fit_transform(user_features)
    
    # Convert to tensor and ensure correct shape
    features_tensor = torch.FloatTensor(features_scaled)
    
    # Ensure we have proper dimensions for pooling
    if features_tensor.dim() == 2:
        features_tensor = features_tensor.unsqueeze(1)  # Add channel dimension
    
    # Handle case where feature dimension might be odd
    if features_tensor.shape[2] % 2 != 0:
        # Pad to make it even for pooling
        padding = torch.zeros(features_tensor.shape[0], features_tensor.shape[1], 1)
        features_tensor = torch.cat([features_tensor, padding], dim=2)
    
    # Create pooling layers
    pool = nn.MaxPool1d(kernel_size=2, stride=2, return_indices=True)
    unpool = nn.MaxUnpool1d(kernel_size=2, stride=2)
    
    # Apply pooling and unpooling
    pooled_output, indices = pool(features_tensor)
    unpooled_features = unpool(pooled_output, indices, output_size=features_tensor.size())
    
    # Remove channel dimension if added
    if unpooled_features.dim() == 3 and unpooled_features.size(1) == 1:
        unpooled_features = unpooled_features.squeeze(1)
    
    # Ensure correct feature dimension matches num_neurons
    if unpooled_features.shape[1] != snn_model.num_neurons:
        # Resize to match expected dimensions
        if unpooled_features.shape[1] > snn_model.num_neurons:
            unpooled_features = unpooled_features[:, :snn_model.num_neurons]
        else:
            # Pad with zeros
            padding_size = snn_model.num_neurons - unpooled_features.shape[1]
            padding = torch.zeros(unpooled_features.shape[0], padding_size)
            unpooled_features = torch.cat([unpooled_features, padding], dim=1)
    
    # Generate spikes
    spike_data = spikegen.rate(unpooled_features, num_steps=num_steps)
    
    # Process through simplified SNN - FIXED: Pass spike_data instead of indices
    spk_rec, mem_rec = run_simple_snn(spike_data, snn_model)
    
    return spk_rec, mem_rec

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
    """Main function with simplified SNN (no polymorphic neurons)."""
    num_neurons = 256
    num_steps = 10
    img_size = 8
    
    print(f"Initializing Simplified SNN with {num_neurons} neurons")
    
    # Initialize components with simplified SNN
    text_processor = NeuronAwareTextProcessor(num_neurons)
    snn_model = SimpleSNN(num_neurons)
    
    print("="*60)
    print("SIMPLIFIED SNN TEXT GENERATOR")
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
        
        # Process user input through simplified SNN
        spk_rec, mem_rec = process_user_input_through_snn(
            filename, user_input, text_processor, snn_model, num_steps
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