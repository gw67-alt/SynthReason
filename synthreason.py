import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
from collections import defaultdict, Counter
import random
import math
import pickle
import os
from pathlib import Path

KB_LEN = -1

class PyTorchHeavyDutyCycleProbabilityManager:
    """Heavy duty cycle manager for probability distributions with neural feedback - Pure PyTorch."""
    def __init__(self, cycle_length=32, duty_ratio=0.8, decay_rate=0.7, device='cpu'):
        self.cycle_length = cycle_length
        self.duty_ratio = duty_ratio
        self.decay_rate = decay_rate
        self.cycle_position = torch.tensor(0.0, device=device)
        self.active_threshold = cycle_length * duty_ratio
        self.probability_buffer = []
        self.cycle_history = []
        self.thermal_accumulator = torch.tensor(0.0, device=device)
        self.neural_feedback_gain = 0.2
        self.device = device
        
        print(f"🔧 PyTorch Heavy Duty Cycle Manager initialized:")
        print(f"   Device: {device}")
        print(f"   Cycle Length: {cycle_length}")
        print(f"   Duty Ratio: {duty_ratio:.2f} ({duty_ratio*100:.1f}%)")
        print(f"   Active Threshold: {self.active_threshold:.1f}")
        
    def is_active_phase(self):
        """Check if we're in the active phase of the duty cycle."""
        return self.cycle_position < self.active_threshold
        
    def get_duty_cycle_modulation(self):
        """Get current duty cycle modulation factor - PyTorch tensors."""
        if self.is_active_phase():
            progress = self.cycle_position / max(self.active_threshold, 1e-8)
            modulation = 0.5 + 0.5 * torch.sin(progress * torch.pi)
        else:
            inactive_progress = (self.cycle_position - self.active_threshold) / max(self.cycle_length - self.active_threshold, 1e-8)
            modulation = 0.1 * torch.exp(-3 * inactive_progress)
            
        return modulation
        
    def apply_thermal_feedback(self, neural_activity):
        """Apply thermal feedback based on neural activity - PyTorch."""
        if isinstance(neural_activity, torch.Tensor):
            activity_level = neural_activity.mean()
        else:
            activity_level = torch.tensor(np.mean(neural_activity), device=self.device)
        
        # Accumulate thermal energy
        self.thermal_accumulator += activity_level * 0.1
        self.thermal_accumulator *= self.decay_rate
        
        # Thermal modulation affects cycle timing
        thermal_speedup = 1.0 + self.thermal_accumulator * 0.2
        return thermal_speedup
        
    def update_cycle(self, neural_activity=None):
        """Update the duty cycle position with neural feedback - PyTorch."""
        speedup = torch.tensor(1.0, device=self.device)
        if neural_activity is not None:
            speedup = self.apply_thermal_feedback(neural_activity)
            
        self.cycle_position += speedup
        
        if self.cycle_position >= self.cycle_length:
            self.cycle_history.append({
                'thermal_peak': self.thermal_accumulator.item(),
                'avg_activity': np.mean(self.probability_buffer) if self.probability_buffer else 0.0
            })
            self.cycle_position = torch.tensor(0.0, device=self.device)
            self.probability_buffer.clear()
            
            if len(self.cycle_history) > 10:
                self.cycle_history.pop(0)
                
    def modulate_probabilities(self, base_probabilities, neural_activity=None):
        """Apply heavy duty cycle modulation to probabilities - Pure PyTorch."""
        self.update_cycle(neural_activity)
        
        modulation = self.get_duty_cycle_modulation()
        
        if isinstance(base_probabilities, torch.Tensor):
            self.probability_buffer.append(base_probabilities.mean().item())
            modulated = base_probabilities * modulation
        else:
            base_probs_tensor = torch.tensor(base_probabilities, device=self.device, dtype=torch.float32)
            avg_prob = base_probs_tensor.mean().item() if len(base_probabilities) > 0 else 0.0
            self.probability_buffer.append(avg_prob)
            modulated = base_probs_tensor * modulation
            
        # Neural feedback amplification during active phase
        if self.is_active_phase() and neural_activity is not None:
            if isinstance(neural_activity, torch.Tensor):
                feedback_boost = 1.0 + (neural_activity.mean() * self.neural_feedback_gain * modulation)
            else:
                activity_tensor = torch.tensor(np.mean(neural_activity), device=self.device)
                feedback_boost = 1.0 + (activity_tensor * self.neural_feedback_gain * modulation)
            modulated = modulated * feedback_boost
            
        return modulated
        
    def get_cycle_statistics(self):
        """Get current cycle statistics - PyTorch compatible."""
        return {
            'current_position': self.cycle_position.item(),
            'cycle_progress': (self.cycle_position / self.cycle_length).item(),
            'is_active': self.is_active_phase(),
            'modulation_factor': self.get_duty_cycle_modulation().item(),
            'thermal_level': self.thermal_accumulator.item(),
            'completed_cycles': len(self.cycle_history)
        }

class PyTorchLeakyIntegrateFireNeuron(nn.Module):
    """Custom Leaky Integrate-and-Fire neuron implementation in pure PyTorch."""
    def __init__(self, tau_mem=10.0, tau_syn=5.0, v_thresh=1.0, v_reset=0.0):
        super().__init__()
        self.tau_mem = tau_mem
        self.tau_syn = tau_syn
        self.v_thresh = v_thresh
        self.v_reset = v_reset
        
        # Decay factors
        self.beta = torch.exp(torch.tensor(-1.0 / tau_mem))
        self.alpha = torch.exp(torch.tensor(-1.0 / tau_syn))
        
    def forward(self, x, state=None):
        """
        Forward pass of LIF neuron.
        x: input current (batch_size, num_neurons)
        state: tuple of (membrane_potential, synaptic_current)
        Returns: (spikes, new_state)
        """
        batch_size, num_neurons = x.shape
        device = x.device
        
        if state is None:
            v_mem = torch.zeros(batch_size, num_neurons, device=device)
            i_syn = torch.zeros(batch_size, num_neurons, device=device)
        else:
            v_mem, i_syn = state
            
        # Update synaptic current
        i_syn = self.alpha * i_syn + x
        
        # Update membrane potential
        v_mem = self.beta * v_mem + i_syn
        
        # Generate spikes
        spikes = (v_mem >= self.v_thresh).float()
        
        # Reset membrane potential where spikes occurred
        v_mem = v_mem * (1 - spikes) + self.v_reset * spikes
        
        return spikes, (v_mem, i_syn)

class PyTorchSimpleSNN(nn.Module):
    """Pure PyTorch SNN with heavy duty cycle probability integration."""
    def __init__(self, num_neurons, device='cpu'):
        super().__init__()
        self.num_neurons = num_neurons
        self.device = device
        
        # Network layers
        self.input_layer = nn.Linear(num_neurons, num_neurons).to(device)
        self.lif_neurons = PyTorchLeakyIntegrateFireNeuron().to(device)
        self.global_adaptation = nn.Parameter(torch.ones(1, device=device) * 0.5)
        
        # Heavy duty cycle integration
        self.duty_cycle_manager = PyTorchHeavyDutyCycleProbabilityManager(device=device)
        self.probability_gate = nn.Linear(num_neurons, num_neurons).to(device)
        
        # Initialize neuron state
        self.neuron_state = None
        
    def forward(self, x, reset_state=False):
        """Forward pass with duty cycle integration."""
        if reset_state:
            self.neuron_state = None
            
        # Ensure input has correct shape and is on right device
        if x.dim() == 1:
            x = x.unsqueeze(0)
        x = x.to(self.device)
        
        x_processed = self.input_layer(x)
        
        # Apply probability gating with duty cycle modulation
        prob_weights = torch.sigmoid(self.probability_gate(x_processed))
        modulated_weights = self.duty_cycle_manager.modulate_probabilities(
            prob_weights, 
            neural_activity=x_processed
        )
        
        # Apply modulated probabilities to input
        x_modulated = x_processed * modulated_weights
        
        # Process through LIF neurons
        spikes, self.neuron_state = self.lif_neurons(x_modulated, self.neuron_state)
        
        # Apply global adaptation with duty cycle influence
        cycle_stats = self.duty_cycle_manager.get_cycle_statistics()
        adaptive_gain = self.global_adaptation * (1 + cycle_stats['modulation_factor'])
        spikes = spikes * adaptive_gain
        
        return spikes.squeeze(0), self.neuron_state
    
    def reset_neurons(self):
        """Reset neuron states."""
        self.neuron_state = None
        
    def get_duty_cycle_status(self):
        """Get current duty cycle status for monitoring."""
        return self.duty_cycle_manager.get_cycle_statistics()

class PyTorchSpikeGenerator:
    """Pure PyTorch spike generation utilities."""
    @staticmethod
    def rate_encoding(data, num_steps=10, max_rate=100):
        """
        Convert continuous data to rate-encoded spikes.
        data: torch.Tensor of shape (batch_size, features)
        num_steps: number of time steps
        max_rate: maximum firing rate
        """
        device = data.device
        batch_size, num_features = data.shape
        
        # Normalize data to [0, 1]
        data_norm = torch.clamp(data, 0, 1)
        
        # Generate spike probabilities
        spike_probs = data_norm * max_rate / 100.0
        
        # Generate random numbers and compare with probabilities
        spikes = torch.zeros(num_steps, batch_size, num_features, device=device)
        for t in range(num_steps):
            rand_vals = torch.rand_like(spike_probs)
            spikes[t] = (rand_vals < spike_probs).float()
            
        return spikes
    
    @staticmethod
    def temporal_encoding(data, num_steps=10):
        """
        Convert data to temporal encoding where higher values spike earlier.
        """
        device = data.device
        batch_size, num_features = data.shape
        
        # Normalize data to [0, 1]
        data_norm = torch.clamp(data, 0, 1)
        
        # Convert to spike times (earlier for higher values)
        spike_times = ((1 - data_norm) * (num_steps - 1)).long()
        
        # Generate spike trains
        spikes = torch.zeros(num_steps, batch_size, num_features, device=device)
        for b in range(batch_size):
            for f in range(num_features):
                t = spike_times[b, f].item()
                if t < num_steps:
                    spikes[t, b, f] = 1.0
                    
        return spikes

def run_pytorch_snn(spike_data, snn_model):
    """Run the PyTorch SNN with duty cycle tracking."""
    spk_rec, mem_rec = [], []
    snn_model.reset_neurons()
    
    print("🚀 Processing through PyTorch SNN with heavy duty cycle...")
    
    num_steps = spike_data.shape[0]
    for step in range(num_steps):
        if spike_data.shape[1] > 0:
            # Get input data for this time step
            if spike_data.dim() == 3:  # (time_steps, batch, features)
                input_data = spike_data[step, 0, :]
            else:  # (time_steps, features)
                input_data = spike_data[step, :]
            
            # Process through SNN
            spk, neuron_state = snn_model(input_data)
            spk_rec.append(spk)
            
            # Extract membrane potential from neuron state
            if neuron_state is not None and len(neuron_state) >= 1:
                mem_rec.append(neuron_state[0].squeeze(0))  # membrane potential
            else:
                mem_rec.append(torch.zeros_like(spk))
            
            # Log duty cycle status periodically
            if step % 20 == 0:
                duty_state = snn_model.get_duty_cycle_status()
                phase = "ACTIVE" if duty_state['is_active'] else "INACTIVE"
                print(f"   Step {step}: Phase={phase}, Modulation={duty_state['modulation_factor']:.3f}")
    
    if spk_rec:
        spk_rec = torch.stack(spk_rec)
        mem_rec = torch.stack(mem_rec)
    else:
        spk_rec = torch.zeros(1, snn_model.num_neurons)
        mem_rec = torch.zeros(1, snn_model.num_neurons)
    
    print(f"✅ PyTorch SNN processing complete. Output shapes: spk={spk_rec.shape}, mem={mem_rec.shape}")
    return spk_rec, mem_rec

class PyTorchNeuronAwareTextProcessor:
    """Text processor with PyTorch-native heavy duty cycle probability enhancement."""
    def __init__(self, num_neurons=256, device='cpu'):
        self.num_neurons = num_neurons
        self.device = device
        self.vocab = {}
        self.word_to_idx = {}
        self.idx_to_word = {}
        self.bigram_counts = Counter()
        self.transition_matrix = None
        self.transition_probs = None
        self.duty_cycle_manager = PyTorchHeavyDutyCycleProbabilityManager(
            cycle_length=64, 
            duty_ratio=0.75, 
            decay_rate=0.8,
            device=device
        )
    
    def load_and_process_text(self, file_path="test.txt"):
        """Load and process text data."""
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
        """Create transition matrix with PyTorch tensors and heavy duty cycle enhancement."""
        vocab_size = len(self.word_to_idx)
        self.transition_matrix = torch.zeros((vocab_size, vocab_size), device=self.device)
        
        # Fill transition matrix
        for (w1, w2), count in self.bigram_counts.items():
            if w1 in self.word_to_idx and w2 in self.word_to_idx:
                i, j = self.word_to_idx[w1], self.word_to_idx[w2]
                self.transition_matrix[i, j] += count
        
        # Normalize rows to get initial probabilities
        row_sums = self.transition_matrix.sum(dim=1, keepdim=True)
        row_sums = torch.where(row_sums == 0, torch.ones_like(row_sums), row_sums)
        initial_probs = self.transition_matrix / row_sums
        
        # Apply heavy duty cycle modulation to transition probabilities
        modulated_probs = self.duty_cycle_manager.modulate_probabilities(
            initial_probs, 
            neural_activity=initial_probs
        )
        
        # Apply double negative logic with duty cycle enhancement
        first_negation = 1 - modulated_probs
        epsilon = 1e-8
        first_negation = torch.clamp(first_negation, min=epsilon)
        
        # Second negation with duty cycle boost
        self.transition_probs = 1 - first_negation
        duty_boost = 1.0 + self.duty_cycle_manager.get_duty_cycle_modulation() * 0.1
        self.transition_probs = torch.where(self.transition_probs > epsilon, 
                                          self.transition_probs * duty_boost, 
                                          self.transition_probs)
        
        # Re-normalize
        final_sums = self.transition_probs.sum(dim=1, keepdim=True)
        final_sums = torch.where(final_sums == 0, torch.ones_like(final_sums), final_sums)
        self.transition_probs = self.transition_probs / final_sums
    
    def words_to_neural_features(self, words, user_input=None, max_words=50):
        """Convert words to neural-compatible PyTorch tensors with heavy duty cycle integration."""
        # Process words into feature vectors (similar to original but return torch tensors)
        features = []
        
        if user_input:
            user_words = user_input.lower().split()
            combined_words = user_words + words[:max(0, max_words-len(user_words))]
        else:
            combined_words = words[:max_words]
        
        for i, word in enumerate(combined_words):
            word_idx = self.word_to_idx.get(word, 0)
            
            # Create basic features
            feature_vector = [
                word_idx / max(len(self.word_to_idx), 1),
                len(word) / 20.0,
                i / max(len(combined_words), 1)
            ]
            
            # Apply duty cycle modulation
            duty_modulation = self.duty_cycle_manager.get_duty_cycle_modulation()
            feature_vector = [f * (1 + duty_modulation.item()) for f in feature_vector]
            
            # Pad to neuron count
            while len(feature_vector) < self.num_neurons:
                feature_vector.append(math.sin(len(feature_vector) * word_idx / 10.0) * duty_modulation.item())
            feature_vector = feature_vector[:self.num_neurons]
            features.append(feature_vector)
        
        return torch.tensor(features, dtype=torch.float32, device=self.device)

class PyTorchGraphConvolution(nn.Module):
    """Simple Graph Convolution implementation in PyTorch."""
    def __init__(self, in_features, out_features):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = nn.Parameter(torch.FloatTensor(in_features, out_features))
        self.bias = nn.Parameter(torch.FloatTensor(out_features))
        self.reset_parameters()
        
    def reset_parameters(self):
        std = 1. / math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-std, std)
        self.bias.data.uniform_(-std, std)
        
    def forward(self, input, adj):
        support = torch.mm(input, self.weight)
        output = torch.spmm(adj, support)
        return output + self.bias

class PyTorchFGCN(nn.Module):
    """Pure PyTorch Graph Convolutional Network for neural data processing."""
    def __init__(self, in_dim, hidden_dim=64, out_dim=32):
        super().__init__()
        self.gc1 = PyTorchGraphConvolution(in_dim, hidden_dim)
        self.gc2 = PyTorchGraphConvolution(hidden_dim, out_dim)
        self.attn = nn.Linear(out_dim, 1)
        self.dropout = nn.Dropout(0.5)
        
    def forward(self, x, adj):
        x = F.relu(self.gc1(x, adj))
        x = self.dropout(x)
        x = F.relu(self.gc2(x, adj))
        attn_weights = torch.sigmoid(self.attn(x))
        x = x * attn_weights
        return x

def create_pytorch_graph(spk_rec, mem_rec):
    """Create graph adjacency matrix and features from spike/membrane data."""
    print(f"Creating PyTorch graph from spk_rec: {spk_rec.shape}, mem_rec: {mem_rec.shape}")
    
    min_steps = min(spk_rec.shape[0], mem_rec.shape[0])
    spk_rec_aligned = spk_rec[:min_steps]
    mem_rec_aligned = mem_rec[:min_steps]
    min_neurons = min(spk_rec_aligned.shape[1], mem_rec_aligned.shape[1])
    spk_rec_aligned = spk_rec_aligned[:, :min_neurons]
    mem_rec_aligned = mem_rec_aligned[:, :min_neurons]
    
    # Create node features by concatenating spike and membrane data
    node_features = torch.cat([spk_rec_aligned.T, mem_rec_aligned.T], dim=1)
    num_nodes = node_features.shape[0]
    
    # Create adjacency matrix (simple connectivity)
    adj = torch.zeros(num_nodes, num_nodes, device=node_features.device)
    if num_nodes > 1:
        for i in range(num_nodes):
            for j in range(i+1, min(i+4, num_nodes)):
                adj[i, j] = 1.0
                adj[j, i] = 1.0  # Make symmetric
    
    # Add self-loops
    adj += torch.eye(num_nodes, device=node_features.device)
    
    # Normalize adjacency matrix
    degree = adj.sum(dim=1)
    degree_inv_sqrt = torch.pow(degree, -0.5)
    degree_inv_sqrt[torch.isinf(degree_inv_sqrt)] = 0.0
    degree_matrix_inv_sqrt = torch.diag(degree_inv_sqrt)
    adj_normalized = torch.mm(torch.mm(degree_matrix_inv_sqrt, adj), degree_matrix_inv_sqrt)
    
    return node_features, adj_normalized

class PyTorchTextGenerator:
    """Pure PyTorch text generator using neural outputs with error handling."""
    def __init__(self, text_processor):
        self.text_processor = text_processor
        self.transitions = defaultdict(list)
        self.fallback_words = ["the", "and", "to", "of", "a", "in", "is", "it", "you", "that"]
        self.build_transitions()
    
    def build_transitions(self):
        """Build transition probabilities with error checking."""
        if not hasattr(self.text_processor, 'bigram_counts') or not self.text_processor.bigram_counts:
            print("⚠️  Warning: No bigram data available, using fallback transitions")
            for i, word in enumerate(self.fallback_words):
                next_word = self.fallback_words[(i + 1) % len(self.fallback_words)]
                self.transitions[word].append((next_word, 1))
            return
            
        for (w1, w2), count in self.text_processor.bigram_counts.items():
            self.transitions[w1].append((w2, count))
        
        if not self.transitions:
            print("⚠️  Warning: No valid transitions found, using fallback")
            for i, word in enumerate(self.fallback_words):
                next_word = self.fallback_words[(i + 1) % len(self.fallback_words)]
                self.transitions[word].append((next_word, 1))
    
    def get_safe_word_choice(self, candidates=None):
        """Safely choose a word with fallback options."""
        if candidates and len(candidates) > 0:
            try:
                words, weights = zip(*candidates)
                if len(words) > 0:
                    weights = np.array(weights, dtype=float)
                    if weights.sum() > 0:
                        weights = weights / weights.sum()
                        return np.random.choice(words, p=weights)
            except (ValueError, IndexError) as e:
                print(f"⚠️  Word choice error: {e}, using fallback")
        
        available_words = list(self.transitions.keys())
        if available_words:
            return random.choice(available_words)
        else:
            return random.choice(self.fallback_words)
    
    def generate_text_from_neural_output(self, spk_rec, mem_rec, seed_word: str = None, length: int = 50):
        """Generate text based on PyTorch neural output with robust error handling."""
        if not self.transitions:
            return "No training data available for text generation."
        
        # Convert PyTorch tensors to numpy for processing
        if isinstance(spk_rec, torch.Tensor):
            neural_influence = spk_rec.mean(dim=1).detach().cpu().numpy()
        else:
            neural_influence = np.array([0.5])
        
        # Start with seed word or random choice
        if seed_word:
            seed_words = seed_word.split()
            current_word = seed_words[-1] if seed_words else self.get_safe_word_choice()
        else:
            current_word = self.get_safe_word_choice()
        
        generated_words = [current_word]
        
        for i in range(length - 1):
            neural_idx = i % len(neural_influence)
            neural_gate = neural_influence[neural_idx]
            
            candidates = self.transitions.get(current_word, [])
            
            if not candidates:
                current_word = self.get_safe_word_choice()
                if current_word != generated_words[-1]:
                    generated_words.append(current_word)
                continue
            
            next_word = self.get_safe_word_choice(candidates)
            if next_word != generated_words[-1]:
                generated_words.append(next_word)
            current_word = next_word
        
        return ' '.join(generated_words)

class PyTorchUserContextGenerator(PyTorchTextGenerator):
    """User-context-aware text generator with PyTorch backend."""
    def __init__(self, text_processor, fgcn_model=None):
        super().__init__(text_processor)
        self.fgcn_model = fgcn_model
        self.user_context_weight = 0.7
        
    def generate_contextual_text(self, user_input, spk_rec, mem_rec, length=50):
        """Generate text that's contextually aware of user input using PyTorch."""
        if not user_input.strip():
            return self.generate_text_from_neural_output(spk_rec, mem_rec, length=length)
        
        user_words = user_input.lower().split()
        
        # Use FGCN if available
        if self.fgcn_model is not None:
            node_features, adj = create_pytorch_graph(spk_rec, mem_rec)
            with torch.no_grad():
                graph_features = self.fgcn_model(node_features, adj)
        
        # Find starting word from user input
        current_word = None
        for word in reversed(user_words):
            if word in self.transitions:
                current_word = word
                break
        
        if current_word is None:
            current_word = self.get_safe_word_choice()
        
        generated_words = [current_word]
        user_context_strength = 1.0
        context_decay = 0.95
        
        for i in range(length - 1):
            candidates = self.transitions.get(current_word, [])
            
            if not candidates:
                current_word = self.get_safe_word_choice()
                if current_word != generated_words[-1]:
                    generated_words.append(current_word)
                continue
            
            # Boost candidates similar to user words
            contextual_candidates = []
            for word, weight in candidates:
                context_boost = 1.0
                for user_word in user_words:
                    if user_word in word or word in user_word:
                        context_boost += user_context_strength
                contextual_candidates.append((word, weight * context_boost))
            
            next_word = self.get_safe_word_choice(contextual_candidates)
            if next_word != generated_words[-1]:
                generated_words.append(next_word)
            current_word = next_word
            
            user_context_strength *= context_decay
        
        return ' '.join(generated_words)

def process_user_input_pytorch(filename, user_input, text_processor, snn_model, num_steps=10):
    """Process user input through the PyTorch SNN."""
    words = text_processor.load_and_process_text(filename)
    
    print("🔧 Processing through PyTorch Heavy Duty Cycle SNN...")
    
    # Generate features for user input
    user_features = text_processor.words_to_neural_features(
        words, 
        user_input=user_input
    )
    
    # Scale features
    scaler = StandardScaler()
    features_scaled = scaler.fit_transform(user_features.cpu().numpy())
    features_tensor = torch.FloatTensor(features_scaled).to(text_processor.device)
    
    # Ensure correct feature dimension matches num_neurons
    if features_tensor.shape[1] != snn_model.num_neurons:
        if features_tensor.shape[1] > snn_model.num_neurons:
            features_tensor = features_tensor[:, :snn_model.num_neurons]
        else:
            padding_size = snn_model.num_neurons - features_tensor.shape[1]
            padding = torch.zeros(features_tensor.shape[0], padding_size, device=text_processor.device)
            features_tensor = torch.cat([features_tensor, padding], dim=1)
    
    # Generate spikes using PyTorch spike generator
    spike_data = PyTorchSpikeGenerator.rate_encoding(features_tensor, num_steps=num_steps)
    
    # Process through PyTorch SNN
    spk_rec, mem_rec = run_pytorch_snn(spike_data, snn_model)
    
    print("✅ PyTorch Heavy duty cycle SNN processing complete!")
    print(f"📈 Final duty cycle stats: {snn_model.get_duty_cycle_status()}")
    
    return spk_rec, mem_rec

def main_pytorch_implementation():
    """Main function using pure PyTorch implementation."""
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"🔥 Using PyTorch device: {device}")
    
    num_neurons = 256
    num_steps = 10
    
    print(f"Initializing PyTorch SNN with {num_neurons} neurons")
    
    # Initialize components with PyTorch backend
    text_processor = PyTorchNeuronAwareTextProcessor(num_neurons, device=device)
    snn_model = PyTorchSimpleSNN(num_neurons, device=device)
    
    print("="*60)
    print("PURE PYTORCH SNN TEXT GENERATOR")
    print("="*60)
    print("This system processes your input through a PyTorch spiking neural network")
    print("and generates contextually relevant text based on neural patterns.")
    print("Enter text to generate responses.")
    print("="*60)
    
    filename = input("Enter dataset filename: ")
    
    while True:
        user_input = input("\nUSER: ").strip()
        if not user_input:
            print("Please enter some text.")
            continue
            
        if user_input.lower() in ['quit', 'exit']:
            print("Goodbye!")
            break
        
        print(f"\nProcessing input: '{user_input}'")
        print("="*40)
        
        try:
            # Process user input through PyTorch SNN
            spk_rec, mem_rec = process_user_input_pytorch(
                filename, user_input, text_processor, snn_model, num_steps
            )
            
            # Initialize PyTorch FGCN model
            node_features, adj = create_pytorch_graph(spk_rec, mem_rec)
            fgcn_model = PyTorchFGCN(node_features.shape[1]).to(device)
            
            # Create user-context-aware text generator
            context_generator = PyTorchUserContextGenerator(text_processor, fgcn_model)
            
            # Generate contextual response
            contextual_text = context_generator.generate_contextual_text(
                user_input, spk_rec, mem_rec, length=500
            )
            print()
            print("AI:", contextual_text)
            
        except Exception as e:
            print(f"❌ Error processing input: {e}")
            print("Continuing with next input...")

if __name__ == "__main__":
    main_pytorch_implementation()
