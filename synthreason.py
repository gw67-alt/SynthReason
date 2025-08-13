import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.preprocessing import StandardScaler
from collections import defaultdict, Counter
import random
import math
import gc
from pathlib import Path

KB_LEN = -1

class TrainableMemoryOptimizedHeavyDutyCycleManager(nn.Module):
    """Trainable memory-efficient heavy duty cycle manager with learnable parameters."""
    def __init__(self, cycle_length=32, duty_ratio=0.8, decay_rate=0.7, device='cpu', 
                 max_buffer_size=100):
        super().__init__()
        
        # Make core parameters trainable
        self.register_parameter('cycle_length', nn.Parameter(torch.tensor(float(cycle_length))))
        self.register_parameter('duty_ratio', nn.Parameter(torch.tensor(duty_ratio)))
        self.register_parameter('decay_rate', nn.Parameter(torch.tensor(decay_rate)))
        self.register_parameter('neural_feedback_gain', nn.Parameter(torch.tensor(0.2)))
        
        # Trainable cycle position (reset during training)
        self.register_buffer('cycle_position', torch.tensor(0.0, device=device))
        
        # Memory-efficient circular buffers
        self.max_buffer_size = max_buffer_size
        self.probability_buffer = []
        self.cycle_history = []
        self.register_buffer('thermal_accumulator', torch.tensor(0.0, device=device))
        self.device = device
        
        # Running statistics
        self.running_mean = 0.0
        self.running_var = 0.0
        self.sample_count = 0
        
        # Learnable modulation parameters
        self.register_parameter('active_modulation_scale', nn.Parameter(torch.tensor(0.5)))
        self.register_parameter('inactive_modulation_scale', nn.Parameter(torch.tensor(0.1)))
        
    @property
    def active_threshold(self):
        """Compute active threshold from learnable parameters."""
        cycle_length_val = self.cycle_length.item() if hasattr(self.cycle_length, 'item') else float(self.cycle_length)
        duty_ratio_val = self.duty_ratio.item() if hasattr(self.duty_ratio, 'item') else float(self.duty_ratio)
        
        threshold = cycle_length_val * duty_ratio_val
        return torch.clamp(torch.tensor(threshold, device=self.cycle_length.device), 
                          1.0, cycle_length_val - 1.0)
        
    def _update_running_stats(self, value):
        """Update running statistics without storing all values."""
        self.sample_count += 1
        delta = value - self.running_mean
        self.running_mean += delta / self.sample_count
        delta2 = value - self.running_mean
        self.running_var += delta * delta2
        
    def _prune_buffers(self):
        """Keep buffers at manageable size."""
        if len(self.probability_buffer) > self.max_buffer_size:
            self.probability_buffer = self.probability_buffer[-self.max_buffer_size//2:]
        if len(self.cycle_history) > 10:
            self.cycle_history = self.cycle_history[-5:]
        
    def modulate_probabilities(self, base_probabilities, neural_activity=None):
        """Trainable probability modulation with streaming updates."""
        self.cycle_position += 1.0
        
        # Reset cycle when threshold reached
        cycle_reset = (self.cycle_position >= self.cycle_length).float()
        self.cycle_position = self.cycle_position * (1 - cycle_reset)
        
        if cycle_reset.item() > 0:
            self._prune_buffers()
            
        modulation = self.get_duty_cycle_modulation()
        
        if isinstance(base_probabilities, torch.Tensor):
            modulated = base_probabilities * modulation
            avg_prob = modulated.mean().item()
        else:
            base_probs_tensor = torch.tensor(base_probabilities, device=self.device, dtype=torch.float32)
            modulated = base_probs_tensor * modulation
            avg_prob = modulated.mean().item()
            
        self._update_running_stats(avg_prob)
        
        # Store recent samples
        if len(self.probability_buffer) < self.max_buffer_size:
            self.probability_buffer.append(avg_prob)
            
        return modulated
    
    def get_duty_cycle_modulation(self):
        """Trainable duty cycle modulation calculation."""
        active_thresh = self.active_threshold
        
        # Use sigmoid for differentiable phase selection
        is_active = torch.sigmoid(10 * (active_thresh - self.cycle_position))
        
        # Active phase modulation
        progress = self.cycle_position / torch.clamp(active_thresh, min=1e-8)
        active_mod = self.active_modulation_scale + self.active_modulation_scale * torch.sin(progress * torch.pi)
        
        # Inactive phase modulation
        inactive_progress = (self.cycle_position - active_thresh) / torch.clamp(
            self.cycle_length - active_thresh, min=1e-8
        )
        inactive_mod = self.inactive_modulation_scale * torch.exp(-3 * inactive_progress)
        
        # Differentiable combination
        modulation = is_active * active_mod + (1 - is_active) * inactive_mod
        
        return modulation

class TrainableMemoryEfficientLIFNeuron(nn.Module):
    """Trainable memory-efficient LIF neuron with learnable parameters."""
    def __init__(self, tau_mem=10.0, tau_syn=5.0, v_thresh=1.0, v_reset=0.0):
        super().__init__()
        
        # Make neuron parameters trainable
        self.register_parameter('tau_mem', nn.Parameter(torch.tensor(tau_mem)))
        self.register_parameter('tau_syn', nn.Parameter(torch.tensor(tau_syn)))
        self.register_parameter('v_thresh', nn.Parameter(torch.tensor(v_thresh)))
        self.register_parameter('v_reset', nn.Parameter(torch.tensor(v_reset)))
        
    def compute_decay_factors(self):
        """Compute decay factors from trainable time constants."""
        tau_mem_clamped = torch.clamp(self.tau_mem, 1.0, 50.0)
        tau_syn_clamped = torch.clamp(self.tau_syn, 1.0, 50.0)
        
        beta = torch.exp(-1.0 / tau_mem_clamped)
        alpha = torch.exp(-1.0 / tau_syn_clamped)
        
        return beta, alpha
        
    def forward(self, x, state=None):
        """Trainable forward pass with differentiable spike generation."""
        device = x.device
        
        # Handle different input dimensions
        if x.dim() == 1:
            x = x.unsqueeze(0)  # Add batch dimension: (features,) -> (1, features)
        elif x.dim() > 2:
            # Flatten extra dimensions: (batch, ..., features) -> (batch, features)
            x = x.view(x.size(0), -1)
        
        batch_size, num_neurons = x.shape
        
        if state is None:
            v_mem = torch.zeros(batch_size, num_neurons, device=device)
            i_syn = torch.zeros(batch_size, num_neurons, device=device)
        else:
            v_mem, i_syn = state
            
        # Rest of your forward method remains the same...

            
        # Get trainable decay factors
        beta, alpha = self.compute_decay_factors()
        
        # Update dynamics
        i_syn = alpha * i_syn + x
        v_mem = beta * v_mem + i_syn
        
        # Trainable threshold
        thresh_clamped = torch.clamp(self.v_thresh, 0.1, 5.0)
        
        # Differentiable spike generation using straight-through estimator
        if self.training:
            # Sigmoid approximation during training for gradients
            spike_prob = torch.sigmoid(10 * (v_mem - thresh_clamped))
            # Gumbel-softmax for discrete sampling with gradients
            gumbel_noise = -torch.log(-torch.log(torch.rand_like(spike_prob) + 1e-8) + 1e-8)
            spikes = torch.sigmoid((torch.log(spike_prob + 1e-8) - torch.log(1 - spike_prob + 1e-8) + gumbel_noise) / 0.1)
        else:
            # Hard thresholding during inference
            spikes = (v_mem >= thresh_clamped).float()
        
        # Reset mechanism
        reset_clamped = torch.clamp(self.v_reset, -2.0, 2.0)
        v_mem = v_mem * (1 - spikes) + reset_clamped * spikes
        
        return spikes, (v_mem, i_syn)

class TrainableStreamingSNN(nn.Module):
    """Trainable memory-efficient streaming SNN with full gradient support."""
    def __init__(self, num_neurons, device='cpu', chunk_size=32):
        super().__init__()
        self.num_neurons = num_neurons
        self.device = device
        self.chunk_size = chunk_size
        
        # Trainable network layers
        self.input_layer = nn.Linear(num_neurons, num_neurons, bias=True)
        self.hidden_layer = nn.Linear(num_neurons, num_neurons, bias=True)
        self.output_layer = nn.Linear(num_neurons, num_neurons, bias=True)
        
        # Trainable LIF neurons
        self.lif_neurons = TrainableMemoryEfficientLIFNeuron()
        
        # Trainable parameters
        self.global_adaptation = nn.Parameter(torch.ones(1) * 0.5)
        
        # Trainable duty cycle manager
        self.duty_cycle_manager = TrainableMemoryOptimizedHeavyDutyCycleManager(device=device)
        
        # State management
        self.neuron_state = None
        
    def forward_chunk(self, x_chunk):
        """Trainable chunk processing with full gradient flow."""
        if x_chunk.dim() == 1:
            x_chunk = x_chunk.unsqueeze(0)
            
        # Process through trainable layers
        x_processed = F.relu(self.input_layer(x_chunk))
        x_hidden = F.relu(self.hidden_layer(x_processed))
        
        # Probability gating with trainable duty cycle modulation
        prob_weights = torch.sigmoid(x_hidden)
        modulated_weights = self.duty_cycle_manager.modulate_probabilities(
            prob_weights, neural_activity=x_hidden
        )
        
        # Apply modulation
        x_modulated = x_hidden * modulated_weights.unsqueeze(0)
        
        # Process through trainable LIF neurons
        spikes, self.neuron_state = self.lif_neurons(x_modulated, self.neuron_state)
        
        # Final output layer
        output = self.output_layer(spikes)
        
        # Apply trainable global adaptation
        cycle_mod = self.duty_cycle_manager.get_duty_cycle_modulation()
        adapted_output = output * self.global_adaptation * (1 + cycle_mod)
        
        return adapted_output.squeeze(0)
    
    def forward(self, x_sequence):
        """Process full sequence with gradient tracking."""
        outputs = []
        self.reset_neurons()
        
        for x in x_sequence:
            out = self.forward_chunk(x)
            outputs.append(out)
            
        return torch.stack(outputs) if outputs else torch.empty(0, self.num_neurons)
    
    def reset_neurons(self):
        """Reset neuron states."""
        self.neuron_state = None

class TrainableMemoryEfficientTextProcessor(nn.Module):
    """Trainable streaming text processor with learnable embeddings."""
    def __init__(self, num_neurons=256, device='cpu', vocab_limit=5000):
        super().__init__()
        self.num_neurons = num_neurons
        self.device = device
        self.vocab_limit = vocab_limit
        self.word_to_idx = {}
        self.bigram_counts = Counter()
        
        # Trainable word embeddings
        self.word_embeddings = nn.Embedding(vocab_limit + 1, num_neurons // 4)
        self.position_embeddings = nn.Embedding(1000, num_neurons // 4)
        
        # Trainable feature processing
        self.feature_processor = nn.Sequential(
            nn.Linear(num_neurons // 2, num_neurons),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(num_neurons, num_neurons),
            nn.Tanh()
        )
        
        # Cache management (non-trainable)
        self.transition_cache = {}
        self.cache_limit = 1000
        
    def load_and_process_text_streaming(self, file_path="test.txt", chunk_size=1000):
        """Stream process text file to build vocabulary."""
        word_count = 0
        
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                prev_word = None
                words_processed = []
                
                while word_count < KB_LEN if KB_LEN > 0 else True:
                    chunk = f.read(chunk_size * 10)
                    if not chunk:
                        break
                        
                    words = chunk.lower().split()
                    for word in words:
                        if len(self.word_to_idx) < self.vocab_limit:
                            if word not in self.word_to_idx:
                                self.word_to_idx[word] = len(self.word_to_idx)
                        
                        if prev_word is not None:
                            self.bigram_counts[(prev_word, word)] += 1
                            
                        prev_word = word
                        words_processed.append(word)
                        word_count += 1
                        
                        if KB_LEN > 0 and word_count >= KB_LEN:
                            break
                            
                    if len(words_processed) > 10000:
                        words_processed = words_processed[-1000:]
                        
        except FileNotFoundError:
            print(f"⚠️ File {file_path} not found. Creating sample data...")
            sample_words = ["the", "quick", "brown", "fox", "jumps", "over", "lazy", "dog"] * 100
            for i, word in enumerate(sample_words):
                if word not in self.word_to_idx:
                    self.word_to_idx[word] = len(self.word_to_idx)
                if i > 0:
                    prev_word = sample_words[i-1]
                    self.bigram_counts[(prev_word, word)] += 1
            words_processed = sample_words
                        
        print(f"📚 Processed {word_count} words with vocab size {len(self.word_to_idx)}")
        return words_processed[-1000:] if words_processed else []
    
    def get_transition_probs(self, word):
        """Get transition probabilities with caching."""
        if word in self.transition_cache:
            return self.transition_cache[word]
            
        transitions = []
        for (w1, w2), count in self.bigram_counts.items():
            if w1 == word:
                transitions.append((w2, count))
                
        if len(self.transition_cache) >= self.cache_limit:
            keys_to_remove = list(self.transition_cache.keys())[:self.cache_limit//2]
            for k in keys_to_remove:
                del self.transition_cache[k]
                
        self.transition_cache[word] = transitions
        return transitions
    
    def words_to_neural_features_trainable(self, words, max_words=50):
        """Generate trainable features using learnable embeddings."""
        if len(words) > max_words:
            words = words[-max_words:]
            
        # Convert words to indices
        word_indices = []
        for word in words:
            idx = self.word_to_idx.get(word, 0)
            word_indices.append(min(idx, self.vocab_limit))
        
        if not word_indices:
            return torch.zeros(1, self.num_neurons, device=next(self.parameters()).device)
            
        word_indices = torch.tensor(word_indices, device=next(self.parameters()).device)
        position_indices = torch.arange(len(words), device=next(self.parameters()).device)
        
        # Get trainable embeddings
        word_embs = self.word_embeddings(word_indices)
        pos_embs = self.position_embeddings(position_indices)
        
        # Combine embeddings
        combined_embs = torch.cat([word_embs, pos_embs], dim=1)
        
        # Process through trainable layers
        features = self.feature_processor(combined_embs)
        
        return features

class TrainableStreamingTextGenerator(nn.Module):
    """Trainable text generator with neural selection networks."""
    def __init__(self, text_processor, hidden_dim=128, max_transitions_per_word=50):
        super().__init__()
        self.text_processor = text_processor
        self.max_transitions = max_transitions_per_word
        self.fallback_words = ["the", "and", "to", "of", "a", "in", "is", "it", "you", "that"]
        
        # Trainable selection network
        self.selection_network = nn.Sequential(
            nn.Linear(text_processor.num_neurons, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 1),
            nn.Sigmoid()
        )
        
    def forward(self, spk_rec):
        """Process spike recordings to generate selection probabilities."""
        if spk_rec.numel() == 0:
            return torch.zeros(1, device=next(self.parameters()).device)
        
        # Process through selection network
        selection_probs = self.selection_network(spk_rec)
        return selection_probs.squeeze(-1)
    
    def generate_text_trainable(self, spk_rec, seed_word=None, length=50):
        """Generate text using trainable selection."""
        if spk_rec.numel() == 0:
            return "No neural data available for generation."
            
        # Get selection probabilities
        with torch.no_grad():
            selection_probs = self.forward(spk_rec)
        
        current_word = seed_word if seed_word else random.choice(self.fallback_words)
        generated_words = [current_word]
        
        for i in range(length - 1):
            transitions = self.text_processor.get_transition_probs(current_word)
            if not transitions:
                current_word = random.choice(self.fallback_words)
                generated_words.append(current_word)
                continue
                
            transitions = transitions[:self.max_transitions]
            
            # Use neural selection
            prob_idx = min(i, len(selection_probs) - 1)
            neural_influence = selection_probs[prob_idx].item()
            
            if transitions:
                words, weights = zip(*transitions)
                weights = np.array(weights, dtype=float)
                weights = weights * (0.5 + neural_influence)
                
                if weights.sum() > 0:
                    weights = weights / weights.sum()
                    next_word = np.random.choice(words, p=weights)
                else:
                    next_word = random.choice(words)
            else:
                next_word = random.choice(self.fallback_words)
                
            generated_words.append(next_word)
            current_word = next_word
            
        return ' '.join(generated_words)

def create_training_dataset(text_processor, target_length=100):
    """Create training dataset from processed text."""
    # This is a simplified dataset creator
    # In practice, you'd want more sophisticated data preparation
    dataset = []

    for word, idx in text_processor.word_to_idx.items():
        dataset.append(word)
    
   
    return dataset

def train_snn_system(text_processor, snn_model, text_generator, dataset, 
                     epochs=10, lr=0.001, device='cpu'):
    """Comprehensive training loop for the entire SNN system."""
    
    # Combine all trainable parameters
    all_params = (list(text_processor.parameters()) + 
                  list(snn_model.parameters()) + 
                  list(text_generator.parameters()))
    
    optimizer = torch.optim.Adam(all_params, lr=lr, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=3, gamma=0.7)
    
    # Multiple loss functions
    mse_loss = nn.MSELoss()
    prediction_loss = nn.BCEWithLogitsLoss()
    
    print(f"🚀 Starting training with {len(all_params)} parameters...")
    
    for epoch in range(epochs):
        epoch_losses = {'total': 0.0, 'spike': 0.0, 'prediction': 0.0, 'regularization': 0.0}
        
        # Set models to training mode
        text_processor.train()
        snn_model.train()
        text_generator.train()
        
        for batch_idx, words in enumerate(dataset):
            optimizer.zero_grad()
            words = words[:len(words)//3] + words[len(words)//3:] + words[len(words)//3:]
            # Convert words to features
            try:
                features = text_processor.words_to_neural_features_trainable(words)

                
                if features.shape[0] == 0:
                    continue
                    
                # Process through SNN
                spike_outputs = snn_model.forward(features)
                
                # Loss 1: Spike activity regularization (encourage sparse, meaningful spikes)
                target_spike_rate = 0.1  # Target 10% spike rate
                actual_spike_rate = spike_outputs.mean()
                spike_loss = mse_loss(actual_spike_rate, torch.tensor(target_spike_rate, device=device))
                
                # Loss 2: Text generation prediction accuracy
                selection_probs = text_generator.forward(spike_outputs)
                # Create pseudo-targets based on word frequency (more frequent words = higher probability)
                word_frequencies = []
                for word in words:
                    freq = sum(1 for (w1, w2), count in text_processor.bigram_counts.items() if w1 == word or w2 == word)
                    word_frequencies.append(freq)
                
                if word_frequencies:
                    max_freq = max(word_frequencies) if max(word_frequencies) > 0 else 1
                    targets = torch.tensor([f / max_freq for f in word_frequencies], 
                                         device=device, dtype=torch.float32)
                    
                    # Ensure targets match selection_probs length
                    if len(targets) != len(selection_probs):
                        min_len = min(len(targets), len(selection_probs))
                        targets = targets[:min_len]
                        selection_probs = selection_probs[:min_len]
                    
                    if len(targets) > 0:
                        pred_loss = mse_loss(selection_probs, targets)
                    else:
                        pred_loss = torch.tensor(0.0, device=device)
                else:
                    pred_loss = torch.tensor(0.0, device=device)
                
                # Loss 3: L2 regularization
                l2_reg = torch.tensor(0.0, device=device)
                for param in all_params:
                    if param.requires_grad:
                        l2_reg += torch.norm(param)
                
                # Combine losses
                total_loss = spike_loss + 0.5 * pred_loss + 0.0001 * l2_reg
                
                # Backpropagation
                total_loss.backward()
                
                # Gradient clipping
                torch.nn.utils.clip_grad_norm_(all_params, max_norm=1.0)
                
                optimizer.step()
                
                # Track losses
                epoch_losses['total'] += total_loss.item()
                epoch_losses['spike'] += spike_loss.item()
                epoch_losses['prediction'] += pred_loss.item()
                epoch_losses['regularization'] += l2_reg.item() * 0.0001
                
                # Periodic logging
                if batch_idx % 10 == 0:
                    print(f"  Batch {batch_idx}: Loss = {total_loss.item():.6f}")
                    
            except Exception as e:
                print(f"⚠️ Training error on batch {batch_idx}: {e}")
                continue
        
        # Update learning rate
        scheduler.step()
        
        # Print epoch summary
        n_batches = len(dataset)
        avg_losses = {k: v / n_batches for k, v in epoch_losses.items()}
        
        print(f"📊 Epoch {epoch+1}/{epochs} Summary:")
        print(f"   Total Loss: {avg_losses['total']:.6f}")
        print(f"   Spike Loss: {avg_losses['spike']:.6f}")
        print(f"   Prediction Loss: {avg_losses['prediction']:.6f}")
        print(f"   Regularization: {avg_losses['regularization']:.6f}")
        print(f"   Learning Rate: {scheduler.get_last_lr()[0]:.6f}")
        
        # Periodic validation
        if (epoch + 1) % 3 == 0:
            validate_model(text_processor, snn_model, text_generator, device)
    
    print("✅ Training completed!")

def validate_model(text_processor, snn_model, text_generator, device):
    """Validate the trained model."""
    # Set models to evaluation mode
    text_processor.eval()
    snn_model.eval()
    text_generator.eval()
    
    print("🔍 Validation:")
    
    # Test with sample input
    test_words = ["the", "quick", "brown", "fox", "jumps"]
    
    with torch.no_grad():
        try:
            features = text_processor.words_to_neural_features_trainable(test_words)
            spike_outputs = snn_model.forward(features)
            selection_probs = text_generator.forward(spike_outputs)
            
            print(f"   Input: {test_words}")
            print(f"   Spike Activity: {spike_outputs.mean().item():.4f}")
            print(f"   Selection Probs: {selection_probs.mean().item():.4f}")
            
            # Generate sample text
            generated = text_generator.generate_text_trainable(
                spike_outputs, seed_word="the", length=20
            )
            print(f"   Generated: {generated}")
            
        except Exception as e:
            print(f"   Validation Error: {e}")

def process_user_input_trainable(filename, user_input, text_processor, snn_model, 
                                text_generator, chunk_size=10):
    """Process user input with trainable models."""
    words = text_processor.load_and_process_text_streaming(filename)
    
    # Generate trainable features
    features = text_processor.words_to_neural_features_trainable(words)
    
    if features.shape[0] == 0:
        return torch.zeros(1, snn_model.num_neurons), torch.zeros(1, snn_model.num_neurons)
    
    # Process through trainable SNN
    with torch.no_grad():
        spike_outputs = snn_model.forward(features)
    
    mem_rec = torch.zeros_like(spike_outputs)
    
    return spike_outputs, mem_rec

def main_trainable_implementation():
    """Main function with full training support."""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"🔥 Using device: {device}")
    
    # Parameters for trainable system
    num_neurons = 128
    chunk_size = 16
    vocab_limit = 3000
    
    # Initialize trainable components
    text_processor = TrainableMemoryEfficientTextProcessor(
        num_neurons, device=device, vocab_limit=vocab_limit
    ).to(device)
    
    snn_model = TrainableStreamingSNN(
        num_neurons, device=device, chunk_size=chunk_size
    ).to(device)
    
    text_generator = TrainableStreamingTextGenerator(
        text_processor
    ).to(device)
    
    print("="*60)
    print("TRAINABLE MEMORY-OPTIMIZED SNN TEXT GENERATOR")
    print("="*60)
    print("Full gradient-based training support with multiple loss functions")
    print("="*60)
    
    filename = input("Enter dataset filename: ")
    
    # Load and prepare data
    print("📚 Loading and preparing training data...")
    words = text_processor.load_and_process_text_streaming(filename)
    
    # Create training dataset
    dataset = create_training_dataset(text_processor)
    print(f"📊 Created training dataset with {len(dataset)} samples")
    
    # Training phase
    print("\n🚀 Starting training phase...")
    train_snn_system(text_processor, snn_model, text_generator, dataset, 
                     epochs=2, lr=0.001, device=device)
    
    # Testing phase
    print("\n🔍 Testing trained model...")
    
    try:
        with open("questions.conf", 'r', encoding='utf-8') as f:
            questions = f.readlines()
    except FileNotFoundError:
        questions = ["Hello, how are you today?", "What is machine learning?"]
    
    # Set models to evaluation mode
    text_processor.eval()
    snn_model.eval()
    text_generator.eval()
    
    for user_input in questions:
        user_input = user_input.strip()
        if not user_input:
            continue
            
        print(f"\nProcessing: '{user_input}'")
        
        try:
            # Process with trained models
            spk_rec, mem_rec = process_user_input_trainable(
                filename, user_input, text_processor, snn_model, 
                text_generator, chunk_size
            )
            
            # Generate text with trained generator
            response = text_generator.generate_text_trainable(
                spk_rec, seed_word=user_input.split()[-1] if user_input.split() else None,
                length=500
            )
            
            print("AI:", response)
            print(f"📈 Spike Activity: {spk_rec.mean().item():.4f}")
            
            # Force garbage collection
            del spk_rec, mem_rec
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                
        except Exception as e:
            print(f"❌ Error: {e}")
            import traceback
            traceback.print_exc()
            continue

if __name__ == "__main__":
    main_trainable_implementation()
