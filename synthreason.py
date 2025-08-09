import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import Data
from torch_geometric.nn import GCNConv
from torch.utils.data import Dataset, DataLoader
from collections import defaultdict, Counter
import random
import multiprocessing as mp
import os


from datasets import load_dataset

# Set threading and multiprocessing settings
os.environ.setdefault("OMP_NUM_THREADS", "8")
os.environ.setdefault("MKL_NUM_THREADS", "8")
torch.set_num_threads(8)
torch.set_num_interop_threads(4)

KB_len = 99999  # use -1 for unlimited

class DataAwareFGCN(nn.Module):
    """Graph Convolutional Network for neural data processing."""
    def __init__(self, in_dim, hidden_dim=64, out_dim=32):
        super().__init__()
        self.gcn1 = GCNConv(in_dim, hidden_dim)
        self.gcn2 = GCNConv(hidden_dim, out_dim)
        self.attn = nn.Linear(out_dim, 1)
        # Add regression head for training
        self.regression_head = nn.Linear(out_dim, 1)
    
    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        x = F.relu(self.gcn1(x, edge_index))
        x = F.relu(self.gcn2(x, edge_index))
        attn_weights = torch.sigmoid(self.attn(x))
        x = x * attn_weights
        return x

def create_neuron_aligned_graph(spk_rec, mem_rec):
    """Create graph from spike and membrane recordings."""
    min_steps = min(spk_rec.shape[0], mem_rec.shape[0])
    spk_rec_aligned = spk_rec[:min_steps]
    mem_rec_aligned = mem_rec[:min_steps]
    min_neurons = min(spk_rec_aligned.shape[1], mem_rec_aligned.shape[1])
    spk_rec_aligned = spk_rec_aligned[:, :min_neurons]
    mem_rec_aligned = mem_rec[:, :min_neurons]
        
    # FIX: Create consistent 2D features per node
    # Take mean across time steps to get [N, 1] for each recording type
    spk_features = spk_rec_aligned.mean(dim=0, keepdim=True).T  # [N, 1]
    mem_features = mem_rec_aligned.mean(dim=0, keepdim=True).T  # [N, 1]
    node_features = torch.cat([spk_features, mem_features], dim=1)  # [N, 2]
    
    num_nodes = node_features.shape[0]
    
    # Create edges (connect each node to its neighbors)
    edge_index = []
    if num_nodes > 1:
        for i in range(num_nodes):
            for j in range(i+1, min(i+4, num_nodes)):
                edge_index.extend([[i, j], [j, i]])
        
    if edge_index:
        edge_index = torch.tensor(edge_index, dtype=torch.long).t().contiguous()
    else:
        edge_index = torch.empty((2, 0), dtype=torch.long)
        
    data = Data(x=node_features, edge_index=edge_index)
    return data

class GraphSequenceDataset(Dataset):
    """
    Builds per-timestep graphs and next-step targets (predict next spk vector).
    Each item: (Data(x, edge_index), target) where:
      - x: node_features from create_neuron_aligned_graph at time t
      - target: next-step spike vector (shape [num_neurons]) at t+1
    """
    def __init__(self, spk_rec, mem_rec):
        super().__init__()
        assert spk_rec.shape == mem_rec.shape
        self.spk_rec = spk_rec  # [T, N]
        self.mem_rec = mem_rec  # [T, N]
        self.T, self.N = spk_rec.shape
        # We can form T-1 supervised pairs (t -> t+1)
        self.length = max(0, self.T - 1)

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        # Build a graph using a 1-step slice around idx
        spk_slice = self.spk_rec[idx:idx+1, :]   # [1, N]
        mem_slice = self.mem_rec[idx:idx+1, :]   # [1, N]
        data = create_neuron_aligned_graph(spk_slice, mem_slice)  # Data.x shape [N, 2], edge_index [2, E]
        target = self.spk_rec[idx+1, :]  # predict next step spikes [N]
        # We'll regress node-wise, so target per node:
        target = target.unsqueeze(-1)  # [N,1]
        return data, target

def collate_graph_batch(batch):
    # We'll iterate items manually in the train step since each Data has different edge_index.
    datas, targets = zip(*batch)
    return list(datas), torch.stack(targets, dim=0)  # targets [B, N, 1]

def make_loader(dataset, batch_size=8, num_workers=None, pin_memory=None):
    if num_workers is None:
        try:
            cpu_cores = mp.cpu_count()
        except Exception:
            cpu_cores = 4
        num_workers = min(8, max(1, cpu_cores // 2))
    if pin_memory is None:
        pin_memory = torch.cuda.is_available()

    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        persistent_workers=(num_workers > 0),
        prefetch_factor=2 if num_workers > 0 else None,
        pin_memory=pin_memory,
        collate_fn=collate_graph_batch,
    )
    return loader

def save_checkpoint(path, epoch, model, optimizer, scheduler, best_val=None):
    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict() if scheduler else None,
        'best_val': best_val,
    }, path)

def load_checkpoint(path, model, optimizer=None, scheduler=None, map_location='cpu'):
    ckpt = torch.load(path, map_location=map_location)
    model.load_state_dict(ckpt['model_state_dict'])
    if optimizer and 'optimizer_state_dict' in ckpt and ckpt['optimizer_state_dict'] is not None:
        optimizer.load_state_dict(ckpt['optimizer_state_dict'])
    if scheduler and 'scheduler_state_dict' in ckpt and ckpt['scheduler_state_dict'] is not None:
        scheduler.load_state_dict(ckpt['scheduler_state_dict'])
    start_epoch = ckpt.get('epoch', 0) + 1
    best_val = ckpt.get('best_val', None)
    return start_epoch, best_val

def train_one_epoch(model, loader, device, optimizer):
    model.train()
    total_loss = 0.0
    criterion = nn.MSELoss()
    
    for datas, targets in loader:
        # Process items in the batch independently, then average loss
        batch_loss = 0.0
        count = 0
        
        for data, target in zip(datas, targets):  # target [N,1]
            data = data.to(device)
            target = target.to(device)
            pred = model(data)  # [N, out_dim]
            
            # Map to 1-dim per node for regression target
            if pred.shape[-1] != 1:
                # Use regression head
                pred_1d = model.regression_head(pred)
            else:
                pred_1d = pred
            
            loss = criterion(pred_1d, target)
            batch_loss += loss
            count += 1

        batch_loss = batch_loss / max(1, count)

        optimizer.zero_grad()
        batch_loss.backward()
        optimizer.step()

        total_loss += batch_loss.item()
    
    return total_loss / max(1, len(loader))

@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    total_loss = 0.0
    criterion = nn.MSELoss()
    
    for datas, targets in loader:
        batch_loss = 0.0
        count = 0
        
        for data, target in zip(datas, targets):
            data = data.to(device)
            target = target.to(device)
            pred = model(data)
            
            if pred.shape[-1] != 1:
                pred_1d = model.regression_head(pred)
            else:
                pred_1d = pred
            
            loss = criterion(pred_1d, target)
            batch_loss += loss
            count += 1
        
        batch_loss = batch_loss / max(1, count)
        total_loss += batch_loss.item()
    
    return total_loss / max(1, len(loader))

import re
import html
from collections import defaultdict, Counter

class NeuronAwareTextProcessor:
    """Text processor that converts text to neural-compatible features."""
    def __init__(self, num_neurons=16):
        self.num_neurons = num_neurons
        self.vocab = {}
        self.word_to_idx = {}
        self.idx_to_word = {}
        self.bigram_counts = Counter()
        self.transition_matrix = None
        self.transition_probs = None
    
    def clean_text(self, text):
        """Clean text by removing XML tags, HTML entities, and non-natural text patterns while preserving punctuation."""
        if not isinstance(text, str):
            return ""
        
        # HTML decode first
        text = html.unescape(text)
        
        # Remove XML/HTML-like tags including complex ones like your example
        xml_patterns = [
            r'<[^>]*>',  # Basic XML/HTML tags
            r'</[^>]*>',  # Closing tags
            r'<\w+[^>]*>.*?</\w+>',  # Complete tag pairs
            r'<value[^>]*>.*?</value>',  # Specific value tags
            r'<[^>]*choice-type[^>]*>',  # Tags with choice-type attributes
            r'<[^>]*="[^"]*"[^>]*>',  # Tags with quoted attributes
        ]
        
        for pattern in xml_patterns:
            text = re.sub(pattern, ' ', text, flags=re.IGNORECASE | re.DOTALL)
        
        # Remove standalone XML-like fragments
        text = re.sub(r'</?\w+[^>]*/?>', ' ', text)
        
        # Remove attribute-value pairs that might be leftover
        text = re.sub(r'\w+\s*=\s*"[^"]*"', ' ', text)
        text = re.sub(r'\w+\s*=\s*\'[^\']*\'', ' ', text)
        
        # Remove common non-natural patterns
        cleanup_patterns = [
            r'\[.*?\]',  # Bracketed content
            r'\{.*?\}',  # Braced content
            r'^\s*[-*•]\s*',  # List markers at start of line
            r'\s+[-*•]\s+',  # List markers in text
            r'#+\s*',  # Markdown headers
            r'``````',  # Code blocks
            r'`[^`]*`',  # Inline code
            r'https?://\S+',  # URLs
            r'www\.\S+',  # WWW links
            r'\S+@\S+\.\S+',  # Email addresses
            r'\b\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}\b',  # IP addresses
            r'\\\w+',  # LaTeX commands
            r'\$\$.*?\$\$',  # LaTeX math blocks
            r'\$[^$]*\$',  # LaTeX inline math
        ]
        
        for pattern in cleanup_patterns:
            text = re.sub(pattern, ' ', text, flags=re.IGNORECASE | re.DOTALL)
        
        # Clean up whitespace and preserve natural punctuation
        text = re.sub(r'\s+', ' ', text)  # Multiple spaces to single
        
        # UPDATED: Keep all natural punctuation - only remove truly problematic characters
        # Remove only control characters and unusual symbols, keep all standard punctuation
        text = re.sub(r'[^\w\s\.,!?;:\'"()\-–—\[\]/\\&%$#@*+=<>|`~^{}]', ' ', text)
        
        # Normalize excessive punctuation (optional - you can remove this if you want to keep all)
        text = re.sub(r'\.{3,}', '...', text)  # Multiple dots to ellipsis
        text = re.sub(r'[!]{2,}', '!', text)  # Multiple exclamations to single
        text = re.sub(r'[?]{2,}', '?', text)  # Multiple questions to single
        
        # UPDATED: More lenient word filtering - keep punctuation attached to words
        words = text.split()
        filtered_words = []
        for word in words:
            # Much more lenient - keep words up to 50 chars and allow punctuation
            if 1 <= len(word) <= 50:
                # Only filter out words that are purely symbols (like "||||" or "===")
                if not re.match(r'^[^\w\s]{3,}$', word):
                    filtered_words.append(word)
        
        text = ' '.join(filtered_words)
        
        # Final cleanup
        text = text.strip()
        text = re.sub(r'\s+', ' ', text)  # Ensure single spaces
        
        return text
    
    def load_wise_dataset(self):
        """Load the WISE dataset from Hugging Face with enhanced text cleaning."""
        print("Loading WISE dataset from Hugging Face...")
        try:
            # Load the WISE dataset
            dataset = load_dataset("meaningalignment/wise-data")
            print(f"Successfully loaded WISE dataset: {dataset}")
            
            # Extract text content from conversations
            text_content = []
            
            # Process different splits if available
            for split_name in dataset.keys():
                split_data = dataset[split_name]
                print(f"Processing {split_name} split with {len(split_data)} examples")
                
                for example in split_data:
                    # Extract text based on the dataset structure
                    raw_texts = []
                    
                    if 'conversations' in example:
                        for turn in example['conversations']:
                            if 'content' in turn:
                                raw_texts.append(turn['content'])
                    elif 'messages' in example:
                        for message in example['messages']:
                            if 'content' in message:
                                raw_texts.append(message['content'])
                    elif 'text' in example:
                        raw_texts.append(example['text'])
                    elif isinstance(example, dict):
                        # Fallback: extract any string values
                        for value in example.values():
                            if isinstance(value, str) and len(value) > 10:
                                raw_texts.append(value)
                    
                    # Clean each text segment
                    for raw_text in raw_texts:
                        cleaned_text = self.clean_text(raw_text)
                        if cleaned_text and len(cleaned_text) > 10:  # Only keep substantial text
                            text_content.append(cleaned_text)
            
            # Combine all text and limit by KB_len if specified
            combined_text = ' '.join(text_content)
            
            # Additional filtering for very long text
            if KB_len != -1 and len(combined_text) > KB_len:
                # Try to cut at sentence boundaries
                sentences = re.split(r'[.!?]+\s+', combined_text[:KB_len])
                if len(sentences) > 1:
                    combined_text = '. '.join(sentences[:-1]) + '.'
                else:
                    combined_text = combined_text[:KB_len]
            
            print(f"Extracted and cleaned {len(combined_text)} characters from WISE dataset")
            print(f"Sample cleaned text: {combined_text[:200]}...")
            
            return combined_text
            
        except Exception as e:
            print(f"Error loading WISE dataset: {e}")
            print("Falling back to sample text...")
            return "The neural network processes information through spiking patterns. Each neuron contributes to the overall computation. Machine learning algorithms use artificial neural networks to simulate biological processes. Deep learning models can generate text by learning patterns from large datasets. Spiking neural networks offer a more biologically plausible approach to artificial intelligence."
    
    def validate_text_quality(self, text):
        """Validate that the cleaned text maintains good quality."""
        if not text or len(text) < 50:
            return False
        
        words = text.split()
        if len(words) < 10:
            return False
        
        # Check ratio of alphabetic characters
        alpha_chars = sum(1 for c in text if c.isalpha())
        total_chars = len(text.replace(' ', ''))
        if total_chars > 0 and alpha_chars / total_chars < 0.7:
            return False
        
        # Check for reasonable sentence structure
        sentences = re.split(r'[.!?]+', text)
        if len(sentences) < 2:
            return False
        
        return True
    
    def load_and_process_text(self, file_path=None, use_wise_dataset=True):
        if use_wise_dataset:
            content = self.load_wise_dataset()
            
            # Validate the cleaned content
            if not self.validate_text_quality(content):
                print("Warning: Cleaned text quality is low, using fallback...")
                content = "The neural network processes information through spiking patterns. Each neuron contributes to the overall computation. Machine learning algorithms use artificial neural networks to simulate biological processes. Deep learning models can generate text by learning patterns from large datasets. Spiking neural networks offer a more biologically plausible approach to artificial intelligence."
        else:
            # Original file loading logic with cleaning
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    raw_content = f.read()
                    content = self.clean_text(raw_content)
                    if KB_len != -1:
                        content = content[:KB_len]
                    print(f"Loaded and cleaned {len(content)} characters from {file_path}")
            except FileNotFoundError:
                content = "The neural network processes information through spiking patterns. Each neuron contributes to the overall computation. Machine learning algorithms use artificial neural networks to simulate biological processes. Deep learning models can generate text by learning patterns from large datasets. Spiking neural networks offer a more biologically plausible approach to artificial intelligence."
                print("Using sample text")
        
        words = content.lower().split()
        words = [w for w in words if w and len(w) > 0]
        unique_words = list(set(words))
        self.word_to_idx = {word: idx for idx, word in enumerate(unique_words)}
        self.idx_to_word = {idx: word for word, idx in self.word_to_idx.items()}
        
        print(f"Vocabulary size: {len(unique_words)} unique words")
        
        for i in range(len(words) - 1):
            self.bigram_counts[(words[i], words[i+1])] += self.word_to_idx[words[i]]
        
        self.create_transition_matrix_features()
        return words


    
    def create_transition_matrix_features(self):
        """Create a transition matrix and extract statistical features with double negative logic."""
        vocab_size = len(self.word_to_idx)
        self.transition_matrix = np.zeros((vocab_size, vocab_size))
        
        # Fill transition matrix
        for (w1, w2), count in self.bigram_counts.items():
            if w1 in self.word_to_idx and w2 in self.word_to_idx:
                i, j = self.word_to_idx[w1], self.word_to_idx[w2]
                self.transition_matrix[i, j] = count
        
        # Normalize rows to get initial probabilities
        row_sums = self.transition_matrix.sum(axis=1, keepdims=True)
        initial_probs = np.divide(self.transition_matrix, row_sums, 
                                out=np.zeros_like(self.transition_matrix), 
                                where=row_sums!=0)
        
        # Apply double negative logic:
        # First negation: Invert the probabilities (1 - prob)
        first_negation = 1 - initial_probs
        # Add small epsilon to avoid zero probabilities
        epsilon = 1e-8
        first_negation = np.maximum(first_negation, epsilon)
        
        # Second negation: Invert again to reinforce (1 - (1 - prob)) = prob
        # But apply a slight boost to non-zero probabilities for emphasis
        self.transition_probs = 1 - first_negation
        # Boost non-zero probabilities slightly to emphasize likely transitions
        self.transition_probs = np.where(self.transition_probs > epsilon, 
                                        self.transition_probs * 1.1, 
                                        self.transition_probs)
        # Re-normalize to ensure probabilities sum to 1
        final_sums = self.transition_probs.sum(axis=1, keepdims=True)
        self.transition_probs = np.divide(self.transition_probs, final_sums, 
                                        out=np.zeros_like(self.transition_probs), 
                                        where=final_sums!=0)
        
    def get_transition_features(self, word):
        """Extract transition-based features for a word."""
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
            
            # Incoming transitions
            in_transitions = self.transition_probs[:, word_idx]
            in_degree = np.sum(in_transitions > 0)
            features.append(in_degree)
            
            # Centrality measure
            centrality = np.sum(in_transitions)
            features.append(centrality)
        else:
            features.extend([0, 0, 0, 0, 0])
        
        return features
    
    def get_semantic_similarity(self, word, user_input):
        """Simple semantic similarity based on shared characters and context."""
        if not user_input:
            return 0.5
        
        user_words = user_input.lower().split()
        common = 0
        for user_word in user_words:
            common += self.word_to_idx.get(user_word, 0)
        return common
    
    def get_context_features(self, word, prev_word=None, next_word=None):
        """Get contextual features based on bigrams."""
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
        """Convert words to neural-compatible features."""
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
                context_weight = 2.0
                position_weight = 1.0 - (i / len(user_input.split())) if len(user_input.split()) > 0 else 1.0
            else:
                context_weight = 1.0
                position_weight = 0.5
            
            # Apply weights to existing features
            feature_vector = [f * context_weight * word_idx for f in feature_vector]
            
            # Add word embedding-like features
            feature_vector.append(word_idx / len(self.word_to_idx))
            feature_vector.append(len(word) / 20.0)
            
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
                
            if neural_gate > 0.1 and seed_words:
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
                    candidates, graph_features, neural_gate
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
                    context_boost += similarity * context_strength
            
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
            generated_words.append(next_word)
            current_word = next_word
            
            # Decay user context over time
            user_context_strength *= context_decay
        
        return ' '.join(generated_words)

def main_with_user_context_awareness():
    """Main function with user context awareness and training using WISE dataset."""
    num_neurons = 256
    num_steps = 64  # Make longer for training signal
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    print(f"Initializing with {num_neurons} neurons on device: {device}")
    
    # Initialize components
    text_processor = NeuronAwareTextProcessor(num_neurons)
    
    print("="*60)
    print("FGCN TEXT GENERATOR (with WISE dataset)")
    print("="*60)
    print("This system generates contextually relevant text based on user input.")
    print("Using WISE dataset from Hugging Face for training.")
    print("="*60)
    
    # Load WISE dataset
    text_processor.load_and_process_text(use_wise_dataset=True)
    
    # Generate synthetic neural sequences (replace with real data if available)
    print("Generating synthetic neural data...")
    spk_rec_full = torch.rand(num_steps, num_neurons)
    mem_rec_full = torch.rand(num_steps, num_neurons)
    
    # Train/val split
    split = int(0.8 * num_steps)
    spk_train, spk_val = spk_rec_full[:split], spk_rec_full[split:]
    mem_train, mem_val = mem_rec_full[:split], mem_rec_full[split:]
    
    print(f"Creating datasets - train: {spk_train.shape}, val: {spk_val.shape}")
    train_ds = GraphSequenceDataset(spk_train, mem_train)
    val_ds = GraphSequenceDataset(spk_val, mem_val)
    
    train_loader = make_loader(train_ds, batch_size=8, num_workers=4)
    val_loader = make_loader(val_ds, batch_size=8, num_workers=4)
    
    # Initialize FGCN model with correct in_dim from a sample graph
    print("Initializing FGCN model...")
    sample_data = create_neuron_aligned_graph(spk_train[:1], mem_train[:1])
    in_dim = sample_data.x.shape[1]
    fgcn_model = DataAwareFGCN(in_dim=in_dim, hidden_dim=64, out_dim=32).to(device)
    
    # Setup training
    optimizer = torch.optim.Adam(fgcn_model.parameters(), lr=0.001)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.9)
    
    # Train the model
    epochs = 10
    best_val_loss = float('inf')
    
    print(f"Training FGCN for {epochs} epochs...")
    for epoch in range(epochs):
        train_loss = train_one_epoch(fgcn_model, train_loader, device, optimizer)
        val_loss = evaluate(fgcn_model, val_loader, device)
        scheduler.step()
        
        print(f"Epoch {epoch+1}/{epochs} - Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")
        
        # Save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            save_checkpoint('best_fgcn_model.pth', epoch, fgcn_model, optimizer, scheduler, best_val_loss)
            print(f"New best model saved with val loss: {best_val_loss:.4f}")
    
    print("Training completed!")
    print("Now generating responses using WISE-trained model...")
    
    # Interactive generation loop
    while True:
        user_input = input("\nUSER: ").strip()
        if not user_input:
            print("Please enter some text or 'quit' to exit.")
            continue
        
        if user_input.lower() in ['quit', 'exit']:
            break
        
        print(f"\nProcessing input: '{user_input}'")
        print("="*40)
        
        # Use trained model with fresh synthetic data for generation
        spk_rec = torch.rand(10, num_neurons)
        mem_rec = torch.rand(10, num_neurons)
        
        # Create user-context-aware text generator with trained model
        context_generator = UserContextAwareTextGenerator(text_processor, fgcn_model)
        
        # Generate contextual response
        contextual_text = context_generator.generate_contextual_text(
            user_input, spk_rec, mem_rec, length=100
        )
        print("\nAI:", contextual_text)

if __name__ == "__main__":
    main_with_user_context_awareness()
