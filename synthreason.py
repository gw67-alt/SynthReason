import string
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset
import numpy as np
import re
import os
import pickle
import tqdm
from collections import Counter

# Parameters
KB_LIMIT = 999
SEQUENCE_LENGTH = 1
EMBEDDING_DIM = 256
HIDDEN_DIM = 256
LEARNING_RATE = 0.001
NUM_EPOCHS = 5
WINDOW_SIZE = 15  # Size of the window to consider for adjustments

# Neural Network Model for Text Generation
class TextGeneratorNN(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim):
        super(TextGeneratorNN, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.lstm = nn.LSTM(embedding_dim, hidden_dim)
        self.dropout = nn.Dropout(0.2)
        self.fc = nn.Linear(hidden_dim, vocab_size)
        
    def forward(self, x, hidden=None):
        embedded = self.embedding(x)
        if hidden is None:
            lstm_out, hidden = self.lstm(embedded)
        else:
            lstm_out, hidden = self.lstm(embedded, hidden)
        lstm_out = self.dropout(lstm_out)
        output = self.fc(lstm_out)
        return output, hidden

# Enhanced Set Operations Integration with Categories
class SetTheoryModifier:
    def __init__(self):
        # Empty set implementation - used to represent ∅
        self.empty_set = set()
        
        # Set theory operations categorized by concept
        self.set_operations = {
            'empty_not_in': {
                'name': 'z=∅∩∉',
                'description': 'Empty set and not-in operation',
                'active': True,
                'influence_factor': 0.15,
                'empty_boost': 1.7,
                'contradiction_penalty': 0.5
            }
        }
    
    def toggle_operation(self, operation_key):
        """Toggle a specific set operation on/off"""
        if operation_key in self.set_operations:
            self.set_operations[operation_key]['active'] = not self.set_operations[operation_key]['active']
            return f"{operation_key} ({self.set_operations[operation_key]['name']}) is now {'active' if self.set_operations[operation_key]['active'] else 'inactive'}"
        return f"Unknown operation: {operation_key}"
    
    def set_operation_parameter(self, operation_key, param_name, value):
        """Set a parameter value for a specific operation"""
        if operation_key in self.set_operations and param_name in self.set_operations[operation_key]:
            try:
                self.set_operations[operation_key][param_name] = float(value)
                return f"Set {param_name} to {value} for {operation_key}"
            except ValueError:
                return f"Invalid value: {value}. Must be a number."
        return f"Unknown operation or parameter: {operation_key}.{param_name}"
    
    def list_active_operations(self):
        """List all currently active set theory operations"""
        active_ops = [f"{key} ({op['name']}): {op['description']}" 
                     for key, op in self.set_operations.items() 
                     if op['active']]
        if active_ops:
            return "Active set theory operations:\n" + "\n".join(active_ops)
        else:
            return "No set theory operations are currently active"
    
    def get_category_words(self, category):
        """Get words associated with a specific category or set theory concept"""
        try:
            with open(f"{category}.txt", "r", encoding="utf-8") as f:
                return [line.strip() for line in f.readlines()]
        except FileNotFoundError:
            return []
    
    def apply_set_theory_modifiers(self, probs, vocab_inv):
        """Apply multiple set theory concepts to the probability distribution"""
        modified_probs = probs.clone()
        
        # Get category word lists for different concepts
        action_words = self.get_category_words("actions")
        description_words = self.get_category_words("descriptions")
        common_words = self.get_category_words("common")
        diverse_words = self.get_category_words("diverse")
        
        # Apply each active set theory operation
        for op_key, operation in self.set_operations.items():
            if operation['active']:
                # Apply operation-specific modifications
                # ∅∩∉ operation: Boost emptiness, penalize presence
                for i in range(len(modified_probs)):
                    word = vocab_inv[i].lower() if i in vocab_inv else ""
                    if word and any(empty_word not in word for empty_word in description_words):
                        modified_probs[i] *= operation['empty_boost']
                    if word and any(presence_word not in word for presence_word in action_words):
                        modified_probs[i] *= operation['contradiction_penalty']
        
        # Ensure probabilities are valid
        modified_probs = torch.maximum(modified_probs, torch.tensor(0.0))
        sum_probs = modified_probs.sum()
        if sum_probs > 0:
            modified_probs = modified_probs / sum_probs
        else:
            # If all probabilities became zero, revert to original
            modified_probs = probs.clone()
            
        return modified_probs

# Dataset class for PyTorch
class TextDataset(Dataset):
    def __init__(self, text_data, vocab, sequence_length):
        self.sequence_length = sequence_length
        self.vocab = vocab
        self.data = self.preprocess_text(text_data)
        self.sequences = self.create_sequences()
        
    def preprocess_text(self, text):
        text = re.sub(r'[^\w\s]', '', text.lower())
        tokens = text.split()
        return [self.vocab.get(word, self.vocab['<UNK>']) for word in tokens]
    
    def create_sequences(self):
        sequences = []
        for i in range(len(self.data) - self.sequence_length - 1):
            seq = self.data[i:i + self.sequence_length]
            target = self.data[i + self.sequence_length]
            sequences.append((seq, target))
        return sequences
    
    def __len__(self):
        return len(self.sequences)
    
    def __getitem__(self, idx):
        seq, target = self.sequences[idx]
        return torch.tensor(seq), torch.tensor(target)

# Function to calculate character ratios
def calculate_character_ratios(data):
    char_count = {letter: 0 for letter in string.ascii_lowercase}
    for item in data:
        item = item.strip()
        if item:
            first_letter = item[0].lower()
            if first_letter in char_count:
                char_count[first_letter] += 1
    total_items = len(data)
    char_ratios = {char: count / total_items for char, count in char_count.items()}
    return char_ratios

# Training function
def train_model(model, dataset, epochs, learning_rate, device):
    """Train the PyTorch model without using batches"""
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    
    model.train()
    for epoch in range(epochs):
        total_loss = 0
        hidden = None
        
        # Process samples one at a time
        progress_bar = tqdm.tqdm(range(len(dataset)), desc=f'Epoch {epoch+1}/{epochs}')
        for idx in progress_bar:
            inputs, targets = dataset[idx]
            inputs = inputs.unsqueeze(0).to(device)  # Single sample dimension
            targets = targets.to(device)
            
            # Clear gradients
            optimizer.zero_grad()
            
            # Forward pass
            output, hidden = model(inputs, hidden)
            hidden = tuple(h.detach() for h in hidden)  # Detach hidden state
            
            # Reshape output for loss calculation
            output = output.squeeze(0)
            loss = criterion(output, targets.unsqueeze(0))
            
            # Backward pass and optimize
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            progress_bar.set_postfix(loss=total_loss/(idx+1))
            
    return model

# Text generation function with PyTorch model
def generate_text_nn(model, prompt, vocab, vocab_inv, char_ratios, set_modifier, device, 
                     seq_length, max_length, temperature):
    """Generate text using the trained PyTorch model with set theory modifications"""
    model.eval()
    
    # Process prompt
    words = prompt.lower().split()
    input_seq = [vocab.get(word, vocab.get('<UNK>')) for word in words]
    
    # Pad sequence if needed
    while len(input_seq) < seq_length:
        input_seq = [vocab['<PAD>']] + input_seq
    
    # Use only the last seq_length words as input
    input_seq = input_seq[-seq_length:]
    
    # Convert to tensor
    input_tensor = torch.tensor(input_seq).unsqueeze(0).to(device)
    
    generated_text = prompt
    hidden = None
    recent_outputs = []
    
    for _ in range(max_length):
        # Forward pass through the model
        with torch.no_grad():
            output, hidden = model(input_tensor, hidden)
            
        # Get probabilities for the next word
        output = output[-1, :]  # Get the last timestep
        
        # Apply temperature
        output = output / temperature
        
        # Apply set theory modifiers
        probs = torch.softmax(output, dim=-1)
        probs = set_modifier.apply_set_theory_modifiers(probs, vocab_inv)
        
        # Apply character ratio adjustments
        for i in range(len(probs)):
            if i in vocab_inv:
                word = vocab_inv[i]
                if word and len(word) > 0:
                    first_char = word[0].lower()
                    if first_char in char_ratios:
                        probs[i] *= (1.0 + char_ratios[first_char])
        
        # Apply recency bias (avoid repeating recent words)
        for i in range(1, min(WINDOW_SIZE, len(recent_outputs)) + 1):
            if recent_outputs[-i] < len(probs):
                probs[recent_outputs[-i]] *= 0.5  # Reduce probability of recently used words
        
        # Ensure probabilities are valid
        probs = torch.maximum(probs, torch.tensor(0.0))
        if probs.sum() > 0:
            probs = probs / probs.sum()
        
        # Sample from the probability distribution
        next_idx = torch.multinomial(probs, 1).item()
        
        # Get the next word and add it to the generated text
        next_word = vocab_inv.get(next_idx, "<UNK>")
        generated_text += ' ' + next_word
        
        # Update input sequence for next iteration
        input_tensor = torch.tensor([[next_idx]]).to(device)
        recent_outputs.append(next_idx)
        
        # Limit the size of recent_outputs
        if len(recent_outputs) > WINDOW_SIZE:
            recent_outputs.pop(0)
    
    return generated_text

# Save and load functions
def save_model(model, vocab, char_ratios, filepath="text_model.pt"):
    """Save the model and necessary data"""
    checkpoint = {
        'model_state_dict': model.state_dict(),
        'vocab': vocab,
        'char_ratios': char_ratios
    }
    torch.save(checkpoint, filepath)
    print(f"Model saved to {filepath}")

def load_model(filepath="text_model.pt", embedding_dim=EMBEDDING_DIM, hidden_dim=HIDDEN_DIM, device='cuda'):
    """Load the model and necessary data"""
    checkpoint = torch.load(filepath, map_location=device)
    vocab = checkpoint['vocab']
    vocab_size = len(vocab)
    
    model = TextGeneratorNN(vocab_size, embedding_dim, hidden_dim).to(device)
    model.load_state_dict(checkpoint['model_state_dict'])
    
    char_ratios = checkpoint['char_ratios']
    
    return model, vocab, char_ratios

# Main function
def main():
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Try to load existing model, otherwise train a new one
    try:
        print("Attempting to load existing model...")
        model, vocab, char_ratios = load_model(device=device)
        vocab_inv = {idx: word for word, idx in vocab.items()}
        print("Model loaded successfully.")
        
    except:
        print("No existing model found. Preparing to train a new model...")
        
        # Load and preprocess data
        try:
            with open("kb.txt", "r", encoding="utf-8") as f:
                text = ' '.join(f.read().split()[:KB_LIMIT])
            
            # Clean text and calculate character ratios
            text = re.sub(r'\d+', '', text)
            
            # Filter out short words but keep common exceptions
            pattern = r'^[a-zA-Z]{1,2}$'
            exceptions = ['a', 'i', 'to', 'is', 'it', 'an', 'of', 'by', 'he', 'me', 'we', 'be', 'my', 'up', 'do', 'go', 'if', 'no', 'so', 'on', 'at', 'in', 'as', 'or', 'la', 'ah', 'uh', 'ye', 'ab', 'ad', 'ae', 'ba', 'bi', 'bo', 'da', 'ed', 'ef', 'eh', 'el', 'em', 'en', 'er', 'es', 'ex', 'fa', 'hi', 'ho', 'id', 'is', 'jo', 'ka', 'la', 'li', 'lo', 'ma', 'me', 'mi', 'mu', 'na', 'no', 'nu', 'od', 'oe', 'oi', 'om', 'op', 'os', 'ow', 'ox', 'oy', 'pa', 're', 'sh', 'si', 'ta', 'uh', 'um','un', 'up', 'us', 'ut', 'va', 'ye', 'yo']
            filtered_words = [word for word in text.split() if not re.match(pattern, word) or word in exceptions]
            
            # Calculate character ratios
            char_ratios = calculate_character_ratios(filtered_words)
            
            # Build vocabulary
            word_counts = Counter(filtered_words)
            vocab = {'<PAD>': 0, '<UNK>': 1}
            for idx, (word, _) in enumerate(word_counts.most_common(), 2):
                vocab[word] = idx
            vocab_inv = {idx: word for word, idx in vocab.items()}
            vocab_size = len(vocab)
            
            # Create dataset
            text = ' '.join(filtered_words)
            dataset = TextDataset(text, vocab, SEQUENCE_LENGTH)
            
            # Initialize and train model
            print(f"Vocabulary size: {vocab_size}")
            model = TextGeneratorNN(vocab_size, EMBEDDING_DIM, HIDDEN_DIM).to(device)
            print("Training general model...")
            train_model(model, dataset, NUM_EPOCHS, LEARNING_RATE, device)
            save_model(model, vocab, char_ratios)
            
        except FileNotFoundError:
            print("Error: kb.txt file not found. Please ensure the knowledge base file exists.")
            return
    
    # Initialize set theory modifier
    set_modifier = SetTheoryModifier()
    
    print("\nEnhanced Text Generator with PyTorch Neural Networks")

    current_topic = None
    auto_topic_detection = False
    temperature = 1.0
    
    try:
        while True:
            try:
                prompt = input("\nUSER: ")
                                # Generate text
                generated_text = generate_text_nn(
                    model,
                    prompt,
                    vocab,
                    vocab_inv,
                    char_ratios,
                    set_modifier,
                    device,
                    seq_length=SEQUENCE_LENGTH,
                    max_length=250,
                    temperature=temperature,
                )
                
                print("\nAI: ", generated_text)
            except EOFError:
                print("Input stream ended. Exiting...")
                break
    
    except KeyboardInterrupt:
        print("\nProgram interrupted. Exiting...")

if __name__ == "__main__":
    main()
