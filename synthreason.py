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
KB_LIMIT = 3399
SEQUENCE_LENGTH = 1
EMBEDDING_DIM = 256
HIDDEN_DIM = 256
magic = 3 #RuntimeError: Expected hidden[0] size (1, 3, 256), got [1, 6, 256]
LEARNING_RATE = 0.001
NUM_EPOCHS = 5
WINDOW_SIZE = 15

# Neural Network Model for Text Generation
class TextGeneratorNN(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim):
        super(TextGeneratorNN, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, batch_first=True)
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

# Modified training function with Manhattan implication for acceleration
def train_model_manhattan(model, dataset, epochs, learning_rate, device):
    """Train the PyTorch model using Manhattan distance for faster convergence"""
    # Use L1Loss (Manhattan distance) as part of the training objective
    ce_criterion = nn.CrossEntropyLoss()
    l1_criterion = nn.L1Loss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    
    # Enable eager execution for faster iteration
    torch.backends.cudnn.benchmark = True
    
    # Training statistics for Manhattan convergence tracking
    convergence_stats = []
    early_stop_patience = 3
    best_loss = float('inf')
    patience_counter = 0
    
    model.train()
    for epoch in range(epochs):
        total_loss = 0
        hidden = None
        
        # Create mini-batches for faster processing
        batch_size = min(magic, len(dataset) // 10)  # Dynamic batch sizing
        indices = torch.randperm(len(dataset))
        batched_indices = [indices[i:i+batch_size] for i in range(0, len(indices), batch_size)]
        
        progress_bar = tqdm.tqdm(batched_indices, desc=f'Epoch {epoch+1}/{epochs}')
        for batch_idx in progress_bar:
            # Initialize batch tensors
            batch_inputs = []
            batch_targets = []
            
            # Collect batch samples
            for idx in batch_idx:
                inputs, targets = dataset[idx.item()]
                batch_inputs.append(inputs)
                batch_targets.append(targets)
            
            # Convert to tensors and move to device
            batch_inputs = torch.stack(batch_inputs).to(device)
            batch_targets = torch.stack(batch_targets).to(device)
            
            # Clear gradients
            optimizer.zero_grad()
            
            # Forward pass
            output, hidden = model(batch_inputs, hidden)
            hidden = tuple(h.detach() for h in hidden)  # Detach hidden state
            
            # Apply Manhattan implication by combining cross-entropy with L1 regularization
            ce_loss = ce_criterion(output.reshape(-1, output.size(-1)), batch_targets)
            
            # L1 regularization on the output logits (Manhattan component)
            l1_reg = sum(p.abs().sum() for p in model.parameters()) * 0.0001
            
            # Combined loss with Manhattan implication
            loss = ce_loss + l1_reg
            
            # Backward pass and optimize
            loss.backward()
            
            # Apply gradient clipping for stability
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
            optimizer.step()
            
            # Track loss
            total_loss += loss.item()
            avg_loss = total_loss / len(batch_idx)
            progress_bar.set_postfix(loss=avg_loss, l1_component=l1_reg.item())
            
        # Track convergence statistics
        epoch_loss = total_loss / len(batched_indices)
        convergence_stats.append(epoch_loss)
        
        # Manhattan-based early stopping
        if epoch_loss < best_loss:
            best_loss = epoch_loss
            patience_counter = 0
        else:
            patience_counter += 1
            
        if patience_counter >= early_stop_patience:
            print(f"Early stopping at epoch {epoch+1} due to Manhattan convergence")
            break
            
        # Dynamic learning rate adjustment based on Manhattan convergence
        if epoch > 0 and convergence_stats[-1] < convergence_stats[-2]:
            learning_rate *= 0.8
            for param_group in optimizer.param_groups:
                param_group['lr'] = learning_rate
            print(f"Adjusted learning rate to {learning_rate}")
    
    return model

# Text generation function with bilinear adversarial character adjustments
def generate_text_nn(model, prompt, vocab, vocab_inv, device, 
                     seq_length, max_length, temperature=0.9):
    """Generate text using the trained PyTorch model with bilinear adversarial modifications"""
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
    
    # Track character usage for dynamic adjustment
    recent_chars = []
    
    for _ in range(max_length):
        # Forward pass through the model
        with torch.no_grad():
            output, hidden = model(input_tensor, hidden)
            
        # Get probabilities for the next word
        output = output[:, -1, :]  # Get the last timestep
        
        # Apply temperature
        output = output / temperature
        
        # Apply bilinear adversarial modifiers
        probs = torch.softmax(output, dim=-1)
        
        # Apply bilinear adversarial weights to word probabilities
        modified_probs = probs.clone()
        if 1 in vocab_inv and vocab_inv[1] == '<UNK>':  # 1 is typically the UNK token index
                modified_probs[0][1] *= 0.00001  # Multiply by a very small number

        
        # Ensure probabilities are valid
        modified_probs = torch.maximum(modified_probs, torch.tensor(0.0))
        if modified_probs.sum() > 0:
            modified_probs = modified_probs / modified_probs.sum()
        
        # Sample from the modified probability distribution
        next_idx = torch.multinomial(modified_probs.squeeze(), 1).item()
        
        # Get the next word and add it to the generated text
        next_word = vocab_inv.get(next_idx, "<UNK>")
        generated_text += ' ' + next_word
        
        # Update tracking variables
        input_tensor = torch.cat((input_tensor, torch.tensor([[next_idx]]).to(device)), dim=1)
        input_tensor = input_tensor[:, -seq_length:]  # Keep sequence length fixed
        
        recent_outputs.append(next_idx)
        if len(recent_outputs) > WINDOW_SIZE:
            recent_outputs.pop(0)
        
        # Update recent character usage for adaptive adversarial effects
        if next_word and len(next_word) > 0:
            for char in next_word.lower():
                if char in string.ascii_lowercase:
                    recent_chars.append(char)
            
            # Only keep track of the most recent characters
            if len(recent_chars) > 50:
                recent_chars = recent_chars[-50:]
    
    return generated_text

# Save and load functions
def save_model(model, vocab, filepath="text_model_manhattan.pt"):
    """Save the model and necessary data"""
    checkpoint = {
        'model_state_dict': model.state_dict(),
        'vocab': vocab
    }
    torch.save(checkpoint, filepath)
    print(f"Model saved to {filepath}")

def load_model(filepath="text_model_manhattan.pt", embedding_dim=EMBEDDING_DIM, hidden_dim=HIDDEN_DIM, device='cuda'):
    """Load the model and necessary data"""
    checkpoint = torch.load(filepath, map_location=device)
    vocab = checkpoint['vocab']
    vocab_size = len(vocab)
    
    model = TextGeneratorNN(vocab_size, embedding_dim, hidden_dim).to(device)
    model.load_state_dict(checkpoint['model_state_dict'])
    
    return model, vocab

# Main function
def main():
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Try to load existing model, otherwise train a new one
    try:
        print("Attempting to load existing model...")
        model, vocab, char_ratios = load_model(device=device)
        # Convert character ratios to bilinear adversarial format if needed
        if not isinstance(next(iter(char_ratios.values())), dict):
            print("Converting to bilinear adversarial character ratios...")
            # We need text data to calculate bilinear ratios
            # For demonstration, we'll use the vocabulary as a proxy
            vocab_inv = {idx: word for word, idx in vocab.items()}
            vocab_words = list(vocab.keys())
            save_model(model, vocab)
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
            print("Training model with Manhattan implication acceleration...")
            train_model_manhattan(model, dataset, NUM_EPOCHS, LEARNING_RATE, device)
            save_model(model, vocab)
            
        except FileNotFoundError:
            print("Error: kb.txt file not found. Please ensure the knowledge base file exists.")
            return
    
    print("\nEnhanced Text Generator with Manhattan implication")

    temperature = 1.0
    
    try:
        while True:
            try:
                prompt = input("\nUSER: ")
                # Generate text with bilinear adversarial character adjustments
                generated_text = generate_text_nn(
                    model,
                    prompt,
                    vocab,
                    vocab_inv,
                    device,
                    seq_length=SEQUENCE_LENGTH,
                    max_length=250,
                    temperature=temperature
                )
                
                print("\nAI: ", generated_text)
            except EOFError:
                print("Input stream ended. Exiting...")
                break
    
    except KeyboardInterrupt:
        print("\nProgram interrupted. Exiting...")

if __name__ == "__main__":
    main()
