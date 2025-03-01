import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import json
import re
import numpy as np
import os

# Configuration parameters
KB_limit = 9999
sequence_length = 3

def preprocess_text(text):
    """Preprocess text by converting to lowercase and removing special characters."""
    text = text.lower()
    text = re.sub(r'[^a-z0-9\s]', '', text)
    return text.split()

def build_vocabulary(text):
    """Build vocabulary from text data."""
    words = preprocess_text(text)
    word_set = sorted(set(words))  # Sort for deterministic indexing
    vocab = {word: i + 1 for i, word in enumerate(word_set)}
    vocab["<PAD>"] = 0  # Padding token
    vocab["<UNK>"] = len(vocab)  # Unknown token
    return vocab, len(vocab)

class CyberneticsDataset(Dataset):
    def __init__(self, text_data, vocab, sequence_length=sequence_length):
        """Initialize dataset with text data instead of assuming file path."""
        self.data = preprocess_text(text_data[:KB_limit])  # Tokenized text
        self.vocab = vocab
        self.sequence_length = sequence_length

    def __len__(self):
        return max(1, len(self.data) - self.sequence_length)

    def __getitem__(self, idx):
        input_seq = self.data[idx:idx + self.sequence_length]
        target_seq = self.data[idx + 1:idx + self.sequence_length + 1]
        
        # Use <UNK> token for unknown words
        input_tokens = [self.vocab.get(word, self.vocab["<UNK>"]) for word in input_seq]
        target_tokens = [self.vocab.get(word, self.vocab["<UNK>"]) for word in target_seq]

        return torch.tensor(input_tokens, dtype=torch.long), torch.tensor(target_tokens, dtype=torch.long)

def pad_collate(batch):
    """Collate function that properly pads sequences to the same length."""
    inputs, targets = zip(*batch)
    max_len = max(len(seq) for seq in inputs)
    
    inputs_padded = torch.zeros((len(inputs), max_len), dtype=torch.long)
    targets_padded = torch.zeros((len(targets), max_len), dtype=torch.long)  # Initialize with 0 (PAD) not 1

    for i, (inp, tgt) in enumerate(zip(inputs, targets)):
        inputs_padded[i, :len(inp)] = inp
        targets_padded[i, :len(tgt)] = tgt
    
    return inputs_padded, targets_padded

class CyberneticsLSTM(nn.Module):
    def __init__(self, vocab_size, embedding_dim=128, hidden_dim=256):
        super(CyberneticsLSTM, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, batch_first=True)
        self.fc = nn.Linear(hidden_dim, vocab_size)
        
    def forward(self, x, hidden=None):
        """Allow passing hidden state for text generation."""
        embedded = self.embedding(x)
        if hidden is not None:
            out, hidden = self.lstm(embedded, hidden)
        else:
            out, hidden = self.lstm(embedded)
        out = self.fc(out)
        return out, hidden

def train_model(model, text_data, vocab, vocab_size, epochs=10, batch_size=32):
    """Train model with proper batch size and more robust configuration."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    
    dataset = CyberneticsDataset(text_data, vocab)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, collate_fn=pad_collate)

    criterion = nn.CrossEntropyLoss(ignore_index=0)  # Ignore padding tokens
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    best_loss = float('inf')
    
    for epoch in range(epochs):
        model.train()
        total_loss = 0
        for inputs, targets in dataloader:
            inputs, targets = inputs.to(device), targets.to(device)
            
            optimizer.zero_grad()
            outputs, _ = model(inputs)
            
            # Reshape for loss calculation
            outputs = outputs.view(-1, vocab_size)
            targets = targets.view(-1)

            loss = criterion(outputs, targets)
            loss.backward()
            
            # Add gradient clipping to prevent exploding gradients
            torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5)
            
            optimizer.step()
            total_loss += loss.item()

        avg_loss = total_loss / len(dataloader)
        print(f"Epoch {epoch+1}/{epochs}, Loss: {avg_loss:.4f}")
        
        # Save best model
        if avg_loss < best_loss:
            best_loss = avg_loss
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': best_loss,
                'vocab': vocab
            }, "cybernetics_model_best.pth")
    
    # Save final model
    torch.save({
        'epoch': epochs,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': avg_loss,
        'vocab': vocab
    }, "cybernetics_model_final.pth")
    
    return model

def load_model(model_path, vocab_size):
    """Load a saved model."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    checkpoint = torch.load(model_path, map_location=device)
    model = CyberneticsLSTM(vocab_size)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    
    return model, checkpoint['vocab']

def generate_text(model, word_to_index, input_text, sequence_length, generate_length, temperature=0.7):
    """Generate text with improved sampling method."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    
    # Process input
    input_sequence = preprocess_text(input_text)
    index_to_word = {i: word for word, i in word_to_index.items()}
    
    # Handle unknown words with <UNK> token
    input_indices = [word_to_index.get(word, word_to_index["<UNK>"]) for word in input_sequence]
    
    # Pad if necessary
    if len(input_indices) < sequence_length:
        input_indices = [word_to_index["<PAD>"]] * (sequence_length - len(input_indices)) + input_indices
    # Truncate if too long
    elif len(input_indices) > sequence_length:
        input_indices = input_indices[-sequence_length:]

    generated_text = input_sequence[:]
    model.eval()
    
    # Initialize hidden state
    hidden = None

    with torch.no_grad():
        for _ in range(generate_length):
            input_tensor = torch.tensor(input_indices[-sequence_length:], dtype=torch.long).unsqueeze(0).to(device)
            
            # Use the hidden state for continuity
            output, hidden = model(input_tensor, hidden)
            output = output[0, -1] / temperature  # Take last step's prediction
            
            # Apply softmax to get probabilities
            probabilities = torch.softmax(output, dim=-1).cpu().numpy()
            
            # Filter out padding token from sampling
            probabilities[word_to_index["<PAD>"]] = 0
            
            # Renormalize after filtering
            probabilities = probabilities / probabilities.sum()
            
            # Sample from the distribution
            next_index = np.random.choice(len(probabilities), p=probabilities)
            next_word = index_to_word.get(next_index, "<UNK>")
            
            # Skip padding and unknown tokens in the output
            if next_word not in ["<PAD>", "<UNK>"] and next_word != generated_text[-1]:
                generated_text.append(next_word)
                
            input_indices.append(next_index)

    return " ".join(generated_text)

def main():
    """Main function to run the program."""
    # Check if model exists
    model_path = "cybernetics_model_best.pth"
    
    if os.path.exists("test.txt"):
        # Load text data
        with open("test.txt", "r", encoding="utf-8") as f:
            text_data = f.read()[:KB_limit]
        
        # Build vocabulary
        word_to_index, vocab_size = build_vocabulary(text_data)
        
        # Add <UNK> token if not present
        if "<UNK>" not in word_to_index:
            word_to_index["<UNK>"] = len(word_to_index)
            vocab_size = len(word_to_index)
        
        if os.path.exists(model_path):
            print("Loading existing model...")
            model, word_to_index = load_model(model_path, vocab_size)
        else:
            print("Training new model...")
            model = CyberneticsLSTM(vocab_size)
            model = train_model(model, text_data, word_to_index, vocab_size, epochs=10)
        
        # Interactive loop
        print("Model ready! Type 'exit' or 'quit' to end.")
        while True:
            user_input = input("USER: ")
            if user_input.lower() in ["exit", "quit"]:
                break
            
            temperature = 0.7  # Default temperature
            
            # Check for temperature setting in input
            try:
                user_input = re.sub(r'temp=\d+\.?\d*', '', user_input).strip()
                print(f"Using temperature: {temperature}")
            except ValueError:
                pass
            
            generated = generate_text(
                model, 
                word_to_index, 
                user_input, 
                sequence_length, 
                generate_length=250,  # Reduced for faster responses
                temperature=temperature
            )
            
            print("Generated Text:\n", generated)
    else:
        print("Error: test.txt file not found.")

if __name__ == "__main__":
    main()
