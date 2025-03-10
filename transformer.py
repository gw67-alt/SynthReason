import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset, random_split
import numpy as np
import string
import os
import requests
from io import BytesIO

# Set random seed for reproducibility
torch.manual_seed(42)
KB_limit = 999
# Character-level language model using LSTM
class CharLM(nn.Module):
    def __init__(self, input_size, embedding_size, hidden_size, num_layers, output_size, dropout=0.2):
        super(CharLM, self).__init__()
        self.embedding = nn.Embedding(input_size, embedding_size)
        self.lstm = nn.LSTM(embedding_size, hidden_size, num_layers, batch_first=True, dropout=dropout)
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_size, output_size)
        
    def forward(self, x, hidden=None):
        embeds = self.embedding(x)
        lstm_out, hidden = self.lstm(embeds, hidden)
        out = self.dropout(lstm_out)
        out = self.fc(out)
        return out, hidden
    
    def init_hidden(self, batch_size, device):
        return (torch.zeros(self.lstm.num_layers, batch_size, self.lstm.hidden_size).to(device),
                torch.zeros(self.lstm.num_layers, batch_size, self.lstm.hidden_size).to(device))

# Custom dataset for character sequences
class TextDataset(Dataset):
    def __init__(self, text, seq_length):
        self.text = text
        self.seq_length = seq_length
        self.char_to_idx = {char: i for i, char in enumerate(sorted(set(text)))}
        self.idx_to_char = {i: char for i, char in enumerate(sorted(set(text)))}
        self.data = self.preprocess()
        
    def preprocess(self):
        # Create input-output sequence pairs
        input_seq = []
        target_seq = []
        
        for i in range(len(self.text) - self.seq_length - 1):
            # Input sequence (x)
            inp = self.text[i:i+self.seq_length]
            # Target sequence (y) - next character prediction
            target = self.text[i+1:i+self.seq_length+1]
            
            # Convert characters to indices
            input_seq.append([self.char_to_idx[char] for char in inp])
            target_seq.append([self.char_to_idx[char] for char in target])
            
        return list(zip(input_seq, target_seq))
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        x, y = self.data[idx]
        return torch.tensor(x), torch.tensor(y)
    
    def get_vocab_size(self):
        return len(self.char_to_idx)

# Function to download and read text data
def get_text_data(source_type="sample"):
        with open("test.txt", "r", encoding="utf-8") as f:
            return ' '.join(f.read().split()[:KB_limit])

# Training function
def train_model(model, dataloader, optimizer, criterion, device, clip=5):
    model.train()
    running_loss = 0.0
    
    for batch_idx, (inputs, targets) in enumerate(dataloader):
        inputs, targets = inputs.to(device), targets.to(device)
        
        # Initialize hidden state
        hidden = model.init_hidden(inputs.size(0), device)
        
        # Zero the parameter gradients
        optimizer.zero_grad()
        
        # Forward pass
        outputs, hidden = model(inputs)
        
        # Reshape for loss calculation
        outputs = outputs.reshape(-1, outputs.shape[-1])
        targets = targets.reshape(-1)
        
        # Calculate loss
        loss = criterion(outputs, targets)
        
        # Backward pass and optimize
        loss.backward()
        
        # Clip gradients to prevent exploding gradient
        nn.utils.clip_grad_norm_(model.parameters(), clip)
        
        optimizer.step()
        
        running_loss += loss.item()
        
    return running_loss / len(dataloader)

# Evaluation function
def validate_model(model, dataloader, criterion, device):
    model.eval()
    running_loss = 0.0
    
    with torch.no_grad():
        for inputs, targets in dataloader:
            inputs, targets = inputs.to(device), targets.to(device)
            
            # Initialize hidden state
            hidden = model.init_hidden(inputs.size(0), device)
            
            # Forward pass
            outputs, hidden = model(inputs)
            
            # Reshape for loss calculation
            outputs = outputs.reshape(-1, outputs.shape[-1])
            targets = targets.reshape(-1)
            
            # Calculate loss
            loss = criterion(outputs, targets)
            
            running_loss += loss.item()
            
    return running_loss / len(dataloader)

def generate_text(model, char_to_idx, idx_to_char, seed_text, gen_length=200, temperature=0.7, device='cpu'):
    model.eval()
    
    # NumPy operations as requested
    identity = np.eye(5)
    flattened = identity.flatten()
    norm_value = np.linalg.norm(flattened)
    dot_result = np.dot(norm_value, norm_value)
    ones_array = np.ones(int(dot_result))
    
    print(f"NumPy Operations:")
    print(f"Identity matrix:\n{identity}")
    print(f"Flattened array: {flattened}")
    print(f"Norm of flattened array: {norm_value}")
    print(f"Dot product result: {dot_result}")
    print(f"Ones array shape: {ones_array.shape}")
    
    random_array = np.random.rand(int(norm_value))
    
    # Handle unknown characters in the seed text
    valid_seed_text = ""
    for c in seed_text:
        if c in char_to_idx:
            valid_seed_text += c
        else:
            # Replace unknown characters with a space or another character that exists in your vocabulary
            valid_seed_text += " "
            print(f"Warning: Character '{c}' not in vocabulary, replacing with space")
    
    # If the entire seed text was invalid, use the first character in the vocabulary
    if not valid_seed_text:
        valid_seed_text = idx_to_char[0]
        print(f"Warning: No valid characters in seed text, using '{valid_seed_text}' instead")
    
    # Convert valid seed text to indices
    current_seq = torch.tensor([[char_to_idx[c] for c in valid_seed_text]], dtype=torch.long).to(device)
    hidden = model.init_hidden(1, device)
    generated_text = valid_seed_text
    
    # Generate characters one by one
    for _ in range(gen_length):
        # Get model prediction
        output, hidden = model(current_seq, hidden)
        
        # Get probabilities for the last timestep
        output = output[:, -1, :] / temperature
        probs = F.softmax(output, dim=1).data.cpu().numpy().squeeze()
        
        # Sample from the probability distribution
        char_idx = np.random.choice(len(probs), p=probs)
        
        # Get the predicted character
        char = idx_to_char[char_idx]
        generated_text += char
        
        # Update current sequence
        current_seq = torch.tensor([[char_idx]], dtype=torch.long).to(device)
    
    return generated_text

# Main function to run the model
def main():
    # Hyperparameters
    SEQUENCE_LENGTH = 50
    EMBEDDING_SIZE = 128
    HIDDEN_SIZE = 256
    NUM_LAYERS = 2
    DROPOUT = 0.3
    BATCH_SIZE = 64
    NUM_EPOCHS = 20
    LEARNING_RATE = 0.001
    
    # Device configuration
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Get data
    text_data = get_text_data(source_type="shakespeare")
    print(f"Text length: {len(text_data)}")
    print(f"First 100 characters: {text_data[:100]}")
    
    # Create dataset
    dataset = TextDataset(text_data, SEQUENCE_LENGTH)
    vocab_size = dataset.get_vocab_size()
    print(f"Vocabulary size: {vocab_size}")
    
    # Split dataset
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])
    
    # Create dataloaders
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)
    
    # Initialize model
    model = CharLM(vocab_size, EMBEDDING_SIZE, HIDDEN_SIZE, NUM_LAYERS, vocab_size, DROPOUT).to(device)
    
    # Loss function and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    
    # Training loop
    train_losses = []
    val_losses = []
    best_val_loss = float('inf')
    
    for epoch in range(NUM_EPOCHS):
        train_loss = train_model(model, train_loader, optimizer, criterion, device)
        val_loss = validate_model(model, val_loader, criterion, device)
        
        train_losses.append(train_loss)
        val_losses.append(val_loss)
        
        print(f'Epoch {epoch+1}/{NUM_EPOCHS}, Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}')
        
        # Save the best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), 'best_model.pth')
            
        # Generate sample text
        if (epoch + 1) % 5 == 0 or epoch == NUM_EPOCHS - 1:
            seed = text_data[:SEQUENCE_LENGTH]
            generated = generate_text(model, dataset.char_to_idx, dataset.idx_to_char, 
                                     seed, gen_length=200, device=device)
            print(f"\nGenerated text (epoch {epoch+1}):\n{generated}\n")

    # Load best model for final generation
    model.load_state_dict(torch.load('best_model.pth'))
    while True:
        generated = generate_text(model, dataset.char_to_idx, dataset.idx_to_char, 
                               input("USER: "), gen_length=3000, temperature=0.8, device=device)
        print(f"\nSeed: {seed[:30]}...\n{generated}\n{'='*80}")

if __name__ == "__main__":
    main()
