import re
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import random

class TextDataset(Dataset):
    def __init__(self, X, positions, y):
        self.X = X
        self.positions = positions  # New position information
        self.y = y

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.positions[idx], self.y[idx]

class TextGenerator(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim, n_gram_size, max_seq_length):
        super(TextGenerator, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        
        # Position encoding
        self.position_embedding = nn.Embedding(max_seq_length, embedding_dim)
        
        # Combined input size (word embedding + position embedding)
        self.lstm = nn.LSTM(embedding_dim * 2, hidden_dim, batch_first=True)
        self.fc = nn.Linear(hidden_dim, vocab_size)
        
        # Add support for Hamming weight input
        self.hamming_fc = nn.Linear(1, embedding_dim)

    def forward(self, x, positions=None, hamming_weights=None):
        if hamming_weights is not None:
            # Process hamming weights - reshape to have proper dimensions
            hamming_input = hamming_weights.view(-1, 1).float()
            x = self.hamming_fc(hamming_input)
            x = x.unsqueeze(1)  # Add sequence dimension
            return self.fc(x.squeeze(1))
        else:
            # Normal embedding lookup
            word_embeddings = self.embedding(x)
            
            # Add position embeddings if provided
            if positions is not None:
                pos_embeddings = self.position_embedding(positions)
                # Concatenate word and position embeddings
                combined_embeddings = torch.cat((word_embeddings, pos_embeddings), dim=2)
            else:
                # If no positions provided, duplicate word embeddings to maintain dimensions
                combined_embeddings = torch.cat((word_embeddings, torch.zeros_like(word_embeddings)), dim=2)
            
            # Process through LSTM
            lstm_out, _ = self.lstm(combined_embeddings)
            output = self.fc(lstm_out[:, -1, :])
            return output

def train_model(text_file='test.txt', limit=1500):
    with open(text_file, 'r', encoding="utf-8") as file:
        text = ' '.join(file.read().split()[:limit])
    
    words = re.sub(r'\d+', '', text.lower()).split()
    
    # Create vocabulary
    unique_words = list(set(words))
    word_to_index = {word: idx + 1 for idx, word in enumerate(unique_words)}
    index_to_word = {idx + 1: word for idx, word in enumerate(unique_words)}
    
    vocab_size = len(word_to_index) + 1
    print(f"Vocabulary size: {vocab_size}")
    
    N_GRAM_SIZE = 3
    EMBEDDING_DIM = 256
    HIDDEN_DIM = 128
    EPOCHS = 10
    MAX_SEQ_LENGTH = len(words)  # Maximum possible position
    
    # Create training sequences with position information
    sequences = []
    positions = []
    for i, n in enumerate(range(len(words) - N_GRAM_SIZE)):
        seq = [words[n]]+words[i:i + N_GRAM_SIZE]
        sequences.append([word_to_index[word] for word in seq])
        # Store the position of each word in the sequence
        positions.append([n + j for j in range(N_GRAM_SIZE)])
    
    # Split into input and target
    X = [seq[:-1] for seq in sequences]
    y = [seq[-1] for seq in sequences]
    
    # Convert to PyTorch tensors
    X = torch.tensor(X, dtype=torch.long)
    positions = torch.tensor(positions, dtype=torch.long)
    y = torch.tensor(y, dtype=torch.long)
    
    # Create dataset and dataloader
    dataset = TextDataset(X, positions, y)
    dataloader = DataLoader(dataset, batch_size=64, shuffle=True)
    
    # Initialize model with position encoding
    model = TextGenerator(vocab_size, EMBEDDING_DIM, HIDDEN_DIM, N_GRAM_SIZE, MAX_SEQ_LENGTH)
    
    # Training
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.CrossEntropyLoss()

    for epoch in range(EPOCHS):
        model.train()
        total_loss = 0
        for inputs, pos, labels in dataloader:
            optimizer.zero_grad()
            outputs = model(inputs, pos)
            word_occurrences = {word_to_index[word]: words.count(word) for word in unique_words}

            # Compute custom loss
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        
        print(f'Epoch {epoch+1}/{EPOCHS}, Loss: {total_loss/len(dataloader):.4f}')
    
    return model, word_to_index, index_to_word, MAX_SEQ_LENGTH

def generate_text(seed_text, generate_length, model, word_to_index, index_to_word, n_gram_size, max_seq_length, temperature=1.0):
    model.eval()
    generated_text = []
    
    words = seed_text.lower().split()
    current_sequence = [word_to_index[word] if word in word_to_index else random.choice(list(word_to_index.values())) for word in words]
    
    while len(current_sequence) < n_gram_size:
        current_sequence.append(random.choice(list(word_to_index.values())))
    current_sequence = current_sequence[-n_gram_size:]
    
    # Starting positions (we'll use relative positions for generation)
    current_positions = list(range(n_gram_size))
    
    print(f"\nSeed text: {seed_text}")
    print("\nGenerated text: ", end="", flush=True)
    
    position_counter = n_gram_size  # Start after the seed sequence
    
    for _ in range(generate_length):
        input_seq = torch.tensor([current_sequence], dtype=torch.long)
        input_pos = torch.tensor([current_positions], dtype=torch.long)
        
        with torch.no_grad():
            # Get output from model
            output = model(input_seq, input_pos)
            
            # Check output dimensions and handle accordingly
            if len(output.shape) == 2:
                # If output shape is [batch_size, vocab_size]
                logits = output[0] / temperature
            elif len(output.shape) == 3:
                # If output shape is [batch_size, sequence_length, vocab_size]
                logits = output[0, -1, :] / temperature
            else:
                raise ValueError(f"Unexpected output shape: {output.shape}")
            
            # Compute probabilities using softmax
            word_probs = torch.softmax(logits, dim=0)
            
            # Sample based on probabilities
            predicted_word_idx = torch.multinomial(word_probs, 1).item()
            
            if predicted_word_idx in index_to_word:
                predicted_word = index_to_word[predicted_word_idx]
                generated_text.append(predicted_word)
                print(predicted_word + " ", end="", flush=True)
                
                # Update sequence with the predicted word index
                current_sequence.append(predicted_word_idx)
                current_sequence = current_sequence[-n_gram_size:]
                
                # Update position information
                position_counter += 1
                # Ensure we don't exceed max sequence length by using modulo
                current_positions.append(position_counter % max_seq_length)
                current_positions = current_positions[-n_gram_size:]
    
    print("\nGeneration complete.")
    return ' '.join(generated_text)

def save_model(model, word_to_index, index_to_word, n_gram_size, max_seq_length, name="model"):
    model_data = {
        'model_state': model.state_dict(),
        'word_to_index': word_to_index,
        'index_to_word': index_to_word,
        'n_gram_size': n_gram_size,
        'max_seq_length': max_seq_length
    }
    torch.save(model_data, f"{name}.pt")
    print(f"Model saved as {name}.pt")

def load_model(name="model"):
    model_data = torch.load(f"{name}.pt")
    word_to_index = model_data['word_to_index']
    index_to_word = model_data['index_to_word']
    n_gram_size = model_data['n_gram_size']
    max_seq_length = model_data.get('max_seq_length', 10000)  # Default value for compatibility
    
    vocab_size = len(word_to_index) + 1
    EMBEDDING_DIM = 256
    HIDDEN_DIM = 128
    
    model = TextGenerator(vocab_size, EMBEDDING_DIM, HIDDEN_DIM, n_gram_size, max_seq_length)
    model.load_state_dict(model_data['model_state'])
    
    print(f"Model {name}.pt loaded successfully")
    return model, word_to_index, index_to_word, n_gram_size, max_seq_length

# Main function
def main():
    print("Position-Enhanced Text Generator")
    
    while True:
        print("\nMenu:")
        print("1. Train new model")
        print("2. Generate text")
        print("3. Save model")
        print("4. Load model")
        print("5. Exit")
        
        choice = input("Choose an option (1-5): ")
        
        if choice == "1":
            file_name = input("Enter text file name (default: test.txt): ") or "test.txt"

            model, word_to_index, index_to_word, max_seq_length = train_model(file_name)
            print("Model trained successfully")
           
        elif choice == "2":
            if 'model' not in locals():
                print("No model available. Please train or load a model first.")
                continue
            while True:    
                seed = input("Enter seed text: ")
                length = int(input("Enter length to generate (default: 250): ") or "250")
                temperature = float(input("Enter temperature (default: 0.7): ") or "0.7")
                generate_text(seed, length, model, word_to_index, index_to_word, 3, max_seq_length, temperature)
            
        elif choice == "3":
            if 'model' not in locals():
                print("No model available to save")
                continue
                
            name = input("Enter model name (default: model): ") or "model"
            save_model(model, word_to_index, index_to_word, 3, max_seq_length, name)
            
        elif choice == "4":
            name = input("Enter model name to load (default: model): ") or "model"
            model, word_to_index, index_to_word, N_GRAM_SIZE, max_seq_length = load_model(name)
            
        elif choice == "5":
            print("Goodbye!")
            break
            
        else:
            print("Invalid option. Please try again.")

if __name__ == "__main__":
    main()
