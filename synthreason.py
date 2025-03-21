import re
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import random

class TextDataset(Dataset):
    def __init__(self, X, y):
        self.X = X
        self.y = y

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

class TextGenerator(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim, n_gram_size):
        super(TextGenerator, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, batch_first=True)
        self.fc = nn.Linear(hidden_dim, vocab_size)
        
        # Add support for Hamming weight input
        self.hamming_fc = nn.Linear(1, embedding_dim)

    def forward(self, x, hamming_weights=None):
        if hamming_weights is not None:
            # Process hamming weights - reshape to have proper dimensions
            hamming_input = hamming_weights.view(-1, 1).float()
            x = self.hamming_fc(hamming_input)
            x = x.unsqueeze(1)  # Add sequence dimension
        else:
            # Normal embedding lookup
            x = self.embedding(x)
            
        x, _ = self.lstm(x)
        x = self.fc(x[:, -1, :])
        return x

def custom_loss_function(outputs, labels, model, word_occurrences, word_to_index, index_to_word, words):
    criterion = nn.CrossEntropyLoss()
    cross_entropy_loss = criterion(outputs, labels)
    
    # Custom gradient calculation
    indices = list(index_to_word.keys())
    gradient = {}
    for i, idx in enumerate(indices):
        exp_word_occurrence = np.sqrt(word_occurrences[idx])
        if i < len(indices) - 1:
            next_idx = indices[i + 1]
            next_exp_word_occurrence = np.sqrt(word_occurrences[next_idx])
            gradient[idx] = next_idx if exp_word_occurrence == word_occurrences[next_idx] else 0

    # Apply custom gradients to model parameters
    for param in model.parameters():
        if param.grad is not None:
            custom_grad = param.grad.clone()
            custom_grad *= torch.tensor([gradient.get(idx, 0) for idx in range(len(custom_grad))], dtype=torch.float32, device=custom_grad.device)
            param.grad = custom_grad

    return cross_entropy_loss

def train_model(text_file='test.txt', limit=6500):
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
    
    # Create training sequences
    sequences = []
    for i in range(len(words) - N_GRAM_SIZE):
        seq = words[i:i + N_GRAM_SIZE + 1]
        sequences.append([word_to_index[word] for word in seq])
    
    # Split into input and target
    X = [seq[:-1] for seq in sequences]
    y = [seq[-1] for seq in sequences]
    
    # Convert to PyTorch tensors
    X = torch.tensor(X, dtype=torch.long)
    y = torch.tensor(y, dtype=torch.long)
    
    # Create dataset and dataloader
    dataset = TextDataset(X, y)
    dataloader = DataLoader(dataset, batch_size=64, shuffle=True)
    
    # Initialize model
    model = TextGenerator(vocab_size, EMBEDDING_DIM, HIDDEN_DIM, N_GRAM_SIZE)
    
    # Training
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    for epoch in range(EPOCHS):
        model.train()
        total_loss = 0
        for inputs, labels in dataloader:
            optimizer.zero_grad()
            outputs = model(inputs)
            word_occurrences = {word_to_index[word]: words.count(word) for word in unique_words}

            # Compute custom loss
            loss = custom_loss_function(outputs, labels, model, word_occurrences, word_to_index, index_to_word, words)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        
        print(f'Epoch {epoch+1}/{EPOCHS}, Loss: {total_loss/len(dataloader):.4f}')
    
    return model, word_to_index, index_to_word

def generate_text(seed_text, generate_length, model, word_to_index, index_to_word, n_gram_size, temperature=1.0):
    model.eval()
    generated_text = []
    
    words = seed_text.lower().split()
    current_sequence = [word_to_index[word] if word in word_to_index else random.choice(list(word_to_index.values())) for word in words]
    
    while len(current_sequence) < n_gram_size:
        current_sequence.append(random.choice(list(word_to_index.values())))
    current_sequence = current_sequence[-n_gram_size:]
    
    print(f"\nSeed text: {seed_text}")
    print("\nGenerated text: ", end="", flush=True)
    
    for _ in range(generate_length):
        input_seq = torch.tensor([current_sequence], dtype=torch.long)
        with torch.no_grad():
            # Get output from model
            output = model(input_seq)
            
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
    
    print("\nGeneration complete.")
    return ' '.join(generated_text)

def save_model(model, word_to_index, index_to_word, n_gram_size, name="model"):
    model_data = {
        'model_state': model.state_dict(),
        'word_to_index': word_to_index,
        'index_to_word': index_to_word,
        'n_gram_size': n_gram_size
    }
    torch.save(model_data, f"{name}.pt")
    print(f"Model saved as {name}.pt")

def load_model(name="model"):
    model_data = torch.load(f"{name}.pt")
    word_to_index = model_data['word_to_index']
    index_to_word = model_data['index_to_word']
    n_gram_size = model_data['n_gram_size']
    
    vocab_size = len(word_to_index) + 1
    EMBEDDING_DIM = 256
    HIDDEN_DIM = 128
    
    model = TextGenerator(vocab_size, EMBEDDING_DIM, HIDDEN_DIM, n_gram_size)
    model.load_state_dict(model_data['model_state'])
    
    print(f"Model {name}.pt loaded successfully")
    return model, word_to_index, index_to_word, n_gram_size

# Main function
def main():
    print("Simple Text Generator")
    
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

            model, word_to_index, index_to_word = train_model(file_name)
            print("Model trained successfully")
           
        elif choice == "2":
            if 'model' not in locals():
                print("No model available. Please train or load a model first.")
                continue
            while True:    
                seed = input("Enter seed text: ")
                length = 250
                temperature = 0.7
                generate_text(seed, length, model, word_to_index, index_to_word, 3, temperature)
            
        elif choice == "3":
            if 'model' not in locals():
                print("No model available to save")
                continue
                
            name = input("Enter model name (default: model): ") or "model"
            save_model(model, word_to_index, index_to_word, 3, name)
            
        elif choice == "4":
            name = input("Enter model name to load (default: model): ") or "model"
            model, word_to_index, index_to_word, N_GRAM_SIZE = load_model(name)
            
        elif choice == "5":
            print("Goodbye!")
            break
            
        else:
            print("Invalid option. Please try again.")

if __name__ == "__main__":
    main()
