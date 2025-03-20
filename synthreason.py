import numpy as np
import re
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from collections import defaultdict
import random
import os
import json
import pickle
import time

KB_limit = 999 # -1 for unlimited
epochs = 10
generate_length = 500
n_gram_size = 2  # n-gram size (for bigrams)
embedding_dim = 256
hidden_dim = 512
import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class CustomLSTMCell(nn.Module):
    """
    Custom implementation of a single LSTM cell
    """
    def __init__(self, input_size, hidden_size, bias=True):
        super(CustomLSTMCell, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        
        # Combined gates parameters (input, forget, cell, output)
        self.weight_ih = nn.Parameter(torch.Tensor(4 * hidden_size, input_size))
        self.weight_hh = nn.Parameter(torch.Tensor(4 * hidden_size, hidden_size))
        
        if bias:
            self.bias_ih = nn.Parameter(torch.Tensor(4 * hidden_size))
            self.bias_hh = nn.Parameter(torch.Tensor(4 * hidden_size))
        else:
            self.register_parameter('bias_ih', None)
            self.register_parameter('bias_hh', None)
            
        self.reset_parameters()
        
    def reset_parameters(self):
        """Initialize parameters using Xavier uniform initialization"""
        stdv = 1.0 / math.sqrt(self.hidden_size)
        for weight in self.parameters():
            weight.data.uniform_(-stdv, stdv)
            
    def forward(self, x, hidden):
        """
        Forward pass for a single LSTM cell
        
        Args:
            x: Input tensor (batch_size, input_size)
            hidden: Tuple of (h_0, c_0) where both are (batch_size, hidden_size)
            
        Returns:
            h_1, c_1: Updated hidden and cell states
        """
        h_0, c_0 = hidden
        
        # Calculate gates - uses batch matrix multiplication
        gates = F.linear(x, self.weight_ih, self.bias_ih) + F.linear(h_0, self.weight_hh, self.bias_hh)
        
        # Split into different gates
        i, f, g, o = gates.chunk(4, 1)
        
        # Apply activations
        i = torch.sigmoid(i)  # input gate
        f = torch.sigmoid(f)  # forget gate
        g = torch.tanh(g)     # cell gate
        o = torch.sigmoid(o)  # output gate
        
        # Update cell state
        c_1 = f * c_0 + i * g
        
        # Update hidden state
        h_1 = o * torch.tanh(c_1)
        
        return h_1, c_1


class CustomLSTM(nn.Module):
    """
    Custom LSTM layer that processes sequences using CustomLSTMCell
    """
    def __init__(self, input_size, hidden_size, num_layers=1, bias=True, batch_first=False, dropout=0.0):
        super(CustomLSTM, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.bias = bias
        self.batch_first = batch_first
        self.dropout = dropout
        
        # Create a list of LSTM cells for each layer
        self.cells = nn.ModuleList()
        for layer in range(num_layers):
            layer_input_size = input_size if layer == 0 else hidden_size
            cell = CustomLSTMCell(layer_input_size, hidden_size, bias)
            self.cells.append(cell)
            
        self.dropout_layer = nn.Dropout(dropout) if dropout > 0 else None
        
    def forward(self, x, hidden=None):
        """
        Forward pass for the entire LSTM
        
        Args:
            x: Input sequence (seq_len, batch_size, input_size) or 
               (batch_size, seq_len, input_size) if batch_first=True
            hidden: Initial hidden states (num_layers, batch_size, hidden_size) for h and c
            
        Returns:
            output: Sequence of hidden states for each time step
            (h_n, c_n): Final hidden states for each layer
        """
        if self.batch_first:
            # Convert to (seq_len, batch_size, input_size) format
            x = x.transpose(0, 1)
            
        seq_len, batch_size, _ = x.size()
        
        # Initialize hidden states if not provided
        if hidden is None:
            h_zeros = torch.zeros(self.num_layers, batch_size, self.hidden_size, 
                                 device=x.device, dtype=x.dtype)
            c_zeros = torch.zeros(self.num_layers, batch_size, self.hidden_size, 
                                 device=x.device, dtype=x.dtype)
            hidden = (h_zeros, c_zeros)
            
        h_0, c_0 = hidden
        h_n, c_n = [], []
        
        # List to store output at each time step (for return)
        outputs = []
        
        # Process one time step at a time
        for t in range(seq_len):
            x_t = x[t]  # (batch_size, input_size)
            
            # Process through each layer
            for layer in range(self.num_layers):
                # If first time step, use provided hidden state
                if t == 0:
                    h_prev, c_prev = h_0[layer], c_0[layer]
                else:
                    h_prev, c_prev = h_n[layer], c_n[layer]
                    
                # Forward through the cell
                h_t, c_t = self.cells[layer](x_t, (h_prev, c_prev))
                
                # Store new hidden state
                if t == 0:
                    h_n.append(h_t)
                    c_n.append(c_t)
                else:
                    h_n[layer] = h_t
                    c_n[layer] = c_t
                    
                # Update input for next layer
                x_t = h_t
                
                # Apply dropout between layers (but not on the output of the last layer)
                if self.dropout_layer is not None and layer < self.num_layers - 1:
                    x_t = self.dropout_layer(x_t)
                    
            # Store the output from the top layer
            outputs.append(h_t)
            
        # Stack outputs into a tensor
        outputs = torch.stack(outputs)  # (seq_len, batch_size, hidden_size)
        
        # Convert back to batch_first format if needed
        if self.batch_first:
            outputs = outputs.transpose(0, 1)  # (batch_size, seq_len, hidden_size)
            
        # Convert lists to tensors for h_n and c_n
        h_n = torch.stack(h_n)  # (num_layers, batch_size, hidden_size)
        c_n = torch.stack(c_n)  # (num_layers, batch_size, hidden_size)
        
        return outputs, (h_n, c_n)


class RemorphicLSTM(nn.Module):
    """
    Advanced LSTM with remorphic capabilities for enhanced discretion and
    adaptability in sequence modeling.
    """
    def __init__(self, input_size, hidden_size, num_layers=1, batch_first=False, remorphic_factor=0.3):
        super(RemorphicLSTM, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.batch_first = batch_first
        self.remorphic_factor = remorphic_factor
        
        # Base LSTM implementation
        self.lstm_base = CustomLSTM(input_size, hidden_size, num_layers, batch_first=batch_first)
        
        # Remorphic extension components
        self.transform_gate = nn.Linear(hidden_size, hidden_size)
        self.context_modulator = nn.Linear(hidden_size, hidden_size)
        self.output_mixer = nn.Linear(hidden_size * 2, hidden_size)
        
    def forward(self, x, hidden=None, apply_remorphic=True):
        """
        Forward pass with optional remorphic transformation
        
        Args:
            x: Input sequence
            hidden: Initial hidden states
            apply_remorphic: Whether to apply remorphic transformation
            
        Returns:
            output: Transformed sequence
            (h_n, c_n): Final hidden states
        """
        # Process through base LSTM
        base_output, (h_n, c_n) = self.lstm_base(x, hidden)
        
        if not apply_remorphic:
            return base_output, (h_n, c_n)
        
        # Apply remorphic transformation
        if self.batch_first:
            # Get sequence length dimension
            seq_dim = 1
        else:
            seq_dim = 0
            
        # Calculate transformation gate values
        transform_values = self.transform_gate(torch.sigmoid(base_output))
        
        # Create context-aware modulation
        context_mod = self.context_modulator(base_output)
        context_mod = torch.exp(context_mod-base_output*transform_values)
        
        # Mix original and transformed representations
        blended_output = transform_values * base_output + (1 - transform_values) * context_mod
        
        # Apply final mixer with residual connection
        concat_output = torch.cat([base_output, blended_output], dim=-1)
        remorphic_output = self.output_mixer(concat_output) + base_output
        
        return remorphic_output, (h_n, c_n)


# Example usage in TextGenerator class
class TextGenerator(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim, n_gram_size):
        super(TextGenerator, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        
        # Replace standard LSTM with custom implementation
        self.lstm = RemorphicLSTM(embedding_dim, hidden_dim, batch_first=True)
        
        # Linear transformation layers
        self.linear_include = nn.Linear(hidden_dim, hidden_dim)
        self.fc = nn.Linear(hidden_dim, vocab_size)
    
    def forward(self, x):
        batch_size = x.size(0)
        
        # Get embeddings
        embedded = self.embedding(x)  # Shape: (batch_size, n_gram_size, embedding_dim)
        
        # Process with custom LSTM
        lstm_out, _ = self.lstm(embedded)  # Shape: (batch_size, n_gram_size, hidden_dim)
        
        # Take the last output: (batch_size, hidden_dim)
        lstm_out = lstm_out[:, -1, :]
        
        # Apply Linear Include to LSTM output
        lstm_out = self.linear_include(lstm_out)  # Shape: (batch_size, hidden_dim)
        
        # Final prediction
        output = self.fc(lstm_out)  # Shape: (batch_size, vocab_size)
        return output

class TextDataset(Dataset):
    def __init__(self, X, y):
        self.X = X
        self.y = y

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

def generate_text(seed_text, generate_length, model, word_to_index, index_to_word, n_gram_size):
    """
    Generate text using zeta zeros transformation.
    """
    model.eval()
    generated_text = []
    
    words = seed_text.lower().split()
    current_sequence = []
    for word in words:
        if word in word_to_index:
            current_sequence.append(word_to_index[word])
        else:
            current_sequence.append(random.choice(list(word_to_index.values())))
    
    while len(current_sequence) < n_gram_size:
        current_sequence.append(random.choice(list(word_to_index.values())))
    
    current_sequence = current_sequence[-n_gram_size:]
    
    # Print the seed text first
    print(f"\nSeed text: {seed_text}")
    print("\nGenerated text: ", end="", flush=True)
    
    for _ in range(generate_length):
                
        # Use the transformed sequence for prediction
        input_seq = torch.tensor([transformed_sequence], dtype=torch.long)
        
        with torch.no_grad():
            output = model(input_seq)
        
        probabilities = torch.nn.functional.softmax(output, dim=1)
        
        # Sample from the distribution
        predicted_word_idx = torch.multinomial(probabilities[0], 1).item()
        
        if predicted_word_idx in index_to_word:
            predicted_word = index_to_word[predicted_word_idx]
            generated_text.append(predicted_word)
            
            # Print the word immediately and flush the output
            print(predicted_word + " ", end="", flush=True)
            

            
            # Update the current sequence
            current_sequence.append(predicted_word_idx)
            current_sequence = current_sequence[-n_gram_size:]
        else:
            continue
    
    print("\nGeneration complete.")
    return ' '.join(generated_text)

# Function to save model and variables
def save_model(model, word_to_index, index_to_word, n_gram_size, embedding_dim, hidden_dim, filename="text_generator"):
    # Create directory if it doesn't exist
    os.makedirs("saved_models", exist_ok=True)
    
    # Save model state
    torch.save(model.state_dict(), f"saved_models/{filename}_model.pth")
    
    # Save vocabulary and parameters
    model_data = {
        "word_to_index": word_to_index,
        "index_to_word": {int(k): v for k, v in index_to_word.items()},  # Convert keys to int for JSON
        "n_gram_size": n_gram_size,
        "embedding_dim": embedding_dim,
        "hidden_dim": hidden_dim,
        "vocab_size": len(word_to_index) + 1
    }
    
    # Save vocabulary and parameters using pickle
    with open(f"saved_models/{filename}_data.pkl", "wb") as f:
        pickle.dump(model_data, f)
    
    print(f"Model and data saved to saved_models/{filename}_*")

# Function to load model and variables
def load_model(filename="text_generator"):
    try:
        # Load vocabulary and parameters
        with open(f"saved_models/{filename}_data.pkl", "rb") as f:
            model_data = pickle.load(f)
        
        # Extract data
        word_to_index = model_data["word_to_index"]
        index_to_word = model_data["index_to_word"]
        n_gram_size = model_data["n_gram_size"]
        embedding_dim = model_data["embedding_dim"]
        hidden_dim = model_data["hidden_dim"]
        vocab_size = model_data["vocab_size"]
        
        # Create model with loaded parameters
        model = TextGenerator(vocab_size, embedding_dim, hidden_dim, n_gram_size)
        
        # Load model state
        model.load_state_dict(torch.load(f"saved_models/{filename}_model.pth"))
        
        print(f"Model loaded from saved_models/{filename}_*")
        return model, word_to_index, index_to_word, n_gram_size, embedding_dim, hidden_dim
    
    except FileNotFoundError:
        print(f"No saved model found at saved_models/{filename}_*")
        return None, None, None, None, None, None

def train_new_model():
    with open('test.txt', 'r', encoding="utf-8") as file:
        text = ' '.join(file.read().split()[:KB_limit])
    words = re.sub(r'\d+', '', text).split()
    filtered_words = [word for word in words if len(word) > 1 or word in ['a', 'i'] and not word.isdigit()]
    
    # Create vocabulary
    unique_words = list(set(filtered_words))
    word_to_index = {word: idx + 1 for idx, word in enumerate(unique_words)}
    index_to_word = {idx + 1: word for idx, word in enumerate(unique_words)}
    
    # Define parameters for sequence generation
    vocab_size = len(word_to_index)
    max_sequence_length = n_gram_size + 1  # Length of input + output
    max_combinations = 100  # Limit to avoid memory explosion
    
    print(f"Vocabulary size: {vocab_size}")
    print(f"Generating combinations with sequence length {max_sequence_length}...")
    
    sequences_idx = []
    
    # Generate combinations using indices directly to save memory
    import itertools
    
    # Generate all possible combinations of word indices
    print("Generating combinations...")
    word_indices = list(range(1, vocab_size + 1))
    
    # If vocab is large, sample a subset to make combinations manageable
    # If vocab is large, use binomial distribution to determine sampling
    if vocab_size > 20:
        import numpy as np
        
        # Set sample size
        sample_size = min(20, vocab_size)
        
        # Use binomial distribution to get probabilities for each word
        # p=0.5 gives equal probability, n=1 for each trial
        probabilities = np.random.binomial(n=1, p=0.5, size=vocab_size)
        
        # Select words based on binomial outcomes
        word_indices_sample = [word_indices[i] for i in range(len(word_indices)) 
                              if probabilities[i] == 1]
        
        # If we didn't get enough samples, supplement with random ones
        if len(word_indices_sample) < sample_size:
            remaining_indices = [idx for idx in word_indices if idx not in word_indices_sample]
            additional_samples = random.sample(remaining_indices, 
                                             sample_size - len(word_indices_sample))
            word_indices_sample.extend(additional_samples)
        
        print(f"Vocabulary too large. Sampling {sample_size} words using binomial distribution.")
        combinations = list(itertools.product(word_indices_sample, repeat=max_sequence_length))
    else:
        combinations = list(itertools.product(word_indices, repeat=max_sequence_length))
    
    # Limit the number of combinations to avoid memory issues
    combinations = combinations[:max_combinations]
    
    # Add actual word sequences from the text
    for i in range(len(filtered_words) - n_gram_size):
        # Get n_gram_size + 1 consecutive words
        word_sequence = filtered_words[i:i + n_gram_size + 1]
        
        # Convert words to their indices
        try:
            index_sequence = [word_to_index[word] for word in word_sequence]
            combinations.append(index_sequence)
        except KeyError:
            # Skip if any word is not in our vocabulary
            continue
    
    # Convert to format needed for training
    sequences_idx = [list(combo) for combo in combinations]
    

    # Split into input (X) and target (y)
    X = [seq[:-1] for seq in sequences_idx]  # Input: all but last word
    y = [seq[-1] for seq in sequences_idx]   # Output: last word
    
    print(f"Training with {len(X)} input-output pairs...")
    
    # Convert to PyTorch tensors
    X = torch.tensor(X, dtype=torch.long)
    y = torch.tensor(y, dtype=torch.long)
    
    dataset = TextDataset(X, y)
    dataloader = DataLoader(dataset, batch_size=64, shuffle=True)

    vocab_size = len(word_to_index) + 1 
    model = TextGenerator(vocab_size, embedding_dim, hidden_dim, n_gram_size)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        for inputs, labels in dataloader:
            optimizer.zero_grad()
            
            outputs = model(inputs)
            
            loss = criterion(outputs, labels)
            loss.backward()
            
            optimizer.step()
            
            running_loss += loss.item()
        
        print(f'Epoch [{epoch+1}/{epochs}], Loss: {running_loss / len(dataloader)}')
    
    return model, word_to_index, index_to_word

def generate_text(seed_text, generate_length, model, word_to_index, index_to_word, n_gram_size):
    """
    Generate text without any transformation or serial control.
    """
    model.eval()
    generated_text = []
    
    words = seed_text.lower().split()
    current_sequence = []
    for word in words:
        if word in word_to_index:
            current_sequence.append(word_to_index[word])
        else:
            current_sequence.append(random.choice(list(word_to_index.values())))
    
    while len(current_sequence) < n_gram_size:
        current_sequence.append(random.choice(list(word_to_index.values())))
    
    current_sequence = current_sequence[-n_gram_size:]
    
    # Print the seed text first
    print(f"\nSeed text: {seed_text}")
    print("\nGenerated text: ", end="", flush=True)
    
    for _ in range(generate_length):
        input_seq = torch.tensor([current_sequence], dtype=torch.long)
        
        with torch.no_grad():
            output = model(input_seq)
        
        probabilities = torch.nn.functional.softmax(output, dim=1)
        
        # Sample from the distribution
        predicted_word_idx = torch.multinomial(probabilities[0], 1).item()
        
        if predicted_word_idx in index_to_word:
            predicted_word = index_to_word[predicted_word_idx]
            generated_text.append(predicted_word)
            
            # Print the word immediately and flush the output
            print(predicted_word + " ", end="", flush=True)
            
            # Update the current sequence
            current_sequence.append(predicted_word_idx)
            current_sequence = current_sequence[-n_gram_size:]
        else:
            continue
    
    print("\nGeneration complete.")
    return ' '.join(generated_text)

def list_saved_models():
    if not os.path.exists("saved_models"):
        print("No saved models directory found.")
        return []
    
    models = []
    for file in os.listdir("saved_models"):
        if file.endswith("_model.pth"):
            models.append(file.replace("_model.pth", ""))
    
    return models

def main():
    # Try to load existing model
    model, word_to_index, index_to_word, loaded_n_gram_size, loaded_embedding_dim, loaded_hidden_dim = load_model()
    
    # If no model loaded, train a new one
    if model is None:
        print("No saved model found. Training a new model...")
        model, word_to_index, index_to_word = train_new_model()
        # Use global variables for n_gram_size, embedding_dim, hidden_dim
        current_n_gram_size = n_gram_size
        current_embedding_dim = embedding_dim
        current_hidden_dim = hidden_dim
    else:
        # Use loaded parameters
        current_n_gram_size = loaded_n_gram_size
        current_embedding_dim = loaded_embedding_dim
        current_hidden_dim = loaded_hidden_dim
    
    while True:
        print("\nText Generation Interface")
        print("1. Generate text")
        print("2. Train new model")
        print("3. Save model")
        print("4. Load model")
        print("5. List saved models")
        print("6. Exit")
        
        choice = input("Enter your choice (1-6): ")
        
        if choice == "1":
            while True:
                seed_text = input("Enter seed text: ")
                generate_length = 500
                generated_text = generate_text(seed_text, generate_length, model, 
                                               word_to_index, index_to_word, 
                                               current_n_gram_size)
        elif choice == "2":
            confirm = input("This will train a new model. Continue? (y/n): ")
            if confirm.lower() == 'y':
                model, word_to_index, index_to_word = train_new_model()
                current_n_gram_size = n_gram_size
                current_embedding_dim = embedding_dim
                current_hidden_dim = hidden_dim
                print("New model trained successfully.")
        elif choice == "3":
            filename = input("Enter filename to save model (default: text_generator): ")
            if not filename:
                filename = "text_generator"
            save_model(model, word_to_index, index_to_word, current_n_gram_size, 
                       current_embedding_dim, current_hidden_dim, filename)
        elif choice == "4":
            available_models = list_saved_models()
            if available_models:
                print("\nAvailable models:")
                for i, model_name in enumerate(available_models):
                    print(f"{i+1}. {model_name}")
                
                model_idx = input("Enter model number to load (or press Enter for default): ")
                if model_idx and model_idx.isdigit() and 1 <= int(model_idx) <= len(available_models):
                    filename = available_models[int(model_idx) - 1]
                else:
                    filename = "text_generator"
                
                model, word_to_index, index_to_word, current_n_gram_size, current_embedding_dim, current_hidden_dim = load_model(filename)
                if model is None:
                    print("Failed to load model. Using current model.")
            else:
                print("No saved models found.")
        elif choice == "5":
            available_models = list_saved_models()
            if available_models:
                print("\nAvailable models:")
                for model_name in available_models:
                    print(f"- {model_name}")
            else:
                print("No saved models found.")
        elif choice == "6":
            print("Exiting...")
            break
        else:
            print("Invalid choice. Please try again.")

if __name__ == "__main__":
    main()
