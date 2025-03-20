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
        
        # Linear Include - applies on LSTM output, not embedding directly
        self.linear_include = nn.Linear(hidden_dim, hidden_dim)
        
        self.fc = nn.Linear(hidden_dim, vocab_size)
    
    def forward(self, x):
        batch_size = x.size(0)
        
        # Get embeddings
        embedded = self.embedding(x)  # Shape: (batch_size, n_gram_size, embedding_dim)
        
        # Process with LSTM
        lstm_out, _ = self.lstm(embedded)  # Shape: (batch_size, n_gram_size, hidden_dim)
        
        # Take the last output: (batch_size, hidden_dim)
        lstm_out = lstm_out[:, -1, :]
        
        # Apply Linear Include to LSTM output
        lstm_out = self.linear_include(lstm_out)  # Shape: (batch_size, hidden_dim)
        
        # Final prediction
        output = self.fc(lstm_out)  # Shape: (batch_size, vocab_size)
        return output

def generate_text(seed_text, generate_length, model, word_to_index, index_to_word, n_gram_size):
    """
    Generate text based on seed text.
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
    text = re.sub(r'\d+', '', text)
    words = re.findall(r'\b\w+\b', text.lower())
    filtered_words = [word for word in words if len(word) > 1 or word in ['a', 'i'] and not word.isdigit()]
    
    # Create vocabulary
    unique_words = list(set(filtered_words))
    word_to_index = {word: idx + 1 for idx, word in enumerate(unique_words)}
    index_to_word = {idx + 1: word for idx, word in enumerate(unique_words)}
    
    # Define parameters for sequence generation
    vocab_size = len(word_to_index)
    
    print(f"Vocabulary size: {vocab_size}")
    print(f"Creating sequences with n-gram size {n_gram_size}...")
    
    sequences_idx = []
    
    # Add actual word sequences from the text
    for i in range(len(filtered_words) - n_gram_size):
        # Get n_gram_size + 1 consecutive words
        word_sequence = filtered_words[i:i + n_gram_size + 1]
        
        # Convert words to their indices
        try:
            index_sequence = [word_to_index[word] for word in word_sequence]
            sequences_idx.append(index_sequence)
        except KeyError:
            # Skip if any word is not in our vocabulary
            continue
    
    print(f"Created {len(sequences_idx)} sequences from the text.")
    
    # Calculate distance to end of vocabulary for each word in each sequence
    modified_sequences = []
    for seq in sequences_idx:
        modified_seq = []
        for word_idx in seq:
            the_index = word_to_index['the']
            # Calculate distance from current index to the end of vocab
            distance_to_end = vocab_size - the_index
            # Store the original index and the distance as a tuple
            modified_seq.append((word_idx, distance_to_end))
        modified_sequences.append(modified_seq)
    
    # Add the distance information to the dataset (for reference during training)
    distance_info = {}
    for i, seq in enumerate(modified_sequences):
        distance_info[i] = [word_tuple[1] for word_tuple in seq]
    
    # Split into input (X) and target (y)
    X = [seq[:-1] for seq in sequences_idx]  # Input: all but last word
    y = [seq[-1] for seq in sequences_idx]   # Output: last word
    
    print(f"Training with {len(X)} input-output pairs...")
    
    # Converting to PyTorch tensors
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
    
    # Store the distance information in the model metadata
    model.distance_info = distance_info
    
    return model, word_to_index, index_to_word

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
        print("5. Train new model")
        print("6. List saved models")
        print("7. Exit")
        
        choice = input("Enter your choice (1-7): ")
        
        if choice == "1":
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
            confirm = input("This will overwrite the current model. Continue? (y/n): ")
            if confirm.lower() == 'y':
                model, word_to_index, index_to_word = train_new_model()
                current_n_gram_size = n_gram_size
                current_embedding_dim = embedding_dim
                current_hidden_dim = hidden_dim
                print("New model trained successfully.")
            
        elif choice == "6":
            available_models = list_saved_models()
            if available_models:
                print("\nAvailable models:")
                for model_name in available_models:
                    print(f"- {model_name}")
            else:
                print("No saved models found.")
                
        elif choice == "7":
            print("Exiting...")
            break
            
        else:
            print("Invalid choice. Please try again.")

if __name__ == "__main__":
    main()
