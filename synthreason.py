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
    word_to_index = {word: idx + 1 for idx, word in enumerate(set(filtered_words))}
    index_to_word = {idx + 1: word for idx, word in enumerate(set(filtered_words))}
    sequences = []
    for i in range(len(filtered_words) - n_gram_size-5):
        sequences.append(filtered_words[i:i + n_gram_size + 1]+filtered_words[i+3:i])
    sequences_idx = [[word_to_index[word] for word in sequence] for sequence in sequences]
    X = [seq[:-1] for seq in sequences_idx]  # Input: all but last word
    y = [seq[-1] for seq in sequences_idx]  # Output: last word
    X = torch.tensor(X)
    y = torch.tensor(y)
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
    
    for _ in range(generate_length):
        input_seq = torch.tensor([current_sequence], dtype=torch.long)
        
        with torch.no_grad():
            output = model(input_seq)
        
        probabilities = torch.nn.functional.softmax(output, dim=1)
        
        predicted_word_idx = torch.multinomial(probabilities[0], 1).item()
        
        if predicted_word_idx in index_to_word:
            predicted_word = index_to_word[predicted_word_idx]
            generated_text.append(predicted_word)
            
            current_sequence.append(predicted_word_idx)
            current_sequence = current_sequence[-n_gram_size:]
        else:
            continue
    
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
        print("2. Save model")
        print("3. Load model")
        print("4. Train new model")
        print("5. List saved models")
        print("6. Exit")
        
        choice = input("Enter your choice (1-6): ")
        
        if choice == "1":
            while True:
                seed_text = input("Enter seed text: ")
                generated_text = generate_text(seed_text, generate_length, model, word_to_index, index_to_word, current_n_gram_size)
                print("\nGenerated Text:")
                print(generated_text)
            
        elif choice == "2":
            filename = input("Enter filename to save model (default: text_generator): ")
            if not filename:
                filename = "text_generator"
            save_model(model, word_to_index, index_to_word, current_n_gram_size, current_embedding_dim, current_hidden_dim, filename)
            
        elif choice == "3":
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
            
        elif choice == "4":
            confirm = input("This will overwrite the current model. Continue? (y/n): ")
            if confirm.lower() == 'y':
                model, word_to_index, index_to_word = train_new_model()
                current_n_gram_size = n_gram_size
                current_embedding_dim = embedding_dim
                current_hidden_dim = hidden_dim
                print("New model trained successfully.")
            
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
