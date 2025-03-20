import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import random
import os
import pickle
import re
import itertools

# Configurations
KB_limit = 999  # -1 for unlimited
epochs = 10
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
        self.fc = nn.Linear(hidden_dim, vocab_size)

    def forward(self, x):
        embedded = self.embedding(x)
        lstm_out, _ = self.lstm(embedded)
        lstm_out = lstm_out[:, -1, :]
        output = self.fc(lstm_out)
        return output

def generate_text(seed_text, generate_length, model, word_to_index, index_to_word, n_gram_size):
    model.eval()
    generated_text = []
    words = seed_text.lower().split()
    current_sequence = [word_to_index.get(word, random.choice(list(word_to_index.values()))) for word in words]
    current_sequence = (current_sequence + [random.choice(list(word_to_index.values()))] * n_gram_size)[-n_gram_size:]
    print(f"\nSeed text: {seed_text}\nGenerated text: ", end="", flush=True)
    for _ in range(generate_length):
        input_seq = torch.tensor([current_sequence], dtype=torch.long)
        with torch.no_grad():
            output = model(input_seq)
        probabilities = torch.nn.functional.softmax(output, dim=1)
        predicted_word_idx = torch.multinomial(probabilities[0], 1).item()
        predicted_word = index_to_word.get(predicted_word_idx, "")
        generated_text.append(predicted_word)
        print(predicted_word + " ", end="", flush=True)
        current_sequence.append(predicted_word_idx)
        current_sequence = current_sequence[-n_gram_size:]
    print("\nGeneration complete.")
    return ' '.join(generated_text)

def save_model(model, word_to_index, index_to_word, filename="text_generator"):
    os.makedirs("saved_models", exist_ok=True)
    torch.save(model.state_dict(), f"saved_models/{filename}_model.pth")
    with open(f"saved_models/{filename}_data.pkl", "wb") as f:
        pickle.dump({"word_to_index": word_to_index, "index_to_word": index_to_word}, f)
    print(f"Model and data saved to saved_models/{filename}_*")

def load_model(filename="text_generator"):
    try:
        with open(f"saved_models/{filename}_data.pkl", "rb") as f:
            data = pickle.load(f)
        model = TextGenerator(len(data["word_to_index"]) + 1, embedding_dim, hidden_dim, n_gram_size)
        model.load_state_dict(torch.load(f"saved_models/{filename}_model.pth"))
        print(f"Model loaded from saved_models/{filename}_*")
        return model, data["word_to_index"], data["index_to_word"]
    except FileNotFoundError:
        print(f"No saved model found at saved_models/{filename}_*")
        return None, None, None

def train_new_model():
    with open('test.txt', 'r', encoding="utf-8") as file:
        text = ' '.join(file.read().split()[:KB_limit])
    words = re.sub(r'\d+', '', text).split()
    words = [word for word in words if len(word) > 1 or word in ['a', 'i']]
    unique_words = list(set(words))
    word_to_index = {word: idx + 1 for idx, word in enumerate(unique_words)}
    index_to_word = {idx + 1: word for idx, word in enumerate(unique_words)}
    
    sequences_idx = []
    for i in range(len(words) - n_gram_size):
        seq = words[i:i + n_gram_size + 1]
        try:
            sequences_idx.append([word_to_index[word] for word in seq])
        except KeyError:
            continue
    
    random.shuffle(sequences_idx)
    
    X = torch.tensor([seq[:-1] for seq in sequences_idx], dtype=torch.long)
    y = torch.tensor([seq[-1] for seq in sequences_idx], dtype=torch.long)
    dataset = TextDataset(X, y)
    dataloader = DataLoader(dataset, batch_size=64, shuffle=True)
    
    model = TextGenerator(len(word_to_index) + 1, embedding_dim, hidden_dim, n_gram_size)
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

def list_saved_models():
    if not os.path.exists("saved_models"):
        return []
    return [file.replace("_model.pth", "") for file in os.listdir("saved_models") if file.endswith("_model.pth")]

def main():
    model, word_to_index, index_to_word = load_model()
    if model is None:
        model, word_to_index, index_to_word = train_new_model()
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
            seed_text = input("Enter seed text: ")
            generated_text = generate_text(seed_text, 500, model, word_to_index, index_to_word, n_gram_size)
        elif choice == "2":
            model, word_to_index, index_to_word = train_new_model()
        elif choice == "3":
            filename = input("Enter filename to save model (default: text_generator): ") or "text_generator"
            save_model(model, word_to_index, index_to_word, filename)
        elif choice == "4":
            filename = input("Enter model name to load (default: text_generator): ") or "text_generator"
            model, word_to_index, index_to_word = load_model(filename)
        elif choice == "5":
            print("\nAvailable models:", list_saved_models())
        elif choice == "6":
            break
        else:
            print("Invalid choice. Please try again.")

if __name__ == "__main__":
    main()
