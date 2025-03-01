import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import numpy as np
from tqdm import tqdm
import re
import pickle
KB_limit = -1
# Preprocessing text
def preprocess_text(text):
    text = text.lower()
    text = re.sub(r'[^a-z0-9\s]', '', text)
    return text.split()

# Building vocabulary
def build_vocabulary(text):
    words = preprocess_text(text)
    vocab = {word: i for i, word in enumerate(set(words), start=1)}
    vocab["<PAD>"] = 0  # Padding token
    return vocab, len(vocab)

# Creating sequences
def create_sequences(vocab, words, sequence_length):
    indices = [vocab[word] for word in words if word in vocab]
    sequences = [indices[i:i+sequence_length+1] for i in range(len(indices) - sequence_length)]
    return sequences

# Custom Dataset
class TextDataset(Dataset):
    def __init__(self, sequences):
        self.data = torch.tensor([seq[:-1] for seq in sequences], dtype=torch.long)
        self.targets = torch.tensor([seq[-1] for seq in sequences], dtype=torch.long)
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        return self.data[idx], self.targets[idx]

# Knowledge-Augmented LSTM
class KnowledgeAugmentedLSTM(nn.Module):
    def __init__(self, vocab_size, embedding_dim=128, hidden_dim=256, knowledge_dim=64):
        super(KnowledgeAugmentedLSTM, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.knowledge_embedding = nn.Embedding(vocab_size, knowledge_dim)
        self.lstm = nn.LSTM(embedding_dim + knowledge_dim, hidden_dim, batch_first=True)
        self.fc = nn.Linear(hidden_dim, vocab_size)
    
    def forward(self, x):
        word_embed = self.embedding(x)
        knowledge_embed = self.knowledge_embedding(x)
        combined = torch.cat((word_embed, knowledge_embed), dim=-1)
        lstm_out, _ = self.lstm(combined)
        return self.fc(lstm_out[:, -1, :])

# Custom Loss Function
def custom_loss(output, target):
    log_probs = F.log_softmax(output, dim=-1)
    return F.nll_loss(log_probs, target)

# Training Model
def train_model(model, data_loader, num_epochs=10, lr=0.001):
    optimizer = optim.Adam(model.parameters(), lr=lr)
    model.train()
    for epoch in range(num_epochs):
        total_loss = 0
        progress_bar = tqdm(data_loader, desc=f"Epoch {epoch+1}/{num_epochs}", leave=False)   
        for inputs, targets in progress_bar:
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = custom_loss(outputs, targets)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()     
            # Update progress bar
            progress_bar.set_postfix(loss=total_loss / (progress_bar.n + 1))  
        print(f"Epoch {epoch+1}, Loss: {total_loss/len(data_loader):.4f}") 
    return model

# Save Model & Vocabulary
def save_model_and_vocab(model, vocab, path="model.pth"):
    torch.save({'model_state_dict': model.state_dict(), 'vocab': vocab}, path)

# Load Model & Vocabulary
def load_model_and_vocab(path="model.pth"):
    checkpoint = torch.load(path)
    model = KnowledgeAugmentedLSTM(len(checkpoint['vocab']))
    model.load_state_dict(checkpoint['model_state_dict'])
    return model, checkpoint['vocab']

# Generate Text
def generate_text(model, word_to_index, input_text, sequence_length, generate_length, temperature=1.0):
    input_sequence = preprocess_text(input_text)
    index_to_word = {i: word for word, i in word_to_index.items()}
    input_indices = [word_to_index.get(word, 0) for word in input_sequence]
    if len(input_indices) < sequence_length:
        input_indices = [0] * (sequence_length - len(input_indices)) + input_indices  # Padding
    generated_text = input_sequence[:]
    model.eval()
    with torch.no_grad():
        for _ in range(generate_length):
            input_tensor = torch.tensor(input_indices[-sequence_length:], dtype=torch.long).unsqueeze(0)
            output = model(input_tensor).squeeze(0) / temperature
            probabilities = torch.softmax(output, dim=-1).cpu().numpy()
            next_index = np.random.choice(len(probabilities), p=probabilities)
            next_word = index_to_word.get(next_index, "<UNK>")
            generated_text.append(next_word)
            input_indices.append(next_index)
    return " ".join(generated_text)

# Main Function
def main():
    choice = input("Do you want to (1) train or (2) load a model: ")

    if choice == '1':
        with open("test.txt", encoding="utf-8") as f:
            text_data = ' '.join(f.read().lower().split()[:KB_limit])
        word_to_index, vocab_size = build_vocabulary(text_data)
        sequences = create_sequences(word_to_index, preprocess_text(text_data), 5)
        dataset = TextDataset(sequences)
        data_loader = DataLoader(dataset, batch_size=512, shuffle=True)
        model = KnowledgeAugmentedLSTM(vocab_size)
        trained_model = train_model(model, data_loader, num_epochs=10)
        save_model_and_vocab(trained_model, word_to_index)
        loaded_model, loaded_vocab = trained_model, word_to_index  # Ensure consistency

    elif choice == '2':
        loaded_model, loaded_vocab = load_model_and_vocab()

    else:
        print("Invalid option.")
        return

    while True:
        user_input = input("USER: ")
        if user_input.lower() in ["exit", "quit"]:
            break
        generated = generate_text(loaded_model, loaded_vocab, user_input, sequence_length=5, generate_length=250, temperature=0.7)
        print("Generated Text:\n", generated)

if __name__ == "__main__":
    main()
