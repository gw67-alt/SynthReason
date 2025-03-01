import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import json
import re
import numpy as np
KB_limit = 9999
sequence_length = 3
def preprocess_text(text):
    text = text.lower()
    text = re.sub(r'[^a-z0-9\s]', '', text)
    return text.split()

def build_vocabulary(text):
    words = preprocess_text(text)
    vocab = {word: i + 1 for i, word in enumerate(set(words))}
    vocab["<PAD>"] = 0  # Padding token
    return vocab, len(vocab)

class CyberneticsDataset(Dataset):
    def __init__(self, file_path, vocab, sequence_length=sequence_length):
        with open(file_path, "r", encoding="utf-8") as f:
            self.data = preprocess_text(f.read()[:KB_limit])  # Tokenized text

        self.vocab = vocab
        self.sequence_length = sequence_length

    def __len__(self):
        return max(1, len(self.data) - self.sequence_length)

    def __getitem__(self, idx):
        input_seq = self.data[idx:idx + self.sequence_length]
        target_seq = self.data[idx + 1:idx + self.sequence_length + 1]
        
        input_tokens = [self.vocab.get(word, 0) for word in input_seq]
        target_tokens = [self.vocab.get(word, 0) for word in target_seq]

        return torch.tensor(input_tokens, dtype=torch.long), torch.tensor(target_tokens, dtype=torch.long)

def pad_collate(batch):
    inputs, targets = zip(*batch)
    max_len = max(len(seq) for seq in inputs)
    
    inputs_padded = torch.zeros((len(inputs), max_len), dtype=torch.long)
    targets_padded = torch.zeros((len(targets), max_len), dtype=torch.long)

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

    def forward(self, x):
        embedded = self.embedding(x)
        out, _ = self.lstm(embedded)
        out = self.fc(out)
        return out

def train_model(model, vocab, vocab_size, epochs=10, batch_size=2):
    dataset = CyberneticsDataset("test.txt", vocab)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, collate_fn=pad_collate)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    for epoch in range(epochs):
        total_loss = 0
        for inputs, targets in dataloader:
            optimizer.zero_grad()
            outputs = model(inputs)
            outputs = outputs.reshape(-1, vocab_size)
            targets = targets.reshape(-1)

            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        print(f"Epoch {epoch+1}, Loss: {total_loss / len(dataloader)}")

    torch.save(model.state_dict(), "cybernetics_model.pth")

def generate_text(model, word_to_index, input_text, sequence_length, generate_length, temperature=1.0):
    input_sequence = preprocess_text(input_text)
    index_to_word = {i: word for word, i in word_to_index.items()}
    input_indices = [word_to_index.get(word, 0) for word in input_sequence]

    if len(input_indices) < sequence_length:
        input_indices = [0] * (sequence_length - len(input_indices)) + input_indices  

    generated_text = input_sequence[:]
    model.eval()

    with torch.no_grad():
        for _ in range(generate_length):
            input_tensor = torch.tensor(input_indices[-sequence_length:], dtype=torch.long).unsqueeze(0)

            output = model(input_tensor).squeeze(0)
            output = output[-1] / temperature  # Take last step's prediction

            probabilities = torch.softmax(output, dim=-1).cpu().numpy()

            next_index = np.random.choice(len(probabilities), p=probabilities)
            next_word = index_to_word.get(next_index, "<UNK>")

            generated_text.append(next_word)
            input_indices.append(next_index)

    return " ".join(generated_text)

# Load text data
with open("test.txt", encoding="utf-8") as f:
    text_data = ' '.join(f.read().lower().split()[:KB_limit])

word_to_index, vocab_size = build_vocabulary(text_data)

# Train the model
model = CyberneticsLSTM(vocab_size)
train_model(model, word_to_index, vocab_size, epochs=10, batch_size=sequence_length+1)

while True:
    user_input = input("USER: ")
    if user_input.lower() in ["exit", "quit"]:
        break
    generated = generate_text(model, word_to_index, user_input, sequence_length, generate_length=250, temperature=0.7)
    print("Generated Text:\n", generated)
