import torch
import torch.nn as nn
import torch.optim as optim
from collections import Counter
import numpy as np
import re

KB_limit = 9999
# Model Definition
class CyberneticsLSTM(nn.Module):
    def __init__(self, vocab_size, embedding_dim=128, hidden_dim=256):
        super(CyberneticsLSTM, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, batch_first=True)
        self.fc = nn.Linear(hidden_dim, vocab_size)

    def forward(self, x, hidden=None):
        embedded = self.embedding(x)
        out, hidden = self.lstm(embedded, hidden)
        out = self.fc(out)
        return out, hidden

# Create input_sequences and target_sequences correctly
def create_sequences(text_data, vocab, sequence_length):
    data = preprocess_text(text_data, vocab)
    input_sequences = []
    target_sequences = []
    for i in range(len(data) - sequence_length):
        input_seq = data[i:i + sequence_length]
        target_seq = data[i + 1:i + sequence_length + 1]
        input_sequences.append(torch.tensor(input_seq, dtype=torch.long))
        target_sequences.append(torch.tensor(target_seq, dtype=torch.long))
    return input_sequences, target_sequences

# Preprocess the text data
def preprocess_text(text, vocab):
    text = re.sub(r'[^\w\s]', '', text.lower())
    tokens = text.split()
    return [vocab.get(word, vocab['<UNK>']) for word in tokens]

def prepare_batch(input_sequences, target_sequences, batch_size):
    num_batches = len(input_sequences) // batch_size
    for i in range(num_batches):
        start_idx = i * batch_size
        end_idx = (i + 1) * batch_size
        input_batch = torch.stack(input_sequences[start_idx:end_idx])
        target_batch = torch.stack(target_sequences[start_idx:end_idx])
        yield input_batch, target_batch

def train_model(model, train_data, num_epochs=10, batch_size=32, learning_rate=0.001):
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    model.train()
    input_sequences, target_sequences, vocab = train_data
    for epoch in range(num_epochs):
        total_loss = 0
        for input_batch, target_batch in prepare_batch(input_sequences, target_sequences, batch_size):
            optimizer.zero_grad()
            output, _ = model(input_batch)
            output = output.view(-1, output.size(-1))
            target_batch = target_batch.view(-1)
            loss = criterion(output, target_batch)
            total_loss += loss.item()
            loss.backward()
            optimizer.step()
        print(f"Epoch {epoch+1}/{num_epochs}")

# Text generation function
def generate_text(model, prompt, vocab, seq_length=5, max_length=250, temperature=1.0):
    vocab_inv = {idx: word for word, idx in vocab.items()}
    input_indices = [vocab[word] for word in prompt.lower().split() if word in vocab]
    while len(input_indices) < seq_length:
        input_indices = [vocab['<PAD>']] + input_indices
    model.eval()
    input_tensor = torch.tensor(input_indices).unsqueeze(0)
    generated_text = prompt
    hidden = None
    for _ in range(max_length):
        with torch.no_grad():
            output, hidden = model(input_tensor, hidden)
            output = output[:, -1, :]
            output = output / temperature
            probabilities = torch.nn.functional.softmax(output, dim=-1).squeeze().cpu().numpy()
            next_word_idx = np.random.choice(len(vocab), p=probabilities)
            next_word = vocab_inv[next_word_idx]
            generated_text += ' ' + next_word
            input_tensor = torch.cat((input_tensor, torch.tensor([[next_word_idx]])), dim=1)
    return generated_text

# Example text data
with open("test.txt", "r", encoding="utf-8") as f:
    text = ' '.join(f.read().split()[:KB_limit])

# Build vocabulary efficiently
text_processed = re.sub(r'[^\w\s]', '', text.lower())
tokens = text_processed.split()
word_counts = Counter(tokens)
vocab = {word: idx for idx, (word, _) in enumerate(word_counts.items(), 1)}
vocab['<PAD>'] = 0
vocab['<UNK>'] = len(vocab)

# Create model instance
model = CyberneticsLSTM(vocab_size=len(vocab))
input_sequences, target_sequences = create_sequences(text_processed, vocab, 3)

# Train the model with the processed data
train_model(model, (input_sequences, target_sequences, vocab))

# Example of generating text
while True:
    prompt = input("USER: ")
    generated_text = generate_text(model, prompt, vocab, seq_length=3, max_length=250, temperature=0.7)
    print("Generated text:\n", generated_text)
