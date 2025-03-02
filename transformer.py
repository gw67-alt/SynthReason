import torch
from torch import nn, optim
from collections import Counter
import numpy as np
import re
import os
import json

# Hyperparameters
KB_LIMIT = 1999  # -1 for unlimited
SEQUENCE_LENGTH = 2
TEMPERATURE = 1.0
EMBEDDING_DIM = 50
HIDDEN_DIM = 128
EPOCHS = 10
LEARNING_RATE = 0.001

# Preprocess the text data
def preprocess_text(text):
    text = re.sub(r'[^\w\s]', '', text.lower())
    tokens = text.split()
    return tokens

# Create sequences and targets
def create_sequences(text_data, vocab, sequence_length):
    data = [vocab.get(word, vocab['<UNK>']) for word in preprocess_text(text_data)]
    sequences = []
    targets = []

    for i in range(len(data) - sequence_length):
        input_seq = data[i:i + sequence_length]
        target_word = data[i + sequence_length]
        sequences.append(input_seq)
        targets.append(target_word)

    return sequences, targets

# Neural Network for Markov Model
class MarkovModel(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim):
        super(MarkovModel, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, batch_first=True)
        self.fc = nn.Linear(hidden_dim, vocab_size)

    def forward(self, x):
        embeds = self.embedding(x)
        lstm_out, _ = self.lstm(embeds)
        lstm_out = lstm_out[:, -1, :]
        output = self.fc(lstm_out)
        return output

# Train the model
def train_model(model, sequences, targets, vocab_size, epochs, learning_rate):
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    for epoch in range(epochs):
        model.train()
        total_loss = 0

        for i in range(len(sequences)):
            inputs = torch.tensor(sequences[i]).unsqueeze(0)
            labels = torch.tensor([targets[i]])

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        print(f'Epoch {epoch + 1}/{epochs}, Loss: {total_loss / len(sequences)}')

# Generate text using the neural Markov model with temperature sampling
def generate_text(model, prompt, vocab, vocab_inv, seq_length=2, max_length=250, temperature=1.0):
    input_indices = [vocab.get(word, vocab['<UNK>']) for word in prompt.lower().split()]

    while len(input_indices) < seq_length:
        input_indices = [vocab['<PAD>']] + input_indices

    generated_text = prompt

    model.eval()
    with torch.no_grad():
        for _ in range(max_length):
            inputs = torch.tensor(input_indices[-seq_length:]).unsqueeze(0)
            outputs = model(inputs)
            probs = torch.softmax(outputs / temperature, dim=1).squeeze().cpu().numpy()

            next_word_idx = np.random.choice(len(probs), p=probs)
            next_word = vocab_inv[next_word_idx]

            if next_word == '<PAD>':
                break

            generated_text += ' ' + next_word
            input_indices.append(next_word_idx)

    return generated_text

# Save the model and vocabulary
def save_model(vocab, model, vocab_path="vocab.json", model_path="model.pth"):
    with open(vocab_path, 'w') as f:
        json.dump(vocab, f)
    torch.save(model.state_dict(), model_path)

# Load the model and vocabulary
def load_model(vocab_path="vocab.json", model_path="model.pth", embedding_dim=50, hidden_dim=128):
    with open(vocab_path, 'r') as f:
        vocab = json.load(f)
    vocab_size = len(vocab)
    model = MarkovModel(vocab_size, embedding_dim, hidden_dim)
    model.load_state_dict(torch.load(model_path))
    return vocab, model

# Load text data
with open("test.txt", "r", encoding="utf-8") as f:
    text = ' '.join(f.read().split()[:KB_LIMIT])

try:
    vocab, model = load_model()
    vocab_inv = {idx: word for word, idx in vocab.items()}

except:
    # Build vocabulary
    tokens = preprocess_text(text)
    word_counts = Counter(tokens)
    vocab = {word: idx for idx, (word, _) in enumerate(word_counts.items(), 1)}
    vocab['<PAD>'] = 0
    vocab['<UNK>'] = len(vocab)  # Add unknown token
    vocab_inv = {idx: word for word, idx in vocab.items()}

    # Create input sequences and targets
    sequences, targets = create_sequences(text, vocab, SEQUENCE_LENGTH)

    # Initialize and train the model
    model = MarkovModel(len(vocab), EMBEDDING_DIM, HIDDEN_DIM)
    train_model(model, sequences, targets, len(vocab), EPOCHS, LEARNING_RATE)

    # Save the model

    save_model(vocab, model)

# Interactive Text Generation (Markovian)
while True:
    prompt = input("USER: ")
    generated_text = generate_text(model, prompt, vocab, vocab_inv, seq_length=SEQUENCE_LENGTH, max_length=250, temperature=TEMPERATURE)
    print("Generated text:\n", generated_text)
