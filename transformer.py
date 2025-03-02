import torch
import torch.nn as nn
import torch.optim as optim
from collections import Counter
import numpy as np
import re
import random
import os
from tqdm import tqdm

# Hyperparameters
KB_LIMIT = 20000
SEQUENCE_LENGTH = 1
NUM_GENERATIONS = 10
POPULATION_SIZE = 3
MUTATION_RATE = 10.01
BATCH_SIZE = 1024
LEARNING_RATE = 0.001
NUM_EPOCHS = 5
EMBEDDING_DIM = 16
HIDDEN_DIM = 32
NUM_LAYERS = 3

# LSTM Model with EANT-based mutation
class CyberneticsEANT(nn.Module):
    def __init__(self, vocab_size, embedding_dim=EMBEDDING_DIM, hidden_dim=HIDDEN_DIM, num_layers=NUM_LAYERS):
        super(CyberneticsEANT, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_dim, vocab_size)

    def forward(self, x, hidden=None):
        embedded = self.embedding(x)
        out, hidden = self.lstm(embedded, hidden)
        out = self.fc(out)
        return out, hidden

    def mutate(self):
        """Applies random mutation to the model."""
        with torch.no_grad():
            for param in self.parameters():
                if torch.rand(1).item() < MUTATION_RATE:
                    param += torch.randn_like(param) * 0.1  # Small mutation

# Preprocess the text data
def preprocess_text(text, vocab):
    text = re.sub(r'[^\w\s]', '', text.lower())
    tokens = text.split()
    return [vocab.get(word, vocab['<UNK>']) for word in tokens]

def create_sequences(text_data, vocab, sequence_length):
    data = preprocess_text(text_data, vocab)
    input_sequences = []
    target_sequences = []
    transition_dict = {}

    for i in range(len(data) - sequence_length):
        input_seq = tuple(data[i:i + sequence_length])
        target_word = data[i + sequence_length]

        input_sequences.append(torch.tensor(input_seq, dtype=torch.long))
        target_sequences.append(torch.tensor(target_word, dtype=torch.long))

        # Markov Transition Table
        if input_seq not in transition_dict:
            transition_dict[input_seq] = Counter()
        transition_dict[input_seq][target_word] += 1  # Increment occurrence

    return input_sequences, target_sequences, transition_dict

def prepare_batch(input_sequences, target_sequences, batch_size):
    num_batches = len(input_sequences) // batch_size
    for i in range(num_batches):
        start_idx = i * batch_size
        end_idx = (i + 1) * batch_size
        input_batch = torch.stack(input_sequences[start_idx:end_idx])
        target_batch = torch.stack(target_sequences[start_idx:end_idx])
        yield input_batch, target_batch

def train_model(model, train_data, num_epochs=NUM_EPOCHS):
    """Trains a single model and returns its final loss."""
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    model.train()
    input_sequences, target_sequences, vocab = train_data
    total_loss = 0

    for epoch in range(num_epochs):
        epoch_loss = 0
        batches = list(prepare_batch(input_sequences, target_sequences, BATCH_SIZE))
        for input_batch, target_batch in batches:
            optimizer.zero_grad()
            output, _ = model(input_batch)
            output = output.view(-1, output.size(-1))
            target_batch = target_batch.view(-1)
            loss = criterion(output, target_batch)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
        total_loss += epoch_loss
    return total_loss / num_epochs

def evolve_population(population, train_data, num_generations=NUM_GENERATIONS):
    """Evolves the population using EANT (mutation and survival)."""
    for generation in tqdm(range(num_generations), desc="Generations"):
        fitness_scores = []

        # Train each model and store its loss
        for i, model in enumerate(population):
            try:
                loss = train_model(model, train_data)
                fitness_scores.append((model, loss))
            except Exception as e:
                print(f"Error training model {i}: {e}")

        if not fitness_scores:
            print(f"Generation {generation+1}: No valid models! Using fallback.")
            return population[0]

        fitness_scores.sort(key=lambda x: x[1])

        if len(fitness_scores) < 2:
            print(f"Warning: Only one model left. Stopping evolution.")
            return fitness_scores[0][0]

        top_models = [m for m, _ in fitness_scores[:max(2, len(fitness_scores) // 2)]]

        new_population = []
        for model in top_models:
            new_model = CyberneticsEANT(len(vocab), model.embedding.embedding_dim, model.lstm.hidden_size)
            new_model.load_state_dict(model.state_dict())
            new_model.mutate()
            new_population.append(new_model)

        population = new_population if new_population else top_models

    return population[0]

def generate_text(model, prompt, vocab, transition_dict, seq_length=3, max_length=250):
    vocab_inv = {idx: word for word, idx in vocab.items()}
    input_indices = [vocab.get(word, vocab['<UNK>']) for word in prompt.lower().split()]

    while len(input_indices) < seq_length:
        input_indices = [vocab['<PAD>']] + input_indices

    generated_text = prompt

    for _ in range(max_length):
        input_tuple = tuple(input_indices[-seq_length:])

        if input_tuple in transition_dict:
            counts = transition_dict[input_tuple]
            words = list(counts.keys())
            probs = np.array(list(counts.values()), dtype=float)
            probs /= probs.sum()

            next_word_idx = np.random.choice(words, p=probs)
        else:
            next_word_idx = vocab['<UNK>']

        next_word = vocab_inv.get(next_word_idx, '<UNK>')
        generated_text += ' ' + next_word
        input_indices.append(next_word_idx)

    return generated_text

def save_model(model, filepath):
    torch.save(model.state_dict(), filepath)

def load_model(model, filepath):
    model.load_state_dict(torch.load(filepath))
    return model

# Load text data
with open("test.txt", "r", encoding="utf-8") as f:
    text = ' '.join(f.read().split()[:KB_LIMIT])

# Build vocabulary
text_processed = re.sub(r'\d', '', text.lower())
tokens = text_processed.split()
word_counts = Counter(tokens)
vocab = {word: idx for idx, (word, _) in enumerate(word_counts.items(), 1)}
vocab['<PAD>'] = 0
vocab['<UNK>'] = len(vocab)
# Create input sequences and transition matrix
input_sequences, target_sequences, transition_dict = create_sequences(text_processed, vocab, SEQUENCE_LENGTH)
population = [CyberneticsEANT(len(vocab)) for _ in range(POPULATION_SIZE)]

model_filepath = "best_model.pth"

# Check if a saved model exists
if os.path.exists(model_filepath):
    print("Loading saved model...")
    best_model = CyberneticsEANT(len(vocab))
    best_model = load_model(best_model, model_filepath)
    print("Model loaded.")
else:
    print("Starting evolution...")
    best_model = evolve_population(population, (input_sequences, target_sequences, vocab))
    print("Evolution complete!")
    print("Saving best model...")
    save_model(best_model, model_filepath)
    print("Model saved.")

# Interactive Text Generation (Markovian)
while True:
    prompt = input("USER: ")
    generated_text = generate_text(best_model, prompt, vocab, transition_dict, seq_length=SEQUENCE_LENGTH, max_length=250)
    print("Generated text:\n", generated_text)
