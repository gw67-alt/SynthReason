import torch
import torch.nn as nn
import torch.optim as optim
from collections import Counter
import numpy as np
import re
import random

# Hyperparameters
KB_LIMIT = 999
SEQUENCE_LENGTH = 3
NUM_GENERATIONS = 10
POPULATION_SIZE = 10
MUTATION_RATE = 0.2
BATCH_SIZE = 32
LEARNING_RATE = 0.001
NUM_EPOCHS = 5

# LSTM Model with EANT-based mutation
class CyberneticsEANT(nn.Module):
    def __init__(self, vocab_size, embedding_dim=128, hidden_dim=256, num_layers=1):
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

def prepare_batch(input_sequences, target_sequences, batch_size):
    num_batches = len(input_sequences) // batch_size
    for i in range(num_batches):
        start_idx = i * batch_size
        end_idx = (i + 10) * batch_size
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
        for input_batch, target_batch in prepare_batch(input_sequences, target_sequences, BATCH_SIZE):
            optimizer.zero_grad()
            output, _ = model(input_batch)
            output = output.view(-1, output.size(-1))
            target_batch = target_batch.view(-1)
            loss = criterion(output, target_batch)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
        total_loss += epoch_loss

    return total_loss / num_epochs  # Average loss

def evolve_population(population, train_data, num_generations=NUM_GENERATIONS):
    """Evolves the population using EANT (mutation and survival)."""
    for generation in range(num_generations):
        fitness_scores = []

        # Train each model and store its loss
        for i, model in enumerate(population):
            try:
                loss = train_model(model, train_data)
                fitness_scores.append((model, loss))
            except Exception as e:
                print(f"Error training model {i}: {e}")

        if not fitness_scores:  # Ensure at least one valid model exists
            print(f"Generation {generation+1}: No valid models! Using fallback.")
            return population[0]  # Return the first model as a fallback to avoid crash

        # Sort by loss (lower is better)
        fitness_scores.sort(key=lambda x: x[1])

        print(f"Generation {generation+1}: Best Loss = {fitness_scores[0][1]}")

        # Ensure we have at least two models to continue
        if len(fitness_scores) < 2:
            print(f"Warning: Only one model left. Stopping evolution.")
            return fitness_scores[0][0]  # Return the best model we have

        # Select top-performing models
        top_models = [m for m, _ in fitness_scores[:max(2, len(fitness_scores)//2)]]

        # Create the next generation through mutation
        new_population = []
        for model in top_models:
            new_model = CyberneticsEANT(len(vocab), model.embedding.embedding_dim, model.lstm.hidden_size)
            new_model.load_state_dict(model.state_dict())
            new_model.mutate()  # Apply mutation
            new_population.append(new_model)

        population = new_population if new_population else top_models  # Ensure population isn't empty

    return population[0]

# Text generation function
def generate_text(model, prompt, vocab, seq_length=5, max_length=250, temperature=1.0):
    vocab_inv = {idx: word for word, idx in vocab.items()}
    input_indices = [vocab.get(word, vocab['<UNK>']) for word in prompt.lower().split()]
    while len(input_indices) < seq_length:
        input_indices = [vocab['<PAD>']] + input_indices
    model.eval()
    input_tensor = torch.tensor(input_indices).unsqueeze(0)
    generated_text = prompt
    hidden = None

    for _ in range(max_length):
        with torch.no_grad():
            output, hidden = model(input_tensor, hidden)
            output = output[:, -1, :] / temperature
            probabilities = torch.nn.functional.softmax(output, dim=-1).squeeze().cpu().numpy()
            next_word_idx = np.random.choice(len(vocab), p=probabilities)
            next_word = vocab_inv.get(next_word_idx, '<UNK>')
            generated_text += ' ' + next_word
            input_tensor = torch.cat((input_tensor, torch.tensor([[next_word_idx]])), dim=1)

    return generated_text

# Load text data
with open("test.txt", "r", encoding="utf-8") as f:
    text = ' '.join(f.read().split()[:KB_LIMIT])

# Build vocabulary
text_processed = re.sub(r'[^\w\s]', '', text.lower())
tokens = text_processed.split()
word_counts = Counter(tokens)
vocab = {word: idx for idx, (word, _) in enumerate(word_counts.items(), 1)}
vocab['<PAD>'] = 0
vocab['<UNK>'] = len(vocab)

# Create input sequences
input_sequences, target_sequences = create_sequences(text_processed, vocab, SEQUENCE_LENGTH)

# Create initial population
population = [CyberneticsEANT(len(vocab)) for _ in range(POPULATION_SIZE)]

# Evolve the population
best_model = evolve_population(population, (input_sequences, target_sequences, vocab))

# Interactive Text Generation
while True:
    prompt = input("USER: ")
    generated_text = generate_text(best_model, prompt, vocab, seq_length=SEQUENCE_LENGTH, max_length=250, temperature=0.7)
    print("Generated text:\n", generated_text)
