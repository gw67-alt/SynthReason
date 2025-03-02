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
KB_LIMIT = 10000 # -1 for unlimited
SEQUENCE_LENGTH = 1

# Preprocess the text data
def preprocess_text(text, vocab):
    text = re.sub(r'[^\w\s]', '', text.lower())
    tokens = text.split()
    return [vocab[word] for word in tokens if word in vocab]

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

    return transition_dict

def generate_text(prompt, vocab, transition_dict, seq_length=3, max_length=250):
    vocab_inv = {idx: word for word, idx in vocab.items()}
    input_indices = [vocab[word] for word in prompt.lower().split() if word in vocab]

    while len(input_indices) < seq_length:
        input_indices = [vocab['<PAD>']] + input_indices

    generated_text = prompt
    next_word_idx
    for _ in range(max_length):
        input_tuple = tuple(input_indices[-seq_length:])

        if input_tuple in transition_dict:
            counts = transition_dict[input_tuple]
            words = list(counts.keys())
            probs = np.array(list(counts.values()), dtype=float)
            probs /= probs.sum()

            next_word_idx = np.random.choice(words, p=probs)
        next_word = vocab_inv[next_word_idx]
        generated_text += ' ' + next_word
        input_indices.append(next_word_idx)

    return generated_text

# Load text data
with open("test.txt", "r", encoding="utf-8") as f:
    text = ' '.join(f.read().split()[:KB_LIMIT])

# Build vocabulary
tokens = text.split()
word_counts = Counter(tokens)
vocab = {word: idx for idx, (word, _) in enumerate(word_counts.items(), 1)}
vocab['<PAD>'] = 0

# Create input sequences and transition matrix
transition_dict = create_sequences(text, vocab, SEQUENCE_LENGTH)

# Interactive Text Generation (Markovian)
while True:
    prompt = input("USER: ")
    generated_text = generate_text(prompt, vocab, transition_dict, seq_length=SEQUENCE_LENGTH, max_length=250)
    print("Generated text:\n", generated_text)
