import torch
from collections import Counter
import numpy as np
import re
import random
import os

# Hyperparameters
KB_LIMIT = -1  # -1 for unlimited
SEQUENCE_LENGTH = 1
DECAY_FACTOR = 0.9  # Decay factor for stable diffusion
WINDOW_SIZE = 5000  # Size of the window to consider for adjustments

# Preprocess the text data
def preprocess_text(text, vocab):
    text = re.sub(r'[^\w\s]', '', text.lower())
    tokens = text.split()
    return [vocab[word] for word in tokens if word in vocab]

# Create sequences and normalize transition probabilities
def create_sequences(text_data, vocab, sequence_length):
    data = preprocess_text(text_data, vocab)
    transition_dict = {}

    for i in range(len(data) - sequence_length):
        input_seq = tuple(data[i:i + sequence_length])
        target_word = data[i + sequence_length]

        if input_seq not in transition_dict:
            transition_dict[input_seq] = Counter()
        transition_dict[input_seq][target_word] += 1  # Increment occurrence

    # Normalize transition probabilities
    for key, counter in transition_dict.items():
        total = sum(counter.values())
        transition_dict[key] = {k: v / total for k, v in counter.items()}  # Normalization

    return transition_dict

# Generate text using Markov chain with adjustments
def generate_text(prompt, vocab, transition_dict, seq_length=3, max_length=250):
    vocab_inv = {idx: word for word, idx in vocab.items()}
    input_indices = [vocab[word] for word in prompt.lower().split() if word in vocab]

    while len(input_indices) < seq_length:
        input_indices = [vocab['<PAD>']] + input_indices

    generated_text = prompt
    recent_transitions = []

    for _ in range(max_length):
        input_tuple = tuple(input_indices[-seq_length:])

        if input_tuple in transition_dict:
            probs_dict = transition_dict[input_tuple]
            words = list(probs_dict.keys())
            probs = np.array(list(probs_dict.values()), dtype=float)

            # Adjust probabilities considering recent transitions
            for i in range(1, min(WINDOW_SIZE, len(recent_transitions)) + 1):
                past_transition = recent_transitions[-i]
                decay = DECAY_FACTOR ** i
                if past_transition in words:
                    probs[words.index(past_transition)] -= decay

            # Ensure probabilities are non-negative and normalize them
            probs = np.maximum(probs, 0)
            probs /= probs.sum()  # Re-normalize probabilities

            next_word_idx = np.random.choice(words, p=probs)  # Sampling from normalized probabilities
            next_word = vocab_inv[next_word_idx]
        else:
            break

        generated_text += ' ' + next_word
        input_indices.append(next_word_idx)
        input_tuple = tuple(input_indices[-seq_length:])

        # Update recent transitions
        recent_transitions.append(next_word_idx)
        if len(recent_transitions) > WINDOW_SIZE:
            recent_transitions.pop(0)

    return generated_text

# Load text data
with open("xaa", "r", encoding="utf-8") as f:
    text = ' '.join(f.read().split()[:KB_LIMIT])

# Build vocabulary
tokens = text.split()
word_counts = Counter(tokens)
vocab = {word: idx for idx, (word, _) in enumerate(word_counts.items(), 1)}
vocab['<PAD>'] = 0

# Create input sequences and transition matrix with normalized probabilities
transition_dict = create_sequences(text, vocab, SEQUENCE_LENGTH)

# Interactive Text Generation (Markovian)
while True:
    prompt = input("USER: ")
    generated_text = generate_text(prompt, vocab, transition_dict, seq_length=SEQUENCE_LENGTH, max_length=250)
    print("Generated text:\n", generated_text)
