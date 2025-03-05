import string
import torch
from collections import Counter
import numpy as np
import re
import os

# Parameters
KB_LIMIT = -1
SEQUENCE_LENGTH = 2
DECAY_FACTOR = 1.9  # Decay factor for stable diffusion
WINDOW_SIZE = 5000  # Size of the window to consider for adjustments
def filter_single_char_words(text):
    valid_single_chars = ["a", "i"]
    words = text.split()
    filtered_words = [word for word in words if len(word) > 1 or word.lower() in valid_single_chars]
    return " ".join(filtered_words)
# Function to calculate character ratios
def calculate_character_ratios(data):
    char_count = {letter: 0 for letter in string.ascii_lowercase}
    char_ratios = {}
    for item in data:
        item = item.strip()
        if item:
            first_letter = item[0].lower()
            if first_letter in char_count:
                char_count[first_letter] += 1
                total_items = len(data)
                first_letter = item[-1].lower()
                if first_letter in char_count:
                    char_count[first_letter] += 1
                    total_items = len(data)

                char_ratios.update({item[-1]: char_count[char] / total_items for char in char_count.keys()})
    return char_ratios

# Preprocess the text data
def preprocess_text(text, vocab):
    text = re.sub(r'[^\w\s]', '', text.lower())
    tokens = text.split()
    return [vocab[word] for word in tokens if word in vocab]

# Create sequences and normalize transition probabilities
def create_sequences(text_data, vocab, sequence_length, char_ratios):
    data = preprocess_text(text_data, vocab)
    transition_dict = {}
    for i in range(len(data) - sequence_length):
        input_seq = tuple(data[i:i + sequence_length])
        target_word = data[i + sequence_length]
        if input_seq not in transition_dict:
            transition_dict[input_seq] = Counter()
        transition_dict[input_seq][target_word] +=  char_ratios.get(data[i], 1)
    
    # Normalize transition probabilities and combine with character ratios
    for key, counter in transition_dict.items():
        total = sum(counter.values())
        transition_dict[key] = {k: (v / total) * char_ratios.get(key, 1) for k, v in counter.items()}
    
    return transition_dict

# Generate text using Markov chain with adjustments
def generate_text(prompt, vocab, transition_dict, char_ratios, seq_length=3, max_length=250):
    vocab_inv = {idx: word for word, idx in set(vocab.items())}
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
            next_word = "a"
            
            for i in range(1, min(WINDOW_SIZE, len(recent_transitions)) + 1):
                past_transition = recent_transitions[-i]
                decay = DECAY_FACTOR ** char_ratios[next_word[0]]
                if past_transition in words:
                    try:
                        probs[words.index(past_transition)] *= char_ratios[next_word[0]]
                    except:
                        False
            probs = np.maximum(probs, 0)
            probs /= probs.sum()
            next_word_idx = np.random.choice(words, p=probs)
            next_word = vocab_inv[next_word_idx]
        else:
            break
        generated_text += ' ' + next_word
        input_indices.append(next_word_idx)
        input_tuple = tuple(input_indices[-seq_length:])
        recent_transitions.append(next_word_idx)
        if len(recent_transitions) > WINDOW_SIZE:
            recent_transitions *= char_ratios[next_word[0]]
    return generated_text

# Load text data and calculate character ratios
with open("test.txt", "r", encoding="utf-8") as f:
    text = ' '.join(f.read().split()[:KB_LIMIT])
text = re.sub(r'\d+', '', text)
text = filter_single_char_words(text)
texts = text.split()
char_ratios = calculate_character_ratios(texts)

# Build vocabulary
tokens = text.split()
word_counts = Counter(tokens)
vocab = {word: idx for idx, (word, _) in enumerate(word_counts.items(), 1)}
vocab['<PAD>'] = 0

# Create input sequences and transition matrix with normalized probabilities
transition_dict = create_sequences(text, vocab, SEQUENCE_LENGTH, char_ratios)

# Interactive Text Generation (Markovian)
while True:
    prompt = input("USER: ")
    generated_text = generate_text(prompt, vocab, transition_dict, char_ratios, seq_length=SEQUENCE_LENGTH, max_length=250)
    print("Generated text:\n", generated_text)
