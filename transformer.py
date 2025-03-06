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
                char_ratios.update({char: char_count[char] / total_items for char in char_count.keys()})
    return char_ratios

# Preprocess the text data
def preprocess_text(text, vocab):
    text = re.sub(r'[^\w\s]', '', text.lower())
    tokens = text.split()
    return [vocab[word] for word in tokens if word in vocab]

# Create sequences and normalize transition probabilities
def create_sequences(text_data, vocab, sequence_length):
    data = preprocess_text(text_data, vocab)
    transition_dict = {}
    
    # Get the inverse vocabulary for looking up words by index
    vocab_inv = {idx: word for word, idx in vocab.items()}
    
    # Calculate character ratios from the original text data
    text_words = text_data.lower().split()
    char_ratios = calculate_character_ratios(text_words)
    
    for i in range(len(data) - sequence_length):
        # Get the current word and apply character ratio weighting
        current_word = vocab_inv.get(data[i], "")
        current_weight = 1.0
        
        # Apply character ratio to the current position (data[i])
        if current_word and len(current_word) > 0:
            first_char = current_word[0].lower()
            if first_char in char_ratios:
                current_weight = 1.0 + char_ratios[first_char]  # Boost weight based on character ratio
        
        input_seq = tuple(data[i:i + sequence_length])
        target_word = data[i + sequence_length]
        
        if input_seq not in transition_dict:
            transition_dict[input_seq] = Counter()
        
        # Apply the character ratio weight to this transition
        transition_dict[input_seq][target_word] += current_weight
    
    # Normalize transition probabilities
    for key, counter in transition_dict.items():
        total = sum(counter.values())
        transition_dict[key] = {k: (v / total) for k, v in counter.items()}
    
    return transition_dict

# Generate text using Markov chain with adjustments
def generate_text(prompt, vocab, transition_dict, seq_length=3, max_length=250):
    vocab_inv = {idx: word for word, idx in set(vocab.items())}
    input_indices = [vocab[word] for word in prompt.lower().split() if word in vocab]
    
    # Handle unknown words in prompt
    if not input_indices:
        print("Warning: None of the words in your prompt are in the vocabulary.")
        input_indices = [0]  # Use padding token
    
    while len(input_indices) < seq_length:
        input_indices = [vocab.get('<PAD>', 0)] + input_indices
    
    generated_text = prompt
    recent_transitions = []

    for _ in range(max_length):
        input_tuple = tuple(input_indices[-seq_length:])
        if input_tuple in transition_dict:
            probs_dict = transition_dict[input_tuple]
            if not probs_dict:
                break  # No transitions available
                
            words = list(probs_dict.keys())
            probs = np.array(list(probs_dict.values()), dtype=float)
            
            # Get current word and apply character ratio
            current_idx = input_indices[-1]
            current_word = vocab_inv.get(current_idx, "a")
            char_ratios = calculate_character_ratios(generated_text.split())

            # Apply character ratio adjustments to recent transitions
            for i in range(1, min(WINDOW_SIZE, len(recent_transitions)) + 1):
                past_transition = recent_transitions[-i]
                if past_transition in words:
                    try:
                        # Use first letter of current word for ratio
                        if current_word and len(current_word) > 0:
                            first_char = current_word[0].lower()
                            if first_char in char_ratios:
                                probs[words.index(past_transition)] *= char_ratios[first_char]
                    except:
                        pass
            
            # Ensure valid probabilities
            probs = np.maximum(probs, 0)
            if probs.sum() > 0:
                probs /= probs.sum()
            else:
                break  # No valid transitions
                
            next_word_idx = np.random.choice(words, p=probs)
            next_word = vocab_inv[next_word_idx]
        else:
            # No matching sequence found
            break
            
        generated_text += ' ' + next_word
        input_indices.append(next_word_idx)
        recent_transitions.append(next_word_idx)
        
        # Trim recent transitions if needed
        if len(recent_transitions) > WINDOW_SIZE:
            recent_transitions.pop(0)
            
    return generated_text

def main():
    # Check if the kb.txt file exists
    # Load text data and calculate character ratios
    try:
        with open("kb.txt", "r", encoding="utf-8") as f:
            text = ' '.join(f.read().split()[:KB_LIMIT])
        text = re.sub(r'\d+', '', text)
        text = filter_single_char_words(text)
        
        # Build vocabulary
        tokens = text.split()
        word_counts = Counter(tokens)
        vocab = {word: idx for idx, (word, _) in enumerate(word_counts.items(), 1)}
        vocab['<PAD>'] = 0
        
        # Create input sequences and transition matrix with normalized probabilities
        transition_dict = create_sequences(text, vocab, SEQUENCE_LENGTH)
        
        print("Model trained on your text data!")
        print("Enter prompts to generate text (type 'exit' to quit):")
        
        # Interactive Text Generation (Markovian)
        while True:
            prompt = input("\nUSER: ")
            if prompt.lower() == 'exit':
                break
                
            generated_text = generate_text(prompt, vocab, transition_dict, seq_length=SEQUENCE_LENGTH, max_length=250)
            print("\nGenerated text:\n", generated_text)
            
    except Exception as e:
        print(f"An error occurred: {str(e)}")

if __name__ == "__main__":
    main()