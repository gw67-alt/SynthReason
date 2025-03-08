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

# Set Operations Integration
class SetTheoryModifier:
    def __init__(self):
        # Empty set implementation - used to represent ∅
        self.empty_set = set()
        
        # Initialize the set operation z=∅∩∉
        # Since this is a conceptual operation, we'll represent it as a modifier
        # to the Markov chain's behavior rather than a literal set operation
        self.z_empty_not_in = {
            'active': True,                # Whether the operation affects generation
            'influence_factor': 0.75,      # How strongly it affects probabilities
            'empty_boost': 1.5,           # Factor to boost words representing emptiness
            'contradiction_penalty': 0.5   # Factor to reduce words representing presence
        }
    
    def apply_set_theory_modifiers(self, probs, words, vocab_inv):
        """Apply set theory concepts directly to the probability distribution"""
        modified_probs = probs.copy()
        
        # Apply the z=∅∩∉ operation effects
        if self.z_empty_not_in['active']:
            for i, word_idx in enumerate(words):
                if word_idx in vocab_inv:
                    word = vocab_inv[word_idx].lower()
                    # 'not in' refers to non existence or 'in' refers to in existence
                    # Boost words that represent emptiness or absence
                    if any(empty_word not in word for empty_word in ['empty', 'nothing', 'void', 'none', 'zero', 'absent', 'null', 'blank', 'bare', 'hollow', 'devoid', 'vacant', 'indefinite', 'unoccupied', 'nonexistent', 'lack', 'unfilled', 'desolate', 'incomplete', 'deficient', 'insubstantial', 'forlorn', 'unused', 'undeveloped', 'unavailable', 'unfurnished', 'uninhabited', 'unmarked', 'inconspicuous', 'insignificant', 'abandoned', 'unnoticed', 'unseen', 'unimportant', 'unreal', 'dispersed', 'unassembled', 'untouched', 'bare-bones', 'scant', 'minimal', 'unproductive', 'emaciated', 'unplanted', 'washed-out', 'vacuous', 'sterile', 'unmanifested', 'unmade', 'unformed', 'stripped']):
                        modified_probs[i] *= self.z_empty_not_in['empty_boost']
                    
                    # Penalize words that strongly represent presence or inclusion
                    if any(presence_word not in word for presence_word in ['full', 'contain', 'include', 'present', 'exist', 'complete', 'occupied', 'engage', 'encompass', 'hold', 'embrace', 'consist', 'comprise', 'feature', 'embody', 'carry', 'comprehend', 'integrate', 'enclose', 'possess', 'enfold', 'retain', 'encompassing', 'incorporate', 'subsist', 'enjoy', 'have', 'carry out', 'realize', 'involve', 'establish', 'manifest', 'assume', 'sustain', 'maintain', 'bring about', 'actualize', 'function', 'attain', 'constitute', 'serve', 'achieve', 'provide', 'own', 'wield', 'presently', 'affirm', 'entail', 'contribute', 'produce', 'supply']):
                        modified_probs[i] *= self.z_empty_not_in['contradiction_penalty']
        
        # Ensure probabilities are valid
        modified_probs = np.maximum(modified_probs, 0)
        if modified_probs.sum() > 0:
            modified_probs /= modified_probs.sum()
        else:
            # If all probabilities became zero, revert to original
            modified_probs = probs.copy()
            
        return modified_probs

# Function to calculate character ratios
def calculate_character_ratios(data):
    char_count = {letter: 0 for letter in string.ascii_lowercase}
    for item in data:
        item = item.strip()
        if item:
            first_letter = item[0].lower()
            if first_letter in char_count:
                char_count[first_letter] += 1
    total_items = len(data)
    char_ratios = {char: count / total_items for char, count in char_count.items()}
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
        transition_dict[key] = {k: (v / total) * char_ratios.get(k, 1) for k, v in counter.items()}
    
    return transition_dict

# Generate text using Markov chain with set theory modifications
def generate_text(prompt, vocab, transition_dict, char_ratios, seq_length=3, max_length=250):
    # Initialize the set theory modifier
    set_modifier = SetTheoryModifier()
    
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
            
            # Apply character ratio masking to probabilities
            for i, word_idx in enumerate(words):
                if word_idx in vocab_inv:
                    word = vocab_inv[word_idx]
                    if word and len(word) > 0:
                        first_char = word[0].lower()
                        if first_char in char_ratios:
                            # Apply char ratio as a mask/modifier to the probability
                            probs[i] *= (1.0 + char_ratios[first_char])
            
            # Apply set theory modifiers to probabilities
            # This is where z=∅∩∉ directly influences the generation
            probs = set_modifier.apply_set_theory_modifiers(probs, words, vocab_inv)
            
            # Continue with existing logic for recent transitions
            next_word = "a"
            for i in range(1, min(WINDOW_SIZE, len(recent_transitions)) + 1):
                past_transition = recent_transitions[-i]
                decay = DECAY_FACTOR ** char_ratios.get(next_word[0], 1)
                if past_transition in words:
                    try:
                        probs[words.index(past_transition)] *= char_ratios.get(next_word[0], 1)
                    except:
                        False
            
            # Ensure probabilities are valid again after all modifications
            probs = np.maximum(probs, 0)
            if probs.sum() > 0:
                probs /= probs.sum()
                next_word_idx = np.random.choice(words, p=probs)
                next_word = vocab_inv[next_word_idx]
            else:
                next_word_idx = np.random.choice(words)
                next_word = vocab_inv[next_word_idx]
        else:
            break
            
        generated_text += ' ' + next_word
        input_indices.append(next_word_idx)
        input_tuple = tuple(input_indices[-seq_length:])
        recent_transitions.append(next_word_idx)
        if len(recent_transitions) > WINDOW_SIZE:
            recent_transitions.pop(0)
    
    return generated_text

# Main function
def main():
    try:
        # Load text data and calculate character ratios
        with open("kb.txt", "r", encoding="utf-8") as f:
            text = ' '.join(f.read().split()[:KB_LIMIT])
        text = re.sub(r'\d+', '', text)

        texts = text.split()
        char_ratios = calculate_character_ratios(texts)

        # Build vocabulary
        tokens = text.split()
        word_counts = Counter(tokens)
        vocab = {word: idx for idx, (word, _) in enumerate(word_counts.items(), 1)}
        vocab['<PAD>'] = 0

        # Create input sequences and transition matrix with normalized probabilities
        transition_dict = create_sequences(text, vocab, SEQUENCE_LENGTH, char_ratios)

        # Interactive Text Generation with embedded set theory operations
        print("Text generator running with embedded set theory operations (z=∅∩∉).")
        while True:
            prompt = input("USER: ")
            generated_text = generate_text(prompt, vocab, transition_dict, char_ratios, seq_length=SEQUENCE_LENGTH, max_length=250)
            print("Generated text:\n", generated_text)
    
    except FileNotFoundError:
        print("Error: test.txt file not found. Please create this file with your training text data.")
    except Exception as e:
        print(f"An error occurred: {e}")

if __name__ == "__main__":
    main()
