import json
import numpy as np
import re
from collections import defaultdict
import random

KB_limit = 99999
length = 500
# Step 1: Load unstructured text from a file
with open('test.txt', 'r', encoding="utf-8") as file:
    text = ' '.join(file.read().split()[:KB_limit])
words = re.findall(r'\b\w+\b', text.lower())  # Convert to lowercase for case-insensitive matching

# Filter out numbers and single-character words except 'a' and 'i'
words = [word for word in words if len(word) > 1 or word in ['a', 'i'] and not word.isdigit()]

# Step 3: Create a word-to-index mapping (using a dictionary)
word_to_index = {word: idx + 1 for idx, word in enumerate(set(words))}

# Reverse the word-to-index for converting index back to word
index_to_word = {v: k for k, v in word_to_index.items()}

# Step 4: Generate training sequences (n-grams)
n_gram_size = 3  # For bi-grams, we use n = 2
n_grams = []

# Create bi-grams from the text
for i in range(len(words) - n_gram_size + 1):
    n_grams.append(tuple(words[i:i + n_gram_size]))

# Step 5: Create a transition matrix based on the n-grams
transition_matrix = defaultdict(list)

# Populate the transition matrix
for n_gram in n_grams:
    # The first word of the n-gram is the "current word"
    # The second word of the n-gram is the "next word"
    transition_matrix[n_gram[0]].append(n_gram[1])

# Step 6: Generate text based on the transition matrix
generated_text = []

# Randomly select the starting word (first word of the sequence)
current_word = random.choice(words)
generated_text.append(current_word)

# Generate the next words based on the transition matrix
for _ in range(length):  # Generate 50 words
    if current_word in transition_matrix:
        next_word = random.choice(transition_matrix[current_word])
        generated_text.append(next_word)
        current_word = next_word
    else:
        break  # Stop if we don't have any more transitions
# Step 7: Output the generated text (a matrix of words)
print("Generated Text:")

print(' '.join(generated_text))
