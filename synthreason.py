import json
import numpy as np
import re
from collections import defaultdict
import random

KB_limit = 9999
generate_length = 500
# Step 1: Load unstructured text from a file
with open('test.txt', 'r', encoding="utf-8") as file:
    text = ' '.join(file.read().split()[:KB_limit])

# Step 2: Process the text and tokenize it into words
# We will split the text by non-alphabetical characters (punctuation, whitespace)
# Filter out numbers and single-character words except for 'a' and 'i'
words = re.findall(r'\b\w+\b', text.lower())  # Convert to lowercase for case-insensitive matching

# Filter out numbers and single-character words except 'a' and 'i'
filtered_words = [word for word in words if len(word) > 1 or word in ['a', 'i'] and not word.isdigit()]

# Step 3: Create a word-to-index mapping (using a dictionary)
word_to_index = {word: idx + 1 for idx, word in enumerate(set(filtered_words))}

# Reverse the word-to-index for converting index back to word
index_to_word = {v: k for k, v in word_to_index.items()}

# Step 4: Create a square matrix (size of the unique words)
n = len(word_to_index)
intersection_matrix = np.zeros((n, n), dtype=int)

# Step 5: Fill the matrix with set intersection of word indices
unique_words = list(word_to_index.keys())
for i in range(n-1):
    for j in range(n):
        # Get the words at index i and j
        word_i = unique_words[i]
        word_j = unique_words[j]
        
        # Find intersection of sets of word_i and word_j
        # We treat each word as a "set" and check intersection of sets
        # Instead of creating sets of word indices, we directly check for word intersection
        intersection_result = 1 if word_i == word_j else 0
        
        intersection_matrix[i][j] = intersection_result

# Step 6: Generate n-grams (bigrams) from the filtered words
n_gram_size = 2  # For bi-grams, we use n = 2
n_grams = []

# Create bi-grams from the filtered text
for i in range(len(filtered_words) - n_gram_size + 1):
    n_grams.append(tuple(filtered_words[i:i + n_gram_size]))

# Step 7: Create a transition matrix based on the n-grams
transition_matrix = defaultdict(list)

# Populate the transition matrix with bigrams
for n_gram in n_grams:
    transition_matrix[n_gram[0]].append(n_gram[1])

# Step 8: Generate text based on the transition matrix
generated_text = []

# Randomly select the starting word (first word of the sequence)
current_word = random.choice(filtered_words)
generated_text.append(current_word)

# Generate the next words based on the transition matrix
for _ in range(generate_length):  # Generate 50 words
    if current_word in transition_matrix:
        next_word = random.choice(transition_matrix[current_word])
        generated_text.append(next_word)
        current_word = next_word
    else:
        break  # Stop if we don't have any more transitions

# Step 9: Output the generated text
print("Generated Text:")
print(' '.join(generated_text))
