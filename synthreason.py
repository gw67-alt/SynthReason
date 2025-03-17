import numpy as np
import re
from collections import defaultdict
import random

KB_limit = 99999
generate_length = 500
n_gram_size = 2  # n-gram size (for bigrams)

# Step 1: Load unstructured text from a file
with open('test.txt', 'r', encoding="utf-8") as file:
    text = ' '.join(file.read().split()[:KB_limit])

# Filter out numbers and single-character words except for 'a' and 'i'
text = re.sub(r'\d+', '', text)  # Removes all numbers

words = re.findall(r'\b\w+\b', text.lower())  # Convert to lowercase for case-insensitive matching

# Filter out numbers and single-character words except 'a' and 'i'
filtered_words = [word for word in words if len(word) > 1 or word in ['a', 'i'] and not word.isdigit()]

# Step 2: Create a word-to-index mapping (using a dictionary)
word_to_index = {word: idx + 1 for idx, word in enumerate(set(filtered_words))}

# Reverse the word-to-index for converting index back to word
index_to_word = {v: k for k, v in word_to_index.items()}

# Step 3: Generate n-grams (bigrams) from the filtered words
n_grams = []
for i in range(len(filtered_words) - n_gram_size + 1):
    n_grams.append(tuple(filtered_words[i:i + n_gram_size]))

# Step 4: Create a transition matrix based on the n-grams (counting transitions)
transition_matrix = defaultdict(int)  # Using an int map to count occurrences
word_counts = defaultdict(int)

for n_gram in n_grams:
    word_counts[n_gram[0]] += 1
    transition_matrix[n_gram] += 1

# Step 5: Convert counts to probabilities
transition_probabilities = defaultdict(lambda: defaultdict(float))

for word in word_counts:
    total_count = word_counts[word]
    for next_word in transition_matrix:
        if next_word[0] == word:
            transition_probabilities[word][next_word[1]] = transition_matrix[(word, next_word[1])] / total_count

# Step 6: Intergrade the probabilities with additional factors
def intergrade_probabilities(word, transition_probabilities):
    """
    This function blends different probabilistic models, here we just use a simple blending approach.
    You can modify this to incorporate other dynamic elements, e.g., external knowledge or semantic weighting.
    """
    probabilities = transition_probabilities[word]
    
    # Adding some form of "external influence" (e.g., bias to favor certain words)
    # For example, adding a factor that makes some words more likely:
    for next_word in probabilities:
        probabilities[next_word] *= 1.1  # Adding 10% bias to each transition
    
    # Normalize probabilities
    total_prob = sum(probabilities.values())
    for next_word in probabilities:
        probabilities[next_word] /= total_prob

    return probabilities

# Step 7: Generate text based on probabilistic transition model
while True:
    generated_text = []

    # Step 8: Allow user input for seed text
    seed_text = input("USER: ").lower().split()  # User-provided seed text (converted to lowercase)

    # Ensure the seed text is in the filtered words list
    valid_seed = [word for word in seed_text if word in filtered_words]

    if valid_seed:
        # Use the last n-gram size words from the seed text
        current_sequence = valid_seed[-n_gram_size:]  # Take last n words of seed for n-gram size
    else:
        # If no valid seed word, use a random word from the filtered list
        current_sequence = [random.choice(filtered_words)]

    generated_text.extend(current_sequence)  # Add the seed text to the generated text

    # Generate the next words based on the transition probabilities
    for _ in range(generate_length - len(valid_seed)):  # Generate based on remaining length
        if len(current_sequence) == n_gram_size:  # Ensure we have enough words for an n-gram
            word = current_sequence[-1]
            next_word_probs = intergrade_probabilities(word, transition_probabilities)

            # Sample the next word based on the intergraded probabilities
            next_word = random.choices(
                list(next_word_probs.keys()), 
                weights=next_word_probs.values(), 
                k=1
            )[0]

            generated_text.append(next_word)
            current_sequence.append(next_word)  # Add the new word to the sequence
            current_sequence = current_sequence[-n_gram_size:]  # Keep only the last n-gram_size words
        else:
            break  # Stop if we don't have enough words to form a valid n-gram

    # Step 9: Output the generated text
    print("Generated Text:")
    print(' '.join(generated_text))
