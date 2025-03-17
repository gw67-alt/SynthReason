import numpy as np
import re
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from collections import defaultdict
import random

KB_limit = 9999 # -1 for unlimited
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

# Step 2: Tokenization (convert words to integer indices)
word_to_index = {word: idx + 1 for idx, word in enumerate(set(filtered_words))}
index_to_word = {idx + 1: word for idx, word in enumerate(set(filtered_words))}

# Step 3: Prepare sequences for training (input-output pairs)
sequences = []
for i in range(len(filtered_words) - n_gram_size-5):
    sequences.append(filtered_words[i:i + n_gram_size + 1]+filtered_words[i+3:i])

# Convert word sequences to index sequences
sequences_idx = [[word_to_index[word] for word in sequence] for sequence in sequences]

X = [seq[:-1] for seq in sequences_idx]  # Input: all but last word
y = [seq[-1] for seq in sequences_idx]  # Output: last word

X = torch.tensor(X)
y = torch.tensor(y)

# Step 4: Create a Dataset class for DataLoader
class TextDataset(Dataset):
    def __init__(self, X, y):
        self.X = X
        self.y = y

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

dataset = TextDataset(X, y)
dataloader = DataLoader(dataset, batch_size=64, shuffle=True)

# Step 5: Define the LSTM model with Linear Include
class TextGenerator(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim, n_gram_size):
        super(TextGenerator, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, batch_first=True)
        
        # Linear Include - applies on LSTM output, not embedding directly
        self.linear_include = nn.Linear(hidden_dim, hidden_dim)
        
        self.fc = nn.Linear(hidden_dim, vocab_size)
    
    def forward(self, x):
        batch_size = x.size(0)
        
        # Get embeddings
        embedded = self.embedding(x)  # Shape: (batch_size, n_gram_size, embedding_dim)
        
        # Process with LSTM
        lstm_out, _ = self.lstm(embedded)  # Shape: (batch_size, n_gram_size, hidden_dim)
        
        # Take the last output: (batch_size, hidden_dim)
        lstm_out = lstm_out[:, -1, :]
        
        # Apply Linear Include to LSTM output
        lstm_out = self.linear_include(lstm_out)  # Shape: (batch_size, hidden_dim)
        
        # Final prediction
        output = self.fc(lstm_out)  # Shape: (batch_size, vocab_size)
        return output

# Step 6: Initialize the model, loss function, and optimizer
vocab_size = len(word_to_index) + 1  # Including padding token (index 0)
embedding_dim = 256
hidden_dim = 512
model = TextGenerator(vocab_size, embedding_dim, hidden_dim, n_gram_size)

# Loss function and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Step 7: Train the model
epochs = 10
for epoch in range(epochs):
    model.train()
    running_loss = 0.0
    for inputs, labels in dataloader:
        optimizer.zero_grad()
        
        # Forward pass
        outputs = model(inputs)
        
        # Calculate loss
        loss = criterion(outputs, labels)
        loss.backward()
        
        # Update weights
        optimizer.step()
        
        running_loss += loss.item()
    
    print(f'Epoch [{epoch+1}/{epochs}], Loss: {running_loss / len(dataloader)}')

# Step 8: Generate text based on the trained model and spatial vector
def generate_text(seed_text, generate_length, model, word_to_index, index_to_word, n_gram_size):
    model.eval()  # Set model to evaluation mode
    generated_text = []
    
    # Convert seed text to sequence of word indices
    words = seed_text.lower().split()
    current_sequence = []
    for word in words:
        if word in word_to_index:
            current_sequence.append(word_to_index[word])
        else:
            # Handle unknown words
            current_sequence.append(random.choice(list(word_to_index.values())))
    
    # Ensure we have exactly n_gram_size words in the sequence
    while len(current_sequence) < n_gram_size:
        current_sequence.append(random.choice(list(word_to_index.values())))
    
    # Keep only the last n_gram_size elements
    current_sequence = current_sequence[-n_gram_size:]
    
    for _ in range(generate_length):
        # Convert the current sequence to a tensor
        input_seq = torch.tensor([current_sequence], dtype=torch.long)
        
        # Predict the next word
        with torch.no_grad():
            output = model(input_seq)
        
        # Apply softmax to get probabilities
        probabilities = torch.nn.functional.softmax(output, dim=1)
        
        # Sample the next word index using torch.multinomial
        predicted_word_idx = torch.multinomial(probabilities[0], 1).item()
        
        # Convert to word
        if predicted_word_idx in index_to_word:
            predicted_word = index_to_word[predicted_word_idx]
            # Add the predicted word to the generated text
            generated_text.append(predicted_word)
            
            # Update the current sequence (shift and add the new word)
            current_sequence.append(predicted_word_idx)
            current_sequence = current_sequence[-n_gram_size:]  # Keep only the last n-gram_size words
        else:
            # Handle unexpected index
            continue
    
    return ' '.join(generated_text)

# Step 9: Get user input and generate text
while True:
    seed_text = input("USER: ")  # Get the seed text from the user
    generated_text = generate_text(seed_text, generate_length, model, word_to_index, index_to_word, n_gram_size)
    print("Generated Text:")
    print(generated_text)
