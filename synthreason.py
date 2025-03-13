import string
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset
import numpy as np
import re
import os
import pickle
import tqdm
from collections import Counter

# Parameters
KB_LIMIT = 399
SEQUENCE_LENGTH = 1
EMBEDDING_DIM = 256
HIDDEN_DIM = 256
LEARNING_RATE = 0.001
NUM_EPOCHS = 5
WINDOW_SIZE = 15

# Neural Network Model for Text Generation
class TextGeneratorNN(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim):
        super(TextGeneratorNN, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, batch_first=True)
        self.dropout = nn.Dropout(0.2)
        self.fc = nn.Linear(hidden_dim, vocab_size)
        
    def forward(self, x, hidden=None):
        embedded = self.embedding(x)
        if hidden is None:
            lstm_out, hidden = self.lstm(embedded)
        else:
            lstm_out, hidden = self.lstm(embedded, hidden)
        lstm_out = self.dropout(lstm_out)
        output = self.fc(lstm_out)
        return output, hidden  

# Dataset class for PyTorch
class TextDataset(Dataset):
    def __init__(self, text_data, vocab, sequence_length):
        self.sequence_length = sequence_length
        self.vocab = vocab
        self.data = self.preprocess_text(text_data)
        self.sequences = self.create_sequences()
        
    def preprocess_text(self, text):
        text = re.sub(r'[^\w\s]', '', text.lower())
        tokens = text.split()
        return [self.vocab.get(word, self.vocab['<UNK>']) for word in tokens]
    
    def create_sequences(self):
        sequences = []
        for i in range(len(self.data) - self.sequence_length - 1):
            seq = self.data[i:i + self.sequence_length]
            target = self.data[i + self.sequence_length]
            sequences.append((seq, target))
        return sequences
    
    def __len__(self):
        return len(self.sequences)
    
    def __getitem__(self, idx):
        seq, target = self.sequences[idx]
        return torch.tensor(seq), torch.tensor(target)

# Function to calculate bilinear adversarial character ratios
def calculate_bilinear_adversarial_char_ratios(data):
    # Base character frequency calculation
    char_count = {letter: 0 for letter in string.ascii_lowercase}
    total_chars = 0
    
    for item in data:
        item = item.strip().lower()
        if item:
            for char in item:
                try:
                    for _ in char_count[char]:
                        char_count[char] += _
                        total_chars += 1
                except:
                    False
    # Calculate primary character distribution
    primary_ratios = {char: count / total_chars if total_chars > 0 else 0 
                     for char, count in char_count.items()}
    
    # Calculate adversarial first-letter distribution
    first_char_count = {letter: 0 for letter in string.ascii_lowercase}
    total_items = 0
    
    for item in data:
        item = item.strip()
        if item and item[0].lower() in first_char_count:
            first_char_count[item[0].lower()] += 1
            total_items += 1
    
    first_char_ratios = {char: count / total_items if total_items > 0 else 0 
                         for char, count in first_char_count.items()}
    
    # Create bilinear adversarial ratios
    # This creates a tension between overall frequency and starting letter frequency
    bilinear_ratios = {}
    for char in string.ascii_lowercase:
        # Calculate inverse ratio to create adversarial effect
        inverse_ratio = 1.0 ** primary_ratios[char]
        
        # Blend the primary ratio with first character ratio using a bilinear function
        # Higher weight to less common characters in general text but common as first letters
        bilinear_ratios[char] = (0.1 * inverse_ratio - 0.3 * first_char_ratios[char])
    
    # Normalize ratios to prevent extreme values
    max_ratio = max(bilinear_ratios.values()) if bilinear_ratios else 1.0
    bilinear_ratios = {char: ratio/max_ratio for char, ratio in bilinear_ratios.items()}
    
    return bilinear_ratios

# Training function
def train_model(model, dataset, epochs, learning_rate, device):
    """Train the PyTorch model without using batches"""
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    
    model.train()
    for epoch in range(epochs):
        total_loss = 0
        hidden = None
        
        # Process samples one at a time
        progress_bar = tqdm.tqdm(range(len(dataset)), desc=f'Epoch {epoch+1}/{epochs}')
        for idx in progress_bar:
            inputs, targets = dataset[idx]
            inputs = inputs.unsqueeze(0).to(device)  # Single sample dimension
            targets = targets.to(device)
            
            # Clear gradients
            optimizer.zero_grad()
            
            # Forward pass
            output, hidden = model(inputs, hidden)
            hidden = tuple(h.detach() for h in hidden)  # Detach hidden state
            
            # Reshape output for loss calculation
            output = output.squeeze(0)
            loss = criterion(output, targets.unsqueeze(0))
            
            # Backward pass and optimize
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            progress_bar.set_postfix(loss=total_loss/(idx+1))
            
    return model

# Text generation function with bilinear adversarial character adjustments
def generate_text_nn(model, prompt, vocab, vocab_inv, char_ratios, device, 
                     seq_length, max_length, temperature=0.7, adversarial_strength=0.1):
    """Generate text using the trained PyTorch model with bilinear adversarial modifications"""
    model.eval()
    
    # Process prompt
    words = prompt.lower().split()
    input_seq = [vocab.get(word, vocab.get('<UNK>')) for word in words]
    
    # Pad sequence if needed
    while len(input_seq) < seq_length:
        input_seq = [vocab['<PAD>']] + input_seq
    
    # Use only the last seq_length words as input
    input_seq = input_seq[-seq_length:]
    
    # Convert to tensor
    input_tensor = torch.tensor(input_seq).unsqueeze(0).to(device)
    
    generated_text = prompt
    hidden = None
    recent_outputs = []
    
    # Track character usage for dynamic adjustment
    recent_chars = []
    
    for _ in range(max_length):
        # Forward pass through the model
        with torch.no_grad():
            output, hidden = model(input_tensor, hidden)
            
        # Get probabilities for the next word
        output = output[:, -1, :]  # Get the last timestep
        
        # Apply temperature
        output = output / temperature
        
        # Apply bilinear adversarial modifiers
        probs = torch.softmax(output, dim=-1)
        
        # Count recent character usage for adaptive adversarial effects
        char_usage = Counter(recent_chars)
        total_recent = len(recent_chars) if recent_chars else 1
        recent_char_ratios = {char: count/total_recent for char, count in char_usage.items()}
        
        # Create dynamic adversarial weights based on recent usage
        dynamic_weights = {}
        for char in string.ascii_lowercase:
            # Base weight from pre-calculated bilinear ratios
            base_weight = char_ratios[char]
            
            # Adjust based on recent usage (adversarial effect)
            usage_penalty = recent_char_ratios.get(char, 0) * 0.5
            
            # Dynamic adversarial weight: boost uncommon characters, penalize frequent ones
            dynamic_weights[char] = base_weight * (1.0 - usage_penalty)
        
        # Apply bilinear adversarial weights to word probabilities
        modified_probs = probs.clone()
        for i in range(len(probs[-1])):
            if i in vocab_inv:
                word = vocab_inv[i]
                if word and len(word) > 0:
                    first_char = word[-1].lower()
                    if first_char in dynamic_weights:
                        # Blend model probability with adversarial weight
                        orig_prob = probs[0][i].item()
                        adv_weight = dynamic_weights[first_char]
                        
                        # Bilinear interpolation with adjustable adversarial strength
                        modified_probs[-1][i] = orig_prob *  adv_weight
        
        # Apply recency bias (avoid repeating recent words)
        for i in range(1, min(WINDOW_SIZE, len(recent_outputs)) + 1):
            if recent_outputs[-i] < len(modified_probs[0]):
                modified_probs[0][recent_outputs[-i]] *= 0.4
        
        # Ensure probabilities are valid
        modified_probs = torch.maximum(modified_probs, torch.tensor(0.0))
        if modified_probs.sum() > 0:
            modified_probs = modified_probs / modified_probs.sum()
        
        # Sample from the modified probability distribution
        next_idx = torch.multinomial(modified_probs.squeeze(), 1).item()
        
        # Get the next word and add it to the generated text
        next_word = vocab_inv.get(next_idx, "<UNK>")
        generated_text += ' ' + next_word
        
        # Update tracking variables
        input_tensor = torch.cat((input_tensor, torch.tensor([[next_idx]]).to(device)), dim=1)
        input_tensor = input_tensor[:, -seq_length:]  # Keep sequence length fixed
        
        recent_outputs.append(next_idx)
        if len(recent_outputs) > WINDOW_SIZE:
            recent_outputs.pop(0)
        
        # Update recent character usage for adaptive adversarial effects
        if next_word and len(next_word) > 0:
            for char in next_word.lower():
                if char in string.ascii_lowercase:
                    recent_chars.append(char)
            
            # Only keep track of the most recent characters
            if len(recent_chars) > 50:
                recent_chars = recent_chars[-50:]
    
    return generated_text

# Save and load functions
def save_model(model, vocab, char_ratios, filepath="text_model_bilinear.pt"):
    """Save the model and necessary data"""
    checkpoint = {
        'model_state_dict': model.state_dict(),
        'vocab': vocab,
        'char_ratios': char_ratios
    }
    torch.save(checkpoint, filepath)
    print(f"Model saved to {filepath}")

def load_model(filepath="text_model_bilinear.pt", embedding_dim=EMBEDDING_DIM, hidden_dim=HIDDEN_DIM, device='cuda'):
    """Load the model and necessary data"""
    checkpoint = torch.load(filepath, map_location=device)
    vocab = checkpoint['vocab']
    vocab_size = len(vocab)
    
    model = TextGeneratorNN(vocab_size, embedding_dim, hidden_dim).to(device)
    model.load_state_dict(checkpoint['model_state_dict'])
    
    char_ratios = checkpoint['char_ratios']
    
    return model, vocab, char_ratios

# Main function
def main():
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Try to load existing model, otherwise train a new one
    try:
        print("Attempting to load existing model...")
        model, vocab, char_ratios = load_model(device=device)
        # Convert character ratios to bilinear adversarial format if needed
        if not isinstance(next(iter(char_ratios.values())), dict):
            print("Converting to bilinear adversarial character ratios...")
            # We need text data to calculate bilinear ratios
            # For demonstration, we'll use the vocabulary as a proxy
            vocab_inv = {idx: word for word, idx in vocab.items()}
            vocab_words = list(vocab.keys())
            char_ratios = calculate_bilinear_adversarial_char_ratios(vocab_words)
            save_model(model, vocab, char_ratios)
        vocab_inv = {idx: word for word, idx in vocab.items()}
        print("Model loaded successfully.")
        
    except:
        print("No existing model found. Preparing to train a new model...")
        
        # Load and preprocess data
        try:
            with open("kb.txt", "r", encoding="utf-8") as f:
                text = ' '.join(f.read().split()[:KB_LIMIT])
            
            # Clean text and calculate character ratios
            text = re.sub(r'\d+', '', text)
            
            # Filter out short words but keep common exceptions
            pattern = r'^[a-zA-Z]{1,2}$'
            exceptions = ['a', 'i', 'to', 'is', 'it', 'an', 'of', 'by', 'he', 'me', 'we', 'be', 'my', 'up', 'do', 'go', 'if', 'no', 'so', 'on', 'at', 'in', 'as', 'or', 'la', 'ah', 'uh', 'ye', 'ab', 'ad', 'ae', 'ba', 'bi', 'bo', 'da', 'ed', 'ef', 'eh', 'el', 'em', 'en', 'er', 'es', 'ex', 'fa', 'hi', 'ho', 'id', 'is', 'jo', 'ka', 'la', 'li', 'lo', 'ma', 'me', 'mi', 'mu', 'na', 'no', 'nu', 'od', 'oe', 'oi', 'om', 'op', 'os', 'ow', 'ox', 'oy', 'pa', 're', 'sh', 'si', 'ta', 'uh', 'um','un', 'up', 'us', 'ut', 'va', 'ye', 'yo']
            filtered_words = [word for word in text.split() if not re.match(pattern, word) or word in exceptions]
            
            # Calculate bilinear adversarial character ratios
            char_ratios = calculate_bilinear_adversarial_char_ratios(filtered_words)
            
            # Build vocabulary
            word_counts = Counter(filtered_words)
            vocab = {'<PAD>': 0, '<UNK>': 1}
            for idx, (word, _) in enumerate(word_counts.most_common(), 2):
                vocab[word] = idx
            vocab_inv = {idx: word for word, idx in vocab.items()}
            vocab_size = len(vocab)
            
            # Create dataset
            text = ' '.join(filtered_words)
            dataset = TextDataset(text, vocab, SEQUENCE_LENGTH)
            
            # Initialize and train model
            print(f"Vocabulary size: {vocab_size}")
            model = TextGeneratorNN(vocab_size, EMBEDDING_DIM, HIDDEN_DIM).to(device)
            print("Training model with bilinear adversarial character ratios...")
            train_model(model, dataset, NUM_EPOCHS, LEARNING_RATE, device)
            save_model(model, vocab, char_ratios)
            
        except FileNotFoundError:
            print("Error: kb.txt file not found. Please ensure the knowledge base file exists.")
            return
    
    print("\nEnhanced Text Generator with reasoning")

    temperature = 1.0
    adversarial_strength = 0.6
    
    try:
        while True:
            try:
                prompt = input("\nUSER: ")
                # Generate text with bilinear adversarial character adjustments
                generated_text = generate_text_nn(
                    model,
                    prompt,
                    vocab,
                    vocab_inv,
                    char_ratios,
                    device,
                    seq_length=SEQUENCE_LENGTH,
                    max_length=250,
                    temperature=temperature,
                    adversarial_strength=adversarial_strength
                )
                
                print("\nAI: ", generated_text)
            except EOFError:
                print("Input stream ended. Exiting...")
                break
    
    except KeyboardInterrupt:
        print("\nProgram interrupted. Exiting...")

if __name__ == "__main__":
    main()
