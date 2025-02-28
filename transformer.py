
#Neural network 2.0 - George W - 22-02-2025
import numpy as np
import pickle
import re
import random
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
from collections import Counter
from textblob import TextBlob

# Constants
KB_MEMORY_UNCOMPRESSED = 50000 # use -1 for unlimited
n = 4  # Use quadgrams for training
num_epochs = 5
generate_length = 100
temperature = 0.7
feedforward_enhancer = KB_MEMORY_UNCOMPRESSED




def analyze_sentiment(text):
    # Create a TextBlob object
    blob = TextBlob(text)
    
    # Get the sentiment analysis
    sentiment = blob.sentiment

    return sentiment.subjectivity
    
    
# Preprocessing and Vocabulary
def preprocess_text(text):
    """Clean and tokenize text."""
    cleaned_text = re.sub(r'[^a-zA-Z\s]', '', text)
    tokens = cleaned_text.split()[:KB_MEMORY_UNCOMPRESSED]
    return [word for word in tokens if len(word) > 1 or word in {"i", "a"}]

def build_vocabulary(text_data):
    """Build vocabulary with word frequencies."""
    tokens = preprocess_text(text_data)
    word_counts = {word: tokens.count(word) for word in set(tokens)}
    if tokens:  # Ensure the tokens list is not empty
        last_word = tokens[-1]
        word_counts[last_word] += feedforward_enhancer

    vocab = sorted(word_counts, key=word_counts.get, reverse=True)
    word_to_index = {word: i for i, word in enumerate(vocab)}
    return word_to_index, len(vocab)

def create_sequences(word_to_index, text, sequence_length):
    """Convert text into sequences."""
    # Encode the text using the word-to-index mapping
    encoded = [word_to_index[word] for word in text if word in word_to_index]
    
    # Create sequences of the specified length
    return [(encoded[i-sequence_length:i], encoded[i]) for i in range(sequence_length, len(encoded))]

# Dataset Class
class TextDataset(Dataset):
    def __init__(self, sequences):
        self.sequences = sequences

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        seq, target = self.sequences[idx]
        return torch.tensor(seq, dtype=torch.long), torch.tensor(target, dtype=torch.long)

# Knowledge-Augmented LSTM Model
class KANEmbedding(nn.Module):
    def __init__(self, vocab_size, embedding_dim, knowledge_dim):
        super().__init__()
        self.word_embedding = nn.Embedding(vocab_size, embedding_dim)
        self.knowledge_embedding = nn.Embedding(vocab_size, knowledge_dim)

    def forward(self, x):
        return torch.cat((self.word_embedding(x), self.knowledge_embedding(x)), dim=-1)
import torch
from torch.autograd import Function

class CustomLossFunction(Function):
    @staticmethod
    def forward(ctx, outputs, targets):
        """
        Forward pass for custom loss.
        Save variables for backward computation using `ctx.save_for_backward`.
        """
        ctx.save_for_backward(outputs, targets)
        # Example: Custom loss calculation (e.g., mean squared error)
        loss = (outputs.min() * targets.min() / -grad_output) / (outputs.min() * targets.mean()) + (-outputs.mean() * targets)*(-outputs / outputs.min() * targets.max()) / (outputs.min() * -outputs.std() / (1 + targets.std()))
        return loss

    @staticmethod
    def backward(ctx, grad_output):
        """
        Backward pass for custom gradients.
        Use saved variables to compute the gradient of the loss with respect to inputs.
        """
        outputs, targets = ctx.saved_tensors

        def grad(x):
            # Dummy gradient function for demonstration purposes
            return grad_output * x  # Replace with actual gradient logic

        # Original terms
        outputs_min = outputs.min()
        targets_min = targets.min()
        grad_output_min = -grad_output.min()
        targets_mean = targets.mean()

        # u(x) and v(x) definitions
        u = (outputs_min * targets_min) / grad_output_min
        v = outputs_min * targets_mean

        # Gradients of u(x) and v(x)
        u_grad = (grad(outputs_min) * targets_min + outputs_min * grad(targets_min)) / grad_output_min

        # Adding linspace to v_grad
        linspace_term = torch.linspace(0, targets.max(), steps=1, device=outputs.device).mean()
        v_grad = grad(outputs_min) * targets_mean + outputs_min * -grad(targets_mean) + linspace_term
        norm_dist = linspace_term / (linspace_term.sum() + 1e-10)
        entropy = -torch.sum(norm_dist * torch.log(norm_dist + 1e-10))
        # Applying the Quotient Rule
        numerator = u_grad * v - u * v_grad
        denominator = v ** 2
        quotient_rule_term = numerator / entropy

        # Final expression for grad_outputs
        grad_outputs = quotient_rule_term + (
            # Add other terms from your formula
            (-outputs.mean() * targets) 
            * (-outputs / outputs_min * targets.max())
            / (outputs_min * -grad_output.mean())
        )
        return grad_outputs, None  # Return gradients for all inputs to forward

class KnowledgeAugmentedLSTM(nn.Module):
    def __init__(self, vocab_size, embedding_dim=1250, knowledge_dim=200, rnn_units=386, dropout_rate=0.7):
        super().__init__()
        self.embedding = KANEmbedding(vocab_size, embedding_dim, knowledge_dim)
        self.lstm = nn.LSTM(embedding_dim + knowledge_dim, rnn_units, batch_first=True)
        self.fc = nn.Linear(rnn_units, vocab_size)
        self.dropout = nn.Dropout(dropout_rate)

    def forward(self, x):
        x = self.embedding(x)
        lstm_out, _ = self.lstm(x)
        return self.fc(self.dropout(lstm_out[:, -1, :]))
        
import torch.nn.functional as F

import torch.nn.functional as F

def train_model(model, data_loader, num_epochs, lr=0.001):
    optimizer = optim.Adam(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()
    model.train()
    
    # Keep track of cumulative sum of inputs for analysis
    cumulative_inputs = None
    batch_count = 0
    
    for epoch in range(num_epochs):
        epoch_loss = 0
        
        for inputs, targets in tqdm(data_loader, desc=f"Epoch {epoch+1}/{num_epochs}"):
            # Process and track cumulative sum of inputs
            batch_count += 1
            
            # Convert input indices to one-hot for meaningful cumulative sum
            batch_size, seq_len = inputs.shape
            vocab_size = model.output_heads[0].weight.shape[0] if hasattr(model, 'output_heads') else model.fc.weight.shape[0]
            
            # Create one-hot representation
            inputs_one_hot = torch.zeros(batch_size, seq_len, vocab_size, device=inputs.device)
            for b in range(batch_size):
                for s in range(seq_len):
                    if inputs[b, s] < vocab_size:  # Ensure index is valid
                        inputs_one_hot[b, s, inputs[b, s]] = 1
            
            # Flatten for cumulative sum
            inputs_flat = inputs_one_hot.sum(dim=0).sum(dim=0)  # Sum across batch and sequence dimensions
            
            # Update cumulative sum
            if cumulative_inputs is None:
                cumulative_inputs = inputs_flat
            else:
                cumulative_inputs += inputs_flat
            
            # Zero gradients at the start of each batch
            optimizer.zero_grad()
            
            # Get model outputs (for multi-target, this returns a list of outputs)
            outputs_list = model(inputs)
            
            # Initialize total loss
            loss = 0
            
            # Single-target case (backward compatibility)
            # Ensure batch sizes match
            assert outputs_list.shape[0] == targets.shape[0], "Batch sizes do not match."
            
            # Calculate loss with variance
            # Convert targets to float for variance calculation
            targets_float = inputs_one_hot.float()
            
            # Calculate standard cross-entropy loss
            
            # Add variance term
            # Using unbiased=False to match numpy's default behavior
            
            # Get model output first as it has gradients
            outputs = model(inputs)
            
            # Apply cumulative input regularization for single target case
            # Note: We'll use the model's outputs for backpropagation, not the topk values
            topk_values, topk_indices = torch.topk(cumulative_inputs, min(256, len(cumulative_inputs)))
            
            # Set outputs_list to be topk_values as requested, but just for inference
            # We don't backpropagate through this
            outputs_list = topk_indices
            
            # Since we're using the model's outputs for gradient calculation
            loss = criterion(outputs, targets)
            
            # We just use the topk_values/outputs_list for analysis, not for gradient computation
            
            # Backpropagate the loss
            loss.backward()
            
            # Update model parameters
            optimizer.step()
            
            # Track total loss for this epoch
            epoch_loss += loss.item()
        
        # Print epoch summary
        print(f"Epoch {epoch+1}, Loss: {epoch_loss:.4f}")
    
    return model
# Save and Load Functions
def save_model_and_vocab(model, word_to_index):
    torch.save(model.state_dict(), 'knowledge_augmented_lstm.mdl')
    with open('vocab.pkl', 'wb') as f:
        pickle.dump(word_to_index, f)
    print("Model and vocabulary saved.")

def load_model_and_vocab(vocab_path='vocab.pkl', model_path='knowledge_augmented_lstm.mdl'):
    with open(vocab_path, 'rb') as f:
        word_to_index = pickle.load(f)
    vocab_size = len(word_to_index)
    model = KnowledgeAugmentedLSTM(vocab_size)
    model.load_state_dict(torch.load(model_path, weights_only= True))
    model.eval()
    print("Model and vocabulary loaded.")
    return model, word_to_index
def generate_text(model, word_to_index, input_text, sequence_length, generate_length, temperature):
    input_sequence = preprocess_text(input_text)
    indices = [word_to_index.get(word, -1) for word in input_sequence if word in word_to_index]

    if not indices:
        return "Input text contains no recognizable words."

    input_tensor = torch.tensor(indices[-sequence_length:], dtype=torch.long).unsqueeze(0)
    reverse_vocab = {i: word for word, i in word_to_index.items()}
    
    generated_text = []
    
    for _ in range(generate_length):
        with torch.no_grad():
            output = model(input_tensor)
            probabilities = torch.softmax(output / temperature, dim=1).squeeze()

            # Add structure to probabilities
            boost_indices = torch.tensor(indices, dtype=torch.long)
            boost_indices = boost_indices[(boost_indices >= 0) & (boost_indices < probabilities.size(0))]

            penalties = torch.ones_like(probabilities)
            subjectivity = analyze_sentiment(' '.join([reverse_vocab.get(idx, "<UNK>") for idx in generated_text]))
            if subjectivity > 0:         
                penalties[boost_indices] = 1.5  # Boost specific tokens
            elif subjectivity < 0:
                penalties[boost_indices] = 0.5  # Penalize specific tokens

            structured_probs = probabilities * penalties

            # Check for invalid values and normalize probabilities
            if torch.any(torch.isnan(structured_probs)) or torch.any(structured_probs < 0):
                return "Error: Invalid probability values encountered."

            structured_probs = structured_probs / structured_probs.sum()

            next_word_idx = torch.multinomial(structured_probs, 1).item()
            generated_text.append(next_word_idx)

            # Update input tensor for next step
            input_tensor = torch.cat(
                (input_tensor[:, 1:], torch.tensor([[next_word_idx]], dtype=torch.long)),
                dim=-1
            )
    print(' '.join([reverse_vocab.get(idx, "<UNK>") for idx in generated_text]))
    print()
    return 


def generate_instruction(model, word_to_index, input_text, sequence_length, generate_length, temperature):
    input_sequence = preprocess_text(input_text)
    indices = [word_to_index.get(word, -1) for word in input_sequence if word in word_to_index]

    if not indices:
        return "Input text contains no recognizable words."

    generated_text = []
    input_tensor = torch.tensor(indices[-sequence_length:], dtype=torch.long).unsqueeze(0)
    reverse_vocab = {i: word for word, i in word_to_index.items()}

    for _ in range(generate_length):
        with torch.no_grad():
            output = model(input_tensor)
            probabilities = torch.softmax(output / temperature, dim=1).squeeze()

            # Add structure to probabilities
            # Boost probabilities for specific tokens
            boost_indices = [word_to_index[word] for word in generated_text if word in word_to_index]
            penalties = torch.ones_like(probabilities)
            penalties[boost_indices] = 1.5  # Boost specific tokens
            structured_probs = probabilities*torch.linspace(0,len(probabilities), steps=1)
            # Normalize probabilities after adding structure
            structured_probs = structured_probs / structured_probs.sum()

    return structured_probs
    
    
class VocabSelectionNetwork(nn.Module):
    def __init__(self, input_size=768, hidden_size=256):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_size // 2, 1),
            nn.Sigmoid()
        )
        
    def forward(self, x):
        return self.network(x)

class VocabSelector:
    def __init__(self, min_freq=2, max_vocab_size=10000):
        self.min_freq = min_freq
        self.max_vocab_size = max_vocab_size
        self.vocab_nn = VocabSelectionNetwork()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.vocab_nn.to(self.device)
        
    def get_word_features(self, word, word_count, total_words, word_contexts):
        """Extract features for a word."""
        # Word length features
        length = len(word)
        norm_length = length / 20  # Normalize length
        
        # Frequency features
        freq = word_count / total_words
        log_freq = np.log1p(word_count)
        norm_log_freq = log_freq / np.log1p(total_words)
        
        # Context features
        context_diversity = len(word_contexts.get(word, set())) / total_words
        
        # Character-level features
        vowels = sum(1 for c in word if c in 'aeiou')
        consonants = length - vowels
        vowel_ratio = vowels / length if length > 0 else 0
        
        # Position features
        positions = word_contexts.get(word, set())
        avg_position = np.mean(list(positions)) if positions else 0
        norm_position = avg_position / total_words if total_words > 0 else 0
        
        # Combine all features
        features = torch.tensor([
            norm_length,
            freq,
            norm_log_freq,
            context_diversity,
            vowel_ratio,
            norm_position,
            consonants / length 
        ], dtype=torch.float32)
        
        # Pad to match input size
        padded = torch.zeros(768, dtype=torch.float32)
        padded[:features.shape[0]] = features
        return padded

    def build_training_data(self, text):
        """Build training data for vocabulary selection."""
        words = preprocess_text(text)
        word_counts = Counter(words)
        total_words = len(words)
        
        # Build context information
        word_contexts = {}
        for i, word in enumerate(words):
            if word not in word_contexts:
                word_contexts[word] = set()
            word_contexts[word].add(i)
        
        # Create training examples
        X = []
        y = []
        
        # Sort words by frequency for balanced sampling
        sorted_words = sorted(word_counts.items(), key=lambda x: x[1], reverse=True)
        
        # Select positive and negative examples
        top_words = set(word for word, count in sorted_words[:self.max_vocab_size])
        
        for word, count in sorted_words:
            if count >= self.min_freq:
                features = self.get_word_features(word, count, total_words, word_contexts)
                X.append(features)
                y.append(1.0 if word in top_words else 0.0)
        
        return torch.stack(X), torch.tensor(y, dtype=torch.float32)

    def train(self, text, epochs=5):
        """Train the vocabulary selection network."""
        X, y = self.build_training_data(text)
        X = X.to(self.device)
        y = y.to(self.device)
        
        optimizer = optim.Adam(self.vocab_nn.parameters())
        criterion = nn.BCELoss()
        
        self.vocab_nn.train()
        for epoch in range(epochs):
            optimizer.zero_grad()
            output = self.vocab_nn(X).squeeze()
            loss = criterion(output, y)
            loss.backward()
            optimizer.step()
            
            print(f"Epoch {epoch + 1}, Loss: {loss.item():.4f}")

    def select_vocabulary(self, text):
        """Select vocabulary using the trained neural network."""
        words = preprocess_text(text)
        word_counts = Counter(words)
        total_words = len(words)
        
        # Build context information
        word_contexts = {}
        for i, word in enumerate(words):
            if word not in word_contexts:
                word_contexts[word] = set()
            word_contexts[word].add(i)
        
        # Get predictions for all words
        word_scores = {}
        self.vocab_nn.eval()
        with torch.no_grad():
            for word, count in word_counts.items():
                if count >= self.min_freq:
                    features = self.get_word_features(word, count, total_words, word_contexts)
                    features = features.to(self.device)
                    score = self.vocab_nn(features.unsqueeze(0)).item()
                    word_scores[word] = score
        
        # Select top words based on neural network scores
        selected_words = sorted(word_scores.items(), key=lambda x: x[1], reverse=True)
        selected_words = selected_words[:self.max_vocab_size]
        
        # Create word_to_index mapping
        word_to_index = {word: i for i, (word, _) in enumerate(selected_words)}
        
        return word_to_index, len(word_to_index)

def build_vocabulary_with_nn(text_data):
    """Build vocabulary using neural network selection."""
    vocab_selector = VocabSelector()
    print("Training vocabulary selection network...")
    vocab_selector.train(text_data)
    print("Selecting vocabulary...")
    word_to_index, vocab_size = vocab_selector.select_vocabulary(text_data)
    
    # Apply feedforward enhancement to last word
    if text_data:
        tokens = preprocess_text(text_data)
        if tokens:
            last_word = tokens[-1]
            if last_word in word_to_index:
                # Move last word to a better position in vocabulary
                current_index = word_to_index[last_word]
                enhanced_index = max(0, current_index - feedforward_enhancer)
                
                # Shift other words to accommodate the enhanced position
                for word in word_to_index:
                    if word_to_index[word] < current_index:
                        word_to_index[word] += 1
                word_to_index[last_word] = enhanced_index
    
    return word_to_index, vocab_size

# Main Function
def main():
    choice = input("Do you want to (1) train or (2) load a model: ")

    if choice == '1':
        print("Reading text file...")
        with open("test.txt", encoding="utf-8") as f:
            text = f.read().lower()

        print("Building vocabulary using neural network...")
        word_to_index, vocab_size = build_vocabulary_with_nn(text)
        
        print(f"Created vocabulary with {vocab_size} words")
        sequences = create_sequences(word_to_index, preprocess_text(text), sequence_length=4)
        dataset = TextDataset(sequences)
        data_loader = DataLoader(dataset, batch_size=256, shuffle=True)

        print("Training main model...")
        model = KnowledgeAugmentedLSTM(vocab_size)
        train_model(model, data_loader, num_epochs=num_epochs)
        save_model_and_vocab(model, word_to_index)
    elif choice == '2':
        model, word_to_index = load_model_and_vocab()
    else:
        print("Invalid option.")
        return

    while True:
        user_input = input("User: ")
        generate_text(model, word_to_index, user_input, 
                                 sequence_length=4, 
                                 generate_length=generate_length, 
                                 temperature=temperature)
if __name__ == "__main__":
    main()
