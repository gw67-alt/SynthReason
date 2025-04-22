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
import math  # Added missing math import

# Constants
KB_MEMORY_UNCOMPRESSED = 10000
n = 4  # Use quadgrams for training
num_epochs = 10
generate_length = 1000
temperature = 0.3
feedforward_enhancer = KB_MEMORY_UNCOMPRESSED

# Preprocessing and Vocabulary
def preprocess_text(text):
    """Clean and tokenize text."""
    cleaned_text = re.sub(r'[^a-zA-Z\s]', '', text)
    tokens = text.split()[:KB_MEMORY_UNCOMPRESSED]
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

# Fixed CustomLossFunction
class CustomLossFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, outputs, targets):
        """
        Forward pass for custom loss.
        Save variables for backward computation using `ctx.save_for_backward`.
        """
        ctx.save_for_backward(outputs, targets)
        # Fixed loss calculation (removed undefined grad_output)
        loss = (outputs.min() * targets.min()) / (outputs.min() * targets.mean()) + \
              (-outputs.mean() * targets)*(-outputs / outputs.min() * targets.max()) / \
              (outputs.min() * -outputs.std() / (1 + targets.std()))
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

        # Applying the Quotient Rule
        numerator = u_grad * v - u * v_grad
        denominator = v ** 2
        quotient_rule_term = numerator / denominator

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
        self.n = n  # Store n as an instance variable for use in generate_text

    def forward(self, x):
        x = self.embedding(x)
        lstm_out, _ = self.lstm(x)
        return self.fc(self.dropout(lstm_out[:, -1, :]))
        
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
            # This works because inputs are typically token indices
            batch_size, seq_len = inputs.shape
            vocab_size = model.fc.weight.shape[0]  # Fixed: removed output_heads reference
            
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
            
            # Get model outputs
            outputs = model(inputs)  # Fixed: no longer handle as list
            
            # Calculate loss
            loss = criterion(outputs, targets)
            
            
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
    model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))  # Fixed: added map_location
    model.eval()
    print("Model and vocabulary loaded.")
    return model, word_to_index

def generate_text(model, word_to_index, input_text, sequence_length, generate_length, temperature):
    input_sequence = preprocess_text(input_text)
    indices = [word_to_index.get(word, -1) for word in input_sequence if word in word_to_index]
    
    # Ensure we have valid word indices
    indices = [idx for idx in indices if idx != -1]

    if not indices:
        return "Input text contains no recognizable words."

    # Initialize generated text list
    generated_text = []
    
    # Pad sequence if needed
    if len(indices) < sequence_length:
        # Pad with zeros if input is shorter than sequence_length
        padding = [0] * (sequence_length - len(indices))
        indices = padding + indices
    
    # Take the last sequence_length tokens for input
    input_tensor = torch.tensor(indices[-sequence_length:], dtype=torch.long).unsqueeze(0)
    reverse_vocab = {i: word for word, i in word_to_index.items()}
    
    # Fixed: Create context from the sequence instead of using self.n
    context = tuple(reverse_vocab.get(idx, "<UNK>") for idx in indices[-n:]) if indices else tuple()
    
    # Generate text
    for _ in range(generate_length):
        with torch.no_grad():
            output = model(input_tensor)
            probabilities = torch.softmax(output / temperature, dim=1).squeeze()
            
            # Add structure to probabilities
            # Boost probabilities for specific tokens
            boost_indices = []
            for word_idx in generated_text:
                if 0 <= word_idx < len(probabilities):
                    boost_indices.append(word_idx)
                    
            penalties = torch.ones_like(probabilities)
            for idx in boost_indices:
                penalties[idx] = 1.5  # Boost specific tokens
                
            structured_probs = probabilities * penalties
            
            # Normalize probabilities after adding structure
            structured_probs = structured_probs / structured_probs.sum()
            
            # --- Start of Symbolic Probability Calculation (simplified version) ---
            # Extract words and their probabilities
            all_words = list(word_to_index.items())  # Fixed: words variable not defined
            all_probs = structured_probs.tolist()
            
            # Filter to keep words with probability above 50% of mean
            mean_prob = torch.mean(structured_probs).item()
            subset_indices = [i for i, p in enumerate(all_probs) if p > mean_prob * 0.5]
            
            # If nothing would remain, use all indices
            if not subset_indices:
                subset_indices = list(range(len(all_probs)))
            
            # Get subset of words and their probabilities
            subset_words = [all_words[i] for i in subset_indices if i < len(all_words)]
            subset_probs = [structured_probs[i].item() for i in subset_indices if i < len(structured_probs)]
            
            # Calculate tensor values (relationship between context and options)
            tensor_values = []
            for word_tuple in subset_words:
                word = word_tuple[0]  # Get the actual word from the (word, index) tuple
                tensor_value = 1.0
                for context_word in context:
                    # Calculate overlap between words (as sets of characters)
                    word_chars = set(word)
                    context_chars = set(context_word)
                    overlap = len(word_chars.intersection(context_chars))
                    tensor_value *= (float(overlap) + 1.0)
                tensor_values.append(tensor_value)
            
            # Check for high probability options
            max_prob = max(subset_probs) if subset_probs else 0
            exists_high_prob = any(p > max_prob * 0.8 for p in subset_probs) if max_prob > 0 else False
            
            # Combine context influence
            last_letters = [w[-1] if w else '' for w in context]
            unique_last_letters = len(set(last_letters))
            context_influence = math.pow(float(unique_last_letters) + 1.0, 3.5)
            
            # Calculate base distribution
            total_prob = sum(subset_probs)
            if total_prob == 0:
                base_distribution = [1.0 / len(subset_probs)] * len(subset_probs) if subset_probs else []
            else:
                base_distribution = [p / total_prob for p in subset_probs]
                
            # Apply weights based on word properties
            word_weights = []
            for word_tuple in subset_words:
                word = word_tuple[0]  # Get the actual word from the tuple
                length_factor = math.log(len(word) + 1.0)
                vowels = sum(1 for c in word if c.lower() in 'aeiou')
                consonants = len(word) - vowels
                vc_ratio = (float(vowels) + 0.1) / (float(consonants) + 0.1)
                first_letter_code = ord(word[0]) % 10 if word else 0
                word_weight = length_factor * (vc_ratio + 0.5) * (float(first_letter_code) + 1.0)
                word_weights.append(word_weight)
                
            # Calculate final adjusted weights
            adjusted_weights = []
            for i in range(len(subset_words)):
                if i >= len(base_distribution):
                    continue  # Safety check
                
                combined = base_distribution[i] * 5.0
                
                if i < len(tensor_values):
                    combined *= math.pow(tensor_values[i], 0.3)
                
                if i < len(word_weights):
                    combined *= word_weights[i] * 0.8
                
                combined *= math.pow(context_influence, 0.4)
                
                if exists_high_prob and base_distribution[i] < 0.3:
                    combined *= 1.5
                
                # Square the final value
                adjusted_weights.append(math.pow(max(0, combined), 2))
            
            # Normalize the final weights
            total_weight = sum(adjusted_weights)
            if total_weight == 0 or not adjusted_weights:
                # Fallback to uniform distribution
                normalized_weights = [1.0 / len(subset_indices)] * len(subset_indices) if subset_indices else []
                # Convert to tensor
                weight_tensor = torch.tensor(normalized_weights)
            else:
                normalized_weights = [w / total_weight for w in adjusted_weights]
                # Convert to tensor
                weight_tensor = torch.tensor(normalized_weights)
            
            # Sample the next word using the normalized weights
            if len(weight_tensor) > 0:
                next_word_idx_in_subset = torch.multinomial(weight_tensor, 1).item() if len(weight_tensor) > 0 else 0
                if next_word_idx_in_subset < len(subset_indices):
                    next_word_idx = subset_indices[next_word_idx_in_subset]
                else:
                    # Fallback if index is out of range
                    next_word_idx = random.choice(subset_indices) if subset_indices else 0
            else:
                # Fallback if weight tensor is empty
                next_word_idx = random.choice(range(len(probabilities))) if len(probabilities) > 0 else 0
            
            # Add the selected word to the generated text
            generated_text.append(next_word_idx)
            
            # Update input tensor for next step
            input_tensor = torch.cat(
                (input_tensor[:, 1:], torch.tensor([[next_word_idx]], dtype=torch.long)),
                dim=1
            )
    
    # Convert indices to words and join them
    return ' '.join([reverse_vocab.get(idx, "<UNK>") for idx in generated_text])


# Main Function
def main():
    choice = input("Do you want to (1) train or (2) load a model: ")

    if choice == '1':
        try:
            with open("test.txt", encoding="utf-8") as f:
                text = f.read().lower()
            
            print(f"Loaded text with {len(text)} characters")
            word_to_index, vocab_size = build_vocabulary(text)
            print(f"Built vocabulary with {vocab_size} unique words")
            
            sequences = create_sequences(word_to_index, preprocess_text(text), sequence_length=n)
            print(f"Created {len(sequences)} training sequences")
            
            dataset = TextDataset(sequences)
            data_loader = DataLoader(dataset, batch_size=256, shuffle=True)
            
            model = KnowledgeAugmentedLSTM(vocab_size)
            print("Starting training...")
            
            train_model(model, data_loader, num_epochs=num_epochs)
            save_model_and_vocab(model, word_to_index)
        except Exception as e:
            print(f"Error during training: {e}")
            return
    elif choice == '2':
        try:
            model, word_to_index = load_model_and_vocab()
        except Exception as e:
            print(f"Error loading model: {e}")
            return
    else:
        print("Invalid option.")
        return

    while True:
        try:
            user_input = input("\nEnter text (or 'quit' to exit): ")
            if user_input.lower() == 'quit':
                break
                
            print("\nGenerating text...")
            generated = generate_text(
                model, 
                word_to_index, 
                user_input, 
                sequence_length=n, 
                generate_length=generate_length, 
                temperature=temperature
            )
            print(f"\nAI: {generated}")
        except KeyboardInterrupt:
            print("\nExiting...")
            break
        except Exception as e:
            print(f"Error: {e}")


if __name__ == "__main__":
    main()