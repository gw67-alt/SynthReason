import numpy as np
import pickle
import re
import random
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

# Check for CUDA availability
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Constants
KB_MEMORY_UNCOMPRESSED = -1 # Use -1 for unlimited
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

class CustomLossFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, outputs, targets):
        """
        Forward pass for custom loss.
        Save variables for backward computation using `ctx.save_for_backward`.
        """
        ctx.save_for_backward(outputs, targets)
        # Example: Custom loss calculation
        loss = (outputs.min() * targets.min()) / (outputs.min() * targets.mean()) + (-outputs.mean() * targets) * (-outputs / outputs.min() * targets.max()) / (outputs.min() * -outputs.std() / (1 + targets.std()))
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

    def forward(self, x):
        x = self.embedding(x)
        lstm_out, _ = self.lstm(x)
        return self.fc(self.dropout(lstm_out[:, -1, :]))
        
def train_model(model, data_loader, num_epochs, lr=0.001):
    model = model.to(device)  # Move model to GPU if available
    optimizer = optim.Adam(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()
    model.train()
    
    # Keep track of cumulative sum of inputs for analysis
    cumulative_inputs = None
    batch_count = 0
    
    for epoch in range(num_epochs):
        epoch_loss = 0
        
        for inputs, targets in tqdm(data_loader, desc=f"Epoch {epoch+1}/{num_epochs}"):
            # Move data to GPU if available
            inputs = inputs.to(device)
            targets = targets.to(device)
            
            # Process and track cumulative sum of inputs
            batch_count += 1
            
            # Convert input indices to one-hot for meaningful cumulative sum
            # This works because inputs are typically token indices
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
            
            if isinstance(outputs_list, list):
                # Multi-target case: Calculate loss for each target position
                for i, output in enumerate(outputs_list):
                    # Get target for this position
                    target_i = targets[:, i]
                    
                    # Ensure batch sizes match
                    assert output.shape[0] == target_i.shape[0], f"Batch sizes do not match for position {i}."
                    
                    # Add loss for this position
                    position_loss = criterion(output, target_i)
                    loss += position_loss
                    
                    # Optional: Apply cumulative input information as a regularization term
                    if epoch > 0:  # Skip first epoch to gather statistics
                        # Get normalized cumulative distribution
                        cum_dist = cumulative_inputs / cumulative_inputs.sum()
                        
                        # Apply regularization based on cumulative distribution
                        # This encourages the model to consider token frequency
                        token_probs = torch.softmax(output, dim=1)
                        reg_strength = 0.01 * (1 - (epoch / num_epochs))  # Decay over time
                        reg_loss = reg_strength * torch.mean(
                            torch.sum(token_probs * torch.log(token_probs / (cum_dist + 1e-10) + 1e-10), dim=1)
                        )
                        loss += reg_loss
            else:
                # Single-target case (backward compatibility)
                # Ensure batch sizes match
                assert outputs_list.shape[0] == targets.shape[0], "Batch sizes do not match."
                
                # Calculate loss
                loss = criterion(outputs_list, targets)
                
                # Apply cumulative input regularization for single target case
                if epoch > 0 and cumulative_inputs is not None:
                    topk_values, topk_indices = torch.topk(cumulative_inputs, 10)
                    for i, (idx, count) in enumerate(zip(topk_indices.tolist(), topk_values.tolist())):
                        norm_dist = cumulative_inputs / cumulative_inputs.sum()
                        entropy = -torch.sum(norm_dist * torch.exp(norm_dist + 1e-10))
                        loss += count*entropy
            
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
    # Save model to CPU to ensure compatibility
    model_cpu = model.to('cpu')
    torch.save(model_cpu.state_dict(), 'knowledge_augmented_lstm.mdl')
    with open('vocab.pkl', 'wb') as f:
        pickle.dump(word_to_index, f)
    print("Model and vocabulary saved.")

def load_model_and_vocab(vocab_path='vocab.pkl', model_path='knowledge_augmented_lstm.mdl'):
    with open(vocab_path, 'rb') as f:
        word_to_index = pickle.load(f)
    vocab_size = len(word_to_index)
    model = KnowledgeAugmentedLSTM(vocab_size)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model = model.to(device)  # Move model to GPU if available
    model.eval()
    print("Model and vocabulary loaded.")
    return model, word_to_index

def generate_text(model, word_to_index, input_text, sequence_length, generate_length, temperature):
    model.eval()  # Set model to evaluation mode
    input_sequence = preprocess_text(input_text)
    indices = [word_to_index.get(word, -1) for word in input_sequence if word in word_to_index]

    if not indices:
        return "Input text contains no recognizable words."

    generated_text = []
    input_tensor = torch.tensor(indices[-sequence_length:], dtype=torch.long).unsqueeze(0).to(device)
    reverse_vocab = {i: word for word, i in word_to_index.items()}

    for _ in range(generate_length):
        with torch.no_grad():
            output = model(input_tensor)
            probabilities = torch.softmax(output / temperature, dim=1).squeeze()

            # Add structure to probabilities
            # Boost probabilities for specific tokens
            boost_indices = [word_to_index[word] for word in generated_text if word in word_to_index]
            penalties = torch.ones_like(probabilities)
            if boost_indices:  # Only apply if there are indices to boost
                penalties[boost_indices] = 1.5  # Boost specific tokens
            structured_probs = probabilities * penalties

            # Normalize probabilities after adding structure
            structured_probs = structured_probs / structured_probs.sum()

            next_word_idx = torch.multinomial(structured_probs, 1).item()
            next_word = reverse_vocab.get(next_word_idx, "<UNK>")
            generated_text.append(next_word)

            # Update input tensor for next step
            next_word_tensor = torch.tensor([[next_word_idx]], dtype=torch.long).to(device)
            input_tensor = torch.cat(
                (input_tensor[:, 1:], next_word_tensor),
                dim=1
            )

    return ' '.join(generated_text)


# Main Function
def main():
    choice = input("Do you want to (1) train or (2) load a model: ")

    if choice == '1':
        with open("xaa", encoding="utf-8") as f:
            text = f.read().lower()

        word_to_index, vocab_size = build_vocabulary(text)
        sequences = create_sequences(word_to_index, preprocess_text(text), sequence_length=4)
        dataset = TextDataset(sequences)
        data_loader = DataLoader(dataset, batch_size=256, shuffle=True)

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
        print("AI:", generate_text(model, word_to_index, user_input, sequence_length=4, generate_length=generate_length, temperature=temperature))

if __name__ == "__main__":
    main()
