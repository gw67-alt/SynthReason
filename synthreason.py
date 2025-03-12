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
KB_LIMIT = 9999
SEQUENCE_LENGTH = 1
EMBEDDING_DIM = 256
HIDDEN_DIM = 256
LEARNING_RATE = 0.001
NUM_EPOCHS = 5
WINDOW_SIZE = 15  # Size of the window to consider for adjustments

# Neural Network Model for Text Generation
class TextGeneratorNN(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim):
        super(TextGeneratorNN, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.lstm = nn.LSTM(embedding_dim, hidden_dim)
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
    
   

# Enhanced Set Operations Integration with Categories
class SetTheoryModifier:
    def __init__(self):
        # Empty set implementation - used to represent ∅
        self.empty_set = set()
        
        # Set theory operations categorized by concept
        self.set_operations = {
            'empty_not_in': {
                'name': 'z=∅∩∉',
                'description': 'Empty set and not-in operation',
                'active': True,
                'influence_factor': 0.15,
                'empty_boost': 1.7,
                'contradiction_penalty': 0.5
            }
        }
    
    def toggle_operation(self, operation_key):
        """Toggle a specific set operation on/off"""
        if operation_key in self.set_operations:
            self.set_operations[operation_key]['active'] = not self.set_operations[operation_key]['active']
            return f"{operation_key} ({self.set_operations[operation_key]['name']}) is now {'active' if self.set_operations[operation_key]['active'] else 'inactive'}"
        return f"Unknown operation: {operation_key}"
    
    def set_operation_parameter(self, operation_key, param_name, value):
        """Set a parameter value for a specific operation"""
        if operation_key in self.set_operations and param_name in self.set_operations[operation_key]:
            try:
                self.set_operations[operation_key][param_name] = float(value)
                return f"Set {param_name} to {value} for {operation_key}"
            except ValueError:
                return f"Invalid value: {value}. Must be a number."
        return f"Unknown operation or parameter: {operation_key}.{param_name}"
    
    def list_active_operations(self):
        """List all currently active set theory operations"""
        active_ops = [f"{key} ({op['name']}): {op['description']}" 
                     for key, op in self.set_operations.items() 
                     if op['active']]
        if active_ops:
            return "Active set theory operations:\n" + "\n".join(active_ops)
        else:
            return "No set theory operations are currently active"
    
    def get_category_words(self, category):
        """Get words associated with a specific category or set theory concept"""
        try:
            with open(f"{category}.txt", "r", encoding="utf-8") as f:
                return [line.strip() for line in f.readlines()]
        except FileNotFoundError:
            return []
    
    def apply_set_theory_modifiers(self, probs, vocab_inv):
        """Apply multiple set theory concepts to the probability distribution"""
        modified_probs = probs.clone()
        
        # Get category word lists for different concepts
        action_words = self.get_category_words("actions")
        description_words = self.get_category_words("descriptions")
        common_words = self.get_category_words("common")
        diverse_words = self.get_category_words("diverse")
        
        # Apply each active set theory operation
        for op_key, operation in self.set_operations.items():
            if operation['active']:
                # Apply operation-specific modifications
                # ∅∩∉ operation: Boost emptiness, penalize presence
                for i in range(len(modified_probs)):
                    word = vocab_inv[i].lower() if i in vocab_inv else ""
                    if word and any(empty_word not in word for empty_word in description_words):
                        modified_probs[i] *= operation['empty_boost']
                    if word and any(presence_word not in word for presence_word in action_words):
                        modified_probs[i] *= operation['contradiction_penalty']
        
        # Ensure probabilities are valid
        modified_probs = torch.maximum(modified_probs, torch.tensor(0.0))
        sum_probs = modified_probs.sum()
        if sum_probs > 0:
            modified_probs = modified_probs / sum_probs
        else:
            # If all probabilities became zero, revert to original
            modified_probs = probs.clone()
            
        return modified_probs

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

# Topic identification
def identify_topic(sequence, vocab_inv, topic_keywords):
    """
    Identify the topic of a sequence based on keywords.
    
    Args:
        sequence: Tensor of word indices
        vocab_inv: Inverse vocabulary (index to word)
        topic_keywords: Dictionary of topics and their associated keywords
        
    Returns:
        str: The identified topic or 'general' if no specific topic is found
    """
    # Convert sequence indices to words
    words = [vocab_inv.get(idx.item(), '') for idx in sequence]
    
    # Check for topic keywords in the sequence
    for topic, keywords in topic_keywords.items():
        if any(keyword in words for keyword in keywords):
            return topic
    
    return 'general'  # Default topic if no specific topic is found

# Topic detection
def detect_topic(text, topic_keywords):
    """
    Automatically detect the most likely topic based on keyword frequency in the input text.
    
    Args:
        text (str): The input text to analyze
        topic_keywords (dict): Dictionary mapping topics to their related keywords
        
    Returns:
        str: The most likely topic or None if no clear topic is detected
        list: Matched keywords for the detected topic
    """
    # Convert input to lowercase and tokenize
    text_lower = text.lower()
    words = re.findall(r'\b\w+\b', text_lower)
    
    # Count keyword matches for each topic
    topic_scores = {topic: 0 for topic in topic_keywords}
    matched_keywords = {topic: [] for topic in topic_keywords}
    
    for word in words:
        for topic, keywords in topic_keywords.items():
            if word in keywords:
                topic_scores[topic] += 1
                matched_keywords[topic].append(word)
    
    # Find topic with highest score
    max_score = 0
    best_topic = None
    
    for topic, score in topic_scores.items():
        if score > max_score:
            max_score = score
            best_topic = topic
    
    # Check for ties and resolve based on keyword specificity
    if best_topic:
        tied_topics = [t for t, s in topic_scores.items() if s == max_score]
        if len(tied_topics) > 1:
            # Resolve ties by checking keyword uniqueness
            topic_uniqueness = {}
            for topic in tied_topics:
                # Count how many other topics contain each matched keyword
                uniqueness_score = 0
                for keyword in matched_keywords[topic]:
                    other_topics_with_keyword = sum(1 for t in topic_keywords if t != topic and keyword in topic_keywords[t])
                    uniqueness_score += 1 / (1 + other_topics_with_keyword)  # More unique keywords score higher
                
                if matched_keywords[topic]:  # Avoid division by zero
                    topic_uniqueness[topic] = uniqueness_score / len(matched_keywords[topic])
                else:
                    topic_uniqueness[topic] = 0
            
            # Select the topic with the most unique keywords
            best_topic = max(topic_uniqueness.items(), key=lambda x: x[1])[0]
    
    # Only return a topic if it has a minimum number of matches
    min_matches = 1  # Adjust this threshold as needed
    if max_score >= min_matches:
        return best_topic, matched_keywords[best_topic] if best_topic else []
    else:
        return None, []

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

# Text generation function with PyTorch model
def generate_text_nn(model, prompt, vocab, vocab_inv, char_ratios, set_modifier, device, 
                     seq_length=SEQUENCE_LENGTH, max_length=250, temperature=1.0, topic_model=None, topic=None):
    """Generate text using the trained PyTorch model with set theory modifications"""
    model.eval()
    if topic_model:
        topic_model.eval()
    
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
    
    for _ in range(max_length):
        # Forward pass through the model
        with torch.no_grad():
            output, hidden = model(input_tensor, hidden)
            
        # Get probabilities for the next word
        output = output[-1, :]  # Get the last timestep
        
        # Apply temperature
        output = output / temperature
        
        # Apply set theory modifiers
        probs = torch.softmax(output, dim=-1)
        probs = set_modifier.apply_set_theory_modifiers(probs, vocab_inv)
        
        # Apply character ratio adjustments
        for i in range(len(probs)):
            if i in vocab_inv:
                word = vocab_inv[i]
                if word and len(word) > 0:
                    first_char = word[0].lower()
                    if first_char in char_ratios:
                        probs[i] *= (1.0 + char_ratios[first_char])
        
        # Apply topic bias if topic model and topic are provided
        if topic_model and topic is not None:
            with torch.no_grad():
                topic_output, _ = topic_model(input_tensor)
                topic_probs = torch.softmax(topic_output[-1, :] / temperature, dim=-1)
                
                # Blend probabilities (70% topic, 30% general)
                probs = 0.3 * probs + 0.7 * topic_probs
        
        # Apply recency bias (avoid repeating recent words)
        for i in range(1, min(WINDOW_SIZE, len(recent_outputs)) + 1):
            if recent_outputs[-i] < len(probs):
                probs[recent_outputs[-i]] *= 0.5  # Reduce probability of recently used words
        
        # Ensure probabilities are valid
        probs = torch.maximum(probs, torch.tensor(0.0))
        if probs.sum() > 0:
            probs = probs / probs.sum()
        
        # Sample from the probability distribution
        next_idx = torch.multinomial(probs, 1).item()
        
        # Get the next word and add it to the generated text
        next_word = vocab_inv.get(next_idx, "<UNK>")
        generated_text += ' ' + next_word
        
        # Update input sequence for next iteration
        input_tensor = torch.tensor([[next_idx]]).to(device)
        recent_outputs.append(next_idx)
        
        # Limit the size of recent_outputs
        if len(recent_outputs) > WINDOW_SIZE:
            recent_outputs.pop(0)
    
    return generated_text

# Save and load functions
def save_model(model, vocab, char_ratios, filepath="text_model.pt"):
    """Save the model and necessary data"""
    checkpoint = {
        'model_state_dict': model.state_dict(),
        'vocab': vocab,
        'char_ratios': char_ratios
    }
    torch.save(checkpoint, filepath)
    print(f"Model saved to {filepath}")

def load_model(filepath="text_model.pt", embedding_dim=EMBEDDING_DIM, hidden_dim=HIDDEN_DIM, device='cuda'):
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
    
    # Define topic keywords
    topic_keywords = {
        "actions": [
            # Creation
            "create", "build", "develop", "generate", "produce", "construct", "design", "invent", "forge", "craft",
            # Movement
            "move", "transfer", "shift", "relocate", "transport", "displace", "advance", "proceed", "travel",
            # Addition
            "add", "append", "attach", "include", "insert", "incorporate", "supplement", "join", "combine", "merge",
            # Possession
            "have", "hold", "keep", "maintain", "possess", "retain", "sustain", "own", "contain", "preserve",
            # Acquisition
            "obtain", "acquire", "gain", "attain", "secure", "procure", "gather", "collect", "amass", "accumulate",
            # Transformation
            "change", "transform", "convert", "modify", "alter", "adjust", "adapt", "revise", "evolve", "transition"
        ],
        
        "descriptions": [
            # Emptiness/Absence
            "empty", "void", "vacant", "hollow", "bare", "barren", "blank", "unfilled", "unoccupied", "deserted", 
            # Lacking
            "absent", "lacking", "missing", "without", "devoid", "deprived", "deficient", "insufficient", "inadequate",
            # Nothingness
            "none", "nothing", "zero", "nil", "nonexistent", "vanished", "gone", "exhausted", "depleted", "spent",
            # Desolation
            "desolate", "abandoned", "forsaken", "neglected", "uninhabited", "isolated", "secluded", "remote", "vacuous",
            # Purity/Clearness
            "pure", "clear", "pristine", "unsullied", "unblemished", "untainted", "clean", "spotless", "immaculate"
        ],
        
        "diverse": [
            # Uniqueness
            "unique", "singular", "unparalleled", "unprecedented", "unmatched", "exclusive", "distinctive", "original",
            # Difference
            "different", "distinct", "unlike", "dissimilar", "contrasting", "divergent", "separate", "discrete",
            # Variety
            "varied", "diverse", "assorted", "miscellaneous", "heterogeneous", "eclectic", "multifarious", "manifold",
            # Rarity
            "rare", "uncommon", "unusual", "exceptional", "extraordinary", "scarce", "infrequent", "sporadic", "exotic",
            # Complexity
            "complex", "intricate", "elaborate", "sophisticated", "multifaceted", "nuanced", "layered", "convoluted"
        ],
        
        "concepts": [
            # Abstract ideas
            "freedom", "justice", "truth", "beauty", "wisdom", "knowledge", "power", "harmony", "balance", "equality",
            # Philosophical
            "existence", "consciousness", "reality", "identity", "purpose", "meaning", "ethics", "morality", "virtue",
            # Social
            "community", "society", "culture", "tradition", "progress", "innovation", "cooperation", "competition"
        ]
    }
    
    # Try to load existing model, otherwise train a new one
    try:
        print("Attempting to load existing model...")
        model, vocab, char_ratios = load_model(device=device)
        vocab_inv = {idx: word for word, idx in vocab.items()}
        print("Model loaded successfully.")
        
        # Load topic-specific models if they exist
        topic_models = {}
        for topic in topic_keywords:
            try:
                topic_model, _, _ = load_model(f"{topic}_model.pt", device=device)
                topic_models[topic] = topic_model
                print(f"Loaded topic model for '{topic}'")
            except:
                print(f"No model found for topic '{topic}'")
    
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
            
            # Calculate character ratios
            char_ratios = calculate_character_ratios(filtered_words)
            
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
            print("Training general model...")
            train_model(model, dataset, NUM_EPOCHS, LEARNING_RATE, device)
            save_model(model, vocab, char_ratios)
            
            # Train topic-specific models
            topic_models = {}
            for topic, keywords in topic_keywords.items():
                print(f"Filtering data for topic '{topic}'...")
                # Create a corpus focused on this topic
                topic_text = []
                for word in filtered_words:
                    if word in keywords:
                        # Include context around the keyword
                        idx = filtered_words.index(word)
                        start = max(0, idx - 50)
                        end = min(len(filtered_words), idx + 50)
                        topic_text.extend(filtered_words[start:end])
                
                if len(topic_text) > 100:  # Only train if we have enough data
                    topic_text = ' '.join(topic_text)
                    topic_dataset = TextDataset(topic_text, vocab, SEQUENCE_LENGTH)
                    
                    print(f"Training model for topic '{topic}'...")
                    topic_model = TextGeneratorNN(vocab_size, EMBEDDING_DIM, HIDDEN_DIM).to(device)
                    train_model(topic_model, topic_dataset, NUM_EPOCHS, LEARNING_RATE, device)
                    save_model(topic_model, vocab, char_ratios, f"{topic}_model.pt")
                    topic_models[topic] = topic_model
                else:
                    print(f"Not enough data for topic '{topic}', skipping.")
        
        except FileNotFoundError:
            print("Error: kb.txt file not found. Please ensure the knowledge base file exists.")
            return
    
    # Initialize set theory modifier
    set_modifier = SetTheoryModifier()
    
    print("\nEnhanced Text Generator with PyTorch Neural Networks")
    print("Available topics:", list(topic_keywords.keys()) + ["general"])
    print("Available commands:")
    print("  /topic <topic>         - Set the current topic")
    print("  /topic auto            - Enable automatic topic detection")
    print("  /topic manual          - Disable automatic topic detection")
    print("  /temp <value>          - Set temperature (0.1-2.0)")
    print("  /exit                  - Exit the program")
    
    current_topic = None
    auto_topic_detection = False
    temperature = 1.0
    
    try:
        while True:
            try:
                prompt = input("\nUSER: ")
                
                # Check for commands
                if prompt.startswith("/"):
                    cmd_parts = prompt.split()
                    cmd = cmd_parts[0].lower()
                    
                    # Topic command
                    if cmd == "/topic" and len(cmd_parts) > 1:
                        requested_topic = cmd_parts[1].lower()
                        
                        if requested_topic == "auto":
                            auto_topic_detection = True
                            print("Automatic topic detection enabled")
                            
                        elif requested_topic == "manual":
                            auto_topic_detection = False
                            print("Automatic topic detection disabled")
                            
                        elif requested_topic in topic_keywords or requested_topic == "general":
                            print(f"Topic set to: {requested_topic}")
                            current_topic = requested_topic if requested_topic != "general" else None
                            auto_topic_detection = False
                            
                        else:
                            print(f"Unknown topic: {requested_topic}")
                            print("Available topics:", list(topic_keywords.keys()) + ["general"])
                    
                    # Temperature command
                    elif cmd == "/temp" and len(cmd_parts) > 1:
                        try:
                            temp_value = float(cmd_parts[1])
                            if 0.1 <= temp_value <= 2.0:
                                temperature = temp_value
                                print(f"Temperature set to: {temperature}")
                            else:
                                print("Temperature must be between 0.1 and 2.0")
                        except ValueError:
                            print("Invalid temperature value")
                    
                    # Exit command
                    elif cmd == "/exit":
                        print("Exiting program.")
                        break
                
                # Generate text
                else:
                    active_topic = current_topic
                    
                    # Auto-detect topic if enabled
                    if auto_topic_detection:
                        detected_topic, matched_keywords = detect_topic(prompt, topic_keywords)
                        
                        if detected_topic:
                            active_topic = detected_topic
                            keyword_display = ", ".join(matched_keywords[:3])
                            if len(matched_keywords) > 3:
                                keyword_display += f" and {len(matched_keywords) - 3} more"
                            print(f"[Auto-detected topic: {detected_topic} based on: {keyword_display}]")
                    
                    # Get the appropriate topic model
                    topic_model = topic_models.get(active_topic) if active_topic else None
                    
                    # Generate text
                    generated_text = generate_text_nn(
                        model,
                        prompt,
                        vocab,
                        vocab_inv,
                        char_ratios,
                        set_modifier,
                        device,
                        seq_length=SEQUENCE_LENGTH,
                        max_length=250,
                        temperature=temperature,
                        topic_model=topic_model,
                        topic=active_topic
                    )
                    
                    print("\nAI: ", generated_text)
            
            except EOFError:
                print("Input stream ended. Exiting...")
                break
    
    except KeyboardInterrupt:
        print("\nProgram interrupted. Exiting...")

if __name__ == "__main__":
    main()
