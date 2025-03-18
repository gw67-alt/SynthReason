import numpy as np
import re
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from collections import defaultdict
import random
import os
import json
import pickle
import serial
import time

KB_limit = 999 # -1 for unlimited
epochs = 10
generate_length = 500
n_gram_size = 2  # n-gram size (for bigrams)
embedding_dim = 256
hidden_dim = 512

class TextDataset(Dataset):
    def __init__(self, X, y):
        self.X = X
        self.y = y

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

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

# Function to save model and variables
def save_model(model, word_to_index, index_to_word, n_gram_size, embedding_dim, hidden_dim, filename="text_generator"):
    # Create directory if it doesn't exist
    os.makedirs("saved_models", exist_ok=True)
    
    # Save model state
    torch.save(model.state_dict(), f"saved_models/{filename}_model.pth")
    
    # Save vocabulary and parameters
    model_data = {
        "word_to_index": word_to_index,
        "index_to_word": {int(k): v for k, v in index_to_word.items()},  # Convert keys to int for JSON
        "n_gram_size": n_gram_size,
        "embedding_dim": embedding_dim,
        "hidden_dim": hidden_dim,
        "vocab_size": len(word_to_index) + 1
    }
    
    # Save vocabulary and parameters using pickle
    with open(f"saved_models/{filename}_data.pkl", "wb") as f:
        pickle.dump(model_data, f)
    
    print(f"Model and data saved to saved_models/{filename}_*")

# Function to load model and variables
def load_model(filename="text_generator"):
    try:
        # Load vocabulary and parameters
        with open(f"saved_models/{filename}_data.pkl", "rb") as f:
            model_data = pickle.load(f)
        
        # Extract data
        word_to_index = model_data["word_to_index"]
        index_to_word = model_data["index_to_word"]
        n_gram_size = model_data["n_gram_size"]
        embedding_dim = model_data["embedding_dim"]
        hidden_dim = model_data["hidden_dim"]
        vocab_size = model_data["vocab_size"]
        
        # Create model with loaded parameters
        model = TextGenerator(vocab_size, embedding_dim, hidden_dim, n_gram_size)
        
        # Load model state
        model.load_state_dict(torch.load(f"saved_models/{filename}_model.pth"))
        
        print(f"Model loaded from saved_models/{filename}_*")
        return model, word_to_index, index_to_word, n_gram_size, embedding_dim, hidden_dim
    
    except FileNotFoundError:
        print(f"No saved model found at saved_models/{filename}_*")
        return None, None, None, None, None, None

def train_new_model():
    with open('test.txt', 'r', encoding="utf-8") as file:
        text = ' '.join(file.read().split()[:KB_limit])
    text = re.sub(r'\d+', '', text)
    words = re.findall(r'\b\w+\b', text.lower())
    filtered_words = [word for word in words if len(word) > 1 or word in ['a', 'i'] and not word.isdigit()]
    word_to_index = {word: idx + 1 for idx, word in enumerate(set(filtered_words))}
    index_to_word = {idx + 1: word for idx, word in enumerate(set(filtered_words))}
    sequences = []
    for i in range(len(filtered_words) - n_gram_size-5):
        sequences.append(filtered_words[i:i + n_gram_size + 1]+filtered_words[i+3:i])
    sequences_idx = [[word_to_index[word] for word in sequence] for sequence in sequences]
    X = [seq[:-1] for seq in sequences_idx]  # Input: all but last word
    y = [seq[-1] for seq in sequences_idx]  # Output: last word
    X = torch.tensor(X)
    y = torch.tensor(y)
    dataset = TextDataset(X, y)
    dataloader = DataLoader(dataset, batch_size=64, shuffle=True)

    vocab_size = len(word_to_index) + 1 
    model = TextGenerator(vocab_size, embedding_dim, hidden_dim, n_gram_size)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        for inputs, labels in dataloader:
            optimizer.zero_grad()
            
            outputs = model(inputs)
            
            loss = criterion(outputs, labels)
            loss.backward()
            
            optimizer.step()
            
            running_loss += loss.item()
        
        print(f'Epoch [{epoch+1}/{epochs}], Loss: {running_loss / len(dataloader)}')
    
    return model, word_to_index, index_to_word

def clear_line():
    # Move the cursor to the beginning of the line and clear the entire line
    print("\033[2K\033[0G", end="", flush=True)

def generate_text_with_serial_rethrow(seed_text, generate_length, model, word_to_index, index_to_word, n_gram_size, serial_port):
    """
    Generate text and update the screen with each word.
    When a 1 is detected from serial, either iterate through the last word or rethrow.
    """
    model.eval()
    generated_text = []
    
    words = seed_text.lower().split()
    current_sequence = []
    for word in words:
        if word in word_to_index:
            current_sequence.append(word_to_index[word])
        else:
            current_sequence.append(random.choice(list(word_to_index.values())))
    
    while len(current_sequence) < n_gram_size:
        current_sequence.append(random.choice(list(word_to_index.values())))
    
    current_sequence = current_sequence[-n_gram_size:]
    
    # Print the seed text first
    print(f"\nSeed text: {seed_text}")
    print("\nGenerated text: ", end="", flush=True)
    
    last_word = None
    last_word_alternatives = []
    currently_rethrowing = False
    current_alternative_index = 0
    
    for _ in range(generate_length):
        # Check for serial input
        if serial_port.in_waiting > 0:
            serial_data = serial_port.readline().decode('utf-8').strip()
            try:
                if int(serial_data) == 1:
                    if not currently_rethrowing and last_word is not None:
                        # Start rethrowing/iterating
                        currently_rethrowing = True
                        print(f"\nDetected '1'. Rethrowing/iterating on last word: '{last_word}'")
                        
                        # Generate alternatives for the last word if we haven't already
                        if not last_word_alternatives:
                            # Get top 5 alternative words
                            input_seq = torch.tensor([current_sequence[:-1]], dtype=torch.long)
                            
                            with torch.no_grad():
                                output = model(input_seq)
                            
                            probabilities = torch.nn.functional.softmax(output, dim=1)[0]
                            top_indices = torch.topk(probabilities, 10).indices.tolist()
                            
                            # Filter out the current word and get valid alternatives
                            last_word_idx = current_sequence[-1]
                            last_word_alternatives = [idx for idx in top_indices if idx != last_word_idx and idx in index_to_word]
                        
                        # If we have alternatives, replace the last word and update the screen
                        if last_word_alternatives:
                            # Clear the current line
                            clear_line()
                            
                            # Get the next alternative
                            new_word_idx = last_word_alternatives[current_alternative_index]
                            new_word = index_to_word[new_word_idx]
                            
                            # Replace the last word in the sequence and generated text
                            current_sequence[-1] = new_word_idx
                            generated_text[-1] = new_word
                            
                            # Display the new word
                            print(f"{new_word} ", end="", flush=True)
                            
                            # Move to the next alternative for the next rethrow
                            current_alternative_index = (current_alternative_index + 1) % len(last_word_alternatives)
                        else:
                            print("\nNo alternatives available for the last word.")
                            currently_rethrowing = False
                    else:
                        # Continue with normal generation
                        currently_rethrowing = False
                        print("\nContinuing with normal generation.")
            except ValueError:
                pass  # Ignore non-integer values
        
        # If we're rethrowing, don't generate a new word
        if currently_rethrowing:
            time.sleep(0.1)
            continue
                
        input_seq = torch.tensor([current_sequence], dtype=torch.long)
        
        with torch.no_grad():
            output = model(input_seq)
        
        probabilities = torch.nn.functional.softmax(output, dim=1)
        
        predicted_word_idx = torch.multinomial(probabilities[0], 1).item()
        
        if predicted_word_idx in index_to_word:
            predicted_word = index_to_word[predicted_word_idx]
            generated_text.append(predicted_word)
            
            # Clear the current line
            clear_line()
            
            # Print the word immediately and flush the output
            print(predicted_word + " ", end="", flush=True)
            
            # Small delay to make the output visible word by word
            time.sleep(0.1)
            
            # Update the current sequence
            current_sequence.append(predicted_word_idx)
            current_sequence = current_sequence[-n_gram_size:]
            
            # Update the last word
            last_word = predicted_word
            last_word_alternatives = []  # Reset alternatives for the new last word
            current_alternative_index = 0
        else:
            continue
    
    print("\nGeneration complete.")
    return ' '.join(generated_text)

def generate_text_with_auto_rethrow(seed_text, generate_length, model, word_to_index, index_to_word, n_gram_size, serial_port):
    """
    Generate text word by word, automatically cycling through alternatives for each position.
    When a '1' is received from serial, keep the current word and move to the next position.
    """
    model.eval()
    generated_text = []
    
    # Process seed text
    words = seed_text.lower().split()
    current_sequence = []
    for word in words:
        if word in word_to_index:
            current_sequence.append(word_to_index[word])
        else:
            current_sequence.append(random.choice(list(word_to_index.values())))
    
    while len(current_sequence) < n_gram_size:
        current_sequence.append(random.choice(list(word_to_index.values())))
    
    current_sequence = current_sequence[-n_gram_size:]
    
    # Print the seed text first
    print(f"\nSeed text: {seed_text}")
    print("\nGenerated text: ", end="", flush=True)
    
    # Print the seed words
    for word in words:
        print(word + " ", end="", flush=True)
        generated_text.append(word)
    
    # Main generation loop
    position = 0
    while position < generate_length:
        # Generate alternatives for the current position
        input_seq = torch.tensor([current_sequence], dtype=torch.long)
        
        with torch.no_grad():
            output = model(input_seq)
        
        probabilities = torch.nn.functional.softmax(output, dim=1)
        
        # Get top 10 alternatives
        top_indices = torch.topk(probabilities[0], 10).indices.tolist()
        alternatives = [index_to_word[idx] for idx in top_indices if idx in index_to_word]
        
        if not alternatives:
            # If no valid alternatives, skip this position
            print("\nNo valid alternatives for this position. Skipping...")
            position += 1
            continue
        
        # Cycle through alternatives until a serial signal is received
        alt_index = 0
        word_confirmed = False
        
        while not word_confirmed:
            # Display the current alternative
            current_word = alternatives[alt_index]
            
            # Clear the current line
            clear_line()
            
            # If this is the first alternative or replacing a previous alternative
            if len(generated_text) > len(words) + position:
                # Remove the last word from display
                print(f"\rGenerated text: {' '.join(generated_text[:-1])}", end=" ", flush=True)
            else:
                print("\rGenerated text: " + " ".join(generated_text), end=" ", flush=True)
            
            # Show the current alternative
            print(current_word + " ", end="", flush=True)
            
            # Wait for serial input or timeout
            start_time = time.time()
            signal_received = False
            
            while time.time() - start_time < 1.0:  # 1-second timeout
                if serial_port.in_waiting > 0:
                    serial_data = serial_port.readline().decode('utf-8').strip()
                    try:
                        if int(serial_data) == 1:
                            # Signal received, keep this word
                            word_confirmed = True
                            signal_received = True
                            break
                    except ValueError:
                        pass  # Ignore non-integer values
                time.sleep(0.1)
            
            if not signal_received:
                # No signal received, move to next alternative
                alt_index = (alt_index + 1) % len(alternatives)
        
        # Word confirmed, add to generated text
        if len(generated_text) > len(words) + position:
            generated_text[-1] = current_word
        else:
            generated_text.append(current_word)
        
        # Update the current sequence
        word_idx = word_to_index[current_word] if current_word in word_to_index else random.choice(list(word_to_index.values()))
        current_sequence.append(word_idx)
        current_sequence = current_sequence[-n_gram_size:]
        
        position += 1
    
    print("\nGeneration complete.")
    return ' '.join(generated_text)

def list_saved_models():
    if not os.path.exists("saved_models"):
        print("No saved models directory found.")
        return []
    
    models = []
    for file in os.listdir("saved_models"):
        if file.endswith("_model.pth"):
            models.append(file.replace("_model.pth", ""))
    
    return models

def list_serial_ports():
    """List available serial ports"""
    import serial.tools.list_ports
    ports = serial.tools.list_ports.comports()
    
    if not ports:
        print("No serial ports found.")
        return []
    
    print("\nAvailable serial ports:")
    port_list = []
    for i, port in enumerate(ports):
        print(f"{i+1}. {port.device}")
        port_list.append(port.device)
    
    return port_list

def main():
    # Try to load existing model
    model, word_to_index, index_to_word, loaded_n_gram_size, loaded_embedding_dim, loaded_hidden_dim = load_model()
    
    # If no model loaded, train a new one
    if model is None:
        print("No saved model found. Training a new model...")
        model, word_to_index, index_to_word = train_new_model()
        # Use global variables for n_gram_size, embedding_dim, hidden_dim
        current_n_gram_size = n_gram_size
        current_embedding_dim = embedding_dim
        current_hidden_dim = hidden_dim
    else:
        # Use loaded parameters
        current_n_gram_size = loaded_n_gram_size
        current_embedding_dim = loaded_embedding_dim
        current_hidden_dim = loaded_hidden_dim
    
    # Serial port setup
    serial_port = None
    
    while True:
        print("\nText Generation Interface")
        print("1. Generate text with manual rethrowing")
        print("2. Generate text with auto-rethrow (wait for signal to keep word)")
        print("3. Save model")
        print("4. Load model")
        print("5. Train new model")
        print("6. List saved models")
        print("7. Setup serial port")
        print("8. Exit")
        
        choice = input("Enter your choice (1-8): ")
        
        if choice == "1":
            if serial_port is None:
                print("Please set up a serial port first (option 7).")
                continue
                
            seed_text = input("Enter seed text: ")
            generated_text = generate_text_with_serial_rethrow(seed_text, generate_length, model, 
                                                             word_to_index, index_to_word, 
                                                             current_n_gram_size, serial_port)
        
        elif choice == "2":
            if serial_port is None:
                print("Please set up a serial port first (option 7).")
                continue
                
            seed_text = input("Enter seed text: ")
            generated_text = generate_text_with_auto_rethrow(seed_text, generate_length, model, 
                                                           word_to_index, index_to_word, 
                                                           current_n_gram_size, serial_port)
            
        elif choice == "3":
            filename = input("Enter filename to save model (default: text_generator): ")
            if not filename:
                filename = "text_generator"
            save_model(model, word_to_index, index_to_word, current_n_gram_size, 
                       current_embedding_dim, current_hidden_dim, filename)
            
        elif choice == "4":
            available_models = list_saved_models()
            if available_models:
                print("\nAvailable models:")
                for i, model_name in enumerate(available_models):
                    print(f"{i+1}. {model_name}")
                
                model_idx = input("Enter model number to load (or press Enter for default): ")
                if model_idx and model_idx.isdigit() and 1 <= int(model_idx) <= len(available_models):
                    filename = available_models[int(model_idx) - 1]
                else:
                    filename = "text_generator"
                
                model, word_to_index, index_to_word, current_n_gram_size, current_embedding_dim, current_hidden_dim = load_model(filename)
                if model is None:
                    print("Failed to load model. Using current model.")
            else:
                print("No saved models found.")
            
        elif choice == "5":
            confirm = input("This will overwrite the current model. Continue? (y/n): ")
            if confirm.lower() == 'y':
                model, word_to_index, index_to_word = train_new_model()
                current_n_gram_size = n_gram_size
                current_embedding_dim = embedding_dim
                current_hidden_dim = hidden_dim
                print("New model trained successfully.")
            
        elif choice == "6":
            available_models = list_saved_models()
            if available_models:
                print("\nAvailable models:")
                for model_name in available_models:
                    print(f"- {model_name}")
            else:
                print("No saved models found.")
                
        elif choice == "7":
            try:
                # Close existing serial port if open
                if serial_port is not None and serial_port.is_open:
                    serial_port.close()
                    print("Previous serial port closed.")
                
                available_ports = list_serial_ports()
                if available_ports:
                    port_idx = input("Enter port number to use (or press Enter to enter manually): ")
                    if port_idx and port_idx.isdigit() and 1 <= int(port_idx) <= len(available_ports):
                        port_name = available_ports[int(port_idx) - 1]
                    else:
                        port_name = input("Enter serial port name (e.g., COM3, /dev/ttyUSB0): ")
                    
                    baud_rate = input("Enter baud rate (default: 9600): ")
                    if not baud_rate or not baud_rate.isdigit():
                        baud_rate = 9600
                    else:
                        baud_rate = int(baud_rate)
                    
                    serial_port = serial.Serial(port_name, baud_rate, timeout=0.1)
                    print(f"Serial port {port_name} opened successfully at {baud_rate} baud.")
                else:
                    port_name = input("Enter serial port name (e.g., COM3, /dev/ttyUSB0): ")
                    baud_rate = input("Enter baud rate (default: 9600): ")
                    if not baud_rate or not baud_rate.isdigit():
                        baud_rate = 9600
                    else:
                        baud_rate = int(baud_rate)
                    
                    serial_port = serial.Serial(port_name, baud_rate, timeout=0.1)
                    print(f"Serial port {port_name} opened successfully at {baud_rate} baud.")
            except Exception as e:
                print(f"Error setting up serial port: {e}")
                serial_port = None
            
        elif choice == "8":
            print("Exiting...")
            # Close serial port if open
            if serial_port is not None and serial_port.is_open:
                serial_port.close()
                print("Serial port closed.")
            break
            
        else:
            print("Invalid choice. Please try again.")

if __name__ == "__main__":
    main()
