import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np

KB_limit = 3999

class CustomModel(nn.Module):
    def __init__(self, embedding_dim, hidden_dim, vocab_size):
        super(CustomModel, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.fc1 = nn.Linear(embedding_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, vocab_size)
        
    def forward(self, x):
        embedded = self.embedding(x)
        embedded = embedded.view(embedded.size(0), -1)  # Flatten
        hidden = F.relu(self.fc1(embedded))
        logits = self.fc2(hidden)
        return logits

def load_text(filename):
    with open(filename, 'r', encoding='utf-8') as file:
        text = ' '.join(file.read().split()[:KB_limit])
    return text

def preprocess_text(text_data, seq_length):
    words = text_data.split()
    word_to_idx = {word: idx for idx, word in enumerate(sorted(set(words)))}
    unk_token = '<unk>'
    word_to_idx[unk_token] = len(word_to_idx)
    idx_to_word = {idx: word for word, idx in word_to_idx.items()}

    input_sequences, target_sequences = [], []

    for i in range(len(words) - seq_length):
        input_seq = words[i:i+seq_length]
        target_seq = words[i+seq_length]
        input_sequences.append([word_to_idx.get(word, word_to_idx[unk_token]) for word in input_seq])
        target_sequences.append(word_to_idx.get(target_seq, word_to_idx[unk_token]))

    return input_sequences, target_sequences, word_to_idx, idx_to_word

def generate_text(model, idx_to_word, word_to_idx, seed_text, seq_length, gen_length=50, device='cpu'):
    model.eval()
    seed_words = seed_text.split()
    seed_words = (['<unk>'] * max(0, seq_length - len(seed_words))) + seed_words[-seq_length:]
    current_seq = [word_to_idx.get(word, word_to_idx['<unk>']) for word in seed_words]
    current_seq = torch.tensor([current_seq], dtype=torch.long).to(device)
    generated_text = " ".join(seed_words).strip()

    for _ in range(gen_length):
        with torch.no_grad():
            output = model(current_seq)
            probabilities = F.softmax(output[0], dim=-1).cpu().numpy()
            predicted_idx = np.random.choice(len(probabilities), p=probabilities)
            predicted_word = idx_to_word[predicted_idx]
            generated_text += " " + predicted_word
            next_word_tensor = torch.tensor([[predicted_idx]], dtype=torch.long).to(device)
            current_seq = torch.cat((current_seq[:, 1:], next_word_tensor), dim=1)

    return generated_text

def create_meaning_nodes(text_data, model, seq_length, word_to_idx, idx_to_word, device):
    meaning_nodes = []
    words = text_data.split()
    for i in range(len(words) - seq_length):
        original_seq = " ".join(words[i:i+seq_length])
        generated_seq = generate_text(model, idx_to_word, word_to_idx, original_seq, seq_length, gen_length=5, device=device)
        meaning_nodes.append((original_seq, generated_seq))
    return meaning_nodes
def train_on_meaning_nodes(model, meaning_nodes, optimizer, criterion, device, word_to_idx):
    model.train()
    total_loss = 0

    for original_text, generated_text in meaning_nodes:
        input_seq = text_to_tensor(original_text, word_to_idx, device)
        target_seq = text_to_tensor(generated_text.split()[0], word_to_idx, device) #Corrected line

        optimizer.zero_grad()
        output = model(input_seq)

        loss = criterion(output, target_seq.squeeze(0))
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    return total_loss / len(meaning_nodes)

def text_to_tensor(text, word_to_idx, device):
    indices = [word_to_idx.get(word, word_to_idx["<unk>"]) for word in text.split()]
    return torch.tensor([indices], dtype=torch.long, device=device)

def main():
    seq_length = 1
    embedding_dim = 1
    hidden_dim = 1
    text_data = load_text('test.txt')
    input_sequences, target_sequences, word_to_idx, idx_to_word = preprocess_text(text_data, seq_length)
    vocab_size = len(word_to_idx)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = CustomModel(embedding_dim, hidden_dim, vocab_size).to(device)
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.CrossEntropyLoss()

    # Train initial model (no batches)
    for epoch in range(5):
        model.train()
        total_loss = 0
        for i in range(len(input_sequences)):
            inputs = torch.tensor([input_sequences[i]], dtype=torch.long).to(device)
            targets = torch.tensor([target_sequences[i]], dtype=torch.long).unsqueeze(0).to(device) #correct line
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets.squeeze(0)) #correct line
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        print(f"Epoch {epoch+1}, Loss: {total_loss / len(input_sequences):.4f}")

    # Generate Meaning Nodes
    meaning_nodes = create_meaning_nodes(text_data, model, seq_length, word_to_idx, idx_to_word, device)

    # Further train on meaning nodes
    loss = train_on_meaning_nodes(model, meaning_nodes, optimizer, criterion, device,word_to_idx)

    # Interactive Generation
    while True:
        seed_text = input("USER: ")
        generated = generate_text(model, idx_to_word, word_to_idx, seed_text, seq_length, gen_length=250, device=device)
        print("\nGenerated Text:")
        print(generated)

if __name__ == "__main__":
    main()
