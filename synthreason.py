import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from collections import Counter

def build_vocab(text):
    words = text.split()
    # Create bigrams as tuples of consecutive words
    bigrams = ['{} {}'.format(words[i], words[i+1]) for i in range(len(words)-1)]
    bigram_counts = Counter(bigrams)
    vocab = sorted(bigram_counts, key=bigram_counts.get, reverse=True)
    vocab.append('<UNK>')

    word_to_idx = {w: i for i, w in enumerate(vocab)}
    idx_to_word = {i: w for w, i in word_to_idx.items()}
    return word_to_idx, idx_to_word

class TextEnv:
    def __init__(self, text, word_to_idx):
        self.words = text.split()
        self.word_to_idx = word_to_idx
        self.idx_to_word = {i: w for w, i in word_to_idx.items()}
        self.reset()

    def reset(self, seed_text=None):
        if seed_text is not None:
            # Find the position of the seed text in the document
            seed_words = seed_text.split()
            found = False
            for i in range(len(self.words) - len(seed_words) + 1):
                if self.words[i:i+len(seed_words)] == seed_words:
                    self.pos = i + len(seed_words) - 1
                    found = True
                    break
            if not found:
                self.pos = 0  # fallback if not found
        else:
            self.pos = 0
        return self._get_state()

    def _get_state(self):
        if self.pos < len(self.words):
            word = self.words[self.pos]
            idx = self.word_to_idx.get(word, 0)
        else:
            idx = 0
        state = np.zeros(len(self.word_to_idx))
        state[idx] = 1
        return state

    def step(self, action_idx):
        done = self.pos >= len(self.words) - 1
        correct_idx = self.word_to_idx.get(self.words[self.pos+1], self.word_to_idx['<UNK>']) if not done else 0
        reward = 1.0 if action_idx == correct_idx else 0.1
        self.pos += 1
        next_state = self._get_state()
        return next_state, reward, self.pos >= len(self.words) - 1

class PolicyNet(nn.Module):
    def __init__(self, vocab_size):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(vocab_size, 256),
            nn.ReLU(),
            nn.Linear(256, vocab_size)
        )

    def forward(self, x):
        return self.fc(x)

    def act(self, state, explore=True):
        logits = self.forward(torch.FloatTensor(state).unsqueeze(0)).squeeze()
        probs = torch.softmax(logits, dim=0)
        if explore:
            action = torch.multinomial(probs, 1).item()
        else:
            action = torch.argmax(probs).item()
        return action, torch.log(probs[action])

def train(seed_text=None):
    with open(input("Filename: "), "r", encoding="utf-8") as f:
        text = ' '.join(f.read().split()[:1999])
    word_to_idx, idx_to_word = build_vocab(text)
    vocab_size = len(word_to_idx)

    env = TextEnv(text, word_to_idx)
    policy = PolicyNet(vocab_size)
    optimizer = optim.Adam(policy.parameters(), lr=0.11)
    gamma = 0.3

    for episode in range(3):
        state = env.reset()
        done = False
        log_probs = []
        rewards = []
        total_reward = 0

        while not done:
            action, log_prob = policy.act(state, explore=True)
            next_state, reward, done = env.step(action)
            log_probs.append(log_prob)
            rewards.append(reward)
            state = next_state
            total_reward += reward

        returns = []
        G = 0
        for r in reversed(rewards):
            G = r + gamma * G
            returns.insert(0, G)
        returns = torch.FloatTensor(returns)

        loss = -sum(log_prob * G for log_prob, G in zip(log_probs, returns))
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if (episode + 1) % 10 == 0:
            print(f"Episode {episode+1}, Total Reward: {total_reward}")

    # Test: generate a sequence from seed
    print("\nGenerated sequence:")
    state = env.reset(seed_text=seed_text)
    if seed_text:
        print(seed_text, end=' ')
    for _ in range(200):
        action, _ = policy.act(state, explore=True)
        print(idx_to_word[action], end=' ')
        state, _, done = env.step(action)
        if done:
            break
    print()

if __name__ == "__main__":
    # Example: pass a seed text here, or set to None
    while True:
        
        train(seed_text=input("USER:"))
