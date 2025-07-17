import torch
import torch.nn as nn
import snntorch as snn
import numpy as np
from torch_geometric.data import Data
from torch_geometric.nn import GCNConv
from sklearn.preprocessing import StandardScaler
import random
from collections import defaultdict
# SNN neuron: Leaky-Integrate and Fire
class PolymorphicNeuron(nn.Module):
    def __init__(self, input_dim, num_modes=256):
        super().__init__()
        self.modes = nn.ModuleList([
            snn.Leaky(beta=_) for _ in range(num_modes)
        ])
        self.mode_selector = nn.Linear(input_dim, num_modes)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x, mem_states=None):
        mode_scores = self.softmax(self.mode_selector(x))
        mode_outs, new_states = [], []
        for i, mode in enumerate(self.modes):
            out, state = mode(x) if mem_states is None else mode(x, mem_states[i])
            mode_outs.append(out)
            new_states.append(state)
        stacked = torch.stack(mode_outs, dim=0)  # (num_modes, batch, features)
        mixed = (stacked * mode_scores.unsqueeze(0)).sum(dim=0)
        return mixed, new_states, mode_scores

# Top-level SNN
class PolymorphicSNN(nn.Module):
    def __init__(self, num_neurons, num_polymorphic):
        super().__init__()
        self.lin = nn.Linear(num_neurons, num_neurons)
        self.regular = snn.Leaky(beta=0.5)
        self.poly = nn.ModuleList([PolymorphicNeuron(num_neurons) for _ in range(num_polymorphic)])

    def forward(self, x, mem=None, pmem=None):
        x = self.lin(x)
        reg_out, reg_mem = self.regular(x) if mem is None else self.regular(x, mem)
        results = []
        poly_memory = []
        for n in self.poly:
            out, mems, sel = n(reg_out)
            results.append(out)
            results.extend(results)
            poly_memory.append(mems)
        if results:
            out = torch.cat([reg_out] + results, dim=-1)
        else:
            out = reg_out
        return out, reg_mem, poly_memory

# Create a simple graph from the spike raster
def create_neuron_graph(spk_rec):
    node_features = spk_rec.T
    num_nodes = node_features.shape[0]
    edge_index = []
    for i in range(num_nodes-1):  # Simple ring
        edge_index.append([i, (i+1)%num_nodes])
        edge_index.append([(i+1)%num_nodes, i])
    edge_index = torch.tensor(edge_index, dtype=torch.long).t()
    data = Data(x=node_features, edge_index=edge_index)
    return data

# GCN to analyze node features
class DataAwareGCN(nn.Module):
    def __init__(self, input_dim, hidden=64, out=32):
        super().__init__()
        self.conv1 = GCNConv(input_dim, hidden)
        self.conv2 = GCNConv(hidden, out)

    def forward(self, data):
        x = torch.relu(self.conv1(data.x, data.edge_index))
        x = torch.relu(self.conv2(x, data.edge_index))
        return x


class TextGenerator:
    def __init__(self, corpus):
        self.transitions = defaultdict(list)
        self.build_transitions(corpus)

    def build_transitions(self, words):
        for i in range(len(words) - 1):
            self.transitions[words[i]].append(words[i + 1])

    def generate_text(self, start_word=None, length=50):
        if not self.transitions:
            return "No data for generation."
        if start_word not in self.transitions:
            current_word = random.choice(list(self.transitions.keys()))
        else:
            current_word = start_word
        words = [current_word]
        for _ in range(length - 1):
            next_words = self.transitions.get(current_word, None)
            if not next_words:
                current_word = random.choice(list(self.transitions.keys()))
            else:
                current_word = random.choice(next_words)
            words.append(current_word)
        return ' '.join(words)
        
def max_psychological_overlap(generator, psychological_words, n=1000):
    max_intersection = 0
    max_text = ""
    max_seed = None

    # Convert psychological words to lowercase set for consistency
    psychological_set = set(w.lower() for w in psychological_words)
    # Choose seed words (could use a predefined list, or random sampling)
    instructions = [
    # cognition and mental processes
    "perception", "attention", "memory", "learning", "problem-solving",
    "reasoning", "judgment", "intelligence", "decision-making", "imagination",
    
    # emotions and affect
    "joy", "sadness", "anxiety", "anger", "fear", "disgust", "surprise",
    "love", "shame", "guilt", "resentment",
    
    # personality traits
    "extraversion", "introversion", "openness", "conscientiousness",
    "agreeableness", "neuroticism", "honesty", "humility", "ambition", "impulsivity",
    
    # motivation and drives
    "achievement", "affiliation", "power", "curiosity", "autonomy", "competence",
    "security", "altruism", "aggression",
    
    # social and interpersonal
    "empathy", "cooperation", "competition", "persuasion", "conformity",
    "prejudice", "stereotype", "attachment", "communication",
    
    # clinical and abnormal
    "depression", "mania", "phobia", "obsession", "compulsion",
    "addiction", "trauma", "delusion", "hallucination", "dissociation",
    
    # developmental
    "attachment", "separation", "maturation", "socialization",
    "identity", "adolescence", "morality",
    
    # behavior
    "conditioning", "habituation", "reinforcement", "extinction",
    "imitation", "avoidance", "repression"
]

    for i in range(n):
        seed = instructions[i % len(instructions)]
        generated = generator.generate_text(start_word=seed, length=230)
        generated_set = set(generated.lower().split())
        intersection = psychological_set & generated_set
        intersect_size = len(intersection)
        if intersect_size > max_intersection:
            max_intersection = intersect_size
            max_text = generated
            max_seed = seed

    print(f"Best instruction word: {max_seed}")
    print("Generated text sample:")
    print(max_text)
# Example runner
def main():
    num_neurons = 256
    num_poly = 8
    steps = 10
    snn = PolymorphicSNN(num_neurons, num_poly)
    features = torch.rand((steps, num_neurons))
    spk_list, mem_list = [], []
    for t in range(steps):
        spk, mem, _ = snn(features[t])
        spk_list.append(spk)
        mem_list.append(mem)
    spk_rec = torch.stack(spk_list)
    gdata = create_neuron_graph(spk_rec)
    gcn = DataAwareGCN(gdata.x.shape[1])
    node_feats = gcn(gdata)
    print("Node features from GCN: ", node_feats.shape)
    
    with open(input("Filename: "), 'r', encoding='utf-8') as f:
        text_corpus = f.read()
   
    generator = TextGenerator(text_corpus.lower().split())
    while True:
        print(max_psychological_overlap(generator, input("USER: "), n=1000))

if __name__ == "__main__":
    main()
