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
        self._init_impl(input_dim, num_modes)

    def _init_impl(self, input_dim, num_modes=256):
        super().__init__()
        self.modes = nn.ModuleList([
            snn.Leaky(beta=i/(num_modes+1)) for i in range(num_modes) # Avoid beta=0
        ])
        self.mode_selector = nn.Linear(input_dim, num_modes)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x, mem_states=None):
        return self._forward_impl(x, mem_states)

    def _forward_impl(self, x, mem_states=None):
        mode_scores = self.softmax(self.mode_selector(x))
        mode_outs, new_states = [], []
        for i, mode in enumerate(self.modes):
            out, state = mode(x) if mem_states is None else mode(x, mem_states[i])
            mode_outs.append(out)
            new_states.append(state)
        
        stacked = torch.stack(mode_outs, dim=0)
        # Apply weighted sum by transposing and broadcasting
        mixed = (stacked.permute(1,0,2) * mode_scores.unsqueeze(-1)).sum(1)
        return mixed, new_states, mode_scores

# Top-level SNN
class PolymorphicSNN(nn.Module):
    def __init__(self, num_neurons, num_polymorphic):
        self._init_impl(num_neurons, num_polymorphic)

    def _init_impl(self, num_neurons, num_polymorphic):
        super().__init__()
        self.lin = nn.Linear(num_neurons, num_neurons)
        self.regular = snn.Leaky(beta=0.5)
        self.poly = nn.ModuleList([PolymorphicNeuron(num_neurons, 8) for _ in range(num_polymorphic)])

    def forward(self, x, mem=None, pmem=None):
        return self._forward_impl(x, mem, pmem)

    def _forward_impl(self, x, mem=None, pmem=None):
        if x.dim() == 1:
            x = x.unsqueeze(0)
        x = self.lin(x)
        reg_out, reg_mem = self.regular(x) if mem is None else self.regular(x, mem)
        
        return reg_out**x,  reg_mem**x,reg_mem

# Create a simple graph from the spike raster
def _create_neuron_graph_impl(spk_rec):
    # node_features shape: (num_nodes, num_node_features)
    node_features = spk_rec.T
    num_nodes = node_features.shape[0]
    edge_index = []
    
    if num_nodes == 0:
        return Data(x=node_features, edge_index=torch.empty((2, 0), dtype=torch.long))
    elif num_nodes == 1:
        # Self-loop for single-node graph
        edge_index = [[0], [0]]
    else:
        # Ring topology for multi-node graph
        src_nodes, dst_nodes = [], []
        for i in range(num_nodes):
            neighbor = (i + 1) % num_nodes
            src_nodes.append(i)
            dst_nodes.append(neighbor)
            src_nodes.append(neighbor)
            dst_nodes.append(i)
        edge_index = [src_nodes, dst_nodes]

    return Data(x=node_features, edge_index=torch.tensor(edge_index, dtype=torch.long))

def create_neuron_graph(spk_rec):
    return _create_neuron_graph_impl(spk_rec)

# GCN to analyze node features
class DataAwareGCN(nn.Module):
    def __init__(self, input_dim, hidden=64, out=32):
        self._init_impl(input_dim, hidden, out)

    def _init_impl(self, input_dim, hidden=64, out=32):
        super().__init__()
        self.conv1 = GCNConv(input_dim, hidden)
        self.conv2 = GCNConv(hidden, out)

    def forward(self, data):
        return self._forward_impl(data)

    def _forward_impl(self, data):
        x = torch.relu(self.conv1(data.x, data.edge_index))
        x = torch.relu(self.conv2(x, data.edge_index))
        return x

class TextGenerator:
    def __init__(self, corpus):
        self._init_impl(corpus)

    def _init_impl(self, corpus):
        self.transitions = defaultdict(list)
        self.build_transitions(corpus)

    def build_transitions(self, words):
        self._build_transitions_impl(words)

    def _build_transitions_impl(self, words):
        for i in range(len(words) - 1):
            self.transitions[words[i]].append(words[i + 1])

    def generate_text(self, start_word=None, length=50):
        return self._generate_text_impl(start_word, length)
        
    def _generate_text_impl(self, start_word=None, length=50):
        if not self.transitions:
            return "No data for generation."
        
        # Ensure start_word is valid
        if start_word not in self.transitions:
            current_word = random.choice(list(self.transitions.keys()))
        else:
            current_word = start_word

        words = [current_word]
        for _ in range(length - 1):
            next_words = self.transitions.get(current_word)
            if not next_words:
                # Failsafe: pick a new random word if current one has no followers
                current_word = random.choice(list(self.transitions.keys()))
            else:
                current_word = random.choice(next_words)
            words.append(current_word)
        return ' '.join(words)

def _max_psychological_overlap_impl(generator, psychological_words, n=1000):
    max_intersection = 0
    max_text = ""
    max_seed = None
    psychological_set = set(w.lower() for w in psychological_words)
    
    # Static list of seed words
    instructions = list(set([
        "perception", "attention", "memory", "learning", "problem-solving",
        "reasoning", "judgment", "intelligence", "decision-making", "imagination",
        "joy", "sadness", "anxiety", "anger", "fear", "disgust", "surprise",
        "love", "shame", "guilt", "resentment", "extraversion", "openness",
        "conscientiousness", "agreeableness", "neuroticism", "honesty", "humility",
        "ambition", "impulsivity", "achievement", "affiliation", "power", "curiosity",
        "autonomy", "competence", "security", "altruism", "aggression", "empathy",
        "cooperation", "competition", "persuasion", "conformity", "prejudice",
        "stereotype", "attachment", "communication", "depression", "mania", "phobia",
        "obsession", "compulsion", "addiction", "trauma", "delusion",
        "hallucination", "dissociation", "separation", "maturation",
        "socialization", "identity", "adolescence", "morality", "conditioning",
        "habituation", "reinforcement", "extinction", "imitation", "avoidance", "repression"
    ]))

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

    print(f"\nBest instruction word: {max_seed}")
    print("Generated text sample:")
    print(max_text)
    # This function prints but does not return a value
    
def max_psychological_overlap(generator, psychological_words, n=1000):
    _max_psychological_overlap_impl(generator, psychological_words, n)

# Example runner
def _main_impl():
    # Use parameters that ensure a multi-node graph for GCN
    num_neurons = 256
    num_poly = 8
    steps = 10
    
    snn = PolymorphicSNN(num_neurons, num_poly)
    features = torch.rand((steps, num_neurons))
    
    spk_list, mem_list = [], []
    for t in range(steps):
        spk, mem, _ = snn(features[t])
        spk_list.append(spk)
        
    spk_rec = torch.cat(spk_list, dim=0)
    
    # Process graph data
    gdata = create_neuron_graph(spk_rec)
    if gdata.num_nodes > 1:
        gcn = DataAwareGCN(gdata.num_node_features)
        node_feats = gcn(gdata)
        print(f"Node features from GCN: {node_feats.shape}")
    else:
        node_feats = gdata.x
        print(f"Single/no node graph, using raw features: {node_feats.shape}")
    
    # Load corpus and initialize generator
    try:
        filename = input("Enter filename for text corpus: ")
        with open(filename, 'r', encoding='utf-8') as f:
            text_corpus = f.read()
    except FileNotFoundError:
        print("File not found. Using default corpus.")
        text_corpus = "the quick brown fox jumps over the lazy dog attention memory learning"
   
    generator = TextGenerator(text_corpus.lower().split())
    
    # Interactive loop
    while True:
        user_input = input("USER: ")
        # Split user input into a list of words for the function
        max_psychological_overlap(generator, user_input.split(), n=1000)

def main():
    _main_impl()

if __name__ == "__main__":
    main()
