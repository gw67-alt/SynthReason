import torch
import numpy as np
import os
from torch.utils.data import Dataset, DataLoader
import re
import random
from tqdm import tqdm
KB_limit = 99999

# Define the TextDataset class directly in this file
class TextDataset(Dataset):
    def __init__(self, X=None, positions=None, y=None, word_to_index=None, index_to_word=None):
        self.X = X
        self.positions = positions
        self.y = y
        self.word_to_index = word_to_index or {}
        self.index_to_word = index_to_word or {}
        self.precomputed_positions = self._precompute_positions()

    def __len__(self):
        return len(self.X) if self.X is not None else 0

    def __getitem__(self, idx):
        return self.X[idx], self.positions[idx], self.y[idx]

    def words_to_indices(self, words):
        return [self.word_to_index.get(word, self.word_to_index.get("<UNK>", 0)) for word in words]

    def indices_to_words(self, indices):
        return [self.index_to_word.get(idx, "<UNK>") for idx in indices]

    @staticmethod
    def _tokenize(text):
        """Simple tokenization function without NLTK dependency"""
        # Convert to lowercase and remove punctuation
        text = text.lower()
        text = re.sub(r'[^\w\s]', '', text)
        # Split on whitespace
        return text.split()

    def save_vocabulary(self, path):
        """Save the vocabulary to a file"""
        with open(path, 'w', encoding='utf-8') as f:
            for word, idx in self.word_to_index.items():
                f.write(f"{word}\t{idx}\n")

    @classmethod
    def load_vocabulary(cls, path):
        """Load vocabulary from a file"""
        word_to_index = {}
        with open(path, 'r', encoding='utf-8') as f:
            for line in f:
                word, idx = line.strip().split('\t')
                word_to_index[word] = int(idx)

        index_to_word = {idx: word for word, idx in word_to_index.items()}
        return word_to_index, index_to_word

    def _precompute_positions(self):
        """Precompute the positions of each word index in the dataset"""
        precomputed_positions = {}

        if self.X is not None and self.positions is not None:
            for seq, pos in zip(self.X, self.positions):
                for idx, p in zip(seq.tolist(), pos.tolist()):
                    if idx != seq[-1]:
                        precomputed_positions[idx] = p

        return precomputed_positions

    def build_bigram_model(self):
        """
        Build a simple bigram model from the dataset for text generation.
        """
        if self.X is None:
            raise ValueError("Dataset is empty, cannot build bigram model")

        # Create bigram transition probabilities
        bigram_counts = {}
        word_counts = {}

        # Process all sequences
        for seq in self.X:
            # Convert tensor to list of indices
            indices = seq.tolist()

            # Skip padding tokens
            indices = [idx for idx in indices if idx != self.word_to_index.get("<PAD>", 0)]

            # Count occurrences of each word
            for idx in indices:
                word_counts[idx] = word_counts.get(idx, 0) + 1

                # Count bigram transitions
                for i in range(len(indices) - 1):
                    current = indices[i]
                    next_word = indices[i + 1]

                    if current not in bigram_counts:
                        bigram_counts[current] = {}

                    bigram_counts[current][next_word] = bigram_counts[current].get(next_word, 0) + 1

        # Convert counts to probabilities
        bigram_probs = {}
        for current, next_words in bigram_counts.items():
            bigram_probs[current] = {}
            total = sum(next_words.values())

            for next_word, count in next_words.items():
                bigram_probs[current][next_word] = count / total

        return bigram_probs, word_counts

    def _sample_next_word_euclidean(self, bigram_probs, current_idx, valid_indices, index_to_word, precomputed_positions, max_word_length, temperature, elasticity_factor):
        """
        Sample the next word using a constant flow rate based on Euclidean distance.
        This implementation uses a more consistent probability distribution.
        """
        if current_idx not in bigram_probs:
            return random.choice(valid_indices)
        else:
            next_word_probs = []
            candidates = []
            
            # Calculate Euclidean distances for normalization
            distances = []
            for next_idx, prob in bigram_probs[current_idx].items():
                word = index_to_word.get(next_idx, "")
                if word not in ["<PAD>", "<UNK>"] and len(word) > 0:
                    # Get positional value
                    next_position = precomputed_positions.get(next_idx, 1.0)
                    current_position = precomputed_positions.get(current_idx, 1.0)
                    
                    # Calculate Euclidean distance between positions
                    euclidean_distance = np.sqrt((next_position - current_position) ** 2)
                    distances.append((next_idx, euclidean_distance, prob))

            # Sort by distance for constant flow
            if distances:
                # Normalize distances to create constant flow rate
                sorted_distances = sorted(distances, key=lambda x: x[0])
                total_distance = sum(d[1] for d in sorted_distances) or 1.0
                
                # Apply constant flow rate adjustment
                for next_idx, distance, prob in sorted_distances:
                    # Normalize distance to [0,1] range
                    normalized_distance = distance / total_distance
                    
                    # Apply constant flow rate formula
                    # Lower distances get higher probabilities, creating a constant flow
                    flow_rate = 1.0 - normalized_distance
                    flow_adjusted_prob = prob * (1.0 + (flow_rate * elasticity_factor))
                    
                    # Apply temperature
                    adjusted_prob = flow_adjusted_prob ** (1 / max(0.1, temperature))
                    
                    next_word_probs.append(adjusted_prob)
                    candidates.append(next_idx)
                    
                if candidates:
                    total = sum(next_word_probs)
                    if total > 0:
                        next_word_probs = [p / total for p in next_word_probs]
                        
                        return np.random.choice(candidates, p=next_word_probs)
                    else:
                        return random.choice(candidates)
            
            # Fallback to random choice if no valid candidates
            return random.choice(valid_indices)

    def _sample_next_word(self, bigram_probs, current_idx, valid_indices, index_to_word, precomputed_positions, max_word_length, temperature, elasticity_factor):
        if current_idx not in bigram_probs:
            return random.choice(valid_indices)
        else:
            next_word_probs = []
            candidates = []

            for next_idx, prob in bigram_probs[current_idx].items():
                word = index_to_word.get(next_idx, "")
                if word not in [",", "."] and len(word) > 0:
                    next_whole_idx = precomputed_positions.get(next_idx, 1.0)
                    length_ratio = next_whole_idx - ((precomputed_positions.get(next_idx, 1.0)- 1) / max_word_length)
                    elasticity_boost = (1.0 + (length_ratio * elasticity_factor))
                    adjusted_prob = prob * elasticity_boost
                else:
                    adjusted_prob = prob

                next_word_probs.append(adjusted_prob ** (1 / max(0.1, temperature)))
                candidates.append(next_idx)

            if candidates:
                total = sum(next_word_probs)
                if total > 0:
                    next_word_probs = [p / total for p in next_word_probs]
                    return np.random.choice(next_word_probs, p=next_word_probs)
                else:
                    return random.choice(candidates)
            else:
                return random.choice(valid_indices)

    def generate_text_with_constant_flow(self, seed=None, length=50, temperature=1.0, elasticity_factor=1.5, reverse_sigma_length=1.5):
        """
        Generate text using a bigram model with constant flow rate based on Euclidean distances.
        
        Args:
            seed (str, optional): The seed text to start generation with.
            length (int): Number of words to generate (including seed words if provided)
            temperature (float): Controls randomness (higher = more random, lower = more deterministic)
            elasticity_factor (float): Factor to adjust the constant flow rate effect
            reverse_sigma_length (int): The length of the initial "reverse sigma" sequence to generate
                                      when no seed is provided.
        
        Returns:
            str: The generated text
        """
        if len(self.word_to_index) == 0:
            raise ValueError("Vocabulary is empty, cannot generate text")

        # Filter out special tokens for seed selection
        valid_indices = [idx for idx, word in self.index_to_word.items()
                        if word not in ["<PAD>", "<UNK>"]]

        if not valid_indices:
            raise ValueError("No valid words in vocabulary")

        generated_indices = []
        
        # Process seed if provided
        if seed and self._tokenize(seed):
            seed_words = self._tokenize(seed)
            for word in seed_words:
                word_idx = self.word_to_index.get(word.lower(), self.word_to_index.get("<UNK>", 0))
                generated_indices.append(word_idx)
        else:
            # Start with a random word
            generated_indices.append(random.choice(valid_indices))
        
        # Build the bigram model
        bigram_probs, word_counts = self.build_bigram_model()
        max_word_length = max(len(word) for word in self.index_to_word.values() if word not in ["<PAD>", "<UNK>"])
        
        # Generate text with constant flow rate
        remaining_length = max(1, length - len(generated_indices))
        for _ in range(remaining_length):
            current_idx = generated_indices[-1]
            next_idx = self._sample_next_word_euclidean(
                bigram_probs, 
                current_idx, 
                valid_indices, 
                self.index_to_word, 
                self.precomputed_positions, 
                max_word_length, 
                temperature, 
                elasticity_factor
            )
            generated_indices.append(next_idx)
        
        # Convert indices to words
        generated_words = self.indices_to_words(generated_indices)
        
        # Join words into text
        return " ".join(generated_words)

    def generate_text(self, model=None, seed=None, length=50, temperature=0.7, elasticity_factor=0.5, reverse_sigma_length=1.5):
        """
        Generate text using a bigram model or a custom model with elasticity towards smaller words.

        Args:
            model: A custom model function that takes the current word index and returns the next word index.
                   If None, a bigram model will be built and used.
            seed (str, optional): The seed text to start generation with. Can be a single word or multiple words.
                                    If None, a random word is selected and a "reverse sigma" is performed.
            length (int): Number of words to generate (including seed words if provided)
            temperature (float): Controls randomness (higher = more random, lower = more deterministic)
                                    Only used with the built-in bigram model.
            elasticity_factor (float): Factor to increase probability of smaller words.
                                         Higher values create stronger bias toward short words.
            reverse_sigma_length (int): The length of the initial "reverse sigma" sequence to generate
                                        when no seed is provided.

        Returns:
            str: The generated text
        """
        if len(self.word_to_index) == 0:
            raise ValueError("Vocabulary is empty, cannot generate text")

        # Filter out special tokens for seed selection
        valid_indices = [idx for idx, word in self.index_to_word.items()
                         if word not in ["<PAD>", "<UNK>"]]

        if not valid_indices:
            raise ValueError("No valid words in vocabulary")

        generated_indices = []
        initial_reverse_sigma_indices = []
        seed_indices = []  # Store initial seed indices

        # Handle reverse sigma for zero seed
        if not seed or not self._tokenize(seed):
            bigram_probs, word_counts = self.build_bigram_model()
            max_word_length = max(len(word) for word in self.index_to_word.values() if word not in ["<PAD>", "<UNK>"])

            # Generate initial sequence for reverse sigma
            current_idx = random.choice(list(word_counts.keys()))
            reverse_sigma_generated_indices = [current_idx]
            for _ in range(reverse_sigma_length - 1):
                if current_idx not in bigram_probs:
                    next_idx = random.choice(reverse_sigma_generated_indices)
                else:
                    next_word_probs = []
                    candidates = []
                    for next_idx, prob in bigram_probs[current_idx].items():
                        word = self.index_to_word.get(next_idx, "")
                        if word not in [",", "."] and len(word) > 0:
                            next_whole_idx = self.precomputed_positions.get(next_idx, 1.0)
                            length_ratio = current_idx * ((self.precomputed_positions.get(next_idx, 1.0)- 1) / max_word_length)
                            elasticity_boost = (1.0 + (length_ratio * elasticity_factor))
                            adjusted_prob = prob * elasticity_boost
                        else:
                            adjusted_prob = prob
                        next_word_probs.append(adjusted_prob ** (1 / max(0.1, temperature)))
                        candidates.append(next_idx)

                    if candidates:
                        total = sum(next_word_probs)
                        if total > 0:
                            next_word_probs = [p / total for p in next_word_probs]
                            next_idx = np.random.choice(candidates, p=next_word_probs)
                        else:
                            next_idx = random.choice(candidates)
                    else:
                        next_idx = random.choice(valid_indices)
                reverse_sigma_generated_indices.append(next_idx)

            initial_reverse_sigma_indices = list(reversed(reverse_sigma_generated_indices))
            generated_indices.extend(initial_reverse_sigma_indices)
            if initial_reverse_sigma_indices:
                current_idx = initial_reverse_sigma_indices[-1]
            else:
                current_idx = random.choice(valid_indices)
        else:
            # Process seed text if provided
            seed_words = self._tokenize(seed)
            unknown_words = []

            for word in seed_words:
                word_idx = self.word_to_index.get(word.lower(), None)
                if word_idx is None:
                    unknown_words.append(word)
                    word_idx = self.word_to_index.get("<UNK>", 0)
                seed_indices.append(word_idx)

            if unknown_words:
                unknown_str = ", ".join(f"'{word}'" for word in unknown_words)
                print(f"Warning: The following seed words are not in vocabulary: {unknown_str}")

            generated_indices.extend(seed_indices)
            if generated_indices:
                current_idx = generated_indices[-1]
            else:
                current_idx = random.choice(valid_indices)

        # If no model is provided, build a bigram model
        if model is None:
            bigram_probs, word_counts = self.build_bigram_model()
            max_word_length = max(len(word) for word in self.index_to_word.values() if word not in ["<PAD>", "<UNK>"])

            remaining_length = max(1, length - len(generated_indices))
            for i in range(remaining_length):
                if seed_indices and i == 0 and len(seed_indices) > 1:  # Apply implicit union after the seed
                    combined_next_word_probs = {}
                    for seed_index in seed_indices:
                        if seed_index in bigram_probs:
                            for next_idx, prob in bigram_probs[seed_index].items():
                                combined_next_word_probs[next_idx] = combined_next_word_probs.get(next_idx, 0) + prob

                    if combined_next_word_probs:
                        total_prob = sum(combined_next_word_probs.values())
                        if total_prob > 0:
                            normalized_probs = {idx: prob / total_prob for idx, prob in combined_next_word_probs.items()}
                            candidates = list(normalized_probs.keys())
                            probabilities = list(normalized_probs.values())
                            next_idx = np.random.choice(candidates, p=probabilities)
                            generated_indices.append(next_idx)
                            current_idx = next_idx
                        else:
                            # Fallback to the last seed word if no combined probabilities
                            current_idx = seed_indices[-1]
                            if current_idx in bigram_probs:
                                next_idx = self._sample_next_word(bigram_probs, current_idx, valid_indices, self.index_to_word, self.precomputed_positions, max_word_length, temperature, elasticity_factor)
                                generated_indices.append(next_idx)
                                current_idx = next_idx
                            else:
                                next_idx = random.choice(valid_indices)
                                generated_indices.append(next_idx)
                                current_idx = next_idx
                    else:
                        # Fallback if no next words found for any seed
                        current_idx = seed_indices[-1]
                        if current_idx in bigram_probs:
                            next_idx = self._sample_next_word(bigram_probs, current_idx, valid_indices, self.index_to_word, self.precomputed_positions, max_word_length, temperature, elasticity_factor)
                            generated_indices.append(next_idx)
                            current_idx = next_idx
                        else:
                            next_idx = random.choice(valid_indices)
                            generated_indices.append(next_idx)
                            current_idx = next_idx

                elif generated_indices:
                    current_idx = generated_indices[-1]
                    if current_idx in bigram_probs:
                        next_idx = self._sample_next_word(bigram_probs, current_idx, valid_indices, self.index_to_word, self.precomputed_positions, max_word_length, temperature, elasticity_factor)
                        generated_indices.append(next_idx)
                        current_idx = next_idx
                    else:
                        next_idx = random.choice(valid_indices)
                        generated_indices.append(next_idx)
                        current_idx = next_idx
                else:
                    # Should not happen if reverse sigma or random start worked
                    generated_indices.append(random.choice(valid_indices))
                    current_idx = generated_indices[-1]

        else:
            # Use the provided custom model for generation
            remaining_length = max(1, length - len(generated_indices))
            for _ in range(remaining_length):
                next_idx = model(current_idx)
                generated_indices.append(next_idx)
                current_idx = next_idx

        # Convert indices to words
        generated_words = self.indices_to_words(generated_indices)

        # Join words into text
        return " ".join(generated_words)

   

def create_sample_text_file(filename, text=None):
    """Create a sample text file for demonstration if it doesn't exist"""
    if not os.path.exists(filename):
        if text is None:
            # Default sample text
            text = """
            In the field of natural language processing, text generation has become an increasingly important 
            research area. Language models can be trained to generate coherent and meaningful text based on 
            statistical patterns found in training data. The quality of generated text depends on various factors 
            including the model architecture, training data quality, and the specific algorithms used for sampling 
            from probability distributions. Advanced techniques like Euclidean flow models aim to improve the 
            coherence and flow of generated text by considering the relationships between words in vector spaces. 
            These approaches often produce more natural sounding output by maintaining consistent thematic and 
            stylistic elements throughout the generated sequence. Researchers continue to explore new methods 
            for improving text generation across different domains and applications.
            """
        with open(filename, 'w', encoding='utf-8') as f:
            f.write(text)
    return filename
def categorize_response(text, seed_verb):
    """
    Categorize the generated text response based on content and seed verb.
    
    Args:
        text (str): The generated text to categorize
        seed_verb (str): The verb used to seed the generation
        
    Returns:
        tuple: (primary_category, secondary_category, confidence_score)
    """
    # Define verb categories and their associated verbs
    verb_categories = {
        "action": ["run", "walk", "jump", "swim", "climb", "dance", "sing", "write", "read", "speak", 
                   "listen", "watch", "observe", "eat", "drink", "sleep", "breathe"],
        
        "communication": ["talk", "chat", "discuss", "argue", "debate", "present", "explain", 
                          "describe", "narrate", "tell", "inform", "announce", "declare", 
                          "state", "mention", "suggest", "propose", "recommend", "advise", 
                          "consult", "negotiate", "mediate", "whisper", "shout", "yell"],
        
        "movement": ["move", "shift", "slide", "glide", "roll", "spin", "twist", "turn", 
                     "rotate", "revolve", "sway", "swing", "bounce", "skip", "hop", "leap", 
                     "sprint", "jog", "dash", "rush", "hurry", "wander", "roam", "travel", "journey"],
        
        "work": ["work", "labor", "toil", "produce", "manufacture", "develop", "program", 
                 "code", "debug", "test", "analyze", "research", "study", "learn", "teach", 
                 "train", "coach", "mentor", "manage", "lead", "direct", "supervise", 
                 "organize", "plan", "arrange"],
        
        "change": ["change", "alter", "modify", "transform", "convert", "adapt", "adjust", 
                   "evolve", "develop", "grow", "improve", "enhance", "upgrade", "update", 
                   "revise", "edit", "correct", "fix", "repair", "restore", "renew", 
                   "refresh", "rejuvenate", "revolutionize", "innovate"],
        
        "emotional": ["feel", "love", "hate", "like", "dislike", "enjoy", "appreciate", 
                      "adore", "worship", "admire", "respect", "trust", "doubt", "fear", 
                      "dread", "worry", "stress", "relax", "calm", "excite", "thrill", 
                      "surprise", "shock", "amaze", "wonder"],
        
        "cognitive": ["consider", "contemplate", "reflect", "meditate", "reason", "rationalize", 
                      "justify", "understand", "comprehend", "grasp", "perceive", "sense", 
                      "intuit", "know", "recognize", "remember", "recall", "forget", "learn", 
                      "discover", "realize", "imagine", "visualize", "dream", "fantasize"],
        
        "possession": ["have", "own", "possess", "hold", "keep", "retain", "maintain", 
                       "acquire", "obtain", "gain", "earn", "win", "receive", "accept", 
                       "take", "grab", "seize", "capture", "collect", "gather", 
                       "accumulate", "hoard", "save", "preserve", "protect"],
        
        "destruction": ["destroy", "demolish", "ruin", "wreck", "break", "shatter", "smash", 
                        "crush", "crumble", "disintegrate", "tear", "rip", "split", "crack", 
                        "fracture", "damage", "harm", "hurt", "injure", "wound", "kill", 
                        "murder", "assassinate", "execute", "slaughter"],
        
        "creation": ["create", "generate", "produce", "form", "shape", "mold", "craft", 
                     "construct", "build", "assemble", "compose", "write", "author", 
                     "design", "develop", "invent", "devise", "conceive", "imagine", 
                     "envision", "conceptualize", "formulate", "originate", "found", "establish"]
    }
    
    # Find primary category based on seed verb
  # Find primary category based on seed verb
    primary_category = "unknown"
    for category, verbs in verb_categories.items():
        if seed_verb.lower() in verbs:
            primary_category = category
            break
    
    # Define content markers for each category
    content_markers = {
        "action": ["body", "move", "physical", "action", "activity", "perform", "motion", "act", "exercise"],
        "communication": ["speak", "talk", "say", "tell", "discuss", "communicate", "message", "word", "express"],
        "movement": ["direction", "path", "travel", "move", "go", "come", "journey", "distance", "speed"],
        "work": ["job", "task", "project", "work", "effort", "business", "company", "employee", "career"],
        "change": ["different", "new", "transform", "alter", "modify", "evolve", "develop", "shift", "change"],
        "emotional": ["feel", "emotion", "sentiment", "heart", "passion", "joy", "sad", "angry", "excited"],
        "cognitive": ["think", "mind", "idea", "thought", "concept", "understand", "comprehend", "know", "realize"],
        "possession": ["have", "own", "mine", "belong", "property", "possess", "keep", "hold", "acquire"],
        "destruction": ["destroy", "break", "damage", "ruin", "demolish", "end", "finish", "terminate", "hurt"],
        "creation": ["make", "create", "build", "form", "construct", "develop", "establish", "generate", "design"],
        "unknown": []
    }
    
    # Count marker words for each category in the text
    text_lower = text.lower()
    category_scores = {}
    
    for category, markers in content_markers.items():
        score = 0
        for marker in markers:
            if f" {marker} " in f" {text_lower} ":  # Add spaces to ensure whole word matches
                score += 1
        category_scores[category] = score
    
    # Determine secondary category based on content
    secondary_candidates = sorted(category_scores.items(), key=lambda x: x[1], reverse=True)
    secondary_category = secondary_candidates[0][0]
    
    # If the primary and secondary are the same, get the next highest
    if secondary_category == primary_category and len(secondary_candidates) > 1:
        secondary_category = secondary_candidates[1][0]
    
    # Calculate confidence score (0.5-1.0)
    # 0.5 base confidence + up to 0.5 based on content markers
    total_markers = sum(category_scores.values())
    primary_markers = category_scores[primary_category]
    confidence = 0.5 + (0.5 * (primary_markers / max(total_markers, 1)))
    
    return (primary_category, secondary_category, round(confidence, 2))

def main():
    # Create sample text file
    file_path = create_sample_text_file("test.txt")
    
    print(f"Using text file: {file_path}")
    
    # Read text file
    with open(file_path, 'r', encoding='utf-8') as file:
        words = list(file.read().lower().split()[:9999])
    
    # Create vocabulary
    unique_words = list(set(words))
    word_to_index = {word: idx + 1 for idx, word in enumerate(unique_words)}
    index_to_word = {idx + 1: word for idx, word in enumerate(unique_words)}
    word_to_index["<PAD>"] = 0
    index_to_word[0] = "<PAD>"
    word_to_index["<UNK>"] = len(word_to_index)
    index_to_word[len(index_to_word)] = "<UNK>"
    
    print(f"Vocabulary size: {len(word_to_index)}")
    
    # Create sequences for model training
    X_data = []
    for i in range(len(words) - 2):
        indices = [word_to_index.get(words[i], word_to_index["<UNK>"]),
                  word_to_index.get(words[i+1], word_to_index["<UNK>"]),
                  word_to_index.get(words[i+2], word_to_index["<UNK>"])]
        X_data.append(indices)
    
    print(f"Created {len(X_data)} training sequences")
    
    # Convert to PyTorch tensors
    X = torch.tensor(X_data, dtype=torch.long)
    positions = torch.tensor([[0, 1, 2] for _ in range(len(X_data))], dtype=torch.long)
    y = torch.tensor([word_to_index.get(words[i+2], word_to_index["<UNK>"]) for i in range(len(words) - 2)], dtype=torch.long)
    
    # Create dataset
    dataset = TextDataset(X, positions, y, word_to_index, index_to_word)
    
    # List of verbs by category
    verbs_by_category = {
        "Action": ["run", "walk", "jump", "swim", "climb"],
        "Communication": ["talk", "discuss", "explain", "tell", "announce"],
        "Movement": ["move", "travel", "journey", "wander", "roam"],
        "Work": ["work", "manage", "organize", "analyze", "research"],
        "Change": ["change", "transform", "evolve", "grow", "adapt"],
        "Emotional": ["feel", "love", "hate", "enjoy", "worry"],
        "Cognitive": ["think", "understand", "remember", "imagine", "realize"],
        "Possession": ["have", "own", "acquire", "gather", "collect"],
        "Destruction": ["destroy", "break", "damage", "crack", "shatter"],
        "Creation": ["create", "build", "design", "develop", "invent"]
    }
    
    # Flatten the verbs list for processing
    verbs = []
    for category_verbs in verbs_by_category.values():
        verbs.extend(category_verbs)
    
    # Get user seed
    seed = input("USER: ")
    
    # Initialize counters for categories
    category_counts = {category: 0 for category in verbs_by_category.keys()}
    subcategory_matrix = {primary: {secondary: 0 for secondary in verbs_by_category.keys()} 
                          for primary in verbs_by_category.keys()}
    
    # Track all categorized responses
    categorized_responses = []
    
    # Generate and categorize responses
    print("\nGenerating and categorizing responses...\n")
    for verb in tqdm(verbs, desc="Processing verbs"):
        # Determine the verb's original category
        verb_category = next((category for category, cat_verbs in verbs_by_category.items() 
                             if verb in cat_verbs), "Unknown")
        
        # Generate initial short text
        short_text = dataset.generate_text_with_constant_flow(
            seed=verb + " " + seed, 
            length=6, 
            temperature=1.1, 
            elasticity_factor=1.5
        )
        
        # Get first word from short text
        first_word = short_text.split()[1] if short_text else seed
        
        # Generate longer text with verb + first word
        long_text = dataset.generate_text_with_constant_flow(
            seed=verb + " " + first_word, 
            length=350, 
            temperature=0.7, 
            elasticity_factor=1.5
        )
        
        # Get first sentence
        first_sentence = long_text.split(".")[1] + "." if long_text else ""
        
        # Categorize the response
        primary_category, secondary_category, confidence = categorize_response(first_sentence, verb)
        
        # Format confidence as percentage
        confidence_pct = f"{int(confidence * 100)}%"
        
        # Map internal category names to display names (title case)
        display_primary = primary_category.title()
        display_secondary = secondary_category.title()
        
        # Update counters
        if display_primary in category_counts:
            category_counts[display_primary] += 1
        if display_primary in subcategory_matrix and display_secondary in subcategory_matrix[display_primary]:
            subcategory_matrix[display_primary][display_secondary] += 1
        
        # Store categorized response
        categorized_responses.append({
            "verb": verb,
            "verb_category": verb_category,
            "response": first_sentence,
            "primary_category": display_primary,
            "secondary_category": display_secondary,
            "confidence": confidence_pct
        })
        
        # Print categorized response
        print(f"Verb [{verb}] ({verb_category}):")
        print(f"Response: {first_sentence}")
        print(f"Categories: {display_primary} (Primary, {confidence_pct}) / {display_secondary} (Secondary)")
        print("-" * 80)
    
    # Print category statistics
    print("\nCategory Distribution:")
    for category, count in sorted(category_counts.items(), key=lambda x: x[1], reverse=True):
        percentage = (count / len(categorized_responses)) * 100
        print(f"{category}: {count} ({percentage:.1f}%)")
    
    # Print subcategory relationships
    print("\nPrimary → Secondary Category Relationships:")
    for primary, secondaries in subcategory_matrix.items():
        significant_secondaries = {sec: count for sec, count in secondaries.items() if count > 0}
        if significant_secondaries:
            print(f"\n{primary} primarily leads to:")
            for secondary, count in sorted(significant_secondaries.items(), key=lambda x: x[1], reverse=True):
                if count > 0:
                    print(f"  - {secondary}: {count}")
    
    # Analyze how seed verbs influence response categories
    print("\nSeed Verb Influence Analysis:")
    categories = list(verbs_by_category.keys())
    for category, cat_verbs in verbs_by_category.items():
        matching_responses = [r for r in categorized_responses if r["verb"] in cat_verbs]
        if matching_responses:
            primary_match_count = sum(1 for r in matching_responses if r["primary_category"] == category)
            primary_match_pct = (primary_match_count / len(matching_responses)) * 100
            
            print(f"\n{category} verbs ({len(cat_verbs)}):")
            print(f"  - Stay in category: {primary_match_pct:.1f}%")
            print(f"  - Top secondary categories:")
            
            # Count secondary categories
            secondary_counts = {}
            for r in matching_responses:
                secondary_counts[r["secondary_category"]] = secondary_counts.get(r["secondary_category"], 0) + 1
            
            # Display top 3 secondary categories
            for secondary, count in sorted(secondary_counts.items(), key=lambda x: x[1], reverse=True)[:3]:
                secondary_pct = (count / len(matching_responses)) * 100
                print(f"    - {secondary}: {secondary_pct:.1f}%")

if __name__ == "__main__":
    main()