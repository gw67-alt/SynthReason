import string
import torch
from collections import Counter
import numpy as np
import re
import os
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed

# Parameters
KB_LIMIT = -1
SEQUENCE_LENGTH = 2
DECAY_FACTOR = 1.9  # Decay factor for stable diffusion
WINDOW_SIZE = 5000  # Size of the window to consider for adjustments

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
    
    def apply_set_theory_modifiers(self, probs, words, vocab_inv):
        """Apply multiple set theory concepts to the probability distribution"""
        modified_probs = probs.copy()
        
        # Get category word lists for different concepts
        action_words = self.get_category_words("actions")
        description_words = self.get_category_words("descriptions")
        common_words = self.get_category_words("common")
        diverse_words = self.get_category_words("diverse")
        
        # Apply each active set theory operation
        for op_key, operation in self.set_operations.items():
                # Apply operation-specific modifications
                if op_key == 'empty_not_in':
                    # ∅∩∉ operation: Boost emptiness, penalize presence
                    for i, word_idx in enumerate(words):
                        if word_idx in vocab_inv:
                            word = vocab_inv[word_idx].lower()
                            if any(empty_word in word for empty_word in description_words):
                                modified_probs[i] *= operation['empty_boost']
                            if any(presence_word in word for presence_word in action_words):
                                modified_probs[i] *= operation['contradiction_penalty']
                
                elif op_key == 'union':
                    # Union operation: Boost diversity, penalize repetition
                    recent_words = set()
                    for i, word_idx in enumerate(words):
                        if word_idx in vocab_inv:
                            word = vocab_inv[word_idx].lower()
                            if any(diverse_word in word for diverse_word in diverse_words):
                                modified_probs[i] *= operation['diversity_boost']
                            if word in recent_words:
                                modified_probs[i] *= operation['repetition_penalty']
                            recent_words.add(word)
                
                elif op_key == 'intersection':
                    # Intersection operation: Boost commonality, penalize divergence
                    for i, word_idx in enumerate(words):
                        if word_idx in vocab_inv:
                            word = vocab_inv[word_idx].lower()
                            if any(common_word in word for common_word in common_words):
                                modified_probs[i] *= operation['commonality_boost']
                            if any(diverse_word in word for diverse_word in diverse_words):
                                modified_probs[i] *= operation['divergence_penalty']
                
                elif op_key == 'complement':
                    # Complement operation: Boost inverse concepts, penalize similarity
                    # This requires knowledge of antonyms, but as a simple approximation:
                    for i, word_idx in enumerate(words):
                        if word_idx in vocab_inv:
                            word = vocab_inv[word_idx].lower()
                            # Simple approximation: boost words with negative prefixes
                            if word.startswith(('un', 'non', 'in', 'dis', 'anti')):
                                modified_probs[i] *= operation['inverse_boost']
                            # Penalize words that are very common (as an approximation of similarity)
                            if word in common_words:
                                modified_probs[i] *= operation['similarity_penalty']
        
        # Ensure probabilities are valid
        modified_probs = np.maximum(modified_probs, 0)
        if modified_probs.sum() > 0:
            modified_probs /= modified_probs.sum()
        else:
            # If all probabilities became zero, revert to original
            modified_probs = probs.copy()
            
        return modified_probs

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

# Preprocess the text data
def preprocess_text(text, vocab):
    text = re.sub(r'[^\w\s]', '', text.lower())
    tokens = text.split()
    return [vocab[word] for word in tokens if word in vocab]

def create_sequences(text_data, vocab, sequence_length, char_ratios, topic_keywords, num_threads=None, progress_callback=None):
    """
    Create sequences and normalize transition probabilities with topic categorization using multiple threads.
    
    Args:
        text_data: The preprocessed text data as a string
        vocab: Dictionary mapping words to indices
        sequence_length: Length of input sequences
        char_ratios: Dictionary with character weighting ratios
        topic_keywords: Dictionary of topics and their associated keywords
        num_threads: Number of threads to use (defaults to CPU count if None)
        progress_callback: Optional callback function to report progress (0-100)
        
    Returns:
        tuple: (transition_dict, topic_transition_dict)
    """
    import threading
    from concurrent.futures import ThreadPoolExecutor, as_completed
    from collections import Counter
    import tqdm
    
    # Preprocess text data
    data = preprocess_text(text_data, vocab)
    
    # Determine chunk size based on number of threads
    if num_threads is None:
        import multiprocessing
        num_threads = multiprocessing.cpu_count()
    
    # Create chunks of data for parallel processing
    chunk_size = max(1, (len(data) - sequence_length) // num_threads)
    chunks = [(i, min(i + chunk_size, len(data) - sequence_length)) 
              for i in range(0, len(data) - sequence_length, chunk_size)]
    
    # Lock for thread-safe updates to shared dictionaries
    lock = threading.Lock()
    
    # Create reverse vocabulary for topic identification
    vocab_inv = {idx: word for word, idx in vocab.items()}
    
    # Shared dictionaries for results
    transition_dict = {}
    topic_transition_dict = {}
    
    # Create progress bar
    total_items = len(data) - sequence_length
    pbar = tqdm.tqdm(total=total_items, desc="Processing sequences", disable=progress_callback is not None)
    processed_items = 0
    
    def process_chunk(start_idx, end_idx):
        """Process a chunk of the data and return local dictionaries"""
        nonlocal processed_items
        local_transition_dict = {}
        local_topic_dict = {}
        local_processed = 0
        
        for i in range(start_idx, end_idx):
            input_seq = tuple(data[i:i + sequence_length])
            target_word = data[i + sequence_length]
            
            # Identify the topic for this sequence
            topic = identify_topic(input_seq, vocab_inv, topic_keywords)
            
            # Update local transition dictionary
            if input_seq not in local_transition_dict:
                local_transition_dict[input_seq] = Counter()
            local_transition_dict[input_seq][target_word] += char_ratios.get(data[i], 1)
            
            # Update local topic-specific transition dictionary
            if topic not in local_topic_dict:
                local_topic_dict[topic] = {}
            if input_seq not in local_topic_dict[topic]:
                local_topic_dict[topic][input_seq] = Counter()
            local_topic_dict[topic][input_seq][target_word] += char_ratios.get(data[i], 1)
            
            local_processed += 1
            
            # Update progress every 1000 items to avoid lock contention
            if local_processed % 1000 == 0:
                with lock:
                    processed_items += 1000
                    pbar.update(1000)
                    if progress_callback:
                        progress_callback(int(processed_items * 100 / total_items))
        
        # Update remaining progress
        with lock:
            remaining = local_processed % 1000
            if remaining > 0:
                processed_items += remaining
                pbar.update(remaining)
                if progress_callback:
                    progress_callback(int(processed_items * 100 / total_items))
        
        return local_transition_dict, local_topic_dict
    
    def merge_results(local_transition_dict, local_topic_dict):
        """Merge local dictionaries into the shared dictionaries"""
        with lock:
            # Merge transition dictionaries
            for input_seq, counter in local_transition_dict.items():
                if input_seq not in transition_dict:
                    transition_dict[input_seq] = Counter()
                transition_dict[input_seq].update(counter)
            
            # Merge topic dictionaries
            for topic, transitions in local_topic_dict.items():
                if topic not in topic_transition_dict:
                    topic_transition_dict[topic] = {}
                
                for input_seq, counter in transitions.items():
                    if input_seq not in topic_transition_dict[topic]:
                        topic_transition_dict[topic][input_seq] = Counter()
                    topic_transition_dict[topic][input_seq].update(counter)
    
    # Process chunks in parallel
    with ThreadPoolExecutor(max_workers=num_threads) as executor:
        futures = [executor.submit(process_chunk, start, end) for start, end in chunks]
        
        for future in as_completed(futures):
            try:
                local_transition_dict, local_topic_dict = future.result()
                merge_results(local_transition_dict, local_topic_dict)
            except Exception as e:
                pbar.write(f"Error processing chunk: {str(e)}")
    
    pbar.close()
    
    # Normalize general transition probabilities
    pbar = tqdm.tqdm(total=len(transition_dict), desc="Normalizing transitions")
    for i, (key, counter) in enumerate(transition_dict.items()):
        total = sum(counter.values())
        if total > 0:  # Avoid division by zero
            transition_dict[key] = {k: (v / total) * char_ratios.get(k, 1) for k, v in counter.items()}
        if i % 100 == 0 and progress_callback:
            progress_callback(int(i * 100 / len(transition_dict)))
        pbar.update(1)
    
    # Normalize topic-specific transition probabilities
    pbar.close()
    total_topic_entries = sum(len(transitions) for transitions in topic_transition_dict.values())
    pbar = tqdm.tqdm(total=total_topic_entries, desc="Normalizing topic transitions")
    
    current = 0
    for topic, transitions in topic_transition_dict.items():
        for key, counter in transitions.items():
            total = sum(counter.values())
            if total > 0:  # Avoid division by zero
                topic_transition_dict[topic][key] = {k: (v / total) * char_ratios.get(k, 1) for k, v in counter.items()}
            current += 1
            if current % 100 == 0 and progress_callback:
                progress_callback(int(current * 100 / total_topic_entries))
            pbar.update(1)
    
    pbar.close()
    
    return transition_dict, topic_transition_dict

def save_transition_dicts(transition_dict, topic_transition_dict, filepath):
    """
    Save transition dictionaries to a file using pickle.
    
    Args:
        transition_dict: The general transition dictionary
        topic_transition_dict: The topic-specific transition dictionary
        filepath: Path to the output file
    """
    import pickle
    import os
    
    # Ensure directory exists
    os.makedirs(os.path.dirname(os.path.abspath(filepath)), exist_ok=True)
    
    data = {
        'transition_dict': transition_dict,
        'topic_transition_dict': topic_transition_dict
    }
    
    print(f"Saving transition dictionaries to {filepath}...")
    with open(filepath, 'wb') as f:
        pickle.dump(data, f)
    print(f"Saved successfully!")

def load_transition_dicts(filepath):
    """
    Load transition dictionaries from a file.
    
    Args:
        filepath: Path to the input file
        
    Returns:
        tuple: (transition_dict, topic_transition_dict)
    """
    import pickle
    import os
    
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"File not found: {filepath}")
    
    print(f"Loading transition dictionaries from {filepath}...")
    with open(filepath, 'rb') as f:
        data = pickle.load(f)
    
    transition_dict = data.get('transition_dict', {})
    topic_transition_dict = data.get('topic_transition_dict', {})
    
    print(f"Loaded {len(transition_dict)} general transitions and {len(topic_transition_dict)} topic-specific transitions")
    return transition_dict, topic_transition_dict


def identify_topic(sequence, vocab_inv, topic_keywords):
    """
    Identify the topic of a sequence based on keywords.
    
    Args:
        sequence: Tuple of word indices
        vocab_inv: Inverse vocabulary (index to word)
        topic_keywords: Dictionary of topics and their associated keywords
        
    Returns:
        str: The identified topic or 'general' if no specific topic is found
    """
    # Convert sequence indices to words
    words = [vocab_inv.get(idx, '') for idx in sequence]
    
    # Check for topic keywords in the sequence
    for topic, keywords in topic_keywords.items():
        if any(keyword in words for keyword in keywords):
            return topic
    
    return 'general'  # Default topic if no specific topic is found

# Generate text using Markov chain with set theory modifications and topic bias
def generate_text(prompt, vocab, transition_dict, char_ratios, set_modifier, topic_transition_dict=None, topic=None, topic_bias=0.7, seq_length=3, max_length=250):
    vocab_inv = {idx: word for word, idx in set(vocab.items())}
    input_indices = [vocab[word] for word in prompt.lower().split() if word in vocab]
    while len(input_indices) < seq_length:
        input_indices = [vocab['<PAD>']] + input_indices
    
    generated_text = prompt
    recent_transitions = []
    
    for _ in range(max_length):
        input_tuple = tuple(input_indices[-seq_length:])
        
        # Determine if we have transitions for this sequence
        has_general = input_tuple in transition_dict
        has_topic = topic and topic in topic_transition_dict and input_tuple in topic_transition_dict[topic]
        
        if has_general or has_topic:
            # If we have topic-specific transitions and general transitions, blend them
            if has_general and has_topic:
                general_probs = transition_dict[input_tuple]
                topic_probs = topic_transition_dict[topic][input_tuple]
                
                # Combine words from both dictionaries
                all_words = set(general_probs.keys()) | set(topic_probs.keys())
                words = list(all_words)
                
                # Initialize probabilities array
                probs = np.zeros(len(words))
                
                # Fill in probabilities, blending between general and topic-specific
                for i, word_idx in enumerate(words):
                    general_prob = general_probs.get(word_idx, 0)
                    topic_prob = topic_probs.get(word_idx, 0)
                    # Blend probabilities using topic_bias
                    probs[i] = (1 - topic_bias) * general_prob + topic_bias * topic_prob
            
            # If we only have general transitions
            elif has_general:
                probs_dict = transition_dict[input_tuple]
                words = list(probs_dict.keys())
                probs = np.array(list(probs_dict.values()), dtype=float)
            
            # If we only have topic-specific transitions
            elif has_topic:
                probs_dict = topic_transition_dict[topic][input_tuple]
                words = list(probs_dict.keys())
                probs = np.array(list(probs_dict.values()), dtype=float)
            
            # Apply character ratio masking to probabilities
            for i, word_idx in enumerate(words):
                if word_idx in vocab_inv:
                    word = vocab_inv[word_idx]
                    if word and len(word) > 0:
                        first_char = word[0].lower()
                        if first_char in char_ratios:
                            # Apply char ratio as a mask/modifier to the probability
                            probs[i] *= (1.0 + char_ratios[first_char])
            
            # Apply set theory modifiers to probabilities
            probs = set_modifier.apply_set_theory_modifiers(probs, words, vocab_inv)
            
            # Continue with existing logic for recent transitions
            next_word = "a"
            for i in range(1, min(WINDOW_SIZE, len(recent_transitions)) + 1):
                past_transition = recent_transitions[-i]
                decay = DECAY_FACTOR ** char_ratios.get(next_word[0], 1)
                if past_transition in words:
                    try:
                        probs[words.index(past_transition)] *= char_ratios.get(next_word[0], 1)
                    except:
                        pass
            
            # Ensure probabilities are valid again after all modifications
            probs = np.maximum(probs, 0)
            if probs.sum() > 0:
                probs /= probs.sum()
                next_word_idx = np.random.choice(words, p=probs)
                next_word = vocab_inv[next_word_idx]
            else:
                next_word_idx = np.random.choice(words)
                next_word = vocab_inv[next_word_idx]
        else:
            break
            
        generated_text += ' ' + next_word
        input_indices.append(next_word_idx)
        input_tuple = tuple(input_indices[-seq_length:])
        recent_transitions.append(next_word_idx)
        if len(recent_transitions) > WINDOW_SIZE:
            recent_transitions.pop(0)
    
    return generated_text

# Create word category files if they don't exist
def ensure_category_files_exist():
    # Define categories and their example words
    categories = {
        "actions": ["create", "move", "add", "include", "insert", "join", "combine", "contain", "exist", "have", 
                   "hold", "keep", "maintain", "possess", "retain", "sustain", "obtain", "acquire", "gain"],
        "descriptions": ["empty", "void", "absent", "lacking", "missing", "without", "none", "nothing", "hollow", 
                        "vacant", "barren", "bare", "blank", "desolate", "devoid", "exhausted", "gone", "vacuous"],
        "common": ["the", "and", "of", "to", "a", "in", "that", "is", "was", "he", "for", "it", "with", "as", 
                  "his", "on", "be", "at", "by", "had", "are", "but", "from", "they", "she", "this", "not"],
        "diverse": ["unique", "distinct", "different", "varied", "assorted", "diverse", "eclectic", "heterogeneous", 
                   "manifold", "miscellaneous", "mixed", "multifarious", "sundry", "unusual", "unlike", "rare"]
    }
    
    # Create files if they don't exist
    for category, words in categories.items():
        filename = f"{category}.txt"
        if not os.path.exists(filename):
            with open(filename, "w", encoding="utf-8") as f:
                f.write("\n".join(words))
            print(f"Created {filename} with default words")
    return categories
# Main function
import re
from collections import Counter

def detect_topic(text, topic_keywords):
    """
    Automatically detect the most likely topic based on keyword frequency in the input text.
    
    Args:
        text (str): The input text to analyze
        topic_keywords (dict): Dictionary mapping topics to their related keywords
        
    Returns:
        str: The most likely topic or None if no clear topic is detected
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

# Modified main loop to incorporate automatic topic detection
def main():
    try:
 # Ensure category files exist
        ensure_category_files_exist()

        # Initialize set theory modifier
        set_modifier = SetTheoryModifier()
        
        # Load text data and calculate character ratios
        with open("kb.txt", "r", encoding="utf-8") as f:
            text = ' '.join(f.read().split()[:KB_LIMIT])
        text = re.sub(r'\d+', '', text)
        pattern = r'^[a-zA-Z]{1,2}$'

        # List of exceptions (words we want to keep)
        exceptions = ['a', 'i', 'to', 'is', 'it', 'an', 'of', 'by', 'he', 'me', 'we', 'be', 'my', 'up', 'do', 'go', 'if', 'no', 'so', 'on', 'at', 'in', 'as', 'or', 'la', 'ah', 'uh', 'ye', 'ab', 'ad', 'ae', 'ba', 'bi', 'bo', 'da', 'ed', 'ef', 'eh', 'el', 'em', 'en', 'er', 'es', 'et', 'ex', 'fa', 'hi', 'ho', 'id', 'is', 'jo', 'ka', 'la', 'li', 'lo', 'ma', 'me', 'mi', 'mu', 'na', 'no', 'nu', 'od', 'oe', 'oi', 'om', 'op', 'os', 'ow', 'ox', 'oy', 'pa', 're', 'sh', 'si', 'ta', 'uh', 'um','un', 'up', 'us', 'ut', 'va', 'ye', 'yo']
        
        # Filter out the short, potentially nonsensical terms, keeping exceptions
        filtered_words = [word for word in text.split() if not re.match(pattern, word) or word in exceptions]
        char_ratios = calculate_character_ratios(filtered_words)

        # Build vocabulary
        tokens = filtered_words
        word_counts = Counter(tokens)
        vocab = {word: idx for idx, (word, _) in enumerate(word_counts.items(), 1)}
        vocab['<PAD>'] = 0
        
        # Define topic keywords (customize these based on your text corpus)
        topic_keywords = {
        "science": [
            "science", "physics", "chemistry", "biology", "experiment", "theory", "research", "data",
            "laboratory", "scientist", "hypothesis", "observation", "methodology", "analysis", "discovery",
            "innovation", "quantum", "molecular", "cellular", "organism", "evolution", "genetics", "neuroscience",
            "astronomy", "geology", "botany", "zoology", "biochemistry", "thermodynamics", "mechanics",
            "relativity", "nuclear", "particle", "element", "compound", "reaction", "ecosystem", "specimen",
            "publication", "peer-review", "academic", "journal", "conference", "symposium", "dissertation",
            "thesis", "doctorate", "laboratory", "microscope", "telescope", "spectrometer", "empirical",
            "theoretical", "applied", "fundamental", "breakthrough", "paradigm", "scientific method"
        ],
        "history": [
            "history", "ancient", "medieval", "century", "war", "kingdom", "empire", "civilization",
            "archaeology", "artifact", "document", "archive", "manuscript", "chronicle", "memoir", "biography",
            "heritage", "legacy", "tradition", "culture", "dynasty", "monarchy", "republic", "revolution",
            "conquest", "colonization", "independence", "nationalism", "imperialism", "feudalism", "renaissance",
            "reformation", "enlightenment", "industrial revolution", "world war", "cold war", "civil war",
            "conflict", "treaty", "peace", "diplomacy", "politics", "ruler", "monarch", "emperor", "king",
            "queen", "president", "prime minister", "dictator", "aristocracy", "nobility", "peasantry",
            "slavery", "serfdom", "migration", "settlement", "conquest", "invasion", "occupation", "liberation",
            "resistance", "rebellion", "uprising", "coup", "genocide", "holocaust", "primary source"
        ],
        "technology": [
            "technology", "computer", "software", "hardware", "internet", "digital", "programming", "code",
            "algorithm", "database", "network", "server", "cloud", "cybersecurity", "encryption", "artificial intelligence",
            "machine learning", "deep learning", "neural network", "robotics", "automation", "virtual reality",
            "augmented reality", "blockchain", "cryptocurrency", "bitcoin", "quantum computing", "nanotechnology",
            "biotechnology", "genetic engineering", "CRISPR", "3D printing", "IoT", "internet of things",
            "semiconductor", "microchip", "processor", "GPU", "CPU", "RAM", "storage", "SSD", "HDD", "USB",
            "bluetooth", "wifi", "broadband", "fiber optic", "mobile", "smartphone", "tablet", "laptop",
            "wearable", "smart home", "drone", "autonomous vehicle", "electric vehicle", "battery", "solar",
            "renewable", "API", "framework", "library", "repository", "Git", "open source", "startup", "tech giant"
        ],
        "art": [
            "art", "painting", "music", "literature", "poetry", "artist", "creative", "aesthetic",
            "sculpture", "drawing", "printmaking", "ceramics", "photography", "film", "theater", "dance",
            "performance", "installation", "conceptual", "abstract", "realism", "impressionism", "expressionism",
            "cubism", "surrealism", "pop art", "minimalism", "contemporary", "modern", "postmodern",
            "renaissance", "baroque", "romantic", "classical", "avant-garde", "composition", "perspective",
            "color theory", "brushstroke", "canvas", "oil paint", "acrylic", "watercolor", "charcoal",
            "gallery", "museum", "exhibition", "curator", "collector", "patron", "commission", "portfolio",
            "critique", "interpretation", "symbolism", "metaphor", "narrative", "theme", "subject matter",
            "portrait", "landscape", "still life", "fresco", "mural", "mosaic", "collage", "mixed media"
        ],
        "philosophy": [
            "philosophy", "ethics", "moral", "existence", "consciousness", "metaphysics", "logic",
            "epistemology", "ontology", "phenomenology", "existentialism", "rationalism", "empiricism",
            "idealism", "materialism", "dualism", "monism", "pragmatism", "stoicism", "nihilism",
            "absurdism", "determinism", "free will", "relativism", "objectivism", "subjectivism",
            "utilitarianism", "deontology", "virtue ethics", "natural law", "social contract",
            "political philosophy", "aesthetics", "philosophy of mind", "philosophy of language",
            "philosophy of science", "philosophy of religion", "hermeneutics", "dialectic", "syllogism",
            "proposition", "inference", "deduction", "induction", "a priori", "a posteriori",
            "transcendence", "immanence", "being", "becoming", "essence", "substance", "accident",
            "causality", "teleology", "axiology", "value theory", "meaning of life", "good and evil"
        ],
        "medicine": [
            "medicine", "health", "disease", "treatment", "doctor", "patient", "diagnosis", "symptom", "cure", "hospital",
            "physician", "surgeon", "nurse", "pharmacist", "therapist", "clinic", "medical", "surgical", "pharmaceutical",
            "anatomy", "physiology", "pathology", "immunology", "neurology", "cardiology", "oncology", "pediatrics",
            "geriatrics", "obstetrics", "gynecology", "psychiatry", "radiology", "anesthesiology", "dermatology",
            "orthopedics", "ophthalmology", "otolaryngology", "urology", "endocrinology", "gastroenterology",
            "surgery", "medication", "prescription", "antibiotic", "vaccine", "chemotherapy", "radiation",
            "therapy", "rehabilitation", "preventive", "palliative", "emergency", "intensive care", "outpatient",
            "inpatient", "primary care", "specialist", "consultation", "referral", "prognosis", "remission",
            "chronic", "acute", "epidemic", "pandemic", "virus", "bacteria", "infection", "inflammation",
            "syndrome", "disorder", "condition", "complication", "medical history", "vital signs", "stethoscope"
        ],
        "psychology": [
            "psychology", "mind", "behavior", "cognitive", "emotion", "therapy", "mental", "personality", "trauma", "subconscious",
            "conscious", "unconscious", "psyche", "psychoanalysis", "psychiatry", "neuropsychology", "developmental",
            "clinical", "abnormal", "social", "positive", "evolutionary", "industrial", "organizational",
            "educational", "perception", "sensation", "learning", "memory", "attention", "motivation",
            "intelligence", "creativity", "reasoning", "language", "thought", "problem-solving", "decision-making",
            "depression", "anxiety", "schizophrenia", "bipolar", "PTSD", "OCD", "ADHD", "autism", "narcissism",
            "psychosis", "neurosis", "psychopath", "sociopath", "CBT", "DBT", "psychotherapy", "counseling",
            "behavior therapy", "gestalt", "humanistic", "existential", "conditioning", "reinforcement",
            "punishment", "stimulus", "response", "defense mechanism", "projection", "denial", "repression",
            "displacement", "sublimation", "attachment theory", "developmental stage", "identity", "self-concept"
        ],
        "economics": [
            "economics", "market", "finance", "economy", "trade", "investment", "capital", "monetary", "fiscal", "inflation",
            "deflation", "recession", "depression", "growth", "GDP", "GNP", "microeconomics", "macroeconomics",
            "supply", "demand", "price", "cost", "profit", "loss", "revenue", "expense", "budget", "deficit",
            "surplus", "debt", "credit", "loan", "interest", "tax", "subsidy", "tariff", "quota", "exchange rate",
            "currency", "dollar", "euro", "yen", "pound", "stock", "bond", "commodity", "futures", "options",
            "derivative", "hedge fund", "mutual fund", "ETF", "IPO", "merger", "acquisition", "bankruptcy",
            "liquidation", "privatization", "nationalization", "globalization", "outsourcing", "offshoring",
            "free market", "planned economy", "mixed economy", "capitalism", "socialism", "communism",
            "keynesian", "monetarist", "austrian school", "chicago school", "elasticity", "equilibrium",
            "perfect competition", "monopoly", "oligopoly", "monopsony", "cartel", "consumer surplus"
        ],
        "politics": [
            "politics", "government", "election", "policy", "democracy", "vote", "legislation", "parliament", "congress", "senator",
            "representative", "legislator", "constitution", "law", "regulation", "executive", "judicial", "legislative",
            "president", "prime minister", "chancellor", "monarch", "dictator", "autocrat", "bureaucracy", "administration",
            "campaign", "candidate", "ballot", "referendum", "initiative", "gerrymander", "lobbying", "special interest",
            "political party", "republican", "democrat", "conservative", "liberal", "progressive", "moderate",
            "left-wing", "right-wing", "centrist", "radical", "populist", "nationalist", "globalist", "isolationist",
            "interventionist", "hawk", "dove", "sovereignty", "citizenship", "immigration", "emigration", "asylum",
            "refugee", "diplomatic", "embassy", "ambassador", "treaty", "alliance", "coalition", "sanction",
            "corruption", "scandal", "impeachment", "coup", "revolution", "protest", "civil disobedience",
            "human rights", "civil rights", "civil liberties", "freedom of speech", "freedom of press",
            "freedom of religion", "discrimination", "equality", "social justice", "public opinion"
        ],
        "environment": [
            "environment", "climate", "ecology", "ecosystem", "sustainability", "conservation", "pollution", "biodiversity", "renewable", "habitat",
            "global warming", "climate change", "greenhouse effect", "carbon footprint", "carbon emission",
            "deforestation", "reforestation", "desertification", "erosion", "drought", "flood", "hurricane",
            "typhoon", "tornado", "wildfire", "earthquake", "tsunami", "landslide", "avalanche", "acid rain",
            "ozone layer", "ozone depletion", "air pollution", "water pollution", "soil pollution", "noise pollution",
            "light pollution", "plastic pollution", "waste management", "recycling", "upcycling", "composting",
            "landfill", "incineration", "toxic waste", "hazardous material", "pesticide", "herbicide", "fertilizer",
            "organic farming", "sustainable agriculture", "permaculture", "agroforestry", "endangered species",
            "extinct species", "invasive species", "keystone species", "indicator species", "food chain", "food web",
            "trophic level", "predator", "prey", "symbiosis", "mutualism", "parasitism", "commensalism",
            "biome", "tundra", "taiga", "temperate forest", "tropical rainforest", "grassland", "savanna", "desert"
        ],
        "education": [
            "education", "learning", "teaching", "school", "university", "student", "classroom", "curriculum", "academic", "pedagogy",
            "professor", "teacher", "instructor", "educator", "faculty", "administrator", "principal", "dean",
            "college", "community college", "vocational school", "trade school", "high school", "middle school",
            "elementary school", "preschool", "kindergarten", "homeschool", "distance learning", "online learning",
            "e-learning", "hybrid learning", "blended learning", "synchronous", "asynchronous", "lecture",
            "seminar", "tutorial", "workshop", "lab", "field trip", "internship", "practicum", "dissertation",
            "thesis", "capstone", "project-based learning", "problem-based learning", "inquiry-based learning",
            "cooperative learning", "collaborative learning", "peer learning", "differentiated instruction",
            "assessment", "evaluation", "test", "exam", "quiz", "assignment", "homework", "grade", "GPA",
            "transcript", "diploma", "degree", "bachelor", "master", "doctorate", "PhD", "scholarship",
            "fellowship", "grant", "financial aid", "student loan", "tuition", "accreditation", "education reform"
        ],
        "architecture": [
            "architecture", "building", "design", "structure", "construction", "architect", "urban", "facade", "interior", "exterior",
            "blueprint", "floor plan", "elevation", "section", "perspective", "rendering", "model", "scale",
            "proportion", "symmetry", "asymmetry", "balance", "harmony", "contrast", "rhythm", "pattern",
            "texture", "material", "concrete", "steel", "glass", "wood", "brick", "stone", "ceramic", "tile",
            "foundation", "column", "beam", "arch", "vault", "dome", "spire", "buttress", "cantilever",
            "skyscraper", "high-rise", "mid-rise", "low-rise", "residential", "commercial", "industrial",
            "institutional", "religious", "educational", "healthcare", "transportation", "infrastructure",
            "landscape architecture", "urban planning", "city planning", "zoning", "land use", "sustainable design",
            "green building", "LEED", "passive house", "net-zero", "energy efficient", "renewable energy",
            "Gothic", "Romanesque", "Renaissance", "Baroque", "Neoclassical", "Victorian", "Art Nouveau",
            "Art Deco", "Bauhaus", "Modernism", "Brutalism", "Postmodernism", "Deconstructivism", "Parametricism"
        ],
        "sports": [
            "sports", "athlete", "competition", "team", "game", "championship", "tournament", "match", "stadium", "coach",
            "player", "referee", "umpire", "official", "captain", "manager", "trainer", "conditioning", "fitness",
            "strength", "endurance", "agility", "speed", "skill", "technique", "strategy", "tactics", "play",
            "score", "goal", "point", "run", "basket", "touchdown", "home run", "penalty", "foul", "violation",
            "offense", "defense", "football", "soccer", "basketball", "baseball", "hockey", "tennis", "golf",
            "volleyball", "rugby", "cricket", "swimming", "diving", "gymnastics", "athletics", "track and field",
            "marathon", "triathlon", "cycling", "boxing", "wrestling", "martial arts", "MMA", "UFC", "skiing",
            "snowboarding", "surfing", "skateboarding", "motorsport", "Formula 1", "NASCAR", "Olympics",
            "Paralympic", "World Cup", "Super Bowl", "World Series", "NBA Finals", "Stanley Cup", "Wimbledon",
            "Masters", "Tour de France", "Commonwealth Games", "X Games", "league", "division", "conference",
            "draft", "trade", "free agent", "contract", "salary cap", "sponsorship", "endorsement", "doping"
        ],
        "food": [
            "food", "cooking", "recipe", "cuisine", "ingredient", "dish", "flavor", "taste", "restaurant", "chef",
            "culinary", "gastronomy", "gourmet", "kitchen", "dining", "meal", "breakfast", "lunch", "dinner",
            "brunch", "supper", "appetizer", "entree", "main course", "dessert", "side dish", "snack",
            "baking", "roasting", "grilling", "broiling", "sautéing", "frying", "steaming", "boiling", "poaching",
            "simmering", "braising", "stewing", "fermenting", "curing", "smoking", "preserving", "canning",
            "spice", "herb", "seasoning", "marinade", "sauce", "condiment", "garnish", "presentation", "plating",
            "menu", "a la carte", "prix fixe", "tasting menu", "buffet", "catering", "takeout", "delivery",
            "fine dining", "casual dining", "fast food", "street food", "food truck", "pop-up", "farm-to-table",
            "organic", "local", "seasonal", "sustainable", "vegan", "vegetarian", "pescatarian", "omnivore",
            "carnivore", "gluten-free", "dairy-free", "nut-free", "allergen", "nutrition", "calorie", "protein",
            "carbohydrate", "fat", "vitamin", "mineral", "processed", "whole food", "junk food", "comfort food"
        ],
        "fashion": [
            "fashion", "clothing", "style", "design", "trend", "textile", "accessory", "runway", "designer", "collection",
            "apparel", "garment", "outfit", "ensemble", "wardrobe", "couture", "haute couture", "ready-to-wear",
            "prêt-à-porter", "bespoke", "tailored", "custom", "vintage", "retro", "contemporary", "seasonal",
            "spring/summer", "fall/winter", "resort", "capsule collection", "fast fashion", "slow fashion",
            "sustainable fashion", "ethical fashion", "upcycled", "recycled", "thrifted", "secondhand",
            "fabric", "cotton", "wool", "silk", "linen", "polyester", "nylon", "rayon", "leather", "suede",
            "denim", "jersey", "tweed", "cashmere", "velvet", "chiffon", "lace", "pattern", "print", "solid",
            "stripe", "plaid", "check", "polka dot", "floral", "abstract", "geometric", "embroidery", "beading",
            "sequin", "appliqué", "fringe", "ruffle", "pleat", "drape", "silhouette", "fit", "proportion",
            "color theory", "fashion week", "model", "supermodel", "photoshoot", "editorial", "lookbook",
            "stylist", "fashion editor", "trend forecaster", "boutique", "department store", "e-commerce"
        ],
        "astronomy": [
            "astronomy", "star", "planet", "galaxy", "universe", "cosmic", "telescope", "orbit", "celestial", "constellation",
            "solar system", "sun", "moon", "mercury", "venus", "earth", "mars", "jupiter", "saturn", "uranus",
            "neptune", "pluto", "dwarf planet", "asteroid", "comet", "meteor", "meteorite", "space", "cosmos",
            "big bang", "expansion", "inflation", "dark matter", "dark energy", "black hole", "event horizon",
            "singularity", "gravity", "gravitational wave", "radiation", "electromagnetic spectrum", "light year",
            "parsec", "astronomical unit", "red shift", "blue shift", "doppler effect", "spectrum", "spectroscopy",
            "nebula", "supernova", "neutron star", "pulsar", "quasar", "white dwarf", "red giant", "brown dwarf",
            "main sequence", "stellar evolution", "stellar nucleosynthesis", "fusion", "fission", "exoplanet",
            "habitable zone", "goldilocks zone", "extraterrestrial", "SETI", "radio telescope", "optical telescope",
            "space telescope", "observatory", "astrophysics", "cosmology", "astrometry", "celestial mechanics",
            "ecliptic", "equinox", "solstice", "eclipse", "lunar phase", "tide", "satellite", "space probe",
            "space station", "rocket", "launch", "reentry", "orbit", "escape velocity", "zero gravity", "weightlessness"
        ],
        "religion": [
            "religion", "faith", "belief", "spiritual", "divine", "sacred", "worship", "ritual", "prayer", "deity",
            "god", "goddess", "creator", "omnipotent", "omniscient", "omnipresent", "transcendent", "immanent",
            "theology", "doctrine", "dogma", "creed", "scripture", "holy text", "revelation", "prophecy",
            "prophet", "messiah", "savior", "saint", "martyr", "apostle", "disciple", "clergy", "priest",
            "pastor", "minister", "rabbi", "imam", "monk", "nun", "missionary", "evangelist", "congregation",
            "church", "temple", "mosque", "synagogue", "shrine", "altar", "pulpit", "pew", "cathedral",
            "basilica", "monastery", "convent", "seminary", "pilgrimage", "pilgrimage site", "holy land",
            "baptism", "communion", "confirmation", "bar mitzvah", "bat mitzvah", "marriage", "funeral",
            "afterlife", "heaven", "hell", "purgatory", "reincarnation", "karma", "nirvana", "enlightenment",
            "salvation", "redemption", "sin", "forgiveness", "grace", "blessing", "miracle", "mysticism",
            "meditation", "contemplation", "prayer", "chant", "mantra", "yoga", "tantra", "asceticism",
            "monotheism", "polytheism", "pantheism", "atheism", "agnosticism", "secularism", "fundamentalism"
        ],
        "linguistics": [
            "linguistics", "language", "grammar", "syntax", "semantic", "phonetic", "dialect", "vocabulary", "morphology", "etymology",
            "phonology", "phoneme", "allophone", "syllable", "stress", "intonation", "prosody", "morpheme",
            "affix", "prefix", "suffix", "root", "stem", "inflection", "derivation", "compound", "word formation",
            "parts of speech", "noun", "verb", "adjective", "adverb", "pronoun", "preposition", "conjunction",
            "interjection", "article", "determiner", "auxiliary", "phrase", "clause", "sentence", "utterance",
            "subject", "predicate", "object", "complement", "modifier", "case", "tense", "aspect", "mood",
            "voice", "person", "number", "gender", "transitivity", "valency", "lexicon", "lexeme", "lexical",
            "idiom", "collocation", "connotation", "denotation", "homonym", "homophone", "homograph",
            "synonym", "antonym", "hypernym", "hyponym", "meronym", "holonym", "polysemy", "ambiguity",
            "pragmatics", "speech act", "implicature", "presupposition", "discourse", "cohesion", "coherence",
            "sociolinguistics", "register", "jargon", "slang", "pidgin", "creole", "bilingualism", "multilingualism",
            "first language", "second language", "native speaker", "language acquisition", "universal grammar"
        ],
        "archaeology": [
            "archaeology", "excavation", "artifact", "ruin", "ancient", "fossil", "civilization", "preservation", "relic", "prehistoric",
            "archaeologist", "anthropologist", "paleontologist", "dig", "site", "field work", "survey", "stratigraphy",
            "layer", "context", "provenience", "dating", "carbon dating", "radiocarbon", "dendrochronology",
            "thermoluminescence", "potassium-argon", "relative dating", "absolute dating", "chronology",
            "timeline", "period", "era", "epoch", "age", "stone age", "bronze age", "iron age", "paleolithic",
            "mesolithic", "neolithic", "chalcolithic", "antiquity", "classical", "medieval", "post-medieval",
            "modern", "cultural resource management", "heritage", "conservation", "restoration", "preservation",
            "curation", "museum", "collection", "exhibit", "archive", "documentation", "field notes", "log",
            "map", "plan", "section", "grid", "GPS", "GIS", "remote sensing", "aerial photography", "satellite imagery",
            "ground-penetrating radar", "magnetometer", "resistivity", "lidar", "photogrammetry", "3D modeling",
            "reconstruction", "experimental archaeology", "ethnoarchaeology", "geoarchaeology", "bioarchaeology",
            "zooarchaeology", "archaeobotany", "lithic analysis", "ceramic analysis", "metallurgy", "human remains"
        ],
        "engineering": [
            "engineering", "mechanical", "electrical", "civil", "design", "prototype", "manufacturing", "industrial", "aerospace", "chemical",
            "engineer", "technician", "drafting", "CAD", "blueprint", "schematic", "diagram", "specification",
            "tolerance", "precision", "accuracy", "measurement", "calibration", "quality control", "quality assurance",
            "stress", "strain", "load", "force", "torque", "pressure", "temperature", "heat transfer", "thermodynamics",
            "fluid dynamics", "hydraulics", "pneumatics", "mechanics", "statics", "dynamics", "kinematics",
            "kinetics", "vibration", "resonance", "frequency", "amplitude", "wavelength", "signal", "noise",
            "circuit", "component", "resistor", "capacitor", "inductor", "transformer", "diode", "transistor",
            "integrated circuit", "microcontroller", "microprocessor", "FPGA", "ASIC", "PCB", "wiring",
            "power supply", "battery", "generator", "motor", "actuator", "sensor", "transducer", "feedback",
            "control system", "automation", "robotics", "mechatronics", "biomedical", "biomechanical",
            "environmental", "nuclear", "petroleum", "mining", "materials science", "nanotechnology",
            "sustainability", "renewable energy", "efficiency", "optimization", "simulation", "modeling",
            "finite element analysis", "computational fluid dynamics", "structural analysis", "failure analysis"
        ],
        "gardening": [
            "gardening", "plant", "flower", "soil", "seed", "grow", "prune", "cultivate", "landscape", "botanical",
            "garden", "yard", "lawn", "bed", "container", "pot", "planter", "greenhouse", "nursery", "horticulture",
            "agriculture", "permaculture", "organic", "biodynamic", "vegetable garden", "herb garden", "flower garden",
            "rock garden", "water garden", "Japanese garden", "English garden", "formal garden", "cottage garden",
            "annual", "perennial", "biennial", "bulb", "rhizome", "tuber", "shrub", "bush", "hedge", "tree",
            "evergreen", "deciduous", "native", "exotic", "invasive", "drought-tolerant", "shade-loving",
            "full sun", "partial shade", "microclimate", "hardiness zone", "frost", "growing season", "dormancy",
            "germination", "seedling", "transplant", "cutting", "division", "layering", "grafting", "budding",
            "propagation", "pollination", "fertilization", "fruit", "vegetable", "herb", "spice", "edible",
            "ornamental", "companion planting", "crop rotation", "succession planting", "intercropping",
            "mulch", "compost", "fertilizer", "nitrogen", "phosphorus", "potassium", "micronutrient",
            "pH level", "acidic", "alkaline", "neutral", "loam", "clay", "sand", "silt", "humus", "drainage"
        ],
        "travel": [
            "travel", "destination", "journey", "tourism", "vacation", "adventure", "tourist", "explore", "itinerary", "landmark",
            "trip", "excursion", "expedition", "tour", "cruise", "safari", "backpacking", "road trip", "staycation",
            "sightseeing", "attraction", "monument", "heritage site", "UNESCO", "natural wonder", "national park",
            "reserve", "sanctuary", "beach", "mountain", "island", "desert", "forest", "jungle", "countryside",
            "rural", "urban", "metropolis", "city", "town", "village", "resort", "accommodation", "hotel",
            "motel", "inn", "hostel", "bed and breakfast", "guesthouse", "Airbnb", "camping", "glamping",
            "transportation", "flight", "airline", "airport", "terminal", "layover", "connecting flight",
            "train", "railway", "station", "bus", "coach", "subway", "metro", "tram", "ferry", "cruise ship",
            "rental car", "taxi", "ride-share", "bicycle", "walking tour", "passport", "visa", "customs",
            "immigration", "international", "domestic", "local", "foreign", "exotic", "remote", "off the beaten path",
            "all-inclusive", "package tour", "guided tour", "self-guided", "solo travel", "family vacation",
            "honeymoon", "business travel", "ecotourism", "sustainable travel", "responsible tourism"
        ],
        "film": [
            "film", "movie", "cinema", "director", "actor", "scene", "screenplay", "production", "cinematography", "studio",
            "actress", "cast", "crew", "producer", "executive producer", "production company", "distributor",
            "box office", "blockbuster", "indie", "low-budget", "high-budget", "script", "screenplay", "screenwriter",
            "adaptation", "original screenplay", "dialogue", "monologue", "narration", "voice-over", "plot",
            "storyline", "narrative", "arc", "character development", "protagonist", "antagonist", "conflict",
            "resolution", "climax", "rising action", "falling action", "exposition", "denouement", "genre",
            "drama", "comedy", "thriller", "horror", "action", "adventure", "science fiction", "fantasy",
            "romance", "musical", "animation", "documentary", "biographical", "historical", "western", "noir",
            "camera", "shot", "angle", "close-up", "medium shot", "long shot", "wide shot", "establishing shot",
            "pan", "tilt", "zoom", "dolly", "tracking shot", "crane shot", "steadicam", "handheld", "montage",
            "cut", "editing", "post-production", "special effects", "visual effects", "CGI", "practical effects",
            "makeup", "costume", "set design", "prop", "location", "soundstage", "green screen", "blue screen",
            "sound design", "foley", "score", "soundtrack", "theme music", "lighting", "color grading", "filter"
        ],
        "photography": [
            "photography", "camera", "image", "photograph", "lens", "exposure", "composition", "portrait", "landscape", "aperture",
            "photographer", "photojournalist", "commercial photographer", "fine art photographer", "amateur",
            "professional", "studio", "location", "shoot", "session", "digital", "film", "analog", "DSLR",
            "mirrorless", "point-and-shoot", "medium format", "large format", "35mm", "full frame", "crop sensor",
            "megapixel", "resolution", "sensor", "ISO", "sensitivity", "noise", "grain", "shutter speed",
            "shutter", "f-stop", "depth of field", "bokeh", "focus", "autofocus", "manual focus", "focal length",
            "wide angle", "telephoto", "prime lens", "zoom lens", "macro", "fisheye", "tilt-shift", "filter",
            "polarizer", "neutral density", "UV filter", "flash", "strobe", "continuous lighting", "diffuser",
            "reflector", "softbox", "umbrella", "tripod", "monopod", "stabilization", "raw", "JPEG", "TIFF",
            "post-processing", "editing", "retouching", "Photoshop", "Lightroom", "contrast", "saturation",
            "vibrance", "hue", "white balance", "temperature", "tint", "highlights", "shadows"
        ],
        "mythology": [
            "mythology", "myth", "legend", "folklore", "deity", "god", "goddess", "pantheon", "hero", "demigod",
            "immortal", "mortal", "divine", "sacred", "underworld", "afterlife", "creation", "origin", "cosmogony",
            "epic", "quest", "journey", "trial", "monster", "creature", "sphinx", "hydra", "minotaur", "cyclops",
            "titan", "giant", "nymph", "fairy", "elf", "dwarf", "dragon", "phoenix", "unicorn", "griffin",
            "centaur", "mermaid", "siren", "trickster", "shapeshifter", "oracle", "prophecy", "fate", "destiny",
            "sacrifice", "ritual", "worship", "offering", "temple", "shrine", "priest", "priestess", "shaman",
            "mystic", "vision", "supernatural", "magic", "spell", "enchantment", "curse", "blessing", "amulet",
            "talisman", "relic", "artifact", "symbol", "archetype", "narrative", "oral tradition", "fable",
            "parable", "allegory", "metaphor", "anthropomorphism", "theogony", "pantheon", "polytheism",
            "monotheism", "ancestor worship", "hero cult", "cultural heritage", "etiology", "moral lesson"
        ],
        "anthropology": [
            "anthropology", "culture", "society", "human", "evolution", "adaptation", "ethnography", "ethnology", "archaeology", "kinship",
            "cultural anthropology", "biological anthropology", "linguistic anthropology", "archaeological anthropology",
            "social structure", "social organization", "tribe", "clan", "band", "chiefdom", "state", "community",
            "ritual", "ceremony", "rite of passage", "initiation", "custom", "tradition", "practice", "belief",
            "worldview", "cosmology", "religion", "magic", "taboo", "totem", "symbol", "meaning", "interpretation",
            "fieldwork", "participant observation", "interview", "informant", "cultural relativism", "ethnocentrism",
            "enculturation", "acculturation", "diffusion", "assimilation", "syncretism", "cultural universals",
            "descent", "lineage", "patrilineal", "matrilineal", "bilateral", "marriage", "family", "household",
            "gender", "sex", "sexuality", "identity", "ethnicity", "race", "indigeneity", "indigenous", "native",
            "colonialism", "postcolonialism", "globalization", "migration", "diaspora", "transnationalism",
            "subsistence", "hunter-gatherer", "pastoral", "horticultural", "agricultural", "industrial", "post-industrial",
            "exchange", "reciprocity", "redistribution", "market", "gift", "commodity", "value", "wealth", "poverty",
            "power", "authority", "leadership", "politics", "law", "conflict", "resolution", "warfare", "peace"
        ],
        "oceanography": [
            "oceanography", "ocean", "sea", "marine", "aquatic", "coastal", "tide", "current", "wave", "depth",
            "physical oceanography", "chemical oceanography", "biological oceanography", "geological oceanography",
            "hydrography", "bathymetry", "hydrodynamics", "hydrology", "limnology", "marine biology", "marine ecology",
            "salinity", "temperature", "density", "pressure", "thermocline", "halocline", "pycnocline", "mixed layer",
            "seawater", "freshwater", "brackish", "estuary", "delta", "bay", "gulf", "strait", "channel", "harbor",
            "continental shelf", "continental slope", "continental rise", "abyssal plain", "mid-ocean ridge", "trench",
            "seamount", "guyot", "atoll", "reef", "coral", "plankton", "phytoplankton", "zooplankton", "nekton", "benthos",
            "pelagic", "neritic", "oceanic", "hadal", "abyssal", "bathyal", "photic zone", "aphotic zone", "euphotic zone",
            "thermohaline circulation", "global conveyor belt", "gyre", "upwelling", "downwelling", "Ekman transport",
            "Coriolis effect", "geostrophic flow", "Rossby wave", "internal wave", "tsunami", "storm surge", "rogue wave",
            "tidal bore", "spring tide", "neap tide", "diurnal tide", "semidiurnal tide", "mixed tide", "tidal range",
            "ocean acidification", "hypoxia", "dead zone", "eutrophication", "red tide", "harmful algal bloom",
            "marine pollution", "oil spill", "plastic pollution", "marine debris", "ocean warming", "sea level rise",
            "coastal erosion", "shoreline", "beach", "dune", "cliff", "wetland", "mangrove", "salt marsh", "seagrass"
        ],
        "business": [
            "business", "company", "corporation", "enterprise", "firm", "organization", "startup", "management", "leadership", "executive",
            "CEO", "CFO", "COO", "CTO", "board of directors", "shareholder", "stakeholder", "investor", "venture capital",
            "private equity", "angel investor", "IPO", "M&A", "merger", "acquisition", "divestiture", "restructuring",
            "strategy", "strategic planning", "business model", "value proposition", "competitive advantage", "core competency",
            "market", "market share", "market segment", "target audience", "customer", "client", "consumer", "B2B", "B2C",
            "marketing", "branding", "advertising", "public relations", "promotion", "sales", "revenue", "profit", "loss",
            "ROI", "ROA", "ROE", "EBITDA", "balance sheet", "income statement", "cash flow", "asset", "liability", "equity",
            "finance", "accounting", "audit", "tax", "compliance", "regulation", "corporate governance", "ethics", "CSR",
            "ESG", "sustainability", "operations", "supply chain", "logistics", "procurement", "inventory", "manufacturing",
            "production", "quality control", "Six Sigma", "lean", "agile", "human resources", "talent acquisition", "recruitment",
            "hiring", "onboarding", "training", "development", "performance management", "compensation", "benefits",
            "organizational behavior", "organizational culture", "change management", "leadership development",
            "entrepreneurship", "innovation", "disruption", "digital transformation", "e-commerce", "retail",
            "wholesale", "franchising", "licensing", "outsourcing", "offshoring", "globalization", "international business"
        ],
        "mathematics": [
            "mathematics", "math", "number", "equation", "formula", "calculation", "computation", "theorem", "proof", "axiom",
            "arithmetic", "algebra", "geometry", "trigonometry", "calculus", "statistics", "probability", "discrete mathematics",
            "topology", "analysis", "differential equations", "numerical analysis", "optimization", "game theory", "set theory",
            "number theory", "cryptography", "logic", "mathematical logic", "boolean algebra", "category theory", "group theory",
            "field theory", "ring theory", "graph theory", "combinatorics", "operations research", "linear algebra", "matrix",
            "vector", "tensor", "scalar", "function", "variable", "constant", "parameter", "coefficient", "exponent", "logarithm",
            "integral", "derivative", "limit", "sequence", "series", "convergence", "divergence", "infinity", "infinitesimal",
            "real number", "complex number", "imaginary number", "rational number", "irrational number", "integer", "natural number",
            "prime number", "composite number", "fraction", "decimal", "percentage", "proportion", "ratio", "average", "mean",
            "median", "mode", "standard deviation", "variance", "correlation", "regression", "distribution", "normal distribution",
            "binomial distribution", "Poisson distribution", "hypothesis testing", "confidence interval", "p-value", "statistical significance",
            "dimension", "coordinate", "Cartesian", "polar", "cylindrical", "spherical", "Euclidean", "non-Euclidean",
            "fractal", "chaos theory", "dynamical system", "algorithm", "computational complexity", "P vs NP", "Turing machine",
            "polynomial", "exponential", "logarithmic", "factorial", "permutation", "combination", "modular arithmetic"
        ],
        "music": [
            "music", "song", "melody", "harmony", "rhythm", "tempo", "beat", "note", "chord", "scale", 
            "musician", "composer", "performer", "conductor", "vocalist", "instrumentalist", "band", "orchestra", "ensemble", "choir",
            "pitch", "timbre", "tone", "frequency", "amplitude", "dynamics", "volume", "forte", "piano", "crescendo",
            "diminuendo", "staccato", "legato", "vibrato", "tremolo", "glissando", "arpeggio", "cadence", "modulation", "transposition",
            "major", "minor", "chromatic", "pentatonic", "diatonic", "atonal", "tonal", "modal", "key signature", "time signature",
            "measure", "bar", "staff", "clef", "treble clef", "bass clef", "notation", "score", "sheet music", "tablature",
            "composition", "arrangement", "improvisation", "interpretation", "performance practice", "technique", "articulation",
            "genre", "style", "classical", "jazz", "rock", "pop", "hip hop", "rap", "electronic", "folk", "world music", "blues",
            "country", "R&B", "soul", "funk", "disco", "techno", "house", "ambient", "experimental", "avant-garde", "indie",
            "instrumental", "vocal", "a cappella", "symphony", "concerto", "sonata", "opera", "aria", "recitative", "musical",
            "soundtrack", "film score", "theme", "leitmotif", "riff", "hook", "verse", "chorus", "bridge", "intro", "outro",
            "solo", "duet", "trio", "quartet", "quintet", "sextet", "septet", "octet", "nonet", "recording", "studio",
            "mixing", "mastering", "production", "producer", "engineer", "live performance", "concert", "recital", "tour"
        ]}
        
        # Create input sequences and transition matrices with normalized probabilities
        # Save to file
        try:
            transition_dict, topic_transition_dict = load_transition_dicts("output/transitions.pkl")
        except:
            transition_dict, topic_transition_dict = create_sequences(text, vocab, SEQUENCE_LENGTH, char_ratios, topic_keywords)
            save_transition_dicts(transition_dict, topic_transition_dict, "output/transitions.pkl")
        
        # Interactive Text Generation with embedded set theory operations and topic selection
        print("Enhanced Text Generator with Set Theory Categories")
        print("Available topics:", list(topic_keywords.keys()) + ["general"])
        print("Available commands:")
        print("  /topic <topic>         - Set the current topic")
        print("  /topic auto            - Enable automatic topic detection")
        print("  /topic manual          - Disable automatic topic detection")
        print("  /exit                  - Exit the program")
        
        current_topic = None
        auto_topic_detection = False
        
        while True:
            prompt = input("USER: ")
            
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
                        auto_topic_detection = False  # Disable auto detection when manually setting topic
                        
                    else:
                        print(f"Unknown topic: {requested_topic}")
                        print("Available topics:", list(topic_keywords.keys()) + ["general"])
                
                # [Your existing command handling code here]
            
            # Generate text
            else:
                active_topic = current_topic
                
                # Auto-detect topic if enabled and no manual topic is set
                if auto_topic_detection or current_topic is None:
                    detected_topic, matched_keywords = detect_topic(prompt, topic_keywords)
                    
                    if detected_topic:
                        if auto_topic_detection or current_topic is None:
                            active_topic = detected_topic
                            keyword_display = ", ".join(matched_keywords[:3])
                            if len(matched_keywords) > 3:
                                keyword_display += f" and {len(matched_keywords) - 3} more"
                            print(f"[Auto-detected topic: {detected_topic} based on: {keyword_display}]")
                
                generated_text = generate_text(
                    prompt, 
                    vocab, 
                    transition_dict, 
                    char_ratios,
                    set_modifier,
                    topic_transition_dict=topic_transition_dict,
                    topic=active_topic,
                    seq_length=SEQUENCE_LENGTH, 
                    max_length=250
                )
                
                print("Generated text:\n", generated_text)
    
    except FileNotFoundError:
        print("Error: test.txt file not found. Please create this file with your training text data.")
    except Exception as e:
        print(f"An error occurred: {e}")
        

if __name__ == "__main__":
    main()
