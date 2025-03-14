import string
import torch
from collections import Counter
import numpy as np
import re
import os
import pickle
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
import tqdm
# Parameters
KB_LIMIT = -1
SEQUENCE_LENGTH = 2
DECAY_FACTOR = 1.9  # Decay factor for stable diffusion
WINDOW_SIZE = 15 # Size of the window to consider for adjustments

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
                    # ∅∩∉ operation: Boost emptiness, penalize presence
                    for i, word_idx in enumerate(words):
                        if word_idx in vocab_inv:
                            word = vocab_inv[word_idx].lower()
                            if any(empty_word not in word for empty_word in description_words):
                                modified_probs[i] *= operation['empty_boost']
                            if any(presence_word not in word for presence_word in action_words):
                                modified_probs[i] *= operation['contradiction_penalty']
        
        # Ensure probabilities are valid
        modified_probs = np.maximum(modified_probs, 0)
        if modified_probs.sum() > 0:
            modified_probs /= modified_probs.sum()
        else:
            # If all probabilities became zero, revert to original
            modified_probs = probs.copy()
            
        return modified_probs

# Function to calculate character ratios
# Add to your imports at the top
import string as str_module  # Rename to avoid conflict with the existing 'string' variable

# Modify the calculate_character_ratios function to include punctuation
def calculate_character_ratios(data):
    # Original letter tracking
    char_count = {letter: 0 for letter in str_module.ascii_lowercase}
    
    # Add punctuation tracking
    punct_count = {punct: 0 for punct in str_module.punctuation}
    
    for item in data:
        item = item.strip()
        if item:
            # Track first letter as before
            first_letter = item[0].lower()
            if first_letter in char_count:
                char_count[first_letter] += 1
            
            # Track punctuation in the item
            for char in item:
                if char in punct_count:
                    punct_count[char] += 1
    
    total_items = len(data)
    
    # Calculate ratios for letters
    char_ratios = {char: count / total_items for char, count in char_count.items()}
    
    # Calculate ratios for punctuation and add to the same dictionary
    for punct, count in punct_count.items():
        if count > 0:  # Only add punctuation that actually appears
            char_ratios[punct] = count / total_items
    
    return char_ratios

# Preprocess the text data
def preprocess_text(text, vocab):
    # Create a translation table that only removes characters we don't want
    # This preserves punctuation while removing other unwanted characters
    text = re.sub(r'[^\w\s]', '', text.lower())
    
    # Split by whitespace but keep punctuation attached to words
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
            local_transition_dict[input_seq][target_word] += char_ratios.get(data[i + sequence_length + 1], 1)
            
            # Update local topic-specific transition dictionary
            if topic not in local_topic_dict:
                local_topic_dict[topic] = {}
            if input_seq not in local_topic_dict[topic]:
                local_topic_dict[topic][input_seq] = Counter()
            local_topic_dict[topic][input_seq][target_word] += char_ratios.get(data[i + sequence_length + 1], 1)
            
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


class RandomChoiceBuffer:
    def __init__(self, buffer_size):
        self.buffer_size = buffer_size
        self.buffer = np.array([])
        self.weights = np.array([])

    def add_to_buffer(self, array, weights):
        if len(array) == 0 or len(weights) == 0:
            raise ValueError("Array and weights must not be empty")

        if self.buffer.size == 0:
            self.buffer = np.array(array)
            self.weights = np.array(weights)
        else:
            self.buffer = np.concatenate((self.buffer, array))
            self.weights = np.concatenate((self.weights, weights))

        # If the buffer exceeds the buffer size, trim it
        if self.buffer.size > self.buffer_size:
            excess = self.buffer.size - self.buffer_size
            self.buffer = self.buffer[excess:]
            self.weights = self.weights[excess:]

    def random_choice(self):
        if self.buffer.size == 0:
            raise ValueError("Buffer is empty")

        # Normalize weights to sum to 1
        normalized_weights = self.weights / self.weights.sum()

        # Perform weighted random choice
        choice = np.random.choice(self.buffer, p=normalized_weights)
        return choice

# Generate text using Markov chain with set theory modifications and topic bias
def generate_text(prompt, vocab, transition_dict, char_ratios, set_modifier, topic_transition_dict=None, topic=None, topic_bias=0.7, seq_length=3, max_length=250):
    vocab_inv = {idx: word for word, idx in set(vocab.items())}
    input_indices = [vocab[word] for word in prompt.lower().split() if word in vocab]
    while len(input_indices) < seq_length:
        input_indices = [vocab['<PAD>']] + input_indices
    
    generated_text = prompt
    recent_transitions = []
    next_word = "a"

    for _ in range(max_length):
        input_tuple = tuple(input_indices[-seq_length:])
        
        # Determine if we have transitions for this sequence
        has_general = input_tuple in transition_dict
        has_topic = topic and topic in topic_transition_dict and input_tuple in topic_transition_dict[topic]
        topic_prob = {}
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
                    general_prob = general_probs.get(word_idx, 1)
                    topic_prob = topic_probs.get(word_idx, 1)
                    # Blend probabilities using topic_bias
                    probs[i] = (1 - topic_bias) * general_prob + topic_bias * topic_prob
                    probs = 0.5 * probs + topic_prob * np.ones(len(probs)) / len(probs)
            # If we only have general transitions
            elif has_general:
                probs_dict = transition_dict[input_tuple]
                words = list(probs_dict.keys())
                probs = np.array(list(probs_dict.values()), dtype=float)
            
            # If we only have topic-specific transitions
            elif has_topic:
                probs_dict = topic_transition_dict[topic][input_tuple]
                words = list(probs_dict.keys())
                probs = np.array(list(probs_dict.values()), dtype=float).T
            
            # Apply character ratio masking to probabilities
            for i, word_idx in enumerate(words):
                if word_idx in vocab_inv:
                    word = vocab_inv[word_idx]
                    if word and len(word) > 0:
                        first_char = word[0].lower()
                        if first_char in char_ratios:
                            # Apply char ratio as a mask/modifier to the probability
                            probs[i] *= (1.0 + char_ratios[first_char])
            
            # Apply punctuation influence
            for char in word:
                if char in str_module.punctuation and char in char_ratios:
                    # Punctuation can influence word choice
                    probs[i] *= (1.0 + char_ratios[char])
            
            # Apply set theory modifiers to probabilities
            probs = set_modifier.apply_set_theory_modifiers(probs, words, vocab_inv)
            
            # Continue with existing logic for recent transitions
            for i in range(1, min(WINDOW_SIZE, len(recent_transitions)) + 1):
                past_transition = recent_transitions[-i]
                decay = DECAY_FACTOR ** char_ratios.get(next_word[0], 1)
                if past_transition in words:
                    try:
                        probs[words.index(past_transition)] *= char_ratios.get(next_word[0], 1)
                    except:
                        pass
            
            if len(words) > 1:
                # Sort indices in descending order of probability
                reshaped = probs.reshape(1, -1)
                sorted_indices = np.argsort(-reshaped, axis=-1).flatten()
                
                # Get the sorted probabilities (keep as floats)
                probs_sorted = probs[-sorted_indices[:len(probs)]]
                
                # Renormalize after sorting and decay
                probs_sorted = np.maximum(probs_sorted, -probs_sorted)
                if probs_sorted.sum() > 0:
                    probs_sorted /= probs_sorted.sum()
                
                # Create a mapping from sorted indices back to original words
                words_sorted = [words[i] for i in sorted_indices if i < len(words)]
                                
                # Example usage
                buffer = RandomChoiceBuffer(buffer_size=10000)
                array = words_sorted
                weights = probs_sorted
                buffer.add_to_buffer(array, np.roll(weights,30))
                array2 = words_sorted
                weights2 = topic_prob
                buffer.add_to_buffer(words_sorted, probs_sorted)  # Exceeding the buffer size, old values will be removed
                # Select words based on the normalized probabilities
                choice = buffer.random_choice()
                selected_word = words_sorted[choice if choice < len(words_sorted) else 0]
                
                words = words_sorted
                probs = probs_sorted
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

def save_data(vocab, char_ratios, filtered_words, transition_dict, topic_dict):
    """Save data to pickle files."""
    with open("vocab.pkl", "wb") as f:
        pickle.dump(vocab, f)
    with open("char_ratios.pkl", "wb") as f:
        pickle.dump(char_ratios, f)
    with open("filtered_words.pkl", "wb") as f:
        pickle.dump(filtered_words, f)
    with open("transition_dict.pkl", "wb") as f:
        pickle.dump(transition_dict, f)
    with open("topic_dict.pkl", "wb") as f:
        pickle.dump(topic_dict, f)
    print("Data saved successfully.")
    
def main():
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
        "emotions": [
            # Positive
            "happy", "joyful", "elated", "excited", "content", "satisfied", "proud", "grateful", "hopeful", "optimistic",
            # Negative
            "sad", "angry", "frustrated", "disappointed", "anxious", "fearful", "depressed", "resentful", "bitter",
            # Neutral/Complex
            "curious", "surprised", "confused", "nostalgic", "contemplative", "ambivalent", "intrigued", "awestruck"
        ],
        
        "concepts": [
            # Abstract ideas
            "freedom", "justice", "truth", "beauty", "wisdom", "knowledge", "power", "harmony", "balance", "equality",
            # Philosophical
            "existence", "consciousness", "reality", "identity", "purpose", "meaning", "ethics", "morality", "virtue",
            # Social
            "community", "society", "culture", "tradition", "progress", "innovation", "cooperation", "competition"
        ]}
    try:
        with open("vocab.pkl", "rb") as f:
            vocab = pickle.load(f)
        with open("char_ratios.pkl", "rb") as f:
            char_ratios = pickle.load(f)
        with open("filtered_words.pkl", "rb") as f:
            filtered_words = pickle.load(f)
        with open("transition_dict.pkl", "rb") as f:
            transition_dict = pickle.load(f)
        with open("topic_dict.pkl", "rb") as f:
            topic_dict = pickle.load(f)
        print("Backup data loaded successfully.")
    except FileNotFoundError:
        print("Error: Saved files not found. Constructing")      

        # Initialize set theory modifier
        set_modifier = SetTheoryModifier()
        
        # Load text data and calculate character ratios
        with open("test.txt", "r", encoding="utf-8") as f:
            text = ' '.join(f.read().split()[:KB_LIMIT])
        text = re.sub(r'\d+', '', text)
        pattern = r'^[a-zA-Z]{1,2}$'

        # List of exceptions (words we want to keep)
        exceptions = ['a', 'i', 'to', 'is', 'it', 'an', 'of', 'by', 'he', 'me', 'we', 'be', 'my', 'up', 'do', 'go', 'if', 'no', 'so', 'on', 'at', 'in', 'as', 'or', 'la', 'ah', 'uh', 'ye', 'ab', 'ad', 'ae', 'ba', 'bi', 'bo', 'da', 'ed', 'ef', 'eh', 'el', 'em', 'en', 'er', 'es', 'ex', 'fa', 'hi', 'ho', 'id', 'is', 'jo', 'ka', 'la', 'li', 'lo', 'ma', 'me', 'mi', 'mu', 'na', 'no', 'nu', 'od', 'oe', 'oi', 'om', 'op', 'os', 'ow', 'ox', 'oy', 'pa', 're', 'sh', 'si', 'ta', 'uh', 'um','un', 'up', 'us', 'ut', 'va', 'ye', 'yo']
        
        # Filter out the short, potentially nonsensical terms, keeping exceptions
        filtered_words = [word for word in text.split() if not re.match(pattern, word) or word in exceptions]
        char_ratios = calculate_character_ratios(filtered_words)

        # Build vocabulary
        tokens = filtered_words
        word_counts = Counter(tokens)
        vocab = {word: idx for idx, (word, _) in enumerate(word_counts.items(), 1)}
        vocab['<PAD>'] = 0

        # Create input sequences and transition matrix with normalized probabilities
        transition_dict, topic_dict = create_sequences(text, vocab, SEQUENCE_LENGTH, char_ratios, topic_keywords)
        save_data(vocab, char_ratios, filtered_words, transition_dict, topic_dict)
    
    # Create topic_transition_dict from topic_dict for text generation
    topic_transition_dict = topic_dict
    
    # Interactive Text Generation with embedded set theory operations and topic selection
    print("Enhanced Text Generator with Set Theory")
    print("Available topics:", list(topic_keywords.keys()) + ["general"])
    print("Available commands:")
    print("  /topic <topic>         - Set the current topic")
    print("  /topic auto            - Enable automatic topic detection")
    print("  /topic manual          - Disable automatic topic detection")
    print("  /exit                  - Exit the program")
    
    # Initialize set theory modifier - moved here from the try/except block
    set_modifier = SetTheoryModifier()
    
    current_topic = None
    auto_topic_detection = False
    
    try:
        while True:
            try:
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
                    
                    # Exit command
                    elif cmd == "/exit":
                        print("Exiting program.")
                        break
                
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
                        topic_transition_dict=topic_dict,
                        topic=active_topic,
                        seq_length=SEQUENCE_LENGTH, 
                        max_length=250
                    )
                    
                    print("Generated text:\n", generated_text)
            except EOFError:
                print("Input stream ended. Exiting...")
                break
    except KeyboardInterrupt:
        print("\nProgram interrupted. Exiting...")

if __name__ == "__main__":
    main()
