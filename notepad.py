import numpy as np
import re
import nltk
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.util import ngrams
from collections import defaultdict, Counter
import tkinter as tk
from tkinter import ttk, scrolledtext, messagebox

class GapFillerAI:
    def __init__(self):
        self.ngram_models = {}  # Will store n-gram models of different sizes
        self.max_ngram = 5      # Maximum n-gram size to consider
        self.min_ngram = 2      # Minimum n-gram size
        self.word_frequencies = Counter()  # Stores overall word frequencies
        self.qa_pairs = []      # Store question-answer pairs for direct lookup
        self.gap_fill_pairs = [] # Store gap-fill pairs (sentence with gap and answer)
        self.complete_statements = []  # Store complete statements for training
        self.sentence_starters = Counter()  # Track words that commonly start sentences
        
    def preprocess_text(self, text):
        """Clean and tokenize the text."""
        # Convert to lowercase and remove excessive whitespace
        text = text.lower()
        text = re.sub(r'\s+', ' ', text).strip()
        
        # Tokenize the text
        try:
            tokens = word_tokenize(text)
        except LookupError:
            # Download necessary NLTK data if not available
            nltk.download('punkt', quiet=True)
            tokens = word_tokenize(text)
            
        return tokens
    
    def parse_training_data(self, training_data):
        """Parse training data in the specific format given."""
        lines = training_data.strip().split('\n')
        i = 0
        
        while i < len(lines):
            line = lines[i].strip()
            if not line:
                i += 1
                continue
                
            # Check if this is a question-answer pair
            if i + 1 < len(lines) and not re.search(r'_{2,}', line):
                question = line
                answer = lines[i+1].strip()
                if answer and not answer.startswith("A ") and not re.search(r'_{2,}', answer):
                    self.qa_pairs.append((question, answer))
                    i += 2
                    continue
            
            # Check if this is a gap-fill statement
            if re.search(r'_{2,}', line) and i + 1 < len(lines):
                gap_statement = line
                answer = lines[i+1].strip()
                if answer and not re.search(r'_{2,}', answer):
                    self.gap_fill_pairs.append((gap_statement, answer))
                    
                    # Also create a complete sentence for n-gram training
                    filled_statement = re.sub(r'_{2,}', answer, gap_statement)
                    self.complete_statements.append(filled_statement)
                    
                    i += 2
                    continue
            
            # If we couldn't match a pattern, just move to the next line
            i += 1
            
        return self.qa_pairs, self.gap_fill_pairs
    
    def build_sentence_starter_model(self, training_text):
        """Identify and count words that commonly start sentences."""
        try:
            sentences = sent_tokenize(training_text)
        except LookupError:
            nltk.download('punkt', quiet=True)
            sentences = sent_tokenize(training_text)
            
        for sentence in sentences:
            tokens = self.preprocess_text(sentence)
            if tokens:  # Check if the sentence has any tokens
                self.sentence_starters[tokens[0]] += 1
    
    def train(self, training_data):
        """Train the model on the provided training data."""
        print("Training the gap filler model...")
        
        self.qa_pairs = []
        self.gap_fill_pairs = []
        self.complete_statements = []
        self.sentence_starters = Counter()
        
        # Parse the training data
        self.parse_training_data(training_data)
        
        # Create a corpus for n-gram training by combining complete statements
        corpus = " ".join(self.complete_statements)
        
        # For each question-answer pair, add the complete question with answer
        for question, answer in self.qa_pairs:
            corpus += f" {question} {answer}."
        
        # Preprocess the corpus
        tokens = self.preprocess_text(corpus)
        
        # Count word frequencies
        self.word_frequencies.update(tokens)
        
        # Build sentence starter model
        self.build_sentence_starter_model(corpus)
        
        # Build n-gram models of different sizes
        for n in range(self.min_ngram, self.max_ngram + 1):
            print(f"Building {n}-gram model...")
            model = defaultdict(Counter)
            
            # Generate n-grams
            token_ngrams = list(ngrams(tokens, n))
            
            # Build the model
            for ngram in token_ngrams:
                context = ngram[:-1]  # All but the last word
                target = ngram[-1]    # The last word
                model[context][target] += 1
            
            # Store the model
            self.ngram_models[n] = model
        
        print(f"Training complete! Learned {len(self.qa_pairs)} Q&A pairs and {len(self.gap_fill_pairs)} gap-fill pairs.")
        print(f"Identified {len(self.sentence_starters)} common sentence starters.")
    
    def predict_next_word(self, text, top_n=5):
        """Predict the next word based on the current text."""
        if not text:
            # If no text is provided, return common sentence starters
            return [word for word, _ in self.sentence_starters.most_common(top_n)]
        
        # Preprocess the text
        tokens = self.preprocess_text(text)
        
        # Generate predictions using different n-gram sizes
        predictions = Counter()
        
        # Start with the largest n-gram and back off to smaller ones
        for n in range(self.max_ngram, self.min_ngram - 1, -1):
            # Get the context (last n-1 words)
            context_size = min(len(tokens), n - 1)
            if context_size > 0:
                context = tuple(tokens[-context_size:])
                if context in self.ngram_models.get(context_size + 1, {}):
                    for word, count in self.ngram_models[context_size + 1][context].items():
                        predictions[word] += count * (n * 2)  # Weight by n-gram size
        
        # If no predictions from n-grams, fall back to word frequency
        if not predictions:
            # If text ends with a complete word, use word frequency
            predictions = self.word_frequencies.copy()
        
        # Return top predictions
        return [word for word, _ in predictions.most_common(top_n)]
        
    def predict_gap(self, sentence, top_n=5):
        """Predict the most likely words to fill a gap in a sentence."""
        # First check if this is a direct match to any of our training examples
        # For question-answer pairs
        for question, answer in self.qa_pairs:
            if question.lower() == sentence.lower():
                return [answer]
        
        # For gap-fill statements
        for gap_statement, answer in self.gap_fill_pairs:
            if gap_statement.lower() == sentence.lower():
                return [answer]
            
            # Try to normalize the gap patterns
            normalized_query = re.sub(r'_{2,}', '____', sentence.lower())
            normalized_training = re.sub(r'_{2,}', '____', gap_statement.lower())
            
            if normalized_query == normalized_training:
                return [answer]
        
        # If no direct match, use n-gram approach
        # Identify the gap (marked by ____)
        gap_pattern = r'_{2,}'
        if not re.search(gap_pattern, sentence):
            return "No gap found in the sentence."
        
        # Replace the gap with a placeholder for tokenization
        sentence_with_placeholder = re.sub(gap_pattern, "GAPMARKER", sentence)
        
        # Tokenize
        tokens = self.preprocess_text(sentence_with_placeholder)
        
        # Find the position of the gap - fix case sensitivity issue
        try:
            gap_position = tokens.index("gapmarker")
        except ValueError:
            # If we can't find the exact placeholder, search for any token containing it
            gap_found = False
            for i, token in enumerate(tokens):
                if "gap" in token:
                    gap_position = i
                    gap_found = True
                    break
            
            if not gap_found:
                return "Error: Could not locate the gap marker in processed text."
        
        # Generate predictions using different n-gram sizes
        predictions = Counter()
        
        # Check if the gap is at the beginning of the sentence
        is_beginning_of_sentence = (gap_position == 0)
        
        if is_beginning_of_sentence:
            # For gaps at the beginning, prioritize sentence starters and
            # use right context (words that follow the gap)
            
            # Add sentence starters with high weight
            for word, count in self.sentence_starters.most_common(top_n * 2):
                predictions[word] += count * 3  # Give sentence starters higher weight
            
            # Use right context for prediction
            for n in range(self.max_ngram, self.min_ngram - 1, -1):
                right_context_size = min(len(tokens) - gap_position - 1, n - 1)
                if right_context_size > 0:
                    right_context = tuple(tokens[gap_position + 1:gap_position + 1 + right_context_size])
                    
                    # Look for patterns in the training data where these right context words appear
                    for model_n, model in self.ngram_models.items():
                        for context, targets in model.items():
                            # Check if any part of this context matches our right context
                            context_list = list(context)
                            right_context_list = list(right_context)
                            
                            if len(context_list) >= len(right_context_list):
                                for i in range(len(context_list) - len(right_context_list) + 1):
                                    if context_list[i:i+len(right_context_list)] == right_context_list:
                                        for word, count in targets.items():
                                            predictions[word] += count * 2
        else:
            # Standard approach for gaps in the middle or end of sentences
            # Start with the largest n-gram and back off to smaller ones
            for n in range(self.max_ngram, self.min_ngram - 1, -1):
                # Left context (words before the gap)
                left_context_size = min(gap_position, n - 1)
                if left_context_size > 0:
                    left_context = tuple(tokens[gap_position - left_context_size:gap_position])
                    if left_context in self.ngram_models.get(left_context_size + 1, {}):
                        for word, count in self.ngram_models[left_context_size + 1][left_context].items():
                            predictions[word] += count * (n * 2)  # Weight by n-gram size
                
                # Right context (words after the gap)
                right_context_size = min(len(tokens) - gap_position - 1, n - 1)
                if right_context_size > 0:
                    right_context = tuple(tokens[gap_position + 1:gap_position + 1 + right_context_size])
                    # We need to search all models for patterns that could predict what comes before this context
                    for model_size, model in self.ngram_models.items():
                        for context, targets in model.items():
                            # Check if the last part of this context matches the beginning of our right context
                            if len(context) > 0 and len(right_context) > 0:
                                context_list = list(context)
                                right_list = list(right_context)
                                min_length = min(len(context_list), len(right_list))
                                if context_list[-min_length:] == right_list[:min_length]:
                                    for word, count in targets.items():
                                        predictions[word] += count * n
        
        # If no predictions from n-grams, fall back to word frequency
        if not predictions:
            if is_beginning_of_sentence:
                # If at beginning, prioritize common sentence starters
                predictions = self.sentence_starters.copy()
                # If still no predictions, use word frequency
                if not predictions:
                    predictions = self.word_frequencies.copy()
            else:
                predictions = self.word_frequencies.copy()
        
        # Return top predictions
        return [word for word, _ in predictions.most_common(top_n)]
    
    def save_trained_data(self, filepath):
        """Save the trained data for later use."""
        import json
        
        data = {
            'qa_pairs': self.qa_pairs,
            'gap_fill_pairs': self.gap_fill_pairs,
            'word_frequencies': dict(self.word_frequencies),
            'sentence_starters': dict(self.sentence_starters)
        }
        
        # Convert n-gram models to a serializable format
        ngram_models_serializable = {}
        for n, model in self.ngram_models.items():
            ngram_models_serializable[n] = {}
            for context, counter in model.items():
                ngram_models_serializable[n][str(context)] = dict(counter)
        
        data['ngram_models'] = ngram_models_serializable
        
        try:
            with open(filepath, 'w', encoding='utf-8') as f:
                json.dump(data, f)
            print(f"Model saved to {filepath}")
        except Exception as e:
            print(f"Error saving model: {e}")
    
    def load_trained_data(self, filepath):
        """Load previously trained data."""
        import json
        
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            self.qa_pairs = data.get('qa_pairs', [])
            self.gap_fill_pairs = data.get('gap_fill_pairs', [])
            self.word_frequencies = Counter(data.get('word_frequencies', {}))
            self.sentence_starters = Counter(data.get('sentence_starters', {}))
            
            # Convert serialized n-gram models back to the proper format
            self.ngram_models = {}
            for n_str, model_data in data.get('ngram_models', {}).items():
                n = int(n_str)
                self.ngram_models[n] = defaultdict(Counter)
                for context_str, counter_dict in model_data.items():
                    # Convert string representation of tuple back to tuple
                    # Remove the "(" and ")" and split by comma
                    context_items = context_str.strip('()').split(', ')
                    if context_items[0] == '':  # Handle empty tuple
                        context = ()
                    else:
                        context = tuple(item.strip("'") for item in context_items)
                    
                    self.ngram_models[n][context] = Counter(counter_dict)
            
            print(f"Model loaded from {filepath}")
            print(f"Loaded {len(self.qa_pairs)} Q&A pairs and {len(self.gap_fill_pairs)} gap-fill pairs.")
            return True
        except Exception as e:
            print(f"Error loading model: {e}")
            return False


class AutoSuggestApp:
    def __init__(self, root, gap_filler):
        self.root = root
        self.root.title("Real-Time Word Prediction")
        self.root.geometry("800x600")
        
        self.gap_filler = gap_filler
        self.suggestion_var = tk.StringVar()
        
        # Create the main frame
        main_frame = ttk.Frame(root, padding="10")
        main_frame.pack(fill=tk.BOTH, expand=True)
        
        # Instructions
        ttk.Label(main_frame, text="Type your text below. Suggestions will appear in real-time.", 
                  font=("Arial", 12)).pack(pady=10)
        
        # Create text area
        self.text_area = scrolledtext.ScrolledText(main_frame, wrap=tk.WORD, 
                                                  width=80, height=15, 
                                                  font=("Arial", 12))
        self.text_area.pack(fill=tk.BOTH, expand=True, pady=10)
        self.text_area.bind("<KeyRelease>", self.on_key_release)
        
        # Suggestions frame
        suggestion_frame = ttk.LabelFrame(main_frame, text="Suggestions", padding="10")
        suggestion_frame.pack(fill=tk.X, pady=10)
        
        # Create buttons for suggestions
        self.suggestion_buttons = []
        for i in range(5):
            btn = ttk.Button(suggestion_frame, text="", width=20, 
                           command=lambda idx=i: self.use_suggestion(idx))
            btn.grid(row=0, column=i, padx=5, pady=5)
            self.suggestion_buttons.append(btn)
        
        # Status bar
        self.status_var = tk.StringVar()
        self.status_var.set("Ready")
        status_bar = ttk.Label(root, textvariable=self.status_var, relief=tk.SUNKEN, anchor=tk.W)
        status_bar.pack(side=tk.BOTTOM, fill=tk.X)
        
        # Focus on the text area
        self.text_area.focus_set()
        
        # Initialize with empty suggestions
        self.update_suggestions([])
    
    def on_key_release(self, event):
        """Handle key release event to update suggestions."""
        # If a special key was pressed (not a character key), skip prediction
        if event.keysym in ('Return', 'BackSpace', 'Delete', 'Escape', 'Tab'):
            # For return key, reset suggestions
            if event.keysym == 'Return':
                self.update_suggestions([])
            return
        
        # Get the current text
        text = self.text_area.get("1.0", "end-1c")
        
        # Update the status bar
        self.status_var.set(f"Processing: {len(text)} characters")
        
        # Get word predictions
        predictions = self.gap_filler.predict_next_word(text)
        
        # Update suggestions
        self.update_suggestions(predictions)
    
    def update_suggestions(self, suggestions):
        """Update the suggestion buttons."""
        for i, btn in enumerate(self.suggestion_buttons):
            if i < len(suggestions):
                btn.config(text=suggestions[i], state=tk.NORMAL)
            else:
                btn.config(text="", state=tk.DISABLED)
    
    def use_suggestion(self, index):
        """Use the selected suggestion."""
        if index < len(self.suggestion_buttons) and self.suggestion_buttons[index]['text']:
            # Get the suggested word
            suggestion = self.suggestion_buttons[index]['text']
            
            # Get the current text
            current_text = self.text_area.get("1.0", "end-1c")
            
            # Check if we should add a space before the suggestion
            if current_text and not current_text.endswith(" "):
                suggestion = " " + suggestion
            
            # Add the suggestion to the text
            self.text_area.insert(tk.END, suggestion + " ")
            
            # Update suggestions for the new text
            new_text = self.text_area.get("1.0", "end-1c")
            predictions = self.gap_filler.predict_next_word(new_text)
            self.update_suggestions(predictions)
            
            # Focus back on the text area
            self.text_area.focus_set()


def main():
    # Create and train the model
    print("Starting the Auto-Suggest Gap Filler...")
    gap_filler = GapFillerAI()
    
    try:
        with open("qa_facts.txt", "r", encoding="utf-8") as f:
            training_data = f.read()
            gap_filler.train(training_data)
    except Exception as e:
        print(f"Initialization error: {e}")
        print("Using minimal default training data...")
        # Minimal training data as fallback
        training_data = "This is a sentence.\nAnother example.\nHow are you?\nI am fine."
        gap_filler.train(training_data)
    
    # Create Tkinter application
    root = tk.Tk()
    app = AutoSuggestApp(root, gap_filler)
    root.mainloop()


if __name__ == "__main__":
    main()