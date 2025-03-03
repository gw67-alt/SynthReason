import numpy as np
import re
import nltk
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.util import ngrams
from collections import defaultdict, Counter

class GapFillerAI:
    def __init__(self):
        self.ngram_models = {}  # Will store n-gram models of different sizes
        self.max_ngram = 5      # Maximum n-gram size to consider
        self.min_ngram = 2      # Minimum n-gram size
        self.word_frequencies = Counter()  # Stores overall word frequencies
        self.qa_pairs = []      # Store question-answer pairs for direct lookup
        self.gap_fill_pairs = [] # Store gap-fill pairs (sentence with gap and answer)
        
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
    
    def train(self, training_data):
        """Train the model on the provided training data."""
        print("Training the gap filler model...")
        
        self.qa_pairs = []
        self.gap_fill_pairs = []
        self.complete_statements = []
        
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
                reversed_model = n + 1
                if right_context in self.ngram_models.get(reversed_model, {}):
                    for word, count in self.ngram_models[reversed_model][right_context].items():
                        predictions[word] += count * n  # Less weight than left context
        
        # If no predictions from n-grams, fall back to word frequency
        if not predictions:
            predictions = self.word_frequencies.copy()
        
        # Return top predictions
        return [word for word, _ in predictions.most_common(top_n)]
    
    def fill_gaps_interactive(self):
        """Interactive mode to fill gaps in sentences."""
        print("Gap Filler AI - Interactive Mode")
        print("Type a sentence with ____ as the gap, a question, or 'quit' to exit")
        
        while True:
            user_input = input("\nEnter a question or sentence with a gap: ").strip()
            
            if user_input.lower() == 'quit':
                break
                
            predictions = self.predict_gap(user_input)
            
            if isinstance(predictions, str):
                print(predictions)
            else:
                print(f"Top predictions:")
                for i, word in enumerate(predictions, 1):
                    print(f"{i}. {word}")
    
    def process_quiz_file(self, file_path):
        """Process a file containing questions with gaps."""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                lines = f.readlines()
            
            questions = []
            for line in lines:
                line = line.strip()
                if line:
                    questions.append(line)
            
            results = []
            for question in questions:
                predictions = self.predict_gap(question)
                if isinstance(predictions, str):
                    results.append((question, [predictions]))
                else:
                    results.append((question, predictions))
            
            return results
                
        except Exception as e:
            print(f"Error processing file: {e}")
            return []
    
    def save_trained_data(self, filepath):
        """Save the trained data for later use."""
        import json
        
        data = {
            'qa_pairs': self.qa_pairs,
            'gap_fill_pairs': self.gap_fill_pairs,
            'word_frequencies': dict(self.word_frequencies)
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

# Example usage
if __name__ == "__main__":
    # Example training data in the specified format
    with open("qa_facts.txt", "r", encoding="utf-8") as f:
        training_data = f.read()
       
    # Create and train the model
    gap_filler = GapFillerAI()
    gap_filler.train(training_data)
    
    # Interactive mode
    gap_filler.fill_gaps_interactive()
    
    # Optional: Save the trained model for later use
    # gap_filler.save_trained_data("trained_gap_filler.json")
    
    # Optional: Load a previously saved model
    # gap_filler.load_trained_data("trained_gap_filler.json")
