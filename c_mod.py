import random
import re
import sys
import os
import tokenize
from io import BytesIO

def random_symbol_replacement(code):
    # Define symbol mappings for arithmetic operators (only using valid Python operators)
    symbol_map = {
        '+': ['+','-','*','/'],            # Addition
        '-': ['+','-','*','/'],            # Subtraction
        '*': ['+','-','*','/'],            # Multiplication
        '**': ['**','*'],          # Exponentiation 
        '/': ['/'],            # Division
        '//': ['//'],          # Floor division
        '%': ['%'],            # Modulo
        '=': ['='],            # Assignment
        '==': ['=='],          # Equality
        '<': ['<','>'],            # Less than
        '<=': ['<=','>='],          # Less than or equal
        '!=': ['!=','=='],          # Not equal
        '>': ['>',"<"],            # Greater than
        '>=': ['>=','<='],          # Greater than or equal
        'and': ['and', 'or'],   # Logical AND, Bitwise AND
        'or': ['or', 'and'],     # Logical OR, Bitwise OR
    }
    
    # Create a list of lines and then process each line
    lines = code.split('\n')
    modified_lines = []
    
    for line in lines:
        # First, handle word operators like 'and' and 'or'
        if random.random() < 0.3:  # 30% chance to replace
            line = re.sub(r'\band\b', '&', line)
        
        if random.random() < 0.3:  # 30% chance to replace
            line = re.sub(r'\bor\b', '|', line)
        
        # Now handle other operators that should remain the same length
        # Use tokenize on this single line
        try:
            line_tokens = list(tokenize.tokenize(BytesIO(line.encode('utf-8')).readline))
            line_bytes = bytearray(line.encode('utf-8'))
            
            # Process tokens in reverse to avoid position shifts
            for token in reversed(line_tokens):
                tok_type = token.type
                tok_string = token.string
                start = token.start[1]
                end = token.end[1]
                
                # Only replace operators that are in our map and don't change length
                if tok_type == tokenize.OP and tok_string in symbol_map:
                    # Choose random replacement and encode it
                    replacement = random.choice(symbol_map[tok_string]).encode('utf-8')
                    
                    # Only replace if the replacement is the same length
                    if len(replacement) == (end - start):
                        line_bytes[start:end] = replacement
                        print(line_bytes[start:end], replacement)
            # Convert back to string
            modified_line = line_bytes.decode('utf-8')
            
        except Exception:
            # If tokenization fails for a line, just keep it as is
            modified_line = line
        
        modified_lines.append(modified_line)
    
    return '\n'.join(modified_lines)

def process_file(input_file, output_file=None, safe_mode=True):
    # Default output filename if not provided
    if output_file is None:
        base_name = os.path.splitext(input_file)[0]
        output_file = f"{base_name}_randomized.py"
    
    try:
        # Read the input file
        with open(input_file, 'r') as f:
            code = f.read()
        
        
        try:
            randomized_code = random_symbol_replacement(code)
        except Exception as e:
            print(f"Error during tokenization, falling back to safe mode: {e}")
            randomized_code = random_symbol_replacement_safe(code)
        
        # Write to output file
        with open(output_file, 'w') as f:
            f.write(randomized_code)
        
        print(f"File processed successfully!")
        print(f"Original: {input_file}")
        print(f"Randomized: {output_file}")
        
        # Validate the output file
        try:
            with open(output_file, 'r') as f:
                compile(f.read(), output_file, 'exec')
            print("Validation: The generated file compiles without syntax errors.")
        except SyntaxError as e:
            print(f"Warning: The generated file has syntax errors: {e}")
        
        return True
    
    except Exception as e:
        print(f"Error processing file: {e}")
        return False

def main():
    print("File-based Random Arithmetic Symbol Replacement")
    print("----------------------------------------------")
    
    if len(sys.argv) > 1:
        # Process files from command line arguments
        input_file = sys.argv[1]
        output_file = sys.argv[2] if len(sys.argv) > 2 else None
        safe_mode = True if len(sys.argv) <= 3 or sys.argv[3].lower() == 'safe' else False
        process_file(input_file, output_file, safe_mode)
    else:
        # Interactive mode
        while True:
            print("\nOptions:")
            print("1. Process a Python file")
            print("2. Exit")
            
            choice = input("Enter your choice (1-3): ")
            
            if choice == '1':
                input_file = input("Enter the path to the Python file: ")
                output_file = input("Enter the output file path (leave blank for default): ")
                output_file = output_file if output_file.strip() else None
                
                process_file(input_file, output_file, safe_mode=False)
            
            elif choice == '2':
                break
            
            else:
                print("Invalid choice, please try again.")

if __name__ == "__main__":
    main()