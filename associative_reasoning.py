import re
import random
from collections import defaultdict
gen_length = 250
class MarkovModel:
    def __init__(self):
        self.model = defaultdict(list)

    def train(self, corpus):
        words = re.findall(r'\b\w+\b', corpus.lower())
        for i in range(len(words) - 1):
            self.model[words[i]].append(words[i + 1])

    def generate(self, seed_word, num_words=1):
        current_word = seed_word
        generated_words = []
        generated_sequence = [current_word]
        for _ in range(num_words):
            if current_word in self.model:
                next_word = random.choice(self.model[current_word])
                generated_words.append(next_word)
                generated_sequence.append(next_word)
                current_word = next_word
            else:
                break
        return generated_words, generated_sequence

def main():
    with open('test.txt', 'r', encoding='utf-8') as file:
        corpus = file.read()

    markov_model = MarkovModel()
    markov_model.train(corpus)
    while True:
        seed_word = input("USER: ").split()
        
        out = []
        output = []
        for seed_word in seed_word:
            for i in range(70):
                num_words = 5  # Number of alternate words to generate

                generated_words, generated_sequence = markov_model.generate(seed_word, num_words)
                # Backtrack to find the wanted word
                if generated_sequence:
                    wanted_word = generated_sequence[-1]
                    seed_word = wanted_word
                    out.append(wanted_word)
            
            # Perform another forward generation to get the distal word
            if out:
                final_seed_word = out[-1]
                distal_generated_words, distal_generated_sequence = markov_model.generate(final_seed_word, num_words)
                out.extend(distal_generated_words)

            output.append(out[-1])
        print(' '.join(output))
if __name__ == "__main__":
    main()
