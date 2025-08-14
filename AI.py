import hashlib
domain = int(input("Enter domain(int): "))
with open(input("Enter filename: "), 'r', encoding='utf-8') as f:
    sample_text = f.read()[:domain]
class HashMapper:
    def __init__(self, domain=100):
        """
        Initialize the hash mapper with a fixed table size.
        - domain: The size of the hash table (should be larger than expected number of words to minimize collisions).
        """
        self.domain = domain
        self.table = [None] * domain  # Stores words at their probed indices
        self.word_to_index = {}  # Dict for word -> index
        self.index_to_word = {}  # Dict for index -> word

    def _compute_hash(self, word):
        """Compute initial hash index for the word using SHA-256."""
        sha_digest = hashlib.sha256(word.encode('utf-8')).hexdigest()
        int_hash = int(sha_digest, 16)
        return int_hash % self.domain

    def _compute_step(self, word):
        """Compute a probing step derived from the SHA-256 hash (for double hashing-like behavior)."""
        sha_digest = hashlib.sha256(word.encode('utf-8')).hexdigest()
        int_hash = int(sha_digest, 16)
        # Derive step from higher bits; ensure it's between 1 and domain-1
        step = 1 + (int_hash // self.domain) % (self.domain - 1)
        return step

    def insert(self, word):
        """
        Insert a word into the hash table using linear probing.
        If the word already exists, do nothing.
        """
        if word in self.word_to_index:
            return  # Already inserted
        index = self._compute_hash(word)
        original_index = index
        probe = 0
        while self.table[index] is not None:
            probe += 1
            index = (original_index + probe) % self.domain
            if probe >= self.domain:
                raise ValueError("Hash table is full! Increase domain.")
        # Insert the word at the probed index
        self.table[index] = word
        self.word_to_index[word] = index
        self.index_to_word[index] = word

    def get_index(self, word):
        """Retrieve the index for a given word."""
        return self.word_to_index.get(word, None)

    def get_word(self, index):
        """Retrieve the word for a given index."""
        return self.index_to_word.get(index, None)

    def insert_words(self, words):
        """Convenience method to insert a list of words."""
        for word in words:
            self.insert(word)

# Example usage
if __name__ == "__main__":
    # Create a HashMapper with a table size (adjusted to 9999 per request)
    mapper = HashMapper(domain)
    text_words = sample_text.split()
    while True:
        # Input first: Get words from user input
        user_input = input("\nEnter your input text or words (space-separated for multiple): ")
        user_words = user_input.split()
        mapper.insert_words(user_words)
        print(f"Inserted {len(user_words)} words from input.")
        mapper.insert_words(text_words)
        print(f"Inserted {len(text_words)} words from text (duplicates ignored).")
        # Customized print: For each user input word, find and print a different word via hash-derived probing
        print("Different words from input via hash:")
        for word in user_words:
            if word not in mapper.word_to_index:
                print(f"'{word}' not in mapper (skipped).")
                continue
            hash_index = mapper._compute_hash(word)
            step = mapper._compute_step(word)
            probe_index = (hash_index + step) % mapper.domain
            start_probe = probe_index
            found = False
            while True:
                if mapper.table[probe_index] is not None and mapper.table[probe_index] != word:
                    print(f"For '{word}' (hash {hash_index}, step {step}), output word: '{mapper.table[probe_index]}' (at index {probe_index})")
                    found = True
                    break
                probe_index = (probe_index + step) % mapper.domain
                if probe_index == start_probe:
                    break  # Full cycle; no different word found
            if not found:
                print(f"No different word found for '{word}' (table may be sparse or isolated).")
