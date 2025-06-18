#include <iostream>
#include <fstream>
#include <vector>
#include <string>
#include <map>
#include <unordered_map>
#include <algorithm>
#include <sstream>
#include <set>
#include <chrono>
#include <iomanip>
#include <random>
#include <cmath>
class ComprehensiveWordToIndexMapper {
private:
    std::unordered_map<std::string, int> word_to_index;
    std::unordered_map<int, std::string> index_to_word;
    std::vector<std::string> all_words;
    std::set<std::string> word_set;
    std::mt19937 rng; // Random number generator

public:
    ComprehensiveWordToIndexMapper() : rng(std::chrono::steady_clock::now().time_since_epoch().count()) {}

    // Load and randomize words from file
    bool loadWordsFromFile(const std::string& filename) {
        std::ifstream file(filename);
        if (!file.is_open()) {
            std::cerr << "Error: Cannot open words file " << filename << std::endl;
            return false;
        }

        std::string word;
        std::cout << "Loading English words from " << filename << "..." << std::endl;

        while (std::getline(file, word)) {
            // Clean the word (remove whitespace, convert to lowercase)
            word.erase(std::remove_if(word.begin(), word.end(), ::isspace), word.end());
            std::transform(word.begin(), word.end(), word.begin(), ::tolower);

            if (!word.empty() && word.find_first_not_of("abcdefghijklmnopqrstuvwxyz") == std::string::npos) {
                // Only add if it's a valid alphabetical word and not already present
                if (word_set.find(word) == word_set.end()) {
                    all_words.push_back(word);
                    word_set.insert(word);
                }
            }
        }

        file.close();

        // Randomize the word order AFTER loading
        randomizeWordOrder();

        // Rebuild the index mappings after randomization
        rebuildIndexMappings();

        std::cout << "Loaded and randomized " << all_words.size() << " unique English words" << std::endl;
        return !all_words.empty();
    }

    // Randomize the order of words using std::shuffle
    void randomizeWordOrder() {
        if (!all_words.empty()) {
            std::cout << "Randomizing word order..." << std::endl;
            std::shuffle(all_words.begin(), all_words.end(), rng);
            std::cout << "Word order randomized successfully!" << std::endl;
        }
    }

    // Rebuild index mappings after randomization
    void rebuildIndexMappings() {
        word_to_index.clear();
        index_to_word.clear();

        for (size_t i = 0; i < all_words.size(); ++i) {
            word_to_index[all_words[i]] = static_cast<int>(i);
            index_to_word[static_cast<int>(i)] = all_words[i];
        }
    }

    // Get word by index (now randomized)
    std::string getWordByIndex(int index) const {
        if (index >= 0 && index < static_cast<int>(all_words.size())) {
            return all_words[index];
        }
        return "unknown";
    }

    // Get random word by modulo (using randomized order)
    std::string getRandomWordByModulo(int value) const {
        if (all_words.empty()) return "empty";
        int index = std::abs(value) % static_cast<int>(all_words.size());
        return all_words[index];
    }

    // Get truly random word
    std::string getRandomWord() {
        if (all_words.empty()) return "empty";
        std::uniform_int_distribution<int> dist(0, static_cast<int>(all_words.size()) - 1);
        return all_words[dist(rng)];
    }

    // Get multiple random words
    std::vector<std::string> getRandomWords(size_t count) {
        std::vector<std::string> random_words;
        if (all_words.empty()) return random_words;

        std::uniform_int_distribution<int> dist(0, static_cast<int>(all_words.size()) - 1);

        for (size_t i = 0; i < count; ++i) {
            random_words.push_back(all_words[dist(rng)]);
        }

        return random_words;
    }

    // Re-randomize words during runtime
    void reshuffleWords() {
        randomizeWordOrder();
        rebuildIndexMappings();
        std::cout << "Words reshuffled!" << std::endl;
    }

    // Get index by word
    int getIndexByWord(const std::string& word) const {
        std::string lowercase_word = word;
        std::transform(lowercase_word.begin(), lowercase_word.end(),
                      lowercase_word.begin(), ::tolower);

        auto it = word_to_index.find(lowercase_word);
        return (it != word_to_index.end()) ? it->second : -1;
    }

    // Check if word exists
    bool wordExists(const std::string& word) const {
        std::string lowercase_word = word;
        std::transform(lowercase_word.begin(), lowercase_word.end(),
                      lowercase_word.begin(), ::tolower);
        return word_set.find(lowercase_word) != word_set.end();
    }

    // Get total word count
    size_t getTotalWords() const {
        return all_words.size();
    }

    // Display statistics about loaded words
    void displayWordStatistics() const {
        std::cout << "\n=== Word Statistics ===" << std::endl;
        std::cout << "Total words loaded: " << all_words.size() << std::endl;

        // Show first 10 words after randomization
        std::cout << "\nFirst 10 words after randomization:" << std::endl;
        for (size_t i = 0; i < std::min(size_t(10), all_words.size()); ++i) {
            std::cout << "  " << i << ": " << all_words[i] << std::endl;
        }
    }
};


// Standalone EnhancedNaturalTextModuloProcessor
class EnhancedNaturalTextModuloProcessor {
private:
    ComprehensiveWordToIndexMapper word_mapper;
    std::vector<int> data;
    std::vector<std::string> natural_text;
    size_t n;
    size_t partition_size;

public:
    EnhancedNaturalTextModuloProcessor(size_t n_, size_t part_size, const std::string& words_file)
        : n(n_), partition_size(part_size) {
        data.resize(n);
        natural_text.resize(n);

        if (!word_mapper.loadWordsFromFile(words_file)) {
            std::cerr << "Failed to load words file. Using default word lists." << std::endl;
        } else {
            word_mapper.displayWordStatistics();
        }
    }

    // Generate natural sentence using comprehensive word list
    std::string generateEnhancedNaturalSentence(int modulo_value, size_t index, int divisor) {
        if (word_mapper.getTotalWords() == 0) {
            return "Default fallback sentence.";
        }

        // Use modulo operations to select words from the comprehensive list
        std::string word1 = word_mapper.getRandomWordByModulo(modulo_value);
        std::string word2 = word_mapper.getRandomWordByModulo(modulo_value * 3 + index);
        std::string word3 = word_mapper.getRandomWordByModulo(modulo_value * 7 + divisor);
        std::string word4 = word_mapper.getRandomWordByModulo(modulo_value * 11 + index * 2);

        // Create different sentence structures
        int structure = modulo_value % 5;

        switch (structure) {
            case 0:
                return "The " + word1 + " " + word2 + " creates " + word3 + " in the " + word4 + ".";
            case 1:
                return "When " + word1 + " meets " + word2 + ", " + word3 + " becomes " + word4 + ".";
            case 2:
                return "Through " + word1 + " and " + word2 + ", we discover " + word3 + " " + word4 + ".";
            case 3:
                return "The essence of " + word1 + " transforms " + word2 + " into " + word3 + " " + word4 + ".";
            default:
                return "In the realm of " + word1 + ", " + word2 + " " + word3 + " with " + word4 + ".";
        }
    }

    // Modular multiplication stub
    int modularMultiplication(int a, int b, int mod) {
        return (a * b) % mod;
    }

    // Main text generation method
    void performModuloAndGenerateText(const std::vector<int>& divisors) {
        std::cout << "=== Enhanced Text Generation with " << word_mapper.getTotalWords()
                  << " English Words ===" << std::endl;

        auto start_time = std::chrono::high_resolution_clock::now();

        for (size_t div_idx = 0; div_idx < divisors.size(); ++div_idx) {
            int divisor = divisors[div_idx];
            std::cout << "Applying modulo " << divisor << " and generating enhanced text..." << std::endl;

            for (size_t i = 0; i < data.size(); ++i) {
                if (div_idx == 0) {
                    data[i] = static_cast<int>(i) % divisor;
                } else {
                    data[i] = modularMultiplication(data[i], divisor, divisor + 1);
                }

                // Use enhanced text generation
                natural_text[i] = generateEnhancedNaturalSentence(data[i], i, divisor);
            }

            double progress = ((div_idx + 1.0) / divisors.size()) * 100.0;
            std::cout << "Progress: " << std::fixed << std::setprecision(1) << progress << "%" << std::endl;
        }

        auto end_time = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time);
        std::cout << "Enhanced text generation completed in " << duration.count() << "ms" << std::endl;
    }

    // Display sample generated text
    void displaySampleNaturalText(size_t count) {
        std::cout << "\nSample generated text:" << std::endl;
        for (size_t i = 0; i < std::min(count, natural_text.size()); ++i) {
            std::cout << "  " << i << ": " << natural_text[i] << std::endl;
        }
    }

    // Stub: Save natural text to file
    void saveNaturalTextToFile(const std::string& filename) {
        std::ofstream out(filename);
        for (const auto& line : natural_text) {
            out << line << '\n';
        }
        std::cout << "Saved natural text to " << filename << std::endl;
    }

    // Stub: Themed partition files
    void createThemedPartitionFiles(const std::string&) {
        // Stub: Implement as needed
    }

    // Stub: Merge partitions into story
    void mergePartitionsIntoStory(const std::string&, const std::string&) {
        // Stub: Implement as needed
    }

    // Stub: Generate word cloud data
    void generateWordCloudData(const std::string&) {
        // Stub: Implement as needed
    }
};

int main() {
    const size_t n = 500000;
    const size_t partition_size = 5000;

    std::cout << "🎨 Enhanced Natural Text Generation with Comprehensive English Dictionary 🎨" << std::endl;
    std::cout << std::string(80, '═') << std::endl;

    // Initialize enhanced processor with words file
    EnhancedNaturalTextModuloProcessor processor(n, partition_size, "words_alpha.txt");

    // Perform modulo operations and generate natural text
    std::vector<int> divisors = {7,67,243,23546,1346,346,2346,346,346,1346,1234,6216,1346};
    processor.performModuloAndGenerateText(divisors);

    // Display sample generated text
    processor.displaySampleNaturalText(8);

    // Save natural text to formatted file
    processor.saveNaturalTextToFile("enhanced_natural_text_collection.txt");

    // Create themed partition files
    processor.createThemedPartitionFiles("enhanced_themed_partition");

    // Merge partitions into a cohesive story
    processor.mergePartitionsIntoStory("enhanced_themed_partition", "enhanced_mathematical_chronicles.txt");

    // Generate word cloud data
    processor.generateWordCloudData("enhanced_word_cloud_data.csv");

    std::cout << "\n🎉 Enhanced Natural Text Generation Complete! 🎉" << std::endl;

    return 0;
}
