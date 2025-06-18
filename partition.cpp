#include <iostream>
#include <fstream>
#include <vector>
#include <cmath>
#include <memory>
#include <string>
#include <algorithm>

class ModuloProcessor {
private:
    std::vector<int> data;
    size_t partition_size;
    size_t total_partitions;
    std::string current_filename;
    
public:
    ModuloProcessor(size_t n, size_t part_size) 
        : data(n), partition_size(part_size) {
        total_partitions = std::ceil(static_cast<double>(n) / partition_size);
    }
    
    // Perform modulo operations across the entire space
    void performModuloOperations(int divisor) {
        for (size_t i = 0; i < data.size(); ++i) {
            data[i] = static_cast<int>(i) % divisor;
        }
    }
    
    // Save current data state back to file (overwrite existing)
    void saveBackToFile(const std::string& filename) {
        current_filename = filename;
        std::ofstream file(filename, std::ios::binary | std::ios::trunc);
        if (!file.is_open()) {
            std::cerr << "Error opening file for saving: " << filename << std::endl;
            return;
        }
        
        // Write file header with metadata
        size_t data_size = data.size();
        file.write(reinterpret_cast<const char*>(&data_size), sizeof(data_size));
        file.write(reinterpret_cast<const char*>(&partition_size), sizeof(partition_size));
        file.write(reinterpret_cast<const char*>(&total_partitions), sizeof(total_partitions));
        
        // Write partitioned data
        for (size_t partition = 0; partition < total_partitions; ++partition) {
            size_t start_idx = partition * partition_size;
            size_t end_idx = std::min(start_idx + partition_size, data.size());
            
            // Write partition header
            size_t partition_length = end_idx - start_idx;
            file.write(reinterpret_cast<const char*>(&partition_length), sizeof(partition_length));
            
            // Write partition data
            file.write(reinterpret_cast<const char*>(&data[start_idx]), 
                      partition_length * sizeof(int));
        }
        
        file.close();
        std::cout << "Data saved back to file: " << filename << std::endl;
    }
    
    // Append new modulo results to existing file
    void appendToFile(const std::string& filename, int new_divisor) {
        // First perform new modulo operation
        std::vector<int> new_data = data;  // Copy current data
        for (size_t i = 0; i < new_data.size(); ++i) {
            new_data[i] = static_cast<int>(i) % new_divisor;
        }
        
        std::ofstream file(filename, std::ios::binary | std::ios::app);
        if (!file.is_open()) {
            std::cerr << "Error opening file for appending: " << filename << std::endl;
            return;
        }
        
        // Write append marker and divisor info
        int append_marker = -1;  // Special marker for appended data
        file.write(reinterpret_cast<const char*>(&append_marker), sizeof(append_marker));
        file.write(reinterpret_cast<const char*>(&new_divisor), sizeof(new_divisor));
        
        // Write new data in partitions
        for (size_t partition = 0; partition < total_partitions; ++partition) {
            size_t start_idx = partition * partition_size;
            size_t end_idx = std::min(start_idx + partition_size, new_data.size());
            
            size_t partition_length = end_idx - start_idx;
            file.write(reinterpret_cast<const char*>(&partition_length), sizeof(partition_length));
            file.write(reinterpret_cast<const char*>(&new_data[start_idx]), 
                      partition_length * sizeof(int));
        }
        
        file.close();
        std::cout << "New modulo results (divisor=" << new_divisor 
                  << ") appended to file: " << filename << std::endl;
    }
    
    // Update specific partition and save back to file
    void updatePartitionAndSave(size_t partition_idx, const std::vector<int>& new_values) {
        if (partition_idx >= total_partitions) {
            std::cerr << "Invalid partition index: " << partition_idx << std::endl;
            return;
        }
        
        size_t start_idx = partition_idx * partition_size;
        size_t end_idx = std::min(start_idx + partition_size, data.size());
        
        // Update the partition data
        for (size_t i = 0; i < new_values.size() && (start_idx + i) < end_idx; ++i) {
            data[start_idx + i] = new_values[i];
        }
        
        // Save back to file
        if (!current_filename.empty()) {
            saveBackToFile(current_filename);
            std::cout << "Partition " << partition_idx << " updated and saved." << std::endl;
        }
    }
    
    // Load data from file with full restoration
    bool loadFromFile(const std::string& filename) {
        std::ifstream file(filename, std::ios::binary);
        if (!file.is_open()) {
            std::cerr << "Error opening file for loading: " << filename << std::endl;
            return false;
        }
        
        current_filename = filename;
        
        // Read file header
        size_t data_size, file_partition_size, file_total_partitions;
        file.read(reinterpret_cast<char*>(&data_size), sizeof(data_size));
        file.read(reinterpret_cast<char*>(&file_partition_size), sizeof(file_partition_size));
        file.read(reinterpret_cast<char*>(&file_total_partitions), sizeof(file_total_partitions));
        
        // Resize data vector and update partition info
        data.resize(data_size);
        partition_size = file_partition_size;
        total_partitions = file_total_partitions;
        
        // Read partitioned data
        for (size_t partition = 0; partition < total_partitions; ++partition) {
            size_t partition_length;
            file.read(reinterpret_cast<char*>(&partition_length), sizeof(partition_length));
            
            size_t start_idx = partition * partition_size;
            file.read(reinterpret_cast<char*>(&data[start_idx]), 
                     partition_length * sizeof(int));
        }
        
        file.close();
        std::cout << "Data loaded from file: " << filename << std::endl;
        return true;
    }
    
    // Save specific partitions to separate files
    void savePartitionsToSeparateFiles(const std::string& base_filename) {
        for (size_t partition = 0; partition < total_partitions; ++partition) {
            std::string partition_filename = base_filename + "_partition_" + 
                                           std::to_string(partition) + ".bin";
            
            std::ofstream file(partition_filename, std::ios::binary);
            if (!file.is_open()) {
                std::cerr << "Error creating partition file: " << partition_filename << std::endl;
                continue;
            }
            
            size_t start_idx = partition * partition_size;
            size_t end_idx = std::min(start_idx + partition_size, data.size());
            size_t partition_length = end_idx - start_idx;
            
            // Write partition metadata
            file.write(reinterpret_cast<const char*>(&partition), sizeof(partition));
            file.write(reinterpret_cast<const char*>(&partition_length), sizeof(partition_length));
            
            // Write partition data
            file.write(reinterpret_cast<const char*>(&data[start_idx]), 
                      partition_length * sizeof(int));
            
            file.close();
        }
        
        std::cout << "Saved " << total_partitions << " partitions to separate files." << std::endl;
    }
    
    // Merge partition files back into main file
    void mergePartitionFiles(const std::string& base_filename, const std::string& output_filename) {
        std::ofstream output_file(output_filename, std::ios::binary);
        if (!output_file.is_open()) {
            std::cerr << "Error creating output file: " << output_filename << std::endl;
            return;
        }
        
        // Write file header
        size_t data_size = data.size();
        output_file.write(reinterpret_cast<const char*>(&data_size), sizeof(data_size));
        output_file.write(reinterpret_cast<const char*>(&partition_size), sizeof(partition_size));
        output_file.write(reinterpret_cast<const char*>(&total_partitions), sizeof(total_partitions));
        
        // Read and merge partition files
        for (size_t partition = 0; partition < total_partitions; ++partition) {
            std::string partition_filename = base_filename + "_partition_" + 
                                           std::to_string(partition) + ".bin";
            
            std::ifstream partition_file(partition_filename, std::ios::binary);
            if (!partition_file.is_open()) {
                std::cerr << "Warning: Could not open " << partition_filename << std::endl;
                continue;
            }
            
            // Read partition metadata
            size_t file_partition_idx, partition_length;
            partition_file.read(reinterpret_cast<char*>(&file_partition_idx), sizeof(file_partition_idx));
            partition_file.read(reinterpret_cast<char*>(&partition_length), sizeof(partition_length));
            
            // Write partition header to output
            output_file.write(reinterpret_cast<const char*>(&partition_length), sizeof(partition_length));
            
            // Copy partition data
            std::vector<int> partition_data(partition_length);
            partition_file.read(reinterpret_cast<char*>(partition_data.data()), 
                              partition_length * sizeof(int));
            output_file.write(reinterpret_cast<const char*>(partition_data.data()), 
                            partition_length * sizeof(int));
            
            partition_file.close();
        }
        
        output_file.close();
        std::cout << "Merged partition files into: " << output_filename << std::endl;
    }
    
    // Auto-save functionality with backup
    void enableAutoSave(const std::string& filename, bool create_backup = true) {
        if (create_backup && !current_filename.empty()) {
            std::string backup_filename = current_filename + ".backup";
            saveBackToFile(backup_filename);
            std::cout << "Backup created: " << backup_filename << std::endl;
        }
        
        current_filename = filename;
        saveBackToFile(filename);
    }
    
    // Display current data state
    void displayData(size_t max_elements = 20) const {
        std::cout << "Current data (showing first " << std::min(max_elements, data.size()) 
                  << " elements): ";
        for (size_t i = 0; i < std::min(max_elements, data.size()); ++i) {
            std::cout << data[i] << " ";
        }
        std::cout << std::endl;
    }
    
    // Get partition information
    void printPartitionInfo() const {
        std::cout << "Total elements: " << data.size() << std::endl;
        std::cout << "Partition size: " << partition_size << std::endl;
        std::cout << "Total partitions: " << total_partitions << std::endl;
        std::cout << "Current file: " << (current_filename.empty() ? "None" : current_filename) << std::endl;
    }
};

int main() {
    const size_t n = 1000;
    const size_t partition_size = 100;
    const std::string main_filename = "modulo_data.bin";
    
    // Create processor instance
    ModuloProcessor processor(n, partition_size);
    processor.printPartitionInfo();
    
    // Perform initial modulo operations
    std::cout << "\n=== Initial Modulo Operations ===" << std::endl;
    processor.performModuloOperations(7);
    processor.displayData();
    
    // Save to file
    processor.saveBackToFile(main_filename);
    
    // Perform new operations and save back
    std::cout << "\n=== Updating with New Modulo Operations ===" << std::endl;
    processor.performModuloOperations(13);
    processor.displayData();
    processor.saveBackToFile(main_filename);  // Overwrite with new data
    
    // Append additional results
    std::cout << "\n=== Appending New Results ===" << std::endl;
    processor.appendToFile(main_filename + "_extended", 17);
    
    // Update specific partition
    std::cout << "\n=== Updating Specific Partition ===" << std::endl;
    std::vector<int> new_partition_values = {100, 101, 102, 103, 104};
    processor.updatePartitionAndSave(0, new_partition_values);
    processor.displayData();
    
    // Save partitions to separate files
    std::cout << "\n=== Saving Partitions to Separate Files ===" << std::endl;
    processor.savePartitionsToSeparateFiles("data");
    
    // Test loading from file
    std::cout << "\n=== Testing Load from File ===" << std::endl;
    ModuloProcessor new_processor(1, 1);  // Start with minimal size
    if (new_processor.loadFromFile(main_filename)) {
        new_processor.printPartitionInfo();
        new_processor.displayData();
    }
    
    // Enable auto-save with backup
    std::cout << "\n=== Auto-save with Backup ===" << std::endl;
    processor.performModuloOperations(23);
    processor.enableAutoSave("auto_saved_data.bin", true);
    
    // Merge partition files back
    std::cout << "\n=== Merging Partition Files ===" << std::endl;
    processor.mergePartitionFiles("data", "merged_data.bin");
    
    std::cout << "\n=== Save-back functionality demonstration complete ===" << std::endl;
    
    return 0;
}
