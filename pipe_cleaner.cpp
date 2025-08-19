#include <iostream>
#include <string>
#include <vector>
#include <thread>
#include <chrono>
#include <fstream>      // For file I/O (logging)
#include <iomanip>      // For formatting timestamps (std::put_time)
#include <ctime>        // For std::time_t and std::localtime
#include <cstdio>       // For popen, pclose, fgets

// The ProcessOutputMonitor class is unchanged as it was correct.
class ProcessOutputMonitor {
private:
    std::string command;
    std::string previous_output;
    bool stop_flag = false;
    std::thread monitor_thread;
    std::ofstream log_file;

    void log(const std::string& message) {
        auto now = std::chrono::system_clock::now();
        std::time_t now_time = std::chrono::system_clock::to_time_t(now);
        std::cout << std::put_time(std::localtime(&now_time), "[%Y-%m-%d %H:%M:%S] ")
                  << message << std::endl;
        if (log_file.is_open()) {
            log_file << std::put_time(std::localtime(&now_time), "[%Y-%m-%d %H:%M:%S] ")
                     << message << std::endl;
        }
    }

    void scanLoop() {
        while (!stop_flag) {
            log("[Monitor] Running command: " + command);
            FILE* pipe = popen(command.c_str(), "r");
            if (!pipe) {
                log("Error: popen() failed!");
                return;
            }
            char buffer[128];
            std::string current_output = "";
            while (fgets(buffer, sizeof(buffer), pipe) != nullptr) {
                current_output += buffer;
            }
            pclose(pipe);
            if (previous_output != current_output) {
                log("--- !!! Change Detected !!! ---");
                std::cout << "New output:\n" << current_output << std::endl;
                if (log_file.is_open()) {
                    log_file << "New output:\n" << current_output << std::endl;
                }
                log("---------------------------------");
                previous_output = current_output;
            } else {
                log("[Monitor] No change detected.");
            }
            std::this_thread::sleep_for(std::chrono::seconds(3));
        }
    }

public:
    ProcessOutputMonitor(const std::string& cmd, const std::string& log_filename = "monitor_log.txt")
        : command(cmd) {
        log_file.open(log_filename, std::ios_base::app);
        if (!log_file.is_open()) {
            std::cerr << "Error: Could not open log file: " << log_filename << std::endl;
        }
        log("Monitor initialized for command: '" + command + "'. Logging to " + log_filename);
    }
    ~ProcessOutputMonitor() {
        if (log_file.is_open()) {
            log("Monitor shutting down. Closing log file.");
            log_file.close();
        }
    }
    void start() {
        if (monitor_thread.joinable()) return;
        stop_flag = false;
        log("[Monitor] Starting background monitoring.");
        monitor_thread = std::thread(&ProcessOutputMonitor::scanLoop, this);
    }
    void stop() {
        stop_flag = true;
        if (monitor_thread.joinable()) {
            monitor_thread.join();
        }
    }
};

// --- Main function with the fix ---
int main(int argc, char* argv[]) {
    if (argc < 2) {
        std::cerr << "Usage: " << argv[0] << " \"<command>\" [log_file_path]" << std::endl;
        std::cerr << "Example: " << argv[0] << " \"ls -l\" my_app.log" << std::endl;
        return 1;
    }

    // --- FIX APPLIED HERE ---
    // The original incorrect line was: std::string command_to_run = argv;
    // The corrected logic iterates through the arguments to build the string.
    std::string command_to_run = argv[1];
    std::string log_path = (argc > 2) ? argv[2] : "monitor_log.txt";

    ProcessOutputMonitor monitor(command_to_run, log_path);
    monitor.start();

    std::cout << "\n--- Monitoring in progress. Press Enter to stop. ---\n" << std::endl;
    std::cin.get();

    monitor.stop();
    std::cout << "--- Main thread finished operations ---" << std::endl;

    return 0;
}
