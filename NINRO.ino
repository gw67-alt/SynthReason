// Advanced Curvature Analysis Oscilloscope for Arduino
// Specialized for signal pattern analysis and sign flip detection

#define NUM_CHANNELS 6
#define SAMPLE_RATE_MS 20   // 50Hz sampling
#define HISTORY_SIZE 5     // Number of samples to store
#define DISPLAY_HEIGHT 12   // Character height for waveform
#define DISPLAY_WIDTH 80    // Character width
#define SMOOTHING_WINDOW 7  // Moving average window
#define STATE_SPACE 999999  // Analysis state space

int analogPins[NUM_CHANNELS] = {A0, A1, A2};
float channelHistory[NUM_CHANNELS][HISTORY_SIZE];
float smoothedData[NUM_CHANNELS][HISTORY_SIZE];
float curvatureData[HISTORY_SIZE];
int historyIndex = 0;
unsigned long lastSample = 0;
int selectedChannel = 0;
unsigned long analysisCount = 0;

// Phosphor-style display characters for enhanced visualization
char phosphorChars[] = {};

// Curvature analysis buffers
float signFlips[HISTORY_SIZE];
int flipCount = 0;

void setup() {
  Serial.begin(115200);

  // Initialize all arrays
  for (int ch = 0; ch < NUM_CHANNELS; ch++) {
    for (int i = 0; i < HISTORY_SIZE; i++) {
      channelHistory[ch][i] = 0.0;
      smoothedData[ch][i] = 0.0;
    }
  }

  for (int i = 0; i < HISTORY_SIZE; i++) {
    curvatureData[i] = 0.0;
    signFlips[i] = 0.0;
  }

  Serial.println("CURVATURE ANALYSIS OSCILLOSCOPE");
  Serial.println("================================");
  Serial.println("Commands: 0-5 (channel), r (reset), c (curvature mode)");
  Serial.println("Specialized for pattern analysis and sign flip detection");
  Serial.println("========================================================");
}

void loop() {
  unsigned long currentTime = millis();

  // Handle serial commands
  handleSerialCommands();

  if (currentTime - lastSample >= SAMPLE_RATE_MS) {
    // Sample all channels with enhanced precision
    sampleAllChannels();

    // Apply moving average smoothing
    applySmoothingFilter();

    // Perform curvature analysis
    performCurvatureAnalysis();

    // Detect sign flips and rate of change
    detectSignFlips();

    // Display enhanced waveform
    displayCurvatureOscilloscope();

    historyIndex = (historyIndex + 1) % HISTORY_SIZE;
    analysisCount++;
    lastSample = currentTime;
  }
}

void handleSerialCommands() {
  if (Serial.available() > 0) {
    char cmd = Serial.read();
    if (cmd >= '0' && cmd <= '5') {
      selectedChannel = cmd - '0';
      Serial.print("■ SELECTED CHANNEL A");
      Serial.println(selectedChannel);
    } else if (cmd == 'r') {
      resetAnalysis();
      Serial.println("■ ANALYSIS RESET");
    } else if (cmd == 'c') {
      displayCurvatureStats();
    }
  }
}

void sampleAllChannels() {
  for (int ch = 0; ch < NUM_CHANNELS; ch++) {
    int rawValue = analogRead(analogPins[ch]);
    // Enhanced voltage conversion with better precision
    channelHistory[ch][historyIndex] = (rawValue / 1023.0) * 5.0;

    // Add some noise filtering
    if (historyIndex > 0) {
      float diff = abs(channelHistory[ch][historyIndex] -
                       channelHistory[ch][(historyIndex - 1 + HISTORY_SIZE) % HISTORY_SIZE]);
      if (diff > 0.5) { // Noise threshold
        channelHistory[ch][historyIndex] =
          (channelHistory[ch][historyIndex] +
           channelHistory[ch][(historyIndex - 1 + HISTORY_SIZE) % HISTORY_SIZE]) / 2.0;
      }
    }
  }
}

void applySmoothingFilter() {
  // Apply moving average smoothing similar to Python implementation
  for (int ch = 0; ch < NUM_CHANNELS; ch++) {
    float sum = 0;
    int count = 0;

    for (int i = 0; i < SMOOTHING_WINDOW && i < HISTORY_SIZE; i++) {
      int idx = (historyIndex - i + HISTORY_SIZE) % HISTORY_SIZE;
      sum += channelHistory[ch][idx];
      count++;
    }

    if (count > 0) {
      smoothedData[ch][historyIndex] = sum / count;
    }
  }
}

void performCurvatureAnalysis() {
  // Simplified curvature calculation based on second derivatives
  if (historyIndex >= 2) {
    int curr = historyIndex;
    int prev = (historyIndex - 1 + HISTORY_SIZE) % HISTORY_SIZE;
    int prev2 = (historyIndex - 2 + HISTORY_SIZE) % HISTORY_SIZE;

    float currentVal = smoothedData[selectedChannel][curr];
    float prevVal = smoothedData[selectedChannel][prev];
    float prev2Val = smoothedData[selectedChannel][prev2];

    // Calculate second derivative approximation
    float secondDeriv = currentVal - 2 * prevVal + prev2Val;
    curvatureData[historyIndex] = abs(secondDeriv) * 100; // Scale for visibility
  }
}

void detectSignFlips() {
  flipCount = 0;

  if (historyIndex >= 15) { // Minimum window for analysis
    float diffs[15];
    float signs[14];

    // Calculate differences over last 15 samples
    for (int i = 0; i < 15; i++) {
      int idx1 = (historyIndex - i + HISTORY_SIZE) % HISTORY_SIZE;
      int idx2 = (historyIndex - i - 1 + HISTORY_SIZE) % HISTORY_SIZE;
      diffs[i] = smoothedData[selectedChannel][idx1] - smoothedData[selectedChannel][idx2];
    }

    // Calculate signs
    for (int i = 0; i < 14; i++) {
      if (diffs[i] > 0.01) signs[i] = 1;
      else if (diffs[i] < -0.01) signs[i] = -1;
      else signs[i] = 0;
    }

    // Count sign flips
    for (int i = 0; i < 13; i++) {
      if (signs[i] != signs[i + 1] && signs[i] != 0 && signs[i + 1] != 0) {
        flipCount++;
      }
    }

    // Advanced analysis condition similar to Python code
    if (analysisCount * analysisCount < STATE_SPACE && flipCount > 1) {
      performAdvancedAnalysis();
    }
  }
}

void performAdvancedAnalysis() {
  // Calculate rate of change analysis
  float rateOfChange = 0;
  if (flipCount > 0) {
    rateOfChange = (float)flipCount * curvatureData[historyIndex] * analysisCount;
  }

  Serial.print("■ PATTERN DETECTED: Flips=");
  Serial.print(flipCount);
  Serial.print(" Rate=");
  Serial.print(rateOfChange, 4);
  Serial.print(" Curve=");
  Serial.println(curvatureData[historyIndex], 4);
}

void displayCurvatureOscilloscope() {
  // Enhanced display with phosphor-style effects
  
  // Enhanced statistics display
  displayEnhancedStats();
}

void displayEnhancedStats() {

  // Calculate enhanced statistics
  float currentVal = getCurrentValue(selectedChannel);
  float peakVal = getPeakValue(selectedChannel);
  float rmsVal = getRMSValue(selectedChannel);
  float curvatureAvg = getAverageCurvature();

  //Serial.print("CURRENT: ");
  //Serial.print(currentVal, 3);
  //Serial.print("V │ PEAK: ");
  //Serial.print(peakVal, 3);
  //Serial.print("V │ RMS: ");
  Serial.println(rmsVal, 3);

  //Serial.print(" CUzRVATURE: ");
  //Serial.print(curvatureAvg, 4);
  //Serial.print(" │ FLIPS: ");
 //Serial.print(flipCount);
 //Serial.print(" │ SAMPLES: ");
  //Serial.println(analysisCount);
}

float getCurrentValue(int channel) {
  int currentIndex = (historyIndex - 1 + HISTORY_SIZE) % HISTORY_SIZE;
  return channelHistory[channel][currentIndex];
}

float getPeakValue(int channel) {
  float peak = 0;
  for (int i = 0; i < HISTORY_SIZE; i++) {
    if (abs(channelHistory[channel][i]) > peak) {
      peak = abs(channelHistory[channel][i]);
    }
  }
  return peak;
}

float getRMSValue(int channel) {
  float sum = 0;
  for (int i = 0; i < HISTORY_SIZE; i++) {
    sum += channelHistory[channel][i] * channelHistory[channel][i];
  }
  return sqrt(sum / HISTORY_SIZE);
}

float getAverageCurvature() {
  float sum = 0;
  for (int i = 0; i < HISTORY_SIZE; i++) {
    sum += curvatureData[i];
  }
  return sum / HISTORY_SIZE;
}

void resetAnalysis() {
  for (int ch = 0; ch < NUM_CHANNELS; ch++) {
    for (int i = 0; i < HISTORY_SIZE; i++) {
      channelHistory[ch][i] = 0;
      smoothedData[ch][i] = 0;
    }
  }
  for (int i = 0; i < HISTORY_SIZE; i++) {
    curvatureData[i] = 0;
  }
  analysisCount = 0;
  flipCount = 0;
}

void displayCurvatureStats() {
  Serial.println("█████ CURVATURE ANALYSIS REPORT █████");
  Serial.print("Total Samples: ");
  Serial.println(analysisCount);
  Serial.print("Sign Flips: ");
  Serial.println(flipCount);
  Serial.print("Average Curvature: ");
  Serial.println(getAverageCurvature(), 4);
  Serial.println("██████████████████████████████████████");
}