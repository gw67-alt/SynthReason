import pyopencl as cl
import numpy as np
import random
import math
import sys
from collections import defaultdict, Counter
from tqdm import tqdm
import time
from datasets import load_dataset

KB_limit = -1  # change to -1 for unlimited

class SymbolicMarkovLPC:
    """
    Enhanced Markov chain text generator using symbolic probability distribution ⊆⊗∃·Λρ∑ω·Σø²,
    symbolic training count adjustments ∀λ±ε, and LPC (Linear Predictive Coding) decoding
    with OpenCL acceleration.
    """

    # Static configuration variables
    SENTENCE_END_CHARS = set('.!?')

    def __init__(self, n=2, lpc_order=10, use_opencl=True):
        """
        Initializes the Markov chain model with n-gram size and LPC parameters.
        Args:
            n (int): Size of n-gram context
            lpc_order (int): Order of the LPC prediction (number of coefficients)
            use_opencl (bool): Whether to use OpenCL acceleration when available
        """
        if not isinstance(n, int) or n < 1:
            raise ValueError("n-gram size 'n' must be a positive integer.")
        if not isinstance(lpc_order, int) or lpc_order < 1:
            raise ValueError("LPC order must be a positive integer.")
            
        self.n = n
        self.lpc_order = lpc_order
        self.use_opencl = use_opencl
        
        # Transitions will store floats due to symbolic adjustments
        self.m = {}  # Transitions: {context_tuple: {next_word: adjusted_frequency}}
        self.s = []  # Sentence starting n-grams
        self.all_words = set()  # Store all words seen during training
        
        # LPC-specific data structures
        self.word_to_index = {}  # Mapping from words to numeric indices
        self.index_to_word = {}  # Mapping from numeric indices back to words
        self.lpc_coefficients = None  # Will store LPC coefficients after training
        self.lpc_mean = None  # Will store mean value for LPC prediction
        self.word_vectors = None  # Will store numeric representations of words
        
        # Initialize OpenCL context and queue if available
        self.cl_ctx = None
        self.cl_queue = None
        self.cl_progs = {}
        
        if use_opencl:
            try:
                # Initialize OpenCL
                platforms = cl.get_platforms()
                if platforms:
                    # Try to get GPU device first, fallback to CPU if not available
                    try:
                        gpu_devices = platforms[0].get_devices(device_type=cl.device_type.GPU)
                        if gpu_devices:
                            self.cl_ctx = cl.Context(devices=gpu_devices)
                            self.cl_queue = cl.CommandQueue(self.cl_ctx)
                            print("Using OpenCL with GPU device:", gpu_devices[0].name)
                    except:
                        # Fallback to CPU
                        cpu_devices = platforms[0].get_devices(device_type=cl.device_type.CPU)
                        if cpu_devices:
                            self.cl_ctx = cl.Context(devices=cpu_devices)
                            self.cl_queue = cl.CommandQueue(self.cl_ctx)
                            print("Using OpenCL with CPU device:", cpu_devices[0].name)
                        else:
                            print("No OpenCL devices found, falling back to CPU implementation.")
                else:
                    print("No OpenCL platforms found, falling back to CPU implementation.")
                    self.use_opencl = False
                    
                if self.cl_ctx:
                    # Compile OpenCL kernels
                    self._compile_opencl_kernels()
            except:
                print("Error initializing OpenCL, falling back to CPU implementation.")
                import traceback
                traceback.print_exc()
                self.use_opencl = False

    def _compile_opencl_kernels(self):
        """Compile OpenCL kernels for various LPC operations"""
        if not self.cl_ctx:
            return
            
        # OpenCL kernel for autocorrelation calculation
        autocorr_kernel = """
        __kernel void autocorrelation(
            __global const float* sequence,
            __global float* result,
            const int seq_length,
            const int max_lag
        ) {
            int lag = get_global_id(0);
            if (lag >= max_lag)
                return;
                
            float sum = 0.0f;
            for (int i = 0; i < seq_length - lag; i++) {
                sum += sequence[i] * sequence[i + lag];
            }
            result[lag] = sum;
        }
        """
        
        # OpenCL kernel for LPC prediction
        lpc_prediction_kernel = """
        __kernel void lpc_predict(
            __global const float* past_sequence,
            __global const float* coefficients,
            __global float* result,
            const float mean,
            const int order
        ) {
            int idx = get_global_id(0);
            if (idx != 0) // We only need one result
                return;
                
            float prediction = mean;
            for (int i = 0; i < order; i++) {
                prediction += coefficients[i] * (past_sequence[order-i-1] - mean);
            }
            result[0] = prediction;
        }
        """
        
        # OpenCL kernel for vector distance calculation
        vector_distance_kernel = """
        __kernel void vector_distance(
            __global const float* vectors,
            __global const float* predicted_vector,
            __global float* distances,
            const int vector_dim,
            const int num_vectors
        ) {
            int vector_idx = get_global_id(0);
            if (vector_idx >= num_vectors)
                return;
                
            float sum_sq = 0.0f;
            for (int d = 0; d < vector_dim; d++) {
                float diff = vectors[vector_idx * vector_dim + d] - predicted_vector[d];
                sum_sq += diff * diff;
            }
            distances[vector_idx] = sqrt(sum_sq);
        }
        """
        
        try:
            # Compile the kernels
            self.cl_progs['autocorr'] = cl.Program(self.cl_ctx, autocorr_kernel).build()
            self.cl_progs['lpc_predict'] = cl.Program(self.cl_ctx, lpc_prediction_kernel).build()
            self.cl_progs['vector_distance'] = cl.Program(self.cl_ctx, vector_distance_kernel).build()
            print("OpenCL kernels compiled successfully.")
        except cl.RuntimeError as e:
            print(f"Error compiling OpenCL kernels: {e}")
            self.use_opencl = False

    def _convert_words_to_vectors(self, words):
        """
        Convert words to numeric vectors for LPC analysis.
        Uses character codes and word properties to create numeric representations.
        
        Args:
            words (list): List of unique words from training data
            
        Returns:
            dict: Mapping of words to numeric vectors
        """
        vectors = {}
        for word in words:
            # Create a vector representation based on character codes and word features
            char_codes = [ord(c) for c in word]
            
            # Pad or truncate to a fixed length (e.g., 10)
            fixed_length = 10
            if len(char_codes) > fixed_length:
                char_codes = char_codes[:fixed_length]
            else:
                char_codes = char_codes + [0] * (fixed_length - len(char_codes))
                
            # Add word features
            word_length = len(word)
            vowel_count = sum(1 for c in word if c.lower() in 'aeiou')
            consonant_count = word_length - vowel_count
            
            # Combine all features into a vector
            vector = np.array(char_codes + [word_length, vowel_count, consonant_count], dtype=np.float32)
            vectors[word] = vector
            
        return vectors

    def _calculate_lpc_coefficients_opencl(self, sequence, order):
        """
        Calculate LPC coefficients using OpenCL for acceleration.
        
        Args:
            sequence (np.array): Numeric sequence for analysis
            order (int): Order of LPC prediction
            
        Returns:
            tuple: (coefficients, mean)
        """
        if not self.cl_ctx or not self.cl_queue or 'autocorr' not in self.cl_progs:
            # Fallback to CPU implementation
            return self._calculate_lpc_coefficients_cpu(sequence, order)
            
        # Ensure sequence is long enough
        if len(sequence) < 2 * order + 1:
            padded_sequence = np.pad(sequence, (0, 2 * order + 1 - len(sequence)))
        else:
            padded_sequence = sequence
            
        # Convert to float32 for OpenCL
        seq_np = np.array(padded_sequence, dtype=np.float32)
        seq_length = np.int32(len(seq_np))
        max_lag = np.int32(order + 1)
        
        # Create OpenCL buffers
        seq_buf = cl.Buffer(self.cl_ctx, cl.mem_flags.READ_ONLY | cl.mem_flags.COPY_HOST_PTR, hostbuf=seq_np)
        autocorr_buf = cl.Buffer(self.cl_ctx, cl.mem_flags.WRITE_ONLY, size=max_lag * np.dtype(np.float32).itemsize)
        
        # Execute autocorrelation kernel
        global_size = (int(max_lag),)
        self.cl_progs['autocorr'].autocorrelation(
            self.cl_queue, global_size, None,
            seq_buf, autocorr_buf, seq_length, max_lag
        )
        
        # Read back results
        autocorr = np.empty(max_lag, dtype=np.float32)
        cl.enqueue_copy(self.cl_queue, autocorr, autocorr_buf)
        
        # Free OpenCL buffers
        seq_buf.release()
        autocorr_buf.release()
        
        # Handle potential numerical issues
        if np.isclose(autocorr[0], 0):
            return np.zeros(order, dtype=np.float32), np.mean(sequence).astype(np.float32)
            
        # Levinson-Durbin recursion to solve the Yule-Walker equations (on CPU for now)
        # This part is less parallelizable and more complex for OpenCL
        coeffs = np.zeros(order, dtype=np.float32)
        reflection = np.zeros(order, dtype=np.float32)
        error = autocorr[0]
        
        for i in range(order):
            # Calculate reflection coefficient
            reflection[i] = -autocorr[i+1]
            for j in range(i):
                reflection[i] -= coeffs[j] * autocorr[i-j]
            reflection[i] /= error
            
            # Update LPC coefficients
            coeffs[i] = reflection[i]
            for j in range(i//2):
                temp = coeffs[j]
                coeffs[j] += reflection[i] * coeffs[i-j-1]
                coeffs[i-j-1] += reflection[i] * temp
                
            error *= 1 - reflection[i]**2
            
        return coeffs, np.mean(sequence).astype(np.float32)

    def _calculate_lpc_coefficients_cpu(self, sequence, order):
        """
        Calculate LPC coefficients on CPU (fallback).
        
        Args:
            sequence (np.array): Numeric sequence for analysis
            order (int): Order of LPC prediction
            
        Returns:
            tuple: (coefficients, mean)
        """
        # Ensure sequence is at least 2*order+1 elements long
        if len(sequence) < 2 * order + 1:
            # Pad with zeros if necessary
            padded_sequence = np.pad(sequence, (0, 2 * order + 1 - len(sequence)))
        else:
            padded_sequence = sequence
            
        # Calculate autocorrelation
        autocorr = np.correlate(padded_sequence, padded_sequence, mode='full')
        # Take only the positive lags (including zero lag)
        autocorr = autocorr[len(autocorr)//2:][:order+1]
        
        # Handle potential numerical issues
        if np.isclose(autocorr[0], 0):
            return np.zeros(order, dtype=np.float32), np.mean(sequence).astype(np.float32)
            
        # Levinson-Durbin recursion to solve the Yule-Walker equations
        coeffs = np.zeros(order, dtype=np.float32)
        reflection = np.zeros(order, dtype=np.float32)
        error = autocorr[0]
        
        for i in range(order):
            # Calculate reflection coefficient
            reflection[i] = -autocorr[i+1]
            for j in range(i):
                reflection[i] -= coeffs[j] * autocorr[i-j]
            reflection[i] /= error
            
            # Update LPC coefficients
            coeffs[i] = reflection[i]
            for j in range(i//2):
                temp = coeffs[j]
                coeffs[j] += reflection[i] * coeffs[i-j-1]
                coeffs[i-j-1] += reflection[i] * temp
                
            error *= 1 - reflection[i]**2
            
        return coeffs, np.mean(sequence).astype(np.float32)

    def _predict_with_lpc_opencl(self, past_sequence, coefficients, mean):
        """
        Use LPC coefficients to predict the next value in a sequence using OpenCL.
        
        Args:
            past_sequence (np.array): Recent history of values
            coefficients (np.array): LPC coefficients
            mean (float): Mean value for prediction
            
        Returns:
            float: Predicted next value
        """
        if not self.cl_ctx or not self.cl_queue or 'lpc_predict' not in self.cl_progs:
            # Fallback to CPU implementation
            return self._predict_with_lpc_cpu(past_sequence, coefficients, mean)
            
        order = len(coefficients)
        # If past sequence is shorter than order, pad with zeros
        if len(past_sequence) < order:
            padded_sequence = np.pad(past_sequence, (order - len(past_sequence), 0))
        else:
            padded_sequence = past_sequence[-order:]
            
        # Convert to float32 for OpenCL
        padded_sequence = np.array(padded_sequence, dtype=np.float32)
        coeffs = np.array(coefficients, dtype=np.float32)
        mean_val = np.float32(mean)
        
        # Create OpenCL buffers
        seq_buf = cl.Buffer(self.cl_ctx, cl.mem_flags.READ_ONLY | cl.mem_flags.COPY_HOST_PTR, hostbuf=padded_sequence)
        coeffs_buf = cl.Buffer(self.cl_ctx, cl.mem_flags.READ_ONLY | cl.mem_flags.COPY_HOST_PTR, hostbuf=coeffs)
        result_buf = cl.Buffer(self.cl_ctx, cl.mem_flags.WRITE_ONLY, size=np.dtype(np.float32).itemsize)
        
        # Execute LPC prediction kernel
        global_size = (1,)
        self.cl_progs['lpc_predict'].lpc_predict(
            self.cl_queue, global_size, None,
            seq_buf, coeffs_buf, result_buf, mean_val, np.int32(order)
        )
        
        # Read back result
        result = np.empty(1, dtype=np.float32)
        cl.enqueue_copy(self.cl_queue, result, result_buf)
        
        # Free OpenCL buffers
        seq_buf.release()
        coeffs_buf.release()
        result_buf.release()
        
        return float(result[0])

    def _predict_with_lpc_cpu(self, past_sequence, coefficients, mean):
        """
        Use LPC coefficients to predict the next value in a sequence (CPU fallback).
        
        Args:
            past_sequence (np.array): Recent history of values
            coefficients (np.array): LPC coefficients
            mean (float): Mean value for prediction
            
        Returns:
            float: Predicted next value
        """
        order = len(coefficients)
        # If past sequence is shorter than order, pad with zeros
        if len(past_sequence) < order:
            padded_sequence = np.pad(past_sequence, (order - len(past_sequence), 0))
        else:
            padded_sequence = past_sequence[-order:]
            
        # Apply LPC prediction formula
        prediction = mean
        for i in range(order):
            prediction += coefficients[i] * (padded_sequence[order-i-1] - mean)
            
        return prediction

    def _calculate_vector_distances_opencl(self, word_vectors, predicted_vector):
        """
        Calculate Euclidean distances between word vectors and predicted vector using OpenCL.
        
        Args:
            word_vectors (np.array): Array of word vectors (2D: words x dimensions)
            predicted_vector (np.array): Predicted vector (1D)
            
        Returns:
            np.array: Distances for each word vector
        """
        if not self.cl_ctx or not self.cl_queue or 'vector_distance' not in self.cl_progs:
            # Fallback to CPU implementation
            return np.array([np.linalg.norm(wv - predicted_vector) for wv in word_vectors])
            
        # Convert inputs to float32 for OpenCL
        vectors_np = np.array(word_vectors, dtype=np.float32)
        predicted_np = np.array(predicted_vector, dtype=np.float32)
        
        num_vectors = np.int32(vectors_np.shape[0])
        vector_dim = np.int32(vectors_np.shape[1])
        
        # Create OpenCL buffers
        vectors_buf = cl.Buffer(self.cl_ctx, cl.mem_flags.READ_ONLY | cl.mem_flags.COPY_HOST_PTR, hostbuf=vectors_np)
        predicted_buf = cl.Buffer(self.cl_ctx, cl.mem_flags.READ_ONLY | cl.mem_flags.COPY_HOST_PTR, hostbuf=predicted_np)
        distances_buf = cl.Buffer(self.cl_ctx, cl.mem_flags.WRITE_ONLY, size=num_vectors * np.dtype(np.float32).itemsize)
        
        # Execute distance calculation kernel
        global_size = (int(num_vectors),)
        self.cl_progs['vector_distance'].vector_distance(
            self.cl_queue, global_size, None,
            vectors_buf, predicted_buf, distances_buf, vector_dim, num_vectors
        )
        
        # Read back results
        distances = np.empty(num_vectors, dtype=np.float32)
        cl.enqueue_copy(self.cl_queue, distances, distances_buf)
        
        # Free OpenCL buffers
        vectors_buf.release()
        predicted_buf.release()
        distances_buf.release()
        
        return distances

    def train(self, t):
        """
        Trains the Markov model on the provided text, applying symbolic
        adjustments ∀λ±ε to the transition counts. Also trains LPC model.

        Interpretation:
            ∀: Adjust increment based on global frequency of the next word.
            λ: Adjust increment based on the length of the next word.
            ±ε: Add small random noise to the increment.

        Args:
            t (str): The training text.
        """
        if not isinstance(t, str) or not t:
            raise TypeError("Training data must be a non-empty string.")

        words = t.split()
        num_words = len(words)

        if num_words <= self.n:
            raise ValueError(f"Training data has only {num_words} words, need more than n-gram size {self.n}.")

        print(f"Training on {num_words} words with n={self.n}...")
        print(f"Applying symbolic count adjustments ∀λ±ε...")

        # Calculate global word frequencies first (for ∀)
        print("Calculating global word frequencies (∀)...")
        overall_word_freqs = Counter(words)
        total_word_count = float(num_words)  # Use float for division

        # Build frequency dictionary using defaultdict with float values
        temp_m = defaultdict(lambda: defaultdict(float))
        temp_s = set()  # Use set for faster lookup

        # Track all words seen during training
        self.all_words = set(words)

        # First pass: collect base frequencies with symbolic adjustments ∀λ±ε
        for i in tqdm(range(num_words - self.n), desc="Collecting Adjusted Frequencies (∀λ±ε)"):
            g = tuple(words[i:i+self.n])
            next_word = words[i+self.n]

            # --- Symbolic Adjustment Calculation ---
            base_increment = 1.0

            # ∀ (Universal/Global Influence): Boost based on overall word frequency
            global_freq_factor = 1.0 + math.log1p(overall_word_freqs[next_word] / total_word_count) * 0.5

            # λ (Lambda/Word Property): Boost based on word length
            length_factor = 1.0 + math.log1p(len(next_word)) * 0.1

            # Combine ∀ and λ factors multiplicatively
            symbolic_factor = global_freq_factor * length_factor

            # Apply the combined factor to the base increment
            adjusted_increment = base_increment * symbolic_factor

            # ±ε (Noise/Variability): Add small random noise
            noise = random.uniform(-0.05, 0.05)  # Small epsilon range
            final_increment = adjusted_increment + noise

            # Ensure the increment is positive
            final_increment = max(0.01, final_increment)
            # --- End Symbolic Adjustment ---

            # Record sentence starts
            if i == 0 or any(c in self.SENTENCE_END_CHARS for c in words[i-1][-1:]):
                temp_s.add(g)

            # Increment frequency with the symbolically adjusted value
            temp_m[g][next_word] += final_increment

        # Convert to regular dictionaries for the final model
        self.m = {g: dict(next_words) for g, next_words in temp_m.items()}

        # Store valid sentence starts
        self.s = list(filter(lambda g: g in self.m, temp_s))

        # If no valid starts, use random contexts
        if not self.s and self.m:
            print("Warning: No sentence starts detected based on punctuation. Using random contexts as starting points.")
            self.s = random.sample(list(self.m.keys()), k=min(len(self.m), 100))

        if not self.s and not self.m:
            print("Warning: Model training resulted in no transitions. Check input data and n-gram size.")

        print(f"Training complete. Model has {len(self.m)} contexts and {len(self.s)} sentence starts. Frequencies adjusted by ∀λ±ε.")
        
        # --- LPC Training Phase ---
        print("Starting LPC training phase...")
        
        # Create word-to-index mapping for numeric representation
        unique_words = list(self.all_words)
        self.word_to_index = {word: idx for idx, word in enumerate(unique_words)}
        self.index_to_word = {idx: word for idx, word in enumerate(unique_words)}
        
        # Convert words to numeric vectors for LPC analysis
        print("Converting words to numeric vectors...")
        self.word_vectors = self._convert_words_to_vectors(unique_words)
        
        # Create a numeric sequence from the training text
        print("Creating numeric sequence for LPC analysis...")
        word_sequence = []
        for word in tqdm(words):
            if word in self.word_vectors:
                word_sequence.append(self.word_vectors[word])
        
        if len(word_sequence) < self.lpc_order * 2:
            print(f"Warning: Training sequence too short for LPC order {self.lpc_order}. Using lower order.")
            self.lpc_order = max(1, len(word_sequence) // 2)
        
        # Calculate LPC coefficients for each dimension of the word vectors
        print(f"Calculating LPC coefficients (order {self.lpc_order}) using {'OpenCL' if self.use_opencl else 'CPU'}...")
        
        # Reshape sequence for dimension-wise analysis
        word_sequence = np.array(word_sequence)
        vector_dims = word_sequence.shape[1]
        self.lpc_coefficients = []
        self.lpc_mean = []
        
        for dim in tqdm(range(vector_dims), desc="Computing LPC coefficients for each dimension"):
            dim_sequence = word_sequence[:, dim]
            if self.use_opencl:
                coeffs, mean = self._calculate_lpc_coefficients_opencl(dim_sequence, self.lpc_order)
            else:
                coeffs, mean = self._calculate_lpc_coefficients_cpu(dim_sequence, self.lpc_order)
            self.lpc_coefficients.append(coeffs)
            self.lpc_mean.append(mean)
        print("LPC training complete!")

    def _symbolic_probability(self, context, options):
        """
        Applies the symbolic probability distribution ⊆⊗∃·Λρ∑ω·Σø² to select the next word.

        Args:
            context (tuple): The current context (n-gram)
            options (dict): Possible next words with their *adjusted* frequencies

        Returns:
            tuple: (words, weights) Lists of candidate words and their probabilities
        """
        words = list(options.keys())
        # Frequencies are now floats due to training adjustments
        freqs = list(options.values())

        # Early return if no options
        if not words:
            return [], []

        # --- Start of Symbolic Probability Calculation (⊆⊗∃·Λρ∑ω·Σø²) ---

        # ⊆ (subset) - Filter options (keep frequencies > adjusted threshold or all)
        mean_freq = sum(freqs) / len(freqs) if freqs else 0
        subset_indices = [i for i, f in enumerate(freqs) if f > mean_freq * 0.5]

        # If nothing would remain, use all words
        if not subset_indices:
            subset_indices = list(range(len(words)))

        subsetWords = [words[i] for i in subset_indices]
        subsetFreqs = [freqs[i] for i in subset_indices]

        # If we still have nothing, return empty lists
        if not subsetWords:
            return [], []

        # ⊗ (tensor product) - Relationship between context and options
        tensorValues = []
        for word in context:
            tensorValue = 1.0
            for contextWord in subsetWords:
                overlap = len(set(contextWord) & set(word))
                tensorValue *= (float(overlap) + 1.0)
            tensorValues.append(tensorValue)

        # ∃ (existential) - Check for high probability option
        maxFreq = max(subsetFreqs) if subsetFreqs else 0
        existsHighProb = any(freq > maxFreq * 0.8 for freq in subsetFreqs) if maxFreq > 0 else False

        # Λ (wedge product) - Combine context influence
        lastLetters = [w[-1] if w else '' for w in context]
        uniqueLastLetters = len(set(lastLetters))
        contextInfluence = math.pow(float(uniqueLastLetters) + 1.0, 3.5)

        # ρ (rho) - Base density/distribution (using adjusted frequencies)
        totalFreq = sum(subsetFreqs)
        if totalFreq == 0:
            baseDistribution = [1.0 / len(subsetFreqs)] * len(subsetFreqs) if subsetFreqs else []
        else:
            baseDistribution = [freq / totalFreq for freq in subsetFreqs]

        # ∑ω (sum of weights) - Apply weights based on word properties
        wordWeights = []
        for word in subsetWords:
            lengthFactor = math.log(len(word) + 1.0)
            vowels = sum(1 for c in word if c.lower() in 'aeiou')
            consonants = len(word) - vowels
            vcRatio = (float(vowels) + 0.1) / (float(consonants) + 0.1)
            firstLetterCode = ord(word[0]) % 10 if word else 0
            wordWeight = lengthFactor * (vcRatio + 0.5) * (float(firstLetterCode) + 1.0)
            wordWeights.append(wordWeight)

        # Σø² (sum of squared empty set?) - Entropy/Final Adjustment
        adjustedWeights = []
        for i in range(len(subsetWords)):
            if i >= len(baseDistribution): continue  # Safety check

            combined = baseDistribution[i] * 5.0
            if i < len(tensorValues):
                combined *= math.pow(tensorValues[i], 0.1)

            if i < len(wordWeights):
                combined *= wordWeights[i] * 0.8

            combined *= math.pow(contextInfluence, 0.3)

            if existsHighProb and baseDistribution[i] < 0.3:
                combined *= 1.5

            # Apply the square as indicated by ²
            adjustedWeights.append(math.pow(max(0, combined), 0.7))

        # Normalize the final weights
        totalWeight = sum(adjustedWeights)
        if totalWeight == 0 or not adjustedWeights:
            normalizedWeights = [1.0 / len(subsetWords)] * len(subsetWords) if subsetWords else []
        else:
            normalizedWeights = [w / totalWeight for w in adjustedWeights]

        # --- End of Symbolic Probability Calculation ---

        # Ensure weights length matches words length
        if len(subsetWords) != len(normalizedWeights):
            normalizedWeights = [1.0 / len(subsetWords)] * len(subsetWords) if subsetWords else []

        return subsetWords, normalizedWeights

    def _apply_lpc_decoding(self, context_words, candidate_words, base_weights):
        """
        Apply LPC decoding to adjust word selection probabilities.
        Using OpenCL acceleration when available.
        
        Args:
            context_words (list): List of words in the current context
            candidate_words (list): List of candidate next words
            base_weights (list): Base probabilities from Markov model
            
        Returns:
            list: LPC-adjusted weights
        """
        if not self.lpc_coefficients or not context_words or not candidate_words:
            return base_weights
            
        # Get vector representations of context words
        context_vectors = []
        for word in context_words:
            if word in self.word_vectors:
                context_vectors.append(self.word_vectors[word])
                
        if not context_vectors:
            return base_weights
            
        # Stack vectors for sequence analysis
        context_sequence = np.array(context_vectors)
        
        # Make LPC predictions for each dimension
        vector_dims = context_sequence.shape[1]
        predicted_vector = np.zeros(vector_dims, dtype=np.float32)
        
        for dim in range(vector_dims):
            # Extract the sequence for this dimension
            dim_sequence = context_sequence[:, dim]
            # Make prediction using LPC with OpenCL acceleration if available
            if self.use_opencl:
                predicted_value = self._predict_with_lpc_opencl(
                    dim_sequence, 
                    self.lpc_coefficients[dim], 
                    self.lpc_mean[dim]
                )
            else:
                predicted_value = self._predict_with_lpc_cpu(
                    dim_sequence, 
                    self.lpc_coefficients[dim], 
                    self.lpc_mean[dim]
                )
            predicted_vector[dim] = predicted_value
            
        # Calculate similarity between predicted vector and candidate word vectors using OpenCL
        candidate_vectors = np.array([self.word_vectors.get(word, np.zeros(vector_dims, dtype=np.float32)) 
                                     for word in candidate_words])
        
        # Calculate distances between candidate vectors and predicted vector
        if self.use_opencl:
            distances = self._calculate_vector_distances_opencl(candidate_vectors, predicted_vector)
        else:
            distances = np.array([np.linalg.norm(cv - predicted_vector) for cv in candidate_vectors])
            
        # Convert distances to similarities (smaller distance = higher similarity)
        similarities = 1.0 / (1.0 + distances)
                
        # Normalize similarities
        total_similarity = np.sum(similarities)
        if total_similarity > 0:
            normalized_similarities = similarities / total_similarity
        else:
            normalized_similarities = np.ones(len(similarities)) / len(similarities)
            
        # Combine base weights with LPC-based similarities
        lpc_weight = 0.3  # How much to weight the LPC prediction (0.0 to 1.0)
        combined_weights = []
        
        for i in range(len(base_weights)):
            if i < len(normalized_similarities):
                combined = (1.0 - lpc_weight) * base_weights[i] + lpc_weight * normalized_similarities[i]
                combined_weights.append(combined)
            else:
                combined_weights.append(base_weights[i])
                
        # Normalize the final weights
        total_weight = sum(combined_weights)
        if total_weight > 0:
            return [w / total_weight for w in combined_weights]
        else:
            return [1.0 / len(combined_weights)] * len(combined_weights)

    def gen(self, seed=None, count=100, window_size=20, word_filter=None, use_lpc=True):
        """
        Generates text using the trained Markov model with symbolic probability distribution
        and optional LPC decoding with OpenCL acceleration.

        Args:
            seed (str, optional): Starting sequence of words.
            count (int): Number of words to generate.
            window_size (int): Size of the moving window for word filtering.
            word_filter (callable, optional): A function that takes (word, window_words) and returns
                                             True if the word should be allowed, False otherwise.
            use_lpc (bool): Whether to use LPC decoding to refine word selection
            
        Returns:
            str: Generated text.
        """
        if not self.m:
            raise ValueError("Model not trained. Call train() first.")

        if count < 1:
            raise ValueError("Word count must be positive.")

        if not self.s:
            # Try to recover if training failed silently
            if self.m:
                self.s = random.sample(list(self.m.keys()), k=min(len(self.m), 100))
                if not self.s:  # Still no starts possible
                    raise ValueError("No valid starting contexts available and recovery failed.")
            else:
                raise ValueError("Model has no transitions and no starting contexts.")

        # Process seed if provided
        if seed:
            seed_words = seed.lower().split()
            if len(seed_words) >= self.n:
                # Check if the exact seed context exists
                potential_context = tuple(seed_words[-self.n:])
                if potential_context in self.m:
                    context = potential_context
                    result = seed_words
                else:
                    # Seed context not found, fall back to random start but keep seed words
                    context = random.choice(self.s)
                    result = seed_words + list(context)  # Append random context start
            else:
                # Seed too short, use random start
                context = random.choice(self.s)
                result = list(context)
        else:
            # No seed, use a random start context
            context = random.choice(self.s)
            result = list(context)

        # Generate text
        retry_count = 0
        max_retries = 15  # Increased retries slightly
        max_filter_attempts = 5

        while len(result) < count:  # Check before loop body
            # Check context validity
            if context not in self.m or not self.m[context]:
                retry_count += 1
                if retry_count >= max_retries:
                    break  # Exit generation loop

                # Try finding a new valid context
                possible_starts = [s for s in self.s if s in self.m and self.m[s]]  # Filter for valid starts
                if not possible_starts:
                    # If even starts are bad, try any valid context
                    possible_starts = [c for c in self.m.keys() if self.m[c]]

                if not possible_starts:
                    break  # Exit generation loop

                context = random.choice(possible_starts)
                continue  # Try again with the new context

            # Get weighted options using symbolic probability
            words, weights = self._symbolic_probability(context, self.m[context])

            # Apply LPC decoding if enabled
            if use_lpc and self.lpc_coefficients is not None:
                weights = self._apply_lpc_decoding(context, words, weights)

            # Make sure we have valid options
            if not words or not weights or len(words) != len(weights) or sum(weights) == 0:
                retry_count += 1
                if retry_count >= max_retries:
                    break

                # Try finding a new valid context
                possible_starts = [s for s in self.s if s in self.m and self.m[s]]
                if not possible_starts:
                    possible_starts = [c for c in self.m.keys() if self.m[c]]
                if not possible_starts:
                    break

                context = random.choice(possible_starts)
                continue  # Try again

            # Get current window for filtering
            window_words = result[-min(len(result), window_size):]

            next_word = None  # Initialize next_word

            # Apply word filtering if provided
            if word_filter is not None:
                available_words = words.copy()
                available_weights = weights.copy()
                found_valid_word = False
                filter_attempts = 0

                while available_words and filter_attempts < max_filter_attempts:
                    # Normalize remaining weights
                    current_total_weight = sum(available_weights)
                    if current_total_weight <= 0:  # Check for non-positive weights
                        break  # Cannot sample

                    normalized_weights = [w / current_total_weight for w in available_weights]

                    # Random weighted choice from available options
                    try:
                        chosen_idx = random.choices(
                            range(len(available_words)),
                            weights=normalized_weights,
                            k=1
                        )[0]
                    except ValueError as e:
                        break  # Exit sampling loop

                    candidate_word = available_words[chosen_idx]

                    # Check filter
                    if word_filter(candidate_word, window_words):
                        next_word = candidate_word
                        found_valid_word = True
                        break  # Found a valid word

                    # Remove this word from consideration for next attempt
                    available_words.pop(chosen_idx)
                    available_weights.pop(chosen_idx)
                    filter_attempts += 1

                # If filter failed after attempts, choose from original list respecting weights
                if not found_valid_word:
                    try:
                        next_word = random.choices(words, weights=weights, k=1)[0]
                    except ValueError as e:
                        if words:  # Check if words list is not empty
                            next_word = random.choice(words)
                        else:
                            retry_count += 1  # Count as retry and try new context next iteration
                            continue

            else:
                # No filter, choose based on original weights
                try:
                    next_word = random.choices(words, weights=weights, k=1)[0]
                except ValueError as e:
                    if words:
                        next_word = random.choice(words)
                    else:
                        retry_count += 1
                        continue

            # Check if a word was actually selected
            if next_word is None:
                retry_count += 1
                continue

            # Add the chosen word and update context
            result.append(next_word)
            context = tuple(result[-self.n:])
            retry_count = 0  # Reset retry counter on successful word generation

        # Return exactly 'count' words (or fewer if generation stopped early)
        return ' '.join(result[-count:])


# Example usage
def no_repetition_filter(word, window_words):
    """Prevents a word from being repeated within the window"""
    return word not in window_words

def main():
    # Configuration
    CONFIG = {
        'input_filename': "test.txt",  # Default, will be overridden by input
        'ngram_size': 2,  # Default n-gram size (can be changed)
        'words_to_generate': 150,  # Default generation length
        'window_size': 50,  # Default window size for filter
        'lpc_order': 10,  # Default LPC order
        'use_lpc': True,   # Whether to use LPC decoding
        'use_opencl': True, # Whether to use OpenCL acceleration
        'use_huggingface': True, # Flag to use Hugging Face dataset
        'hf_dataset_name': "wikipedia",
        'hf_subset_name': "20220301.en"
    }

    print(f"--- Symbolic Markov Text Generator with OpenCL-accelerated LPC Decoding ---")
    print(f"--- (Training: ∀λ±ε | Generation: ⊆⊗∃·Λρ∑ω·Σø² + OpenCL LPC) ---")

    txt = ""
    if CONFIG['use_huggingface']:
        print(f"Loading Hugging Face dataset: {CONFIG['hf_dataset_name']}, subset: {CONFIG['hf_subset_name']}")
        try:
            dataset = load_dataset(CONFIG['hf_dataset_name'], CONFIG['hf_subset_name'], split="train", streaming=True,trust_remote_code=True)
            # Take a limited number of samples for demonstration
            sample = next(iter(dataset))
            txt = sample['text']
            print(f"Loaded a sample of text from the dataset (length: {len(txt)} characters).")
        except Exception as e:
            print(f"Error loading Hugging Face dataset: {e}")
            print("Falling back to default 'test.txt' for demonstration.")
            try:
                with open(CONFIG['input_filename'], 'r', encoding='utf-8') as file:
                    txt = ' '.join(file.read().lower().split()[:KB_limit])
                    if not txt:
                        print(f"Error: Input file '{CONFIG['input_filename']}' is empty.")
                        sys.exit(1)
            except FileNotFoundError:
                print(f"Error: Input file '{CONFIG['input_filename']}' not found.")
                sys.exit(1)
            except Exception as e:
                print(f"Error reading file: {e}")
                sys.exit(1)
    else:
        # Get filename from user input
        try:
            filename_to_use = input(f"Enter input filename (default: {CONFIG['input_filename']}): ")
            if not filename_to_use:
                filename_to_use = CONFIG['input_filename']
            CONFIG['input_filename'] = filename_to_use

            with open(CONFIG['input_filename'], 'r', encoding='utf-8') as file:
                txt = ' '.join(file.read().lower().split()[:KB_limit])
                if not txt:
                    print(f"Error: Input file '{CONFIG['input_filename']}' is empty.")
                    sys.exit(1)

        except FileNotFoundError:
            print(f"Error: Input file '{CONFIG['input_filename']}' not found.")
            sys.exit(1)
        except Exception as e:
            print(f"Error reading file: {e}")
            sys.exit(1)

    # Allow user to configure parameters
    try:
        ngram_input = input(f"Enter n-gram size (default: {CONFIG['ngram_size']}): ")
        if ngram_input:
            CONFIG['ngram_size'] = int(ngram_input)

        lpc_input = input(f"Enter LPC order (default: {CONFIG['lpc_order']}): ")
        if lpc_input:
            CONFIG['lpc_order'] = int(lpc_input)

        use_lpc_input = input(f"Use LPC decoding? (y/n, default: {'y' if CONFIG['use_lpc'] else 'n'}): ")
        if use_lpc_input:
            CONFIG['use_lpc'] = use_lpc_input.lower() == 'y'

        use_opencl_input = input(f"Use OpenCL acceleration? (y/n, default: {'y' if CONFIG['use_opencl'] else 'n'}): ")
        if use_opencl_input:
            CONFIG['use_opencl'] = use_opencl_input.lower() == 'y'
    except ValueError:
        print("Invalid input. Using default values.")

    # Initialize and train model
    try:
        model = SymbolicMarkovLPC(CONFIG['ngram_size'], CONFIG['lpc_order'], CONFIG['use_opencl'])
        model.train(txt)
    except ValueError as e:
        print(f"Error during initialization or training: {e}")
        sys.exit(1)
    except Exception as e:  # Catch other potential errors
        print(f"An unexpected error occurred during training: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

    # Interactive generation loop
    if not model.m or not model.s:
        print("\nModel training resulted in no usable data. Cannot generate text.")
    else:
        print(f"\n--- Ready to generate text ---")
        print(f"Using n={model.n}, LPC order={model.lpc_order}, generating {CONFIG['words_to_generate']} words.")
        print(f"LPC decoding is {'enabled' if CONFIG['use_lpc'] else 'disabled'}")
        print(f"OpenCL acceleration is {'enabled' if model.use_opencl else 'disabled'}")
        print(f"Applying filter: no_repetition_filter (window: {CONFIG['window_size']})")
        print(f"Enter seed text (at least {model.n} words recommended) or press Enter for random start. Type 'exit' to quit.")

        while True:
            user_input = input("\nSEED: ")
            if user_input.lower() in ['quit', 'exit', 'q']:
                print("Exiting...")
                break

            CONFIG['words_to_generate'] = 250

            # Generate text with the user's seed or random start
            start_time = time.time() if 'time' in sys.modules else None

            generated_text = model.gen(
                seed=user_input,
                count=CONFIG['words_to_generate'],
                window_size=CONFIG['window_size'],
                word_filter=no_repetition_filter,
                use_lpc=CONFIG['use_lpc']
            )

            end_time = time.time() if 'time' in sys.modules else None

            print("\nGENERATED:")
            print(generated_text)

            if start_time and end_time:
                print(f"\nGeneration took {end_time - start_time:.2f} seconds")

if __name__ == "__main__":
    import time  # Import time for performance measurement
    main()
