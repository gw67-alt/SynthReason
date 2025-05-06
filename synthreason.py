import random
import math
import sys
from collections import defaultdict, Counter
from tqdm import tqdm
import numpy as np
# Removed matplotlib

class SymbolicMarkov:
    """
    Markov chain text generator using symbolic probability distribution ⊆⊗∃·Λρ∑ω·Σø²
    and symbolic training count adjustments ∀λ±ε.
    (Includes modifications for calculating generation metrics + DEBUG prints)
    *** Includes Geometric Distribution Re-weighting ***
    """

    # Static configuration variables
    SENTENCE_END_CHARS = set('.!?')
    METRIC_NAMES = [ # Define names for clarity
        "Word Length", "V/C Ratio", "Num Choices", "Chosen Prob",
        "Entropy", "Context Overlap", "Filter Rejects", "Is Start Word"
    ]

    # Added geom_p parameter to init
    def __init__(self, n=2, geom_p=0.5): # Default geom_p = 0.5
        if not isinstance(n, int) or n < 1: raise ValueError("n must be > 0")
        if not (isinstance(geom_p, float) and 0 < geom_p <= 1.0):
             raise ValueError("geom_p must be a float > 0 and <= 1.0")
        self.n = n
        self.geom_p = geom_p # Store geometric distribution parameter
        self.m = {}
        self.s = []
        self.all_words = set()
        self._overall_word_freqs = Counter()

    def train(self, t):
        # ... (train remains the same) ...
        if not isinstance(t, str) or not t: raise TypeError("Training data empty")
        words = t.split(); num_words = len(words)
        if num_words <= self.n: raise ValueError(f"Need more words than n={self.n}")
        print(f"Training on {num_words} words with n={self.n}...")
        print(f"Applying symbolic count adjustments ∀λ±ε...")
        self._overall_word_freqs = Counter(words); total_word_count = float(num_words)
        temp_m = defaultdict(lambda: defaultdict(float)); temp_s = set()
        self.all_words = set(words)
        for i in tqdm(range(num_words - self.n), desc="Collecting Adjusted Frequencies (∀λ±ε)"):
            g = tuple(words[i:i+self.n]); next_word = words[i+self.n]
            base_increment = 1.0
            global_freq_factor = 1.0 + math.log1p(self._overall_word_freqs[next_word] / total_word_count) * 0.5
            length_factor = 1.0 + math.log1p(len(next_word)) * 0.1
            symbolic_factor = global_freq_factor * length_factor
            adjusted_increment = base_increment * symbolic_factor
            noise = random.uniform(-0.05, 0.05)
            final_increment = max(0.01, adjusted_increment + noise)
            # Identify potential sentence starts based on previous word ending punctuation
            # Ensure previous word exists (i>0) and check its last char
            if i == 0 or (words[i-1] and words[i-1][-1] in self.SENTENCE_END_CHARS):
                 temp_s.add(g)
            temp_m[g][next_word] += final_increment # Use final_increment for adjusted count
        self.m = {g: dict(next_words) for g, next_words in temp_m.items()}
        # Filter starts to ensure they actually exist as keys in the model and have successors
        self.s = list(filter(lambda g: g in self.m and self.m[g], temp_s))
        if not self.s and self.m:
            print("Warning: No valid sentence starts detected based on punctuation. Using random valid contexts.")
            valid_contexts = [ctx for ctx, successors in self.m.items() if successors] # Contexts with followers
            self.s = random.sample(valid_contexts, k=min(len(valid_contexts), 100)) if valid_contexts else []
        if not self.s and not self.m: print("Warning: Model training resulted in no transitions.")
        elif not self.s and self.m: print("Warning: Model trained but no valid starting points found.")
        print(f"Training complete. Model has {len(self.m)} contexts. Found {len(self.s)} valid sentence starts.")


    def _symbolic_probability(self, context, options):
        # Calculate initial weights based on symbolic factors
        words = list(options.keys()); freqs = list(options.values())
        if not words: return [], []

        # --- Initial Symbolic Weighting ---
        mean_freq = sum(freqs) / len(freqs) if freqs else 0
        # Filter for words with frequency above half the mean (or all if none meet criteria)
        subset_indices_map = {i: f for i, f in enumerate(freqs) if f > mean_freq * 0.5}
        if not subset_indices_map:
            subset_indices_map = {i:f for i, f in enumerate(freqs)}

        # Get the words and their corresponding initial frequencies for the subset
        original_indices = list(subset_indices_map.keys())
        subsetWords = [words[i] for i in original_indices]
        subsetFreqs = [subset_indices_map[i] for i in original_indices]

        if not subsetWords: return [], []

        # Context influence based on unique last letters
        lastLetters = [w[-1] if w else '' for w in context]; uniqueLastLetters = len(set(lastLetters))
        contextInfluence = math.pow(float(uniqueLastLetters) + 1.0, 3.5) # Power factor for context

        # Base distribution from subset frequencies
        totalFreq = sum(subsetFreqs)
        if totalFreq <= 1e-9: baseDistribution = [1.0 / len(subsetFreqs)] * len(subsetFreqs) if subsetFreqs else []
        else: baseDistribution = [freq / totalFreq for freq in subsetFreqs]

        # Calculate individual word weights based on features
        wordWeights = []
        for word in subsetWords:
            lengthFactor = math.log(len(word) + 1.0)
            vowels = sum(1 for c in word if c.lower() in 'aeiou'); consonants = len(word) - vowels
            vcRatio = (float(vowels) + 0.1) / (float(consonants) + 0.1) # Smoothed V/C Ratio
            firstLetterCode = ord(word[0]) % 10 if word else 0 # Simple feature from first letter
            wordWeight = lengthFactor * (vcRatio + 0.5) * (float(firstLetterCode) + 1.0)
            wordWeights.append(wordWeight)

        # Combine factors to get initial adjusted weights for the subset
        adjustedWeights = []
        # Check if any frequency significantly dominates
        existsHighProb = any(freq > sum(subsetFreqs) * 0.8 for freq in subsetFreqs) if sum(subsetFreqs) > 0 else False

        for i in range(len(subsetWords)):
            # Start with base probability, scaled up
            combined = baseDistribution[i] * 5.0 # Base scaling factor
            # Multiply by word-specific feature weight
            if i < len(wordWeights): combined *= wordWeights[i] * 0.8 # Word feature scaling
            # Multiply by context influence factor
            combined *= math.pow(contextInfluence, 0.4) # Context influence scaling
            # Boost if a high-probability word exists but this one isn't it (promotes alternatives)
            if existsHighProb and baseDistribution[i] < 0.3: combined *= 1.5 # Diversity boost

            # Apply power law and ensure non-negative
            adjustedWeights.append(math.pow(max(0, combined), 2)) # Squaring emphasizes higher values

        # --- Geometric Distribution Re-weighting ---
        if not subsetWords or not adjustedWeights or len(subsetWords) != len(adjustedWeights):
            print(f"Warning: Inconsistency before geometric re-weighting. Fallback uniform.")
            return subsetWords, ([1.0 / len(subsetWords)] * len(subsetWords) if subsetWords else [])

        # Sort words by their current adjustedWeight to determine rank (k=1 is highest weight)
        # We need ranks associated with the *original* indices within the subset
        indexed_weights = list(enumerate(adjustedWeights)) # [(0, w0), (1, w1), ...] indices within subset
        sorted_by_weight = sorted(indexed_weights, key=lambda item: item[1], reverse=True)

        # Create a map from the subset index to its rank
        ranks = {subset_idx: rank + 1 for rank, (subset_idx, weight) in enumerate(sorted_by_weight)}

        geometric_modified_weights = [0.0] * len(adjustedWeights)
        geom_p = self.geom_p # Use the instance parameter

        for i, original_weight in enumerate(adjustedWeights): # i is index within subset
            if original_weight <= 1e-9: # Skip zero/negligible weights
                 continue
            rank_k = ranks.get(i)
            if rank_k is None: # Should not happen
                 print(f"Warning: Index {i} not found in ranks during geometric mod.")
                 continue

            # Calculate geometric factor P(X=k) = (1-p)^(k-1) * p
            geometric_factor = math.pow(1.0 - geom_p, rank_k - 1) * geom_p

            # Apply the factor by multiplication
            geometric_modified_weights[i] = original_weight * geometric_factor

        # --- Final Normalization ---
        finalTotalWeight = sum(geometric_modified_weights)
        if finalTotalWeight <= 1e-9:
            # Fallback if re-weighting nullified everything (e.g., p=1 and many items, or all weights zero initially)
            print(f"Warning: Geometric modification resulted in zero total weight for context '{context}'. Using uniform fallback over subset.")
            normalizedWeights = [1.0 / len(subsetWords)] * len(subsetWords) if subsetWords else []
        else:
            normalizedWeights = [w / finalTotalWeight for w in geometric_modified_weights]

        # Final sanity check
        if len(subsetWords) != len(normalizedWeights):
             print(f"Warning:_symbolic_probability final mismatch. Fallback uniform.")
             normalizedWeights = [1.0 / len(subsetWords)] * len(subsetWords) if subsetWords else []

        # Return the original subset words and their NEW final normalized weights
        return subsetWords, normalizedWeights


    def _calculate_step_metrics(self, chosen_word, context, candidate_words,
                                 candidate_weights, chosen_prob, filter_rejects, is_start):
        # ... (calculate_step_metrics remains the same) ...
        metrics = [0.0] * 8 # Corresponds to METRIC_NAMES
        metrics[0] = float(len(chosen_word)) # Word Length

        vowels = sum(1 for c in chosen_word if c.lower() in 'aeiou')
        consonants = len(chosen_word) - vowels
        metrics[1] = (float(vowels) + 0.1) / (float(consonants) + 0.1) # Smoothed V/C Ratio

        metrics[2] = float(len(candidate_words)) # Num Choices (before filtering)

        metrics[3] = float(chosen_prob) # Chosen Prob (final probability after filtering/selection)

        # Entropy calculation based on the final candidate weights distribution
        if candidate_weights and sum(candidate_weights) > 1e-9:
            # Ensure weights are normalized probabilities for entropy calc
            probs = np.array(candidate_weights)
            probs_sum = np.sum(probs)
            if probs_sum > 1e-9:
                probs = probs / probs_sum # Normalize
                # Ensure no zero probabilities for log calculation (use small epsilon)
                probs = np.maximum(probs, 1e-12)
                # Entropy H(X) = - sum(p(x) * log(p(x))) - using natural log (nats)
                # Using exp(probs) in the provided code seems unusual, might be a specific variant or typo.
                # Standard Shannon entropy: metrics[4] = -np.sum(probs * np.log(probs))
                # Original code used exp(probs):
                metrics[4] = -np.sum(probs * np.exp(probs)) # Keep original calculation for consistency
            else:
                metrics[4] = 0.0
        else:
            metrics[4] = 0.0 # Entropy is 0 if no choices or zero probability

        # Context Overlap: Average character overlap between chosen word and context words
        total_overlap = 0; count = 0
        if chosen_word and context:
            chosen_set = set(chosen_word)
            for ctx_word in context:
                if ctx_word:
                    total_overlap += len(set(ctx_word) & chosen_set)
                    count += 1
        metrics[5] = float(total_overlap / count) if count > 0 else 0.0

        metrics[6] = float(filter_rejects) # Filter Rejects

        metrics[7] = 1.0 if is_start else 0.0 # Is Start Word

        return metrics

    # Keep original gen for compatibility if needed
    def gen(self, seed=None, count=100, window_size=20, word_filter=None):
          generated_text, _ = self.generate_and_get_metrics(seed, count, window_size, word_filter)
          return generated_text # Or return just the last 'count' words if preferred

    def generate_and_get_metrics(self, seed=None, count=100, window_size=20, word_filter=None):
        """
        Generates text and calculates inference metrics for each step.
        (Includes DEBUG prints and geometric re-weighting)
        """
        if not self.m: raise ValueError("Model not trained.")
        if count < 1: raise ValueError("Word count must be positive.")
        # Ensure self.s is valid before starting
        if not self.s:
            if not self.m: raise ValueError("Model has no transitions and no starting points.")
            # Attempt recovery if self.s is empty but model exists (should be handled in train)
            print("Warning: No valid starting points found during generation init. Attempting recovery.")
            valid_contexts = [ctx for ctx, successors in self.m.items() if successors]
            self.s = random.sample(valid_contexts, k=min(len(valid_contexts), 100)) if valid_contexts else []
            if not self.s: raise ValueError("Cannot start generation: No valid starting contexts found in model even after recovery attempt.")

        context = None
        result = []
        is_start_word_flag = True # Track if the next word starts a sequence

        # --- Seed Processing ---
        if seed:
            seed_words = seed.lower().split()
            if len(seed_words) >= self.n:
                potential_context = tuple(seed_words[-self.n:])
                # Check if seed context is valid *and* has successors in the model
                if potential_context in self.m and self.m[potential_context]:
                    context = potential_context
                    result = seed_words # Start result with the seed words
                    is_start_word_flag = False # Seed provides context, so next word isn't a true start
                    print(f"DEBUG: Starting with seed context: {context}")
                else:
                    print(f"Warning: Seed context {potential_context} not found or has no successors in model. Starting randomly.")
                    result = [] # Discard invalid seed context
            else: # Seed too short
                print("Warning: Seed is shorter than n. Starting randomly.")
                result = [] # Start fresh

        # --- Random Start (if no valid seed context) ---
        if context is None:
            if not self.s: raise ValueError("Cannot start generation: No starting contexts available.")
            context = random.choice(self.s) # self.s should only contain valid contexts now
            result.extend(list(context)) # Start result with the chosen context words
            is_start_word_flag = True # This context choice is the start
            print(f"DEBUG: Starting with random context: {context}")


        # --- Generation loop ---
        generated_metrics = []
        retry_count = 0
        max_retries = 10
        max_filter_attempts = 5
        generated_word_count = 0 # Count words generated *in this loop*

        while generated_word_count < count:

            # --- Check context validity and get candidates ---
            if context not in self.m or not self.m[context]:
                # Context became invalid (e.g., led to a dead end not caught earlier)
                retry_count += 1
                print(f"DEBUG: Current context {context} became invalid or has no successors. Attempting reset ({retry_count}/{max_retries}).")
                if retry_count >= max_retries:
                    print(f"DEBUG: Max retries reached on invalid context. Breaking generation.")
                    break
                # Attempt reset to a known valid starting point
                valid_starts = [s for s in self.s if s in self.m and self.m[s]] # Re-check validity of starts
                if not valid_starts:
                     print("DEBUG: No valid starting points left for reset. Breaking.")
                     break
                context = random.choice(valid_starts)
                result.extend(list(context)) # Add the new context to result
                print(f"DEBUG: Reset context to {context}")
                is_start_word_flag = True
                continue # Retry with new context in next loop iteration

            # Get candidates using the current valid context
            current_options = self.m[context]
            # Now calls the modified probability function
            candidate_words, candidate_weights = self._symbolic_probability(context, current_options)

            # --- Check if candidates are valid ---
            if not candidate_words or not candidate_weights or len(candidate_words) != len(candidate_weights) or sum(candidate_weights) < 1e-9:
                retry_count += 1
                print(f"DEBUG: No valid/weighted candidates returned from _symbolic_probability for {context}. Attempting reset ({retry_count}/{max_retries})")
                if retry_count >= max_retries:
                    print(f"DEBUG: Max retries reached on candidate failure. Breaking generation.")
                    break
                # Reset context to another valid starting point
                valid_starts = [s for s in self.s if s in self.m and self.m[s]]
                if not valid_starts: print("DEBUG: No valid starts left for reset. Breaking."); break
                context = random.choice(valid_starts)
                result.extend(list(context)) # Add the new context
                print(f"DEBUG: Reset context to {context} (due to candidate failure)")
                is_start_word_flag = True
                continue # Retry with new context

            # --- Filtering and Selection ---
            next_word = None
            chosen_prob_value = 0.0 # Probability of the chosen word from the final distribution
            filter_rejections = 0
            window_words = result[-min(len(result), window_size):] # Look at recent history for filter

            # Apply filter or choose directly
            if word_filter is not None:
                # Filter logic remains largely the same, but operates on the potentially geometrically-modified weights
                available_words = candidate_words.copy()
                available_weights = candidate_weights.copy()
                found_valid_word = False
                filter_attempts = 0

                while available_words and filter_attempts < max_filter_attempts:
                    current_total_weight = sum(available_weights)
                    if current_total_weight <= 1e-9: break # No more probability mass to sample from

                    # Normalize remaining weights for sampling
                    normalized_weights = [w / current_total_weight for w in available_weights]

                    try:
                        # Choose an index from the *currently available* list
                        choices = random.choices(range(len(available_words)), weights=normalized_weights, k=1)
                        if not choices: break # Should not happen if current_total_weight > 0
                        chosen_idx_in_available = choices[0]

                        candidate_word = available_words[chosen_idx_in_available]

                        # Apply the filter
                        if word_filter(candidate_word, window_words):
                            next_word = candidate_word
                            # Find the probability of this word in the *original* distribution from _symbolic_probability
                            original_idx = candidate_words.index(candidate_word) # Find index in the unfiltered list
                            chosen_prob_value = candidate_weights[original_idx] # Get its probability
                            found_valid_word = True
                            break # Found a word that passes the filter
                        else:
                            # Word was rejected by filter
                            filter_rejections += 1
                            # Remove rejected word and its weight
                            available_words.pop(chosen_idx_in_available)
                            available_weights.pop(chosen_idx_in_available)
                            filter_attempts += 1

                    except ValueError as e: # E.g., weights don't sum to positive, empty lists
                        print(f"DEBUG: Error during filtered choice sampling: {e}"); break
                    except IndexError as e:
                        print(f"DEBUG: Index error during filtering, likely weight/word mismatch: {e}"); break

                if not found_valid_word:
                    # If filtering removed all options or reached max attempts, fallback needed
                    filter_rejections = filter_attempts # Record attempts made before fallback
                    print(f"DEBUG: Filter removed all options or max attempts reached for context {context}. Fallback choice.")
                    try: # Fallback: Choose from the original unfiltered list
                        if not candidate_words: raise ValueError("No candidates for fallback choice.")
                        # Use original weights from _symbolic_probability for fallback
                        fallback_indices = random.choices(range(len(candidate_words)), weights=candidate_weights, k=1)
                        if not fallback_indices: raise ValueError("Fallback choice returned empty.")
                        chosen_idx = fallback_indices[0]
                        next_word = candidate_words[chosen_idx]
                        chosen_prob_value = candidate_weights[chosen_idx]
                    except Exception as e:
                        print(f"DEBUG: Error during filter fallback choice: {e}. Trying uniform.")
                        # Ultimate fallback: uniform random choice if possible
                        if candidate_words:
                             next_word = random.choice(candidate_words)
                             chosen_prob_value = 1.0 / len(candidate_words) # Approximate probability
                        else:
                             print(f"DEBUG: No candidates even for uniform fallback."); next_word = None # Will trigger reset

            else: # No filter applied
                # print("DEBUG: No filter.")
                try:
                    if not candidate_words: raise ValueError("No candidates for choice (unfiltered).")
                    # Choose directly using the weights from _symbolic_probability
                    chosen_indices = random.choices(range(len(candidate_words)), weights=candidate_weights, k=1)
                    if not chosen_indices: raise ValueError("Unfiltered choice returned empty.")
                    chosen_idx = chosen_indices[0]
                    next_word = candidate_words[chosen_idx]
                    chosen_prob_value = candidate_weights[chosen_idx]
                except Exception as e:
                    print(f"DEBUG: Error during unfiltered choice: {e}. Trying uniform.")
                    if candidate_words:
                        next_word = random.choice(candidate_words)
                        chosen_prob_value = 1.0 / len(candidate_words) # Approximate probability
                    else:
                         print(f"DEBUG: No candidates for uniform choice."); next_word = None # Will trigger reset

            # --- Check selection outcome ---
            if next_word is None:
                # This occurs if even fallbacks failed (e.g., no candidates at all initially)
                print(f"DEBUG: Failed to select any next_word in this iteration for context {context}. Resetting.")
                retry_count += 1
                if retry_count >= max_retries: print(f"DEBUG: Max retries overall. Breaking."); break
                # Reset context
                valid_starts = [s for s in self.s if s in self.m and self.m[s]]
                if not valid_starts: print("DEBUG: No valid starts left for reset. Breaking."); break
                context = random.choice(valid_starts)
                result.extend(list(context)) # Add new context words
                print(f"DEBUG: Reset context to {context} (due to selection failure)")
                is_start_word_flag = True
                continue # Try again with new context

            # --- Selection successful: Calculate metrics and update ---
            step_metrics = self._calculate_step_metrics(
                  chosen_word=next_word, context=context,
                  candidate_words=candidate_words, # Full list from _symbolic
                  candidate_weights=candidate_weights, # Weights from _symbolic
                  chosen_prob=chosen_prob_value, # Probability of the actual chosen word
                  filter_rejects=filter_rejections,
                  is_start=is_start_word_flag
            )
            generated_metrics.append(step_metrics)
            # print(f"DEBUG: Chose: '{next_word}' (Prob: {chosen_prob_value:.4f}) Metrics: {[f'{m:.2f}' for m in step_metrics]}")

            result.append(next_word)
            context = tuple(result[-self.n:]) # Update context
            generated_word_count += 1
            is_start_word_flag = False # Next word is not a start word unless context resets
            retry_count = 0 # Reset retry count on successful generation step

        # --- Finalize ---
        metrics_array = np.array(generated_metrics)

        # Return all words generated including seed/start context for inspection
        final_text = ' '.join(result)
        return final_text, metrics_array


# --- Example Usage (`if __name__ == "__main__":`) ---
def no_repetition_filter(word, window_words):
    """Simple filter: prevents the exact word from appearing in the last `window_size` words."""
    return word not in window_words

if __name__ == "__main__":
    CONFIG = { # Defaults
        'input_filename': "test.txt", 'ngram_size': 2,
        'words_to_generate': 100, # Reduced for quicker testing
        'window_size': 10,
        'geom_p': 0.5 # Default geometric distribution parameter
    }
    print(f"--- Symbolic Markov Generator (with Geometric Re-weighting) ---")
    try: # File/Input handling
        fname = input(f"Input filename (def: {CONFIG['input_filename']}): ") or CONFIG['input_filename']
        CONFIG['input_filename'] = fname
        with open(fname, 'r', encoding='utf-8') as f:
             # Read, lower, split into words, join back with single spaces to normalize whitespace
             txt = ' '.join(f.read().lower().split())
        if not txt: print("Error: Input file empty."); sys.exit(1)
    except FileNotFoundError:
        print(f"Error: File '{CONFIG['input_filename']}' not found. Using sample text.")
        txt = ("this is a very simple sample text it is short but will serve the purpose of demonstrating the generator repeat repeat short text sample this is another sentence the end") # Slightly longer sample
        CONFIG['input_filename'] = "sample_text"
    except Exception as e: print(f"Error reading file: {e}"); sys.exit(1)

    try: # Ngram size
        n_in = input(f"N-gram size (def: {CONFIG['ngram_size']}): ")
        CONFIG['ngram_size'] = int(n_in) if n_in else CONFIG['ngram_size']
    except ValueError: print("Invalid N, using default.")

    try: # Geometric P
        p_in = input(f"Geometric distribution P (0<p<=1, def: {CONFIG['geom_p']}): ")
        if p_in:
             try:
                 p_val = float(p_in)
                 if 0 < p_val <= 1.0: CONFIG['geom_p'] = p_val
                 else: print("Invalid P value, using default.")
             except ValueError: print("Invalid P input, using default.")
    except ValueError: print("Invalid P, using default.")


    try: # Word count
        w_in = input(f"Words to generate (def: {CONFIG['words_to_generate']}): ")
        CONFIG['words_to_generate'] = int(w_in) if w_in else CONFIG['words_to_generate']
    except ValueError: print("Invalid count, using default.")

    # --- Train ---
    model = None
    try:
        # Pass geom_p to the constructor
        model = SymbolicMarkov(n=CONFIG['ngram_size'], geom_p=CONFIG['geom_p'])
        model.train(txt)
    except Exception as e:
        print(f"\n--- Training Error ---")
        print(f"{e}")
        # import traceback # Optional: Uncomment for full stack trace
        # traceback.print_exc()
        sys.exit(1)

    # --- Generate ---
    if not model or not model.m or not model.s: # Check if model is usable after training
        print("\nModel training failed or resulted in no usable data.")
    else:
        print(f"\n--- Ready to generate ---")
        print(f"Model: n={model.n}, geom_p={model.geom_p}")
        print(f"Target: {CONFIG['words_to_generate']} words")
        print(f"Enter seed (>= {model.n} words) or press Enter for random start.")
        print(f"Type 'exit' to quit.")

        while True:
            try:
                user_input = input("\nSEED: ")
                if user_input.lower() in ['quit', 'exit']: break

                # --- Generate and get metrics ---
                generated_text, metrics_data = model.generate_and_get_metrics(
                    seed=user_input or None,
                    count=CONFIG['words_to_generate'],
                    window_size=CONFIG['window_size'],
                    word_filter=no_repetition_filter # Set to None to disable filter
                )

                # --- Display results ---
                print("\n--- Generated Text ---")
                if user_input: print(f"(Seed: '{user_input}')")
                print(generated_text)
                # Optional: Print average metrics
                if metrics_data.size > 0:
                     avg_metrics = np.mean(metrics_data, axis=0)
                     print("\n--- Average Metrics ---")
                     for name, val in zip(model.METRIC_NAMES, avg_metrics):
                         print(f"{name:>15}: {val:.3f}")
                print("-" * 60)

            except KeyboardInterrupt: print("\nExiting."); break
            except ValueError as e: print(f"Generation Error: {e}") # Errors from model logic
            except Exception as e:
                 print(f"Unexpected Error during generation: {e}")
                 import traceback # Optional: Uncomment for full stack trace
                 traceback.print_exc()

    print("\n--- Run Complete ---")
