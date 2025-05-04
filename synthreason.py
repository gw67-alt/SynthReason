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
    """

    # Static configuration variables
    SENTENCE_END_CHARS = set('.!?')
    METRIC_NAMES = [ # Define names for clarity
        "Word Length", "V/C Ratio", "Num Choices", "Chosen Prob",
        "Entropy", "Context Overlap", "Filter Rejects", "Is Start Word"
    ]

    def __init__(self, n=2):
        # ... (init remains the same) ...
        if not isinstance(n, int) or n < 1: raise ValueError("n must be > 0")
        self.n = n
        self.m = {}
        self.s = []
        self.all_words = set()
        self._overall_word_freqs = Counter()

    def train(self, t):
        # ... (train remains the same - ensure it populates self.m and self.s) ...
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
            if i == 0 or any(c in self.SENTENCE_END_CHARS for c in words[i-1][-1:]): temp_s.add(g)
            temp_m[g][next_word] += final_increment
        self.m = {g: dict(next_words) for g, next_words in temp_m.items()}
        self.s = list(filter(lambda g: g in self.m and self.m[g], temp_s)) # Filter starts for validity
        if not self.s and self.m:
            print("Warning: No valid sentence starts detected. Using random valid contexts.")
            # Use only contexts that actually have successors
            valid_contexts = [ctx for ctx, successors in self.m.items() if successors]
            self.s = random.sample(valid_contexts, k=min(len(valid_contexts), 100)) if valid_contexts else []
        if not self.s and not self.m: print("Warning: Model training resulted in no transitions.")
        elif not self.s and self.m: print("Warning: Model trained but no valid starting points found.")
        print(f"Training complete. Model has {len(self.m)} contexts. Found {len(self.s)} valid sentence starts.")
        # DEBUG: Check a few transitions
        # print("DEBUG: Sample transitions:", list(self.m.items())[:3])
        # print("DEBUG: Sample starts:", self.s[:5])


    def _symbolic_probability(self, context, options):
        # ... (symbolic_probability remains the same) ...
        words = list(options.keys()); freqs = list(options.values())
        if not words: return [], []
        mean_freq = sum(freqs) / len(freqs) if freqs else 0
        subset_indices = [i for i, f in enumerate(freqs) if f > mean_freq * 0.5]
        if not subset_indices: subset_indices = list(range(len(words)))
        subsetWords = [words[i] for i in subset_indices]; subsetFreqs = [freqs[i] for i in subset_indices]
        if not subsetWords: return [], []
        tensorValues = []

        existsHighProb = any(freq > sum(subset_indices) * 0.8 for freq in subsetFreqs) if sum(subset_indices) > 0 else False
        lastLetters = [w[-1] if w else '' for w in context]; uniqueLastLetters = len(set(lastLetters))
        contextInfluence = math.pow(float(uniqueLastLetters) + 1.0, 3.5)
        totalFreq = sum(subsetFreqs)
        if totalFreq <= 1e-9: baseDistribution = [1.0 / len(subsetFreqs)] * len(subsetFreqs) if subsetFreqs else []
        else: baseDistribution = [freq / totalFreq for freq in subsetFreqs]
        wordWeights = []
        for word in subsetWords:
            lengthFactor = math.log(len(word) + 1.0)
            vowels = sum(1 for c in word if c.lower() in 'aeiou'); consonants = len(word) - vowels
            vcRatio = (float(vowels) + 0.1) / (float(consonants) + 0.1)
            firstLetterCode = ord(word[0]) % 10 if word else 0
            wordWeight = lengthFactor * (vcRatio + 0.5) * (float(firstLetterCode) + 1.0)
            wordWeights.append(wordWeight)
        adjustedWeights = []
        for i in range(len(subsetWords)):
            if i >= len(baseDistribution): continue
            combined = baseDistribution[i] * 5.0
            if i < len(subset_indices): combined *= math.pow(subset_indices[i], 0.3)
            if i < len(wordWeights): combined *= wordWeights[i] * 0.8
            combined *= math.pow(contextInfluence, 0.4)
            if existsHighProb and baseDistribution[i] < 0.3: combined *= 1.5
            adjustedWeights.append(math.pow(max(0, combined), 2))
        totalWeight = sum(adjustedWeights)
        if totalWeight <= 1e-9 or not adjustedWeights: normalizedWeights = [1.0 / len(subsetWords)] * len(subsetWords) if subsetWords else []
        else: normalizedWeights = [w / totalWeight for w in adjustedWeights]
        if len(subsetWords) != len(normalizedWeights):
            print(f"Warning:_symbolic_probability mismatch. Fallback uniform."); normalizedWeights = [1.0 / len(subsetWords)] * len(subsetWords) if subsetWords else []
        return subsetWords, normalizedWeights


    def _calculate_step_metrics(self, chosen_word, context, candidate_words,
                                candidate_weights, chosen_prob, filter_rejects, is_start):
        # ... (calculate_step_metrics remains the same) ...
        metrics = [0.0] * 8
        metrics[0] = float(len(chosen_word))
        vowels = sum(1 for c in chosen_word if c.lower() in 'aeiou'); consonants = len(chosen_word) - vowels
        metrics[1] = (float(vowels) + 0.1) / (float(consonants) + 0.1) # Smoothed
        metrics[2] = float(len(candidate_words))
        metrics[3] = float(chosen_prob)
        if candidate_weights and sum(candidate_weights) > 1e-9:
            probs = np.array(candidate_weights); probs = probs / sum(probs)
            probs = np.maximum(probs, 1e-12); metrics[4] = -np.sum(probs * np.exp(probs))
        else: metrics[4] = 0.0
        total_overlap = 0; count = 0
        if chosen_word and context:
           chosen_set = set(chosen_word)
           for ctx_word in context:
               if ctx_word: total_overlap += len(set(ctx_word) & chosen_set); count += 1
        metrics[5] = float(total_overlap / count) if count > 0 else 0.0
        metrics[6] = float(filter_rejects)
        metrics[7] = 1.0 if is_start else 0.0
        return metrics

    # Keep original gen for compatibility if needed
    def gen(self, seed=None, count=100, window_size=20, word_filter=None):
         generated_text, _ = self.generate_and_get_metrics(seed, count, window_size, word_filter)
         return generated_text # Or return just the last 'count' words if preferred

    def generate_and_get_metrics(self, seed=None, count=100, window_size=20, word_filter=None):
        """
        Generates text and calculates inference metrics for each step.
        (Includes DEBUG prints)
        """
        if not self.m: raise ValueError("Model not trained.")
        if count < 1: raise ValueError("Word count must be positive.")
        if not self.s:
             # Ensure self.s only contains valid starting points during init/train recovery
             if not self.m: raise ValueError("Model has no transitions and no starting points.")
             # Attempt recovery if needed (although train should handle this)
             valid_contexts = [ctx for ctx, successors in self.m.items() if successors]
             self.s = random.sample(valid_contexts, k=min(len(valid_contexts), 100)) if valid_contexts else []
             if not self.s: raise ValueError("Cannot start generation: No valid starting contexts found in model.")


        context = None
        result = []
        is_start_word_flag = True # Track if the next word starts a sequence

        # --- Seed Processing ---
        if seed:
            seed_words = seed.lower().split()
            if len(seed_words) >= self.n:
                potential_context = tuple(seed_words[-self.n:])
                # Check if seed context is valid *and* has successors
                if potential_context in self.m and self.m[potential_context]:
                    context = potential_context
                    result = seed_words
                    is_start_word_flag = False
                else:
                    result = seed_words # Keep seed words anyway
            else: # Seed too short
                 result = [] # Start fresh

        # --- Random Start (if no valid seed) ---
        if context is None:
            if not self.s: raise ValueError("Cannot start generation: No starting contexts available.")
            context = random.choice(self.s) # self.s should only contain valid contexts now
            result.extend(list(context))
            is_start_word_flag = True # This context choice is the start

        # --- Generation loop ---
        generated_metrics = []
        retry_count = 0
        max_retries = 10 # Lowered for faster debug feedback
        max_filter_attempts = 5
        generated_word_count = 0 # Words generated *in this loop*


        while generated_word_count < count:

            # --- Check context validity and get candidates ---
            # Note: Context validity (exists and has successors) should be ensured by start/reset logic
            if context not in self.m or not self.m[context]:
                 # This block should ideally not be hit if start/reset logic is correct, but acts as a safety net
                 retry_count += 1
                 if retry_count >= max_retries: print(f"DEBUG: Max retries on unexpected invalid context. Breaking."); break
                 # Attempt reset
                 valid_starts = [s for s in self.s if s in self.m and self.m[s]] # Re-check validity
                 context = random.choice(valid_starts)
                 is_start_word_flag = True
                 continue # Retry with new context

            # Get candidates using the current valid context
            current_options = self.m[context]
            candidate_words, candidate_weights = self._symbolic_probability(context, current_options)


            # --- Check if candidates are valid ---
            if not candidate_words or not candidate_weights or len(candidate_words) != len(candidate_weights) or sum(candidate_weights) < 1e-9:
                retry_count += 1
                print(f"DEBUG: No valid/weighted candidates from symbolic_probability for {context}. Retry {retry_count}/{max_retries}")
                # Reset context to another valid starting point
                valid_starts = [s for s in self.s if s in self.m and self.m[s]]
                if not valid_starts: print("DEBUG: No valid starts left for reset. Breaking."); break
                context = random.choice(valid_starts)
                is_start_word_flag = True
                continue # Retry with new context

            # --- Filtering and Selection ---
            next_word = None
            chosen_prob = 0.0
            filter_rejections = 0
            window_words = result[-min(len(result), window_size):]

            # Apply filter or choose directly
            if word_filter is not None:
                available_words = candidate_words.copy(); available_weights = candidate_weights.copy()
                found_valid_word = False; filter_attempts = 0
                # ... (Filtering loop - keep as before, maybe add internal prints if needed) ...
                while available_words and filter_attempts < max_filter_attempts:
                    current_total_weight = sum(available_weights)
                    if current_total_weight <= 1e-9: break
                    normalized_weights = [w / current_total_weight for w in available_weights]
                    try:
                        choices = random.choices(range(len(available_words)), weights=normalized_weights, k=1)
                        if not choices: break
                        chosen_idx_in_available = choices[0]
                        candidate_word = available_words[chosen_idx_in_available]
                        if word_filter(candidate_word, window_words):
                            next_word = candidate_word
                            original_idx = candidate_words.index(candidate_word) # Find original index
                            chosen_prob = candidate_weights[original_idx] # Use original weight
                            found_valid_word = True; break
                        else:
                            filter_rejections += 1
                            available_words.pop(chosen_idx_in_available); available_weights.pop(chosen_idx_in_available)
                            filter_attempts += 1
                    except ValueError as e: print(f"DEBUG: Error during filtered choice: {e}"); break
                    except IndexError: print(f"DEBUG: Index error during filtering, likely weight/word mismatch."); break


                if not found_valid_word:
                    filter_rejections = filter_attempts # Record attempts made
                    try: # Fallback: Choose from original list
                        if not candidate_words: raise ValueError("No candidates for fallback")
                        chosen_indices = random.choices(range(len(candidate_words)), weights=candidate_weights, k=1)
                        if not chosen_indices: raise ValueError("Fallback choice returned empty")
                        chosen_idx = chosen_indices[0]
                        next_word = candidate_words[chosen_idx]; chosen_prob = candidate_weights[chosen_idx]
                    except Exception as e:
                        print(f"DEBUG: Error during fallback choice: {e}. Trying uniform.")
                        if candidate_words: next_word = random.choice(candidate_words); chosen_prob = 1.0 / len(candidate_words)
                        else: print(f"DEBUG: No candidates for uniform fallback."); next_word = None # Will trigger reset

            else: # No filter
                print("DEBUG: No filter.")
                try:
                    if not candidate_words: raise ValueError("No candidates for choice")
                    chosen_indices = random.choices(range(len(candidate_words)), weights=candidate_weights, k=1)
                    if not chosen_indices: raise ValueError("Unfiltered choice returned empty")
                    chosen_idx = chosen_indices[0]
                    next_word = candidate_words[chosen_idx]; chosen_prob = candidate_weights[chosen_idx]
                except Exception as e:
                    print(f"DEBUG: Error during unfiltered choice: {e}. Trying uniform.")
                    if candidate_words: next_word = random.choice(candidate_words); chosen_prob = 1.0 / len(candidate_words)
                    else: print(f"DEBUG: No candidates for uniform choice."); next_word = None # Will trigger reset


            # --- Check selection outcome ---
            if next_word is None:
                print(f"DEBUG: Failed to select next_word in this iteration. Resetting context.")
                retry_count += 1 # Count as a retry even if it wasn't candidate finding
                if retry_count >= max_retries: print(f"DEBUG: Max retries overall. Breaking."); break
                 # Reset context
                valid_starts = [s for s in self.s if s in self.m and self.m[s]]
                if not valid_starts: print("DEBUG: No valid starts left for reset. Breaking."); break
                context = random.choice(valid_starts)
                print(f"DEBUG: Reset context to {context} (due to selection failure)")
                is_start_word_flag = True
                continue # Try again with new context


            # --- Selection successful: Calculate metrics and update ---
            step_metrics = self._calculate_step_metrics(
                 chosen_word=next_word, context=context,
                 candidate_words=candidate_words, candidate_weights=candidate_weights,
                 chosen_prob=chosen_prob, filter_rejects=filter_rejections,
                 is_start=is_start_word_flag
            )
            generated_metrics.append(step_metrics)
            # print(f"DEBUG: Metrics: {[f'{m:.2f}' for m in step_metrics]}")

            result.append(next_word)
            context = tuple(result[-self.n:])
            generated_word_count += 1
            is_start_word_flag = False # Next word is not a start word
            retry_count = 0 # Reset retry count on success

        # --- Finalize ---
        metrics_array = np.array(generated_metrics)

        # Return all words generated including seed/start context for inspection
        final_text = ' '.join(result)
        return final_text, metrics_array


# --- Example Usage (`if __name__ == "__main__":`) ---
def no_repetition_filter(word, window_words):
    return word not in window_words

if __name__ == "__main__":
    CONFIG = { # Defaults
        'input_filename': "test.txt", 'ngram_size': 2,
        'words_to_generate': 250, 'window_size': 10
    }
    print(f"--- Symbolic Markov Generator ---")
    try: # File/Input handling
        fname = input(f"Input filename (def: {CONFIG['input_filename']}): ") or CONFIG['input_filename']
        CONFIG['input_filename'] = fname
        with open(fname, 'r', encoding='utf-8') as f: txt = ' '.join(f.read().lower().split())
        if not txt: print("Error: Input file empty."); sys.exit(1)
    except FileNotFoundError:
        print(f"Error: File '{CONFIG['input_filename']}' not found. Using sample text.")
        txt = ("this is a sample text it is short repeat repeat short text sample this") # Very simple text
    except Exception as e: print(f"Error reading file: {e}"); sys.exit(1)
    try: # Ngram size
        n_in = input(f"N-gram size (def: {CONFIG['ngram_size']}): ")
        CONFIG['ngram_size'] = int(n_in) if n_in else CONFIG['ngram_size']
    except ValueError: print("Invalid N, using default.")
    try: # Word count
         w_in = input(f"Words to generate (def: {CONFIG['words_to_generate']}): ")
         CONFIG['words_to_generate'] = int(w_in) if w_in else CONFIG['words_to_generate']
    except ValueError: print("Invalid count, using default.")

    # --- Train ---
    model = None
    try:
        model = SymbolicMarkov(CONFIG['ngram_size'])
        model.train(txt)
    except Exception as e: print(f"Training Error: {e}"); import traceback; traceback.print_exc(); sys.exit(1)

    # --- Generate ---
    if not model or not model.m or not model.s: # Check if model is usable
        print("\nModel training failed or resulted in no usable data.")
    else:
        print(f"\n--- Ready to generate: n={model.n}, target={CONFIG['words_to_generate']} words ---")
        print(f"Enter seed (>= {model.n} words) or press Enter. Type 'exit' to quit.")
        while True:
            try:
                user_input = input("\nSEED: ")
                if user_input.lower() in ['quit', 'exit']: break

                # --- Generate and get metrics ---
                generated_text, metrics_data = model.generate_and_get_metrics(
                    seed=user_input or None,
                    count=CONFIG['words_to_generate'],
                    window_size=CONFIG['window_size'],
                    word_filter=no_repetition_filter # Set to None to test without filter
                )

                # --- Display results ---
                print("\n--- Generated Text ---")
                if user_input: print(f"(Seed: '{user_input}')")
                print(generated_text)
                print("-" * 60)

            except KeyboardInterrupt: print("\nExiting."); break
            except ValueError as e: print(f"Generation Error: {e}")
            except Exception as e: print(f"Unexpected Error: {e}"); import traceback; traceback.print_exc()

    print("\n--- Run Complete ---")
