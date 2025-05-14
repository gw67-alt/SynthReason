import numpy as np
from collections import defaultdict, Counter
import random # For initial text_state in main

# 1. Define the Environment (Simplified) with Text N-gram Capability
class Environment:
    def __init__(self, state_size=3, text_ngram_size=2):
        self.state_size = state_size
        self.state = np.zeros(state_size)  # Initial state
        self.goal_state = np.array([0, 0, 0, 1])  # Example goal

        # Text N-gram related attributes
        self.text_ngram_size = text_ngram_size
        self.word_to_idx = {}
        self.idx_to_word = [] # List of words, index is ID
        self.vocabulary_set = set() # For quick "in" checks
        self.ngram_counts = defaultdict(Counter)  # Store n-gram counts: context_tuple -> Counter(next_word -> count)
        self.text_state = []  # Current text state (list of words)

    def reset(self):
        self.state = np.zeros(self.state_size)
        self.text_state = []
        return self.state.copy() # Return a copy to avoid external modification

    def step(self, action, text_action=None):
        # Original numeric state transition
        self.state = self.state + action # Assumes action has same shape as state
        reward = -np.sum(np.abs(self.state - self.goal_state)) # L1 distance
        done = np.all(np.isclose(self.state, self.goal_state)) # Use isclose for float comparison

        text_reward_component = 0 # Renamed to avoid conflict with reward variable
        if text_action is not None:
            self.text_state.append(text_action)
            # Dynamically update vocabulary if a new word appears (e.g. from a generative agent)
            # However, current agent samples from known vocab, so this path is less likely for text_action itself
            if text_action not in self.vocabulary_set:
                self.vocabulary_set.add(text_action)
                if text_action not in self.word_to_idx: # Should only happen if vocab is truly dynamic
                    self.idx_to_word.append(text_action)
                    self.word_to_idx[text_action] = len(self.idx_to_word) - 1
                    # Agent's policy matrix would need to be updated if this happens mid-training

            if len(self.text_state) >= self.text_ngram_size:
                ngram = tuple(self.text_state[-self.text_ngram_size:])
                context = ngram[:-1]
                next_word = ngram[-1]
                self.ngram_counts[context][next_word] += 1 # Update counts based on this step

                current_context_counts = self.ngram_counts[context]
                total_for_context = sum(current_context_counts.values())
                if total_for_context > 0:
                    prob = current_context_counts[next_word] / total_for_context
                    text_reward_component = prob # Reward based on empirical frequency

        combined_reward = reward + text_reward_component # Simple additive combination
        return self.state.copy(), combined_reward, done, {"text_state": self.text_state.copy()}

    def get_ngram_probabilities(self, context_tuple):
        """Get probability distribution for next word given context_tuple (tuple of (n-1) words)."""
        # Check if the exact context has been seen and has continuations
        if context_tuple is None or context_tuple not in self.ngram_counts or not self.ngram_counts[context_tuple]:
            # Fallback: uniform distribution over the whole vocabulary
            if not self.idx_to_word: # No vocabulary at all
                return {} # No words to choose from
            num_vocab = len(self.idx_to_word)
            # Ensure num_vocab is not zero before division
            return {word: 1.0 / num_vocab for word in self.idx_to_word} if num_vocab > 0 else {}

        counts = self.ngram_counts[context_tuple]
        total = sum(counts.values())
        
        # This case should be rare if context_tuple is in self.ngram_counts and has entries.
        if total == 0: 
             if not self.idx_to_word: return {}
             num_vocab = len(self.idx_to_word)
             return {word: 1.0 / num_vocab for word in self.idx_to_word} if num_vocab > 0 else {}
        
        return {word: count / total for word, count in counts.items()}

    def train_on_text_corpus(self, corpus_text):
        """Build vocabulary and n-gram model from a text corpus (single string)."""
        words = corpus_text.lower().split() # Simple tokenization

        # Build vocabulary (word_to_idx, idx_to_word, vocabulary_set)
        self.vocabulary_set = set(words)
        self.idx_to_word = sorted(list(self.vocabulary_set)) # Sort for consistent ID assignment
        self.word_to_idx = {word: i for i, word in enumerate(self.idx_to_word)}

        # Process the corpus to build n-gram counts
        self.ngram_counts.clear() # Clear previous counts
        for i in range(len(words) - self.text_ngram_size + 1):
            # N-gram is a tuple of words
            ngram_words = tuple(words[i : i + self.text_ngram_size])
            context = ngram_words[:-1] # Context is the first (n-1) words of the n-gram
            next_word = ngram_words[-1]  # The word to predict
            self.ngram_counts[context][next_word] += 1
        
        print(f"Environment vocabulary size after corpus training: {len(self.idx_to_word)}")


# 2. Define the Agent with Text Capability
class Agent:
    def __init__(self, state_size, action_size, word_to_idx_map, idx_to_word_list, embedding_size=5):
        self.state_size = state_size
        self.action_size = action_size # Should match state_size for current env.step
        
        # Vocabulary information from environment (or pre-defined)
        self.word_to_idx = word_to_idx_map
        self.idx_to_word = idx_to_word_list
        self.vocab_size = len(self.idx_to_word)
        self.embedding_size = embedding_size

        # Matrix for numeric actions policy
        self.P_numeric_policy = np.random.randn(action_size, state_size) * 0.01

        # Text policy embeddings (one embedding vector per word in vocabulary)
        if self.vocab_size > 0:
            self.text_policy_embeddings = np.random.randn(self.vocab_size, self.embedding_size) * 0.01
        else:
            self.text_policy_embeddings = None

        self.learning_rate = 0.01

    def get_action(self, state):
        """Get numeric action based on current numeric state."""
        # Linear policy: action = P @ state
        action = np.dot(self.P_numeric_policy, state)
        return action

    def get_text_action(self, current_numeric_state, current_text_state_words, env_ngram_probs, temperature=1.0):
        """
        Generate a text action (word). 
        Currently, this relies purely on env_ngram_probs from the environment's n-gram model.
        The agent's own text_policy_embeddings are not used for *selecting* the action here.
        """
        if not env_ngram_probs:  # Handles empty dict from get_ngram_probabilities
            return None

        words = list(env_ngram_probs.keys())
        probs_values = np.array(list(env_ngram_probs.values()))

        if not words or probs_values.sum() == 0: # No words or sum of probabilities is zero
            # Fallback: if vocabulary exists, pick a random word uniformly
            return random.choice(self.idx_to_word) if self.idx_to_word else None

        # Normalize probabilities if they don't sum to 1 (e.g. due to float precision)
        if not np.isclose(probs_values.sum(), 1.0):
            probs_values = probs_values / probs_values.sum()
            # Handle case where sum became zero after normalization (e.g. all very small numbers)
            if not np.isclose(probs_values.sum(), 1.0):
                 return random.choice(self.idx_to_word) if self.idx_to_word else None


        # Apply temperature scaling
        if temperature == 0: # Greedy selection (highest probability)
             return words[np.argmax(probs_values)]
        elif temperature != 1.0: # Avoid unnecessary computation for T=1
            # Ensure probs_values are non-negative before power
            probs_scaled = np.power(np.maximum(probs_values, 0), 1.0 / temperature)
            probs_sum = np.sum(probs_scaled)
            if probs_sum == 0 : # If all probabilities became zero after scaling
                return random.choice(words) # Fallback to uniform choice among candidates
            probs_values = probs_scaled / probs_sum # Renormalize

        try:
            # Ensure no NaN or inf in probs_values
            if np.any(np.isnan(probs_values)) or np.any(np.isinf(probs_values)):
                 return random.choice(words) # Fallback
            return np.random.choice(words, p=probs_values)
        except ValueError: # Catch errors from np.random.choice (e.g., if probs still don't sum to 1)
            return random.choice(words) # Fallback to uniform random choice from available words


    def update_matrices(self, grad_P_numeric, grad_text_policy_embeddings=None):
        """Update agent's policy matrices using calculated gradients."""
        self.P_numeric_policy += self.learning_rate * grad_P_numeric
        
        if grad_text_policy_embeddings is not None and self.text_policy_embeddings is not None:
            self.text_policy_embeddings += self.learning_rate * grad_text_policy_embeddings


# 3. Reward Function is implicitly defined within Environment.step()

# 4. Define the Objective Function and Calculate Gradients (Simplified Heuristics)
def calculate_objective_and_gradients(agent, environment, episodes=1):
    total_reward_accumulator = 0
    grad_P_numeric_total = np.zeros_like(agent.P_numeric_policy)
    
    if agent.text_policy_embeddings is not None:
        grad_text_policy_embeddings_total = np.zeros_like(agent.text_policy_embeddings)
    else:
        grad_text_policy_embeddings_total = None

    for _ in range(episodes):
        current_numeric_env_state = environment.reset() # Resets numeric and text state in env
        done = False
        episode_reward_sum = 0 # Renamed
        
        # Store trajectory data
        trajectory_numeric_states = []
        trajectory_numeric_actions = []
        trajectory_chosen_text_actions = [] # Store the chosen word (string) or None
        trajectory_rewards = []

        max_steps_per_episode = 50 # Safety break for non-terminating episodes

        for step_count in range(max_steps_per_episode): # Renamed loop variable
            # Get numeric action
            numeric_action_taken = agent.get_action(current_numeric_env_state)
            
            # Get text action
            chosen_text_action_word = None
            context_tuple_for_ngram = None
            # Determine context for n-gram: (n-1) previous words
            if environment.text_ngram_size > 1:
                if len(environment.text_state) >= environment.text_ngram_size - 1:
                    context_tuple_for_ngram = tuple(environment.text_state[-(environment.text_ngram_size - 1):])
            elif environment.text_ngram_size == 1: # Unigram case, context is empty
                 context_tuple_for_ngram = tuple()
            # If not enough words for context, context_tuple_for_ngram remains None
            # environment.get_ngram_probabilities handles None context by returning uniform distribution.

            ngram_probabilities = environment.get_ngram_probabilities(context_tuple_for_ngram)
            chosen_text_action_word = agent.get_text_action(
                current_numeric_env_state, 
                environment.text_state, # Current sequence of words in env
                ngram_probabilities
            )

            # Environment step
            next_numeric_env_state, reward_value, done, info = environment.step(numeric_action_taken, chosen_text_action_word)
            episode_reward_sum += reward_value

            # Store data for this step
            trajectory_numeric_states.append(current_numeric_env_state.copy())
            trajectory_numeric_actions.append(numeric_action_taken.copy())
            trajectory_chosen_text_actions.append(chosen_text_action_word) # This can be None
            trajectory_rewards.append(reward_value)

            current_numeric_env_state = next_numeric_env_state # Move to next state
            if done:
                break
        
        total_reward_accumulator += episode_reward_sum

        # Calculate gradients heuristically based on the trajectory
        for t in range(len(trajectory_numeric_states)):
            # Gradient for numeric policy P_numeric_policy
            # Heuristic: encourage actions (scaled by state) that led to higher reward
            grad_P_contrib = np.outer(trajectory_numeric_actions[t], trajectory_numeric_states[t])
            grad_P_numeric_total += grad_P_contrib * trajectory_rewards[t]

            # Gradient for text_policy_embeddings
            if grad_text_policy_embeddings_total is not None and trajectory_chosen_text_actions[t] is not None:
                word_str = trajectory_chosen_text_actions[t]
                if word_str in agent.word_to_idx: # Check if word is in agent's known vocabulary
                    word_idx = agent.word_to_idx[word_str]
                    # Heuristic: adjust the embedding of the chosen word based on reward
                    # This adds the scaled reward to each component of the word's embedding vector
                    grad_text_policy_embeddings_total[word_idx, :] += trajectory_rewards[t] * 0.01 # Small factor
    
    num_actual_episodes = episodes
    avg_episode_reward = total_reward_accumulator / num_actual_episodes if num_actual_episodes > 0 else 0
    
    # Normalize gradients by number of episodes
    final_grad_P_numeric = grad_P_numeric_total / num_actual_episodes if num_actual_episodes > 0 else np.zeros_like(agent.P_numeric_policy)
    
    final_grad_text_policy_embeddings = None
    if grad_text_policy_embeddings_total is not None:
        final_grad_text_policy_embeddings = grad_text_policy_embeddings_total / num_actual_episodes if num_actual_episodes > 0 else np.zeros_like(agent.text_policy_embeddings)
        
    return avg_episode_reward, final_grad_P_numeric, final_grad_text_policy_embeddings


# 5. Main Training Loop
def main():
    state_size = 4
    action_size = 4 # Must match state_size for current env.step: self.state = self.state + action
    text_ngram_size = 2  # Using bigrams (context of 1 word)

    # Initialize Environment and train its n-gram model on a corpus
    environment = Environment(state_size, text_ngram_size)
    with open("test.txt", 'r', encoding='utf-8') as f:
        # Read, lower, split into words, join back with single spaces to normalize whitespace
        sample_corpus = ' '.join(f.read().lower().split()) 
    environment.train_on_text_corpus(sample_corpus)

    # Initialize Agent, providing it with vocabulary info from the environment
    agent = Agent(state_size, action_size, 
                  word_to_idx_map=environment.word_to_idx, 
                  idx_to_word_list=environment.idx_to_word,
                  embedding_size=100) # Example embedding size

    episodes_per_iteration = 50 # Number of episodes for averaging gradients
    optimization_steps = 25   # Number of times to update agent's matrices

    print(f"Agent initialized. Vocabulary size: {agent.vocab_size}.")
    if agent.vocab_size == 0:
        print("Warning: Agent vocabulary is empty. Text features may not work.")
        # return # Optionally exit if no vocab
    while True:
        initial_seed_words = input("USER: ").split()
        seed_words = []

        for i in range(optimization_steps):
            avg_reward, grad_P, grad_text_embed = calculate_objective_and_gradients(agent, environment, episodes_per_iteration)
            
            if i % 10 == 0 : # Print status more frequently
                print(f"Optimization Step {i}, Average Reward: {avg_reward:.4f}")
            
            agent.update_matrices(grad_P, grad_text_embed)

        print(f"\n--- Testing Text Generation at Step {i} ---")
        
        test_env_for_generation = Environment(state_size, text_ngram_size)
        # Crucially, give the test environment the n-gram model and vocabulary from the main trained environment
        test_env_for_generation.ngram_counts = environment.ngram_counts
        test_env_for_generation.word_to_idx = environment.word_to_idx
        test_env_for_generation.idx_to_word = environment.idx_to_word
        test_env_for_generation.vocabulary_set = environment.vocabulary_set
        
        
        current_test_numeric_state = test_env_for_generation.reset()
        
        # Seed the text generation
        for word in initial_seed_words: # If vocabulary exists
            if word in agent.word_to_idx:
                seed_words.append(word)
            else:
                seed_words.append(random.choice(agent.idx_to_word))
        
        test_env_for_generation.text_state = seed_words.copy()
        generated_text_sequence = seed_words.copy()
        
        for _ in range(250):  # Generate a sequence of 10 additional words
            numeric_action_for_test = agent.get_action(current_test_numeric_state)
            
            generated_word_for_step = None
            context_for_test_gen = None
            # Determine context based on n-gram size and current text_state in test_env
            if text_ngram_size > 1:
                if len(test_env_for_generation.text_state) >= text_ngram_size - 1:
                    context_for_test_gen = tuple(test_env_for_generation.text_state[-(text_ngram_size - 1):])
            elif text_ngram_size == 1: # Unigram
                context_for_test_gen = tuple()
            
            # Get n-gram probabilities from the (primed) test_env
            ngram_probs_for_test = test_env_for_generation.get_ngram_probabilities(context_for_test_gen)
            
            if ngram_probs_for_test:
                generated_word_for_step = agent.get_text_action(
                    current_test_numeric_state, 
                    test_env_for_generation.text_state, 
                    ngram_probs_for_test, 
                    temperature=0.7 # Use a temperature for some randomness
                )
            
            # Step the test environment
            current_test_numeric_state, _, _, _ = test_env_for_generation.step(numeric_action_for_test, generated_word_for_step)
            # test_env_for_generation.text_state is updated internally by its step method
        
            if generated_word_for_step:
                generated_text_sequence.append(generated_word_for_step)
            else: # Stop generation if no word could be chosen
                break 
        
        print(f"Generated Text Sample: {' '.join(generated_text_sequence)}")
        print("-------------------------------------------")
        
                
if __name__ == "__main__":
    main()
