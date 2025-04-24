import numpy as np
import matplotlib.pyplot as plt
import scipy.signal as signal
import sys
import random
out_limit = 42
KB_limit = 999
try:
    import librosa
    import librosa.display
except ImportError:
    print("Error: librosa library not found.")
    print("Please install it using: pip install librosa")
    sys.exit(1)

try:
    from sklearn.cluster import KMeans
    from sklearn.preprocessing import StandardScaler
    from sklearn.decomposition import PCA
except ImportError:
    print("Error: scikit-learn library not found.")
    print("Please install it using: pip install scikit-learn")
    sys.exit(1)

# --- Provided LPC Functions ---
def compute_lpc_coefficients(signal_data, order=16):
    """
    Compute LPC coefficients using librosa's implementation.
    Returns coefficients a[1] through a[p].
    """
    if len(signal_data) <= order:
        print(f"Warning: Signal length ({len(signal_data)}) is too short for LPC order ({order}). Returning zeros.")
        return np.zeros(order)

    # Normalize signal amplitude (optional but often helpful)
    # signal_data = signal_data / np.max(np.abs(signal_data) + 1e-9)

    # Compute LPC coefficients using librosa
    try:
        # librosa.lpc returns the coefficients including a_0.
        lpc_coeffs_full = librosa.lpc(signal_data.astype(np.float32), order=order)
        # Return only a_1 to a_order.
        return lpc_coeffs_full[1:]
    except Exception as e:
        # Common errors include signal being too short or containing NaNs/Infs
        print(f"Error computing LPC: {e}")
        print("Signal stats: len={}, min={:.2f}, max={:.2f}, mean={:.2f}".format(
            len(signal_data), np.min(signal_data), np.max(signal_data), np.mean(signal_data)
        ))
        # Check for NaNs or Infs which can cause errors
        if np.any(np.isnan(signal_data)) or np.any(np.isinf(signal_data)):
            print("Signal contains NaN or Inf values.")
        return np.zeros(order) # Return zeros or handle error appropriately

def visualize_word_lpc(word, lpc_coeffs, sample_rate, lpc_order):
    """
    Visualize the magnitude spectrum derived from LPC coefficients for a word.
    """
    if lpc_coeffs is None or len(lpc_coeffs) != lpc_order:
        print("Cannot visualize: Invalid LPC coefficients provided.")
        return

    plt.figure(figsize=(10, 6))

    # Construct the full denominator polynomial A(z) coefficients [1, a_1, ..., a_p]
    a_coeffs = np.concatenate(([1], lpc_coeffs))

    # Compute frequency response of the LPC filter H(z) = 1 / A(z)
    w, h = signal.freqz(1, a_coeffs, worN=4096, fs=sample_rate) # Use fs for Hz axis

    # Compute magnitude spectrum in dB
    magnitudes_db = 20 * np.log10(np.abs(h) + 1e-9) # Add epsilon for stability

    # Plot
    plt.plot(w, magnitudes_db)
    plt.title(f"LPC Spectrum for '{word}' (Order {lpc_order})")
    plt.xlabel('Frequency [Hz]')
    plt.ylabel('Magnitude [dB]')
    plt.grid(True)
    plt.ylim(bottom=np.percentile(magnitudes_db, 1) - 10 if len(magnitudes_db)>0 else -80,
            top=np.percentile(magnitudes_db, 99) + 10 if len(magnitudes_db)>0 else 40) # Dynamic Y limits
    plt.tight_layout()
    plt.show()

# --- Signal Generation Functions (Unchanged) ---
def generate_signal(letter):
    sample_rate = 4410
    duration = 0.5
    t = np.linspace(0, duration, int(sample_rate * duration), False)
    frequencies = {
        'A': 220, 'B': 240, 'C': 260, 'D': 280, 'E': 300,
        'F': 320, 'G': 340, 'H': 360, 'I': 380, 'J': 400,
        'K': 420, 'L': 440, 'M': 460, 'N': 480, 'O': 500,
        'P': 520, 'Q': 540, 'R': 560, 'S': 580, 'T': 600,
        'U': 620, 'V': 640, 'W': 660, 'X': 680, 'Y': 700, 'Z': 720
    }
    freq = frequencies.get(letter.upper(), 0)
    if freq == 0:
        signal_data = np.zeros(int(sample_rate * duration))
    else:
        signal_data = np.sin(2 * np.pi * freq * t)
        signal_data += 0.5 * np.sin(2 * np.pi * (freq * 2) * t)
        envelope = np.hanning(len(signal_data))
        signal_data = signal_data * envelope
    return signal_data, sample_rate

def generate_word_signal(word):
    word_signal_parts = []
    sample_rate = 4410
    for letter in word:
        if 'A' <= letter.upper() <= 'Z':
            signal_part, sr = generate_signal(letter.upper())
            word_signal_parts.append(signal_part)
            sample_rate = sr
    if not word_signal_parts:
        return np.array([]), sample_rate
    full_word_signal = np.concatenate(word_signal_parts)
    return full_word_signal, sample_rate

# --- ENHANCED: Feature Extraction Function for 8 Groups ---
def extract_signal_features(signal_data, sample_rate):
    if signal_data is None or signal_data.size == 0:
        return None
    signal_data = signal_data / (np.max(np.abs(signal_data)) + 1e-10)
    peaks_high, _ = signal.find_peaks(np.abs(signal_data), height=0.5)
    peaks_med, _ = signal.find_peaks(np.abs(signal_data), height=0.3)
    peaks_low, _ = signal.find_peaks(np.abs(signal_data), height=0.1)
    peak_count_high = len(peaks_high)
    peak_count_med = len(peaks_med)
    peak_count_low = len(peaks_low)
    signal_duration = len(signal_data) / sample_rate
    peak_density = peak_count_med / signal_duration if signal_duration > 0 else 0
    if peak_count_med > 0:
        widths_50 = signal.peak_widths(np.abs(signal_data), peaks_med, rel_height=0.5)[0]
        widths_25 = signal.peak_widths(np.abs(signal_data), peaks_med, rel_height=0.75)[0]
        widths_75 = signal.peak_widths(np.abs(signal_data), peaks_med, rel_height=0.25)[0]
        mean_width = np.mean(widths_50)
        width_std = np.std(widths_50) if len(widths_50) > 1 else 0
        width_ratio = np.mean(widths_25 / widths_75) if len(widths_75) > 0 and np.all(widths_75 > 0) else 1.0
    else:
        mean_width = 0
        width_std = 0
        width_ratio = 1.0
    try:
        spectral_centroid = librosa.feature.spectral_centroid(y=signal_data, sr=sample_rate)[0].mean()
        spectral_bandwidth = librosa.feature.spectral_bandwidth(y=signal_data, sr=sample_rate)[0].mean()
        spectral_flatness = librosa.feature.spectral_flatness(y=signal_data)[0].mean()
        spectral_rolloff = librosa.feature.spectral_rolloff(y=signal_data, sr=sample_rate)[0].mean()
    except Exception:
        spectral_centroid = 0
        spectral_bandwidth = 0
        spectral_flatness = 0
        spectral_rolloff = 0
    zero_crossings = np.sum(np.abs(np.diff(np.signbit(signal_data).astype(int))))
    zero_crossing_rate = zero_crossings / len(signal_data)
    rms = np.sqrt(np.mean(signal_data**2))
    crest_factor = np.max(np.abs(signal_data)) / (rms + 1e-10)
    envelope = np.abs(signal.hilbert(signal_data))
    env_mean = np.mean(envelope)
    env_std = np.std(envelope)
    env_max = np.max(envelope)
    env_roughness = np.std(np.diff(envelope)) * 100
    try:
        harmonic, percussive = librosa.effects.hpss(signal_data)
        harmonic_ratio = np.sum(harmonic**2) / (np.sum(signal_data**2) + 1e-10)
        percussive_ratio = np.sum(percussive**2) / (np.sum(signal_data**2) + 1e-10)
    except Exception:
        harmonic_ratio = 0.5
        percussive_ratio = 0.5
    features = np.array([
        peak_count_high,
        peak_count_med / max(1, peak_count_low),
        peak_density,
        mean_width,
        width_std,
        width_ratio,
        spectral_centroid,
        spectral_bandwidth,
        spectral_flatness,
        spectral_rolloff,
        zero_crossing_rate,
        rms,
        crest_factor,
        env_mean,
        env_std / (env_mean + 1e-10),
        env_roughness,
        harmonic_ratio,
        percussive_ratio
    ])
    return features

def compute_mfcc_features(signal_data, sample_rate, n_mfcc=13, n_fft=512, hop_length=256):
    if signal_data is None or signal_data.size == 0:
        return None
    signal_data = signal_data.astype(np.float32)
    if np.any(np.isnan(signal_data)) or np.any(np.isinf(signal_data)):
        print(f"Warning: Signal contains NaN or Inf values before MFCC. Length={len(signal_data)}. Returning None.")
        return None
    try:
        mfccs = librosa.feature.mfcc(y=signal_data, sr=sample_rate, n_mfcc=n_mfcc,
                                        n_fft=n_fft, hop_length=hop_length)
        if not np.all(np.isfinite(mfccs)):
            print(f"Warning: MFCC computation resulted in non-finite values. Signal length={len(signal_data)}. Returning None.")
            return None
        mean_mfccs = np.mean(mfccs, axis=1)
        return mean_mfccs
    except Exception as e:
        print(f"Error computing MFCC: {e}")
        print("Signal stats: len={}, min={:.2f}, max={:.2f}, mean={:.2f}".format(
            len(signal_data), np.min(signal_data), np.max(signal_data), np.mean(signal_data)
        ))
        return None

# --- Updated Categorization Logic for 8 Groups ---
def categorize_words(words_to_process, num_clusters=8, use_comprehensive=True, compute_lpc=False, lpc_order=16):
    audio_features = {}
    word_list_for_clustering = []
    audio_signals = {}

    print(f"Processing {len(words_to_process)} words using {'comprehensive' if use_comprehensive else 'MFCC'} features...")
    if compute_lpc:
        print(f"Also computing LPC coefficients (order {lpc_order})...")

    processed_count = 0
    for word in words_to_process:
        word_signal, sample_rate = generate_word_signal(word)
        lpc_coefficients = None
        if compute_lpc:
            lpc_coefficients = compute_lpc_coefficients(word_signal, order=lpc_order)
        audio_signals[word] = (word_signal, sample_rate, lpc_coefficients)

        if word_signal.size == 0:
            continue

        # Compute features based on selected method
        if use_comprehensive:
            features = extract_signal_features(word_signal, sample_rate)
        else:
            features = compute_mfcc_features(word_signal, sample_rate)

        # Check if features were computed successfully
        if features is not None and not np.any(np.isnan(features)) and not np.any(np.isinf(features)):
            audio_features[word] = features
            word_list_for_clustering.append(word)

        processed_count += 1
        if (processed_count % 100 == 0):
            print(f"  Processed {processed_count}/{len(words_to_process)} words...")

    if not audio_features:
        print("No valid audio features could be computed for any word. Cannot categorize.")
        return {}, {}, None, None, {}

    print(f"\nSuccessfully computed features for {len(audio_features)} words.")

    # --- Prepare data for clustering ---
    feature_matrix = np.array([audio_features[word] for word in word_list_for_clustering])

    # Scale features
    print("Scaling features...")
    scaler = StandardScaler()
    scaled_features = scaler.fit_transform(feature_matrix)

    # --- Perform Clustering into 8 groups ---
    actual_num_clusters = min(num_clusters, len(word_list_for_clustering))
    if actual_num_clusters < 2:
        print(f"Only {len(word_list_for_clustering)} words with features, cannot form multiple clusters.")
        return {0: word_list_for_clustering} if len(word_list_for_clustering) == 1 else {}, {}, scaled_features, np.zeros(len(word_list_for_clustering)) if len(word_list_for_clustering) == 1 else None, audio_signals

    print(f"\nClustering features into {actual_num_clusters} groups using K-Means...")
    kmeans = KMeans(n_clusters=actual_num_clusters, random_state=42, n_init=10)
    kmeans.fit(scaled_features)

    # Get cluster assignments
    labels = kmeans.labels_

    # --- Group words based on clusters ---
    word_groups = {}
    for i, word in enumerate(word_list_for_clustering):
        cluster_label = labels[i]
        if cluster_label not in word_groups:
            word_groups[cluster_label] = []
        word_groups[cluster_label].append(word)

    # --- Analyze key characteristics of each cluster ---
    cluster_characteristics = {}
    if use_comprehensive:
        centers = kmeans.cluster_centers_
        centers_original = scaler.inverse_transform(centers)
        for cluster_id in range(len(centers_original)):
            center = centers_original[cluster_id]
            peak_width = center[3]
            peak_count = center[1]
            spectral_flatness = center[8]
            harmonic_ratio = center[16]
            env_roughness = center[15]
            if peak_width < np.percentile(centers_original[:, 3], 25):
                width_category = "very thin peaks"
            elif peak_width < np.percentile(centers_original[:, 3], 50):
                width_category = "thin peaks"
            elif peak_width < np.percentile(centers_original[:, 3], 75):
                width_category = "thick peaks"
            else:
                width_category = "very thick peaks"
            if harmonic_ratio > np.percentile(centers_original[:, 16], 75):
                harmonic_category = "highly harmonic"
            elif harmonic_ratio > np.percentile(centers_original[:, 16], 50):
                harmonic_category = "moderately harmonic"
            else:
                harmonic_category = "less harmonic"
            characteristic = f"{width_category}, {harmonic_category}"
            if peak_count > np.percentile(centers_original[:, 1], 75):
                characteristic += ", many peaks"
            elif peak_count < np.percentile(centers_original[:, 1], 25):
                characteristic += ", few peaks"
            if env_roughness > np.percentile(centers_original[:, 15], 75):
                characteristic += ", rough envelope"
            elif env_roughness < np.percentile(centers_original[:, 15], 25):
                characteristic += ", smooth envelope"
            cluster_characteristics[cluster_id] = characteristic

    return word_groups, cluster_characteristics, scaled_features, labels, audio_signals

# --- Dot Matrix Conversion and Visualization ---
def spectrum_to_dot_matrix(magnitudes_db, num_rows=20, num_cols=64):
    """Converts a magnitude spectrum to a dot matrix representation."""
    if not magnitudes_db.size:
        return np.zeros((num_rows, num_cols))

    min_mag = np.min(magnitudes_db)
    max_mag = np.max(magnitudes_db)
    matrix = np.zeros((num_rows, num_cols), dtype=int)

    # Discretize frequency (assuming magnitudes_db has num_cols points)
    freq_bins = np.linspace(0, len(magnitudes_db) - 1, num_cols, dtype=int)

    for col_idx, freq_bin in enumerate(freq_bins):
        mag_val = magnitudes_db[freq_bin]
        # Discretize magnitude
        row_idx = int(((mag_val - min_mag) / (max_mag - min_mag + 1e-9)) * (num_rows - 1))
        row_idx = np.clip(row_idx, 0, num_rows - 1)
        matrix[num_rows - 1 - row_idx, col_idx] = 1  # Flip row index for visual

    return matrix

def visualize_dot_matrix(matrix, word=""):
    """Visualizes the dot matrix."""
    plt.figure(figsize=(8, 4))
    plt.imshow(matrix, aspect='auto', cmap='binary', origin='upper')
    plt.title(f"Dot Matrix for '{word}'")
    plt.xlabel("Frequency Bins")
    plt.ylabel("Magnitude Levels")
    plt.xticks([])
    plt.yticks([])
    plt.tight_layout()
    plt.show()

def predict_word(new_word, stored_matrices, sample_rate, lpc_order, num_rows=20, num_cols=64):
    """Predicts similar words based on LPC spectrum dot matrix similarity."""
    new_signal, sr = generate_word_signal(new_word)
    new_lpc = compute_lpc_coefficients(new_signal, order=lpc_order)
    if new_lpc is not None:
        new_a_coeffs = np.concatenate(([1], new_lpc))
        w_new, h_new = signal.freqz(1, new_a_coeffs, worN=4096, fs=sr)
        new_magnitudes_db = 20 * np.log10(np.abs(h_new) + 1e-9)
        new_dot_matrix = spectrum_to_dot_matrix(new_magnitudes_db, num_rows, num_cols)

        similarities = {}
        for stored_word, stored_matrix in stored_matrices.items():
            # Example similarity metric: Negative of the sum of absolute differences (Hamming-like)
            similarity = -np.sum(np.abs(new_dot_matrix - stored_matrix))
            similarities[stored_word] = similarity

        sorted_predictions = sorted(similarities.items(), key=lambda item: item[1], reverse=True)
        return sorted_predictions, new_dot_matrix
    else:
        return [], None

# --- Main function ---
def main():
    input_word_file = "words_alpha.txt"
    filename = input("Enter filename or leave blank: ")
    if len(filename) > 0:
        with open(filename, 'r', encoding='utf-8') as file:
            all_words = file.read().upper().split()[:KB_limit]
    else:
        all_words = input("USER: ").upper().split()
    max_words_to_process = len(all_words) # Reduced for demonstration
    print(f"Found {len(all_words)} alphabetic words.")

    # Select words to process
    if max_words_to_process is not None and len(all_words) > max_words_to_process:
        print(f"Taking a random sample of {max_words_to_process} words for processing...")
        word_dictionary = random.sample(all_words, max_words_to_process)
    else:
        print("Processing all words found in the file.")
        word_dictionary = all_words

    # --- Parameters ---
    num_clusters = 8    # Classify into 8 groups
    output_filename = "word_categories_8_groups_with_lpc.txt"
    lpc_order = 16
    use_comprehensive = True
    compute_lpc = True
    num_rows_dot_matrix = 20
    num_cols_dot_matrix = 64

    # --- Perform categorization and LPC analysis ---
    result = categorize_words(word_dictionary, num_clusters=num_clusters, use_comprehensive=use_comprehensive, compute_lpc=compute_lpc, lpc_order=lpc_order)
    if use_comprehensive:
        categorized_groups, cluster_characteristics, scaled_features, labels, audio_signals = result
    else:
        categorized_groups, _, scaled_features, labels, audio_signals = result
        cluster_characteristics = {}

    # --- Output the results to Text File ---
    print(f"\nSaving categorized groups to '{output_filename}'...")
    try:
        with open(output_filename, 'w', encoding='utf-8') as f:
            f.write(f"--- Word Categories (k={num_clusters}) ---\n")
            f.write(f"Processed {len(word_dictionary)} words (sampled from {len(all_words)} total).\n")

            if not categorized_groups:
                f.write("\nNo categories were formed.\n")
            else:
                # Sort clusters by size (largest first)
                sorted_clusters = sorted(categorized_groups.items(),
                                                key=lambda x: len(x[1]), reverse=True)

                for cluster_id, words_in_group in sorted_clusters:
                    sorted_words = sorted(words_in_group)
                    characteristic = ""
                    if cluster_id in cluster_characteristics:
                        characteristic = f" - {cluster_characteristics[cluster_id]}"
                    f.write(f"\nGroup {cluster_id + 1}{characteristic} ({len(words_in_group)} words):\n")
                    f.write(f"  {', '.join(sorted_words)}\n")
        print(f"Successfully saved categories to '{output_filename}'.")
    except IOError as e:
        print(f"\nError: Could not write to file '{output_filename}'.")
        print(f"Reason: {e}")

    # --- Display LPC Spectra and Perform Prediction ---
    stored_matrices = {}
    for word in word_dictionary:
        signal_data, sample_rate, lpc_coeffs = audio_signals.get(word, (None, None, None))
        if lpc_coeffs is not None:
            a_coeffs = np.concatenate(([1], lpc_coeffs))
            w, h = signal.freqz(1, a_coeffs, worN=4096, fs=sample_rate)
            magnitudes_db = 20 * np.log10(np.abs(h) + 1e-9)
            dot_matrix = spectrum_to_dot_matrix(magnitudes_db, num_rows_dot_matrix, num_cols_dot_matrix)
            stored_matrices[word] = dot_matrix

    # --- Interactive Prediction ---
    while True:
        new_word = input("\nEnter a word to predict (or type 'exit' to quit): ").upper()
        if new_word == "EXIT":
            break
        predictions, new_word_matrix = predict_word(new_word, stored_matrices, sample_rate, lpc_order, num_rows_dot_matrix, num_cols_dot_matrix)
        if predictions:
            print(f"\nPredictions for '{new_word}':")
            for word, similarity in predictions[:out_limit]:
                print(f"- {word}: Similarity = {similarity:.2f}")
            if new_word_matrix is not None:
                visualize_dot_matrix(new_word_matrix, new_word)
        else:
            print(f"Could not make predictions for '{new_word}'.  Check the word and ensure it contains only letters A-Z.")

    print("\nCategorization and prediction finished.")

if __name__ == "__main__":
    main()
