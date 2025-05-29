
import os
import glob
import numpy as np
import pandas as pd
import mne

def load_and_create_raw(data_src, sfreq=173.61):
    """
    Loads multiple CSVs from a folder and returns a list of filtered MNE RawArray objects.
    """
    files = glob.glob(data_src)
    raw_list = []

    for file in files:
        df = pd.read_csv(file, header=None)
        data = df.values.flatten()
        data = data[np.newaxis, :]  # Shape = (1, n_samples)

        info = mne.create_info(ch_names=['EEG'], sfreq=sfreq, ch_types=['eeg'])
        raw = mne.io.RawArray(data, info, verbose=False)
        raw.filter(l_freq=0.5, h_freq=30.0, verbose=False)

        raw_list.append(raw)

    return raw_list

def segment_raw_data(raw, segment_length):
    """
    Splits MNE Raw object into multiple segments of given length (in samples).
    """
    data = raw.get_data()[0]
    num_segments = len(data) // segment_length
    segments = []

    for i in range(num_segments):
        segment = data[i*segment_length:(i+1)*segment_length]
        segments.append(segment)

    return segments

def extract_features_from_segment(segment, sfreq):
    """
    Extracts spectral features from a single segment using compute_psd.
    """
    # Create temporary raw object for the segment
    data = np.array([segment])
    info = mne.create_info(ch_names=['EEG'], sfreq=sfreq, ch_types=['eeg'])
    raw = mne.io.RawArray(data, info, verbose=False)

    # Compute PSD (avoid deprecated psd_* functions)
    psd = raw.compute_psd(method='welch', fmin=0.5, fmax=40.0, n_per_seg=len(segment), verbose=False)
    freqs = psd.freqs
    psd_values = psd.get_data()[0]  # shape: (n_freqs,)

    # Normalize for entropy
    psd_norm = psd_values / np.sum(psd_values)

    # Features
    spectral_centroid = np.sum(freqs * psd_norm)
    spectral_spread = np.sqrt(np.sum(((freqs - spectral_centroid) ** 2) * psd_norm))
    spectral_entropy = -np.sum(psd_norm * np.log2(psd_norm + 1e-12))
    peak_idx = np.argmax(psd_values)
    peak_freq = freqs[peak_idx]
    peak_power = psd_values[peak_idx]

    return [spectral_centroid, spectral_spread, spectral_entropy, peak_freq, peak_power]

def extract_feature_matrix(raw_list, segment_length, sfreq):
    """
    Processes a list of raw signals and returns a feature matrix.
    """
    features = []
    for raw in raw_list:
        segments = segment_raw_data(raw, segment_length)
        for seg in segments:
            feats = extract_features_from_segment(seg, sfreq)
            features.append(feats)
    return np.array(features)

# Paths
data_path = "./data/"
fs = 173.61
segment_length = 500  # ~2.88 seconds

# Load
A_raw = load_and_create_raw(os.path.join(data_path, 'A/*'), sfreq=fs)
B_raw = load_and_create_raw(os.path.join(data_path, 'B/*'), sfreq=fs)
C_raw = load_and_create_raw(os.path.join(data_path, 'C/*'), sfreq=fs)
D_raw = load_and_create_raw(os.path.join(data_path, 'D/*'), sfreq=fs)
E_raw = load_and_create_raw(os.path.join(data_path, 'E/*'), sfreq=fs)

# Group
normal_raw = A_raw + B_raw
interictal_raw = C_raw + D_raw
ictal_raw = E_raw

# Extract feature matrices
X_normal = extract_feature_matrix(normal_raw, segment_length, fs)
X_interictal = extract_feature_matrix(interictal_raw, segment_length, fs)
X_ictal = extract_feature_matrix(ictal_raw, segment_length, fs)


# Add class labels
normal_labels = np.zeros(X_normal.shape[0])       # 0 = Normal
interictal_labels = np.ones(X_interictal.shape[0]) # 1 = Interictal
ictal_labels = np.full(X_ictal.shape[0], 2)         # 2 = Ictal

# Combine features and labels
X_all = np.vstack((X_normal, X_interictal, X_ictal))
y_all = np.concatenate((normal_labels, interictal_labels, ictal_labels))

# Create DataFrame
columns = ['SpectralCentroid', 'SpectralSpread', 'SpectralEntropy', 'PeakFrequency', 'PeakPower', 'Label']
df = pd.DataFrame(np.column_stack((X_all, y_all)), columns=columns)


# Save to CSV
output_path = "./eeg_feature_dataset.csv"
df.to_csv(output_path, index=False)

print(f"Dataset saved to {output_path}")
