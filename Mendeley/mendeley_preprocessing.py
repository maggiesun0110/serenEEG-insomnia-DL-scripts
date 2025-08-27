import os
import numpy as np
import mne
from scipy.signal import butter, filtfilt, resample

# === Preprocessing functions ===
def bandpass_filter(data, sf, low=0.5, high=40):
    b, a = butter(N=4, Wn=[low, high], btype='band', fs=sf)
    return filtfilt(b, a, data)

def preprocess(data, orig_sf, target_sf=200):
    data = bandpass_filter(data, orig_sf)
    if orig_sf != target_sf:
        duration = len(data) / orig_sf
        data = resample(data, int(duration * target_sf))
    return data

def normalize_epoch(epoch):
    """Z-score normalize a 1D array."""
    mean = np.mean(epoch)
    std = np.std(epoch) + 1e-6  # avoid division by zero
    return (epoch - mean) / std

# === Config ===
BASE_PATH = "/Users/maggiesun/downloads/research/sereneeg/data/mendeley"
CHANNELS = ["F4A1", "C4A1", "O2A1"]   # must match names in EDF files
SF_TARGET = 200
EPOCH_LEN = 30  # seconds
EPOCH_SAMPLES = SF_TARGET * EPOCH_LEN
RESULTS_PATH = os.path.join("..", "dl_ins_results")
OUTFILE = os.path.join(RESULTS_PATH, "mendeley_raw.npz")

X, y, subject_ids = [], [], []

# === Main loop ===
for label_folder, label in [("normal", 0), ("insomnia", 1)]:
    folder_path = os.path.join(BASE_PATH, label_folder)
    if not os.path.exists(folder_path):
        print(f"❌ Folder not found: {folder_path}")
        continue

    for file in os.listdir(folder_path):
        if not file.endswith(".edf"):
            continue

        file_path = os.path.join(folder_path, file)
        print("Processing:", file_path)

        try:
            raw = mne.io.read_raw_edf(file_path, preload=True, verbose=False)
        except Exception as e:
            print(f"❌ Failed to load {file}: {e}")
            continue

        orig_sf = int(raw.info["sfreq"])
        try:
            raw.pick_channels(CHANNELS)
        except Exception as e:
            print(f"⚠️ Missing channels in {file}: {e}")
            continue

        data = raw.get_data()

        # Preprocess each channel
        preprocessed = [preprocess(data[ch], orig_sf, SF_TARGET) for ch in range(len(CHANNELS))]
        min_len = min(len(ch) for ch in preprocessed)
        n_epochs = min_len // EPOCH_SAMPLES
        if n_epochs == 0:
            print("⚠️ Too short for 30s epochs:", file_path)
            continue

        # Epoching and normalization
        epoch_arrays = [ch[:n_epochs * EPOCH_SAMPLES].reshape(n_epochs, EPOCH_SAMPLES) for ch in preprocessed]
        for i in range(n_epochs):
            epoch = np.stack([normalize_epoch(epoch_arrays[ch_idx][i]) for ch_idx in range(len(CHANNELS))], axis=0)
            X.append(epoch)
            y.append(label)
            subject_ids.append(os.path.splitext(file)[0])

# === Save ===
X = np.array(X, dtype=np.float32)  # (epochs, channels, samples)
y = np.array(y)
subject_ids = np.array(subject_ids)

os.makedirs(RESULTS_PATH, exist_ok=True)
np.savez(OUTFILE, raw_X=X, labels=y, subject_ids=subject_ids)

print("✅ Saved:", OUTFILE)
print("Shape:", X.shape, y.shape, "Subjects:", len(np.unique(subject_ids)))