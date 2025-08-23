#bandpass
#30s, 200Hz
#normalize per epoch (z-score)
   #normzlie = (og-mean)/sdd
import os
import numpy as np
from scipy.io import loadmat
from scipy.signal import butter, filtfilt, resample

# === Preprocessing Functions ===
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
    std = np.std(epoch) + 1e-6  # prevent division by zero
    return (epoch - mean) / std

# === Config ===
BASE_PATH = "/Users/maggiesun/downloads/research/sereneeg/data/ISRUC sleep/data"
RESULTS_PATH = os.path.join("..", "dl_ins_results")
CHANNELS = ["F4_A1", "C4_A1", "O2_A1"]
EPOCH_LEN = 30  
SF_TARGET = 200
EPOCH_SAMPLES = EPOCH_LEN * SF_TARGET
OUTFILE = os.path.join(RESULTS_PATH, "isruc_raw.npz")

X, y, subject_ids = [], [], []

# === Main Loop ===
for subj in sorted(os.listdir(BASE_PATH)):
    subj_path = os.path.join(BASE_PATH, subj)
    if not os.path.isdir(subj_path):
        continue

    print("Processing:", subj)
    mat_file = next((f for f in os.listdir(subj_path) if f.endswith(".mat")), None)
    if not mat_file:
        print("No .mat file found.")
        continue

    mat_path = os.path.join(subj_path, mat_file)
    mat = loadmat(mat_path)

    try:
        raw_data = [mat[ch].flatten() for ch in CHANNELS]
    except KeyError as e:
        print(f"Missing channel {e} in {subj}")
        continue

    orig_sf = mat.get("sampling_frequency", SF_TARGET)
    if isinstance(orig_sf, np.ndarray):
        orig_sf = int(orig_sf.flatten()[0])

    # Preprocess channels
    preprocessed = [preprocess(sig, orig_sf, SF_TARGET) for sig in raw_data]

    min_len = min(len(sig) for sig in preprocessed)
    n_epochs = min_len // EPOCH_SAMPLES
    if n_epochs == 0:
        print("Too short for 30s epochs.")
        continue

    # Epoching
    epoch_arrays = [sig[:n_epochs * EPOCH_SAMPLES].reshape(n_epochs, EPOCH_SAMPLES) for sig in preprocessed]

    # Stack channels per epoch and normalize each channel
    for i in range(n_epochs):
        epoch = np.stack([normalize_epoch(epoch_arrays[ch][i]) for ch in range(len(CHANNELS))], axis=0)
        X.append(epoch)
        y.append(1)  # change label accordingly
        subject_ids.append(subj)

# === Save ===
X = np.array(X)  # shape: (num_epochs, channels, samples)
y = np.array(y)
subject_ids = np.array(subject_ids)

os.makedirs(RESULTS_PATH, exist_ok=True)
np.savez(OUTFILE, raw_X=X, labels=y, subject_ids=subject_ids)

print("Saved:", OUTFILE)
print("Shape:", X.shape, y.shape, "Subjects:", len(np.unique(subject_ids)))