import os
import numpy as np
import mne
from scipy.signal import butter, filtfilt, resample

# === Preprocessing Functions ===
def bandpass_filter(data, sf, low=0.5, high=40):
    b, a = butter(N=4, Wn=[low, high], btype='band', fs=sf)
    return filtfilt(b, a, data)

def notch_filter(data, sf, freq=60.0, Q=30.0):
    """Optional notch filter to remove power line noise."""
    from scipy.signal import iirnotch
    b, a = iirnotch(w0=freq, Q=Q, fs=sf)
    return filtfilt(b, a, data)

def preprocess(data, orig_sf, target_sf=200):
    data = bandpass_filter(data, orig_sf)
    data = notch_filter(data, orig_sf)  
    if orig_sf != target_sf:
        duration = len(data) / orig_sf
        data = resample(data, int(duration * target_sf))
    return data

def normalize_epoch(epoch):
    mean = np.mean(epoch)
    std = np.std(epoch) + 1e-6
    return (epoch - mean) / std

# === Config ===
BASE_PATH = "/Users/maggiesun/downloads/research/serenEEG/data/CapSleep"
RESULTS_PATH = os.path.join("..", "dl_ins_results")
CHANNELS_RAW = ["F4-C4", "C4-A1"]  
TARGET_CHANNELS = ["F4-A1", "C4-A1"]  # final channels
EPOCH_LEN = 30
SF_TARGET = 200
EPOCH_SAMPLES = EPOCH_LEN * SF_TARGET
OUTFILE = os.path.join(RESULTS_PATH, "capsleep_raw.npz")

X, y, subject_ids = [], [], []

# === Main Loop ===
for fname in sorted(os.listdir(BASE_PATH)):
    if not fname.endswith(".edf"):
        continue

    subj_path = os.path.join(BASE_PATH, fname)
    subj_id = os.path.splitext(fname)[0]
    print("Processing:", subj_id)

    try:
        raw = mne.io.read_raw_edf(subj_path, preload=True, verbose="ERROR")
    except Exception as e:
        print(f"Failed to load {fname}: {e}")
        continue

    orig_sf = int(raw.info["sfreq"])

    # Check channels
    missing = [ch for ch in CHANNELS_RAW if ch not in raw.ch_names and ch.replace("-", "_") not in raw.ch_names]
    if missing:
        print(f"Skipping {subj_id}, missing channels: {missing}")
        continue

    # Extract signals
    # F4-C4
    f4c4_name = "F4-C4" if "F4-C4" in raw.ch_names else "F4_C4"
    f4c4 = raw.get_data(picks=f4c4_name)[0]

    # C4-A1
    c4a1_name = "C4-A1" if "C4-A1" in raw.ch_names else "C4_A1"
    c4a1 = raw.get_data(picks=c4a1_name)[0]

    # Re-ref: F4-A1 = F4-C4 + C4-A1
    f4a1 = f4c4 + c4a1

    preprocessed = [
        preprocess(f4a1, orig_sf, SF_TARGET),
        preprocess(c4a1, orig_sf, SF_TARGET)
    ]

    # Epoching
    min_len = min(len(sig) for sig in preprocessed)
    n_epochs = min_len // EPOCH_SAMPLES
    if n_epochs == 0:
        print("Too short for 30s epochs.")
        continue

    epoch_arrays = [sig[:n_epochs * EPOCH_SAMPLES].reshape(n_epochs, EPOCH_SAMPLES) for sig in preprocessed]

    for i in range(n_epochs):
        epoch = np.stack([normalize_epoch(epoch_arrays[ch][i]) for ch in range(len(TARGET_CHANNELS))], axis=0)
        X.append(epoch)
        y.append(1)  # all CapSleep = insomnia
        subject_ids.append(subj_id)

# === Save ===
X = np.array(X)
y = np.array(y)
subject_ids = np.array(subject_ids)

os.makedirs(RESULTS_PATH, exist_ok=True)
np.savez(OUTFILE, raw_X=X, labels=y, subject_ids=subject_ids)

print("Saved:", OUTFILE)
print("Shape:", X.shape, y.shape, "Subjects:", len(np.unique(subject_ids)))