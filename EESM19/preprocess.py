import os
import numpy as np
import mne
from scipy.signal import butter, filtfilt, resample

# =====================
# CONFIG
# =====================
BASE_PATH = "/Volumes/ORICO/SerenEEG/data/EESM19_tmp"
RESULTS_PATH = "/Volumes/ORICO/SerenEEG/dl_ins_results"

SF_TARGET = 200
EPOCH_LEN = 30  # seconds
EPOCH_SAMPLES = SF_TARGET * EPOCH_LEN

OUTFILE = os.path.join(RESULTS_PATH, "eesm19_raw.npz")

# =====================
# SIGNAL PROCESSING
# =====================
def bandpass(sig, sf, low=0.5, high=40):
    b, a = butter(4, [low, high], btype="band", fs=sf)
    return filtfilt(b, a, sig)

def preprocess_channel(sig, orig_sf):
    sig = bandpass(sig, orig_sf)
    if orig_sf != SF_TARGET:
        new_len = int(len(sig) * SF_TARGET / orig_sf)
        sig = resample(sig, new_len)
    return sig

def zscore(x):
    return (x - np.mean(x)) / (np.std(x) + 1e-6)

# =====================
# MAIN
# =====================
X, y, subject_ids = [], [], []
files_seen = 0
epochs_added = 0

for subj in sorted(os.listdir(BASE_PATH)):
    if not subj.startswith("sub-"):
        continue

    subj_path = os.path.join(BASE_PATH, subj)
    print(f"\nProcessing {subj}")

    for root, _, files in os.walk(subj_path):
        for fname in files:
            if not fname.endswith("_task-sleep_acq-earEEG_eeg.set"):
                continue

            files_seen += 1
            eeg_path = os.path.join(root, fname)
            print("  → Loading:", eeg_path)

            # ---- Load ----
            raw = mne.io.read_raw_eeglab(eeg_path, preload=True, verbose=False)

            # ---- Drop EOG ----
            if "EOGr" in raw.ch_names:
                raw.drop_channels(["EOGr"])

            # ---- Bipolar montage (3 channels) ----
            raw = mne.set_bipolar_reference(raw, "ELA", "ELB", ch_name="E1", drop_refs=False)
            raw = mne.set_bipolar_reference(raw, "ERA", "ERB", ch_name="E2", drop_refs=False)
            raw = mne.set_bipolar_reference(raw, "ELT", "ERT", ch_name="E3", drop_refs=False)

            raw.pick_channels(["E1", "E2", "E3"])

            data = raw.get_data()  # (3, time)
            sf = raw.info["sfreq"]

            print("    Raw samples:", data.shape[1], "sf:", sf)

            # ---- Preprocess ----
            processed = [preprocess_channel(ch, sf) for ch in data]

            min_len = min(len(ch) for ch in processed)
            n_epochs = min_len // EPOCH_SAMPLES

            print("    Epochs available:", n_epochs)

            if n_epochs == 0:
                print("    ⚠ Skipping (too short)")
                continue

            # ---- Epoch + normalize ----
            for i in range(n_epochs):
                epoch = np.stack(
                    [
                        zscore(
                            ch[i * EPOCH_SAMPLES : (i + 1) * EPOCH_SAMPLES]
                        )
                        for ch in processed
                    ],
                    axis=0
                )

                X.append(epoch)
                y.append(1)  # healthy
                subject_ids.append(subj)
                epochs_added += 1

print("\n=====================")
print("FILES SEEN:", files_seen)
print("EPOCHS ADDED:", epochs_added)

# =====================
# SAVE
# =====================
X = np.array(X)
y = np.array(y)
subject_ids = np.array(subject_ids)

os.makedirs(RESULTS_PATH, exist_ok=True)

np.savez(
    OUTFILE,
    raw_X=X,
    labels=y,
    subject_ids=subject_ids
)

print("\nSaved:", OUTFILE)
print("Final shape:", X.shape)
print("Subjects:", len(np.unique(subject_ids)))