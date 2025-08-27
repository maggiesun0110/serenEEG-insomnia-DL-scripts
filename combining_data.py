import os
import numpy as np

# === Paths to your preprocessed datasets ===
datasets = {
    "capsleep": "/Users/maggiesun/downloads/research/sereneeg/dl_ins/dl_ins_results/capsleep_raw.npz",
    "isruc": "/Users/maggiesun/downloads/research/sereneeg/dl_ins/dl_ins_results/isruc_raw.npz",
    "mendeley": "/Users/maggiesun/downloads/research/sereneeg/dl_ins/dl_ins_results/mendeley_raw.npz"
}

TARGET_CHANNELS = 3  # F4-A1, C4-A1, O2-A1

# --- Prepare combined .npz storage ---
output_path = "/Users/maggiesun/downloads/research/sereneeg/dl_ins/dl_ins_results/combined_raw.npz"
os.makedirs(os.path.dirname(output_path), exist_ok=True)

X_list = []
y_list = []
subject_ids_list = []

for name, path in datasets.items():
    if not os.path.exists(path):
        print(f"❌ Missing dataset file: {path}")
        continue

    data = np.load(path)
    X = data['raw_X'].astype(np.float32)  # save memory
    y = data['labels']
    subject_ids = data.get('subject_ids', np.array([f"{name}_{i}" for i in range(len(y))]))

    print(f"Loaded {name}: X={X.shape}, y={y.shape}, subjects={len(subject_ids)}")

    # --- Sanity checks ---
    if X.shape[0] != y.shape[0] or X.shape[0] != len(subject_ids):
        print(f"⚠️ Epoch/label/subject mismatch in {name}, skipping")
        continue
    if X.ndim != 3:
        print(f"⚠️ Expected 3D features (epochs, channels, samples) in {name}, got {X.shape}")
        continue
    if np.isnan(X).any():
        print(f"⚠️ NaNs found in {name}, skipping")
        continue

    # --- Pad channels if needed ---
    if X.shape[1] < TARGET_CHANNELS:
        n_missing = TARGET_CHANNELS - X.shape[1]
        pad_shape = (X.shape[0], n_missing, X.shape[2])
        X_pad = np.zeros(pad_shape, dtype=X.dtype)
        X = np.concatenate([X, X_pad], axis=1)
        print(f"⚠️ Padded {name} with {n_missing} zero channel(s) to match {TARGET_CHANNELS} channels")

    # --- Prefix subject IDs with dataset name ---
    subject_ids = np.array([f"{name}_{sid}" for sid in subject_ids])

    X_list.append(X)
    y_list.append(y)
    subject_ids_list.append(subject_ids)

# --- Concatenate in chunks to avoid memory spikes ---
X_combined = np.concatenate(X_list, axis=0)
y_combined = np.concatenate(y_list, axis=0)
subject_ids_combined = np.concatenate(subject_ids_list, axis=0)

print(f"\n✅ Combined dataset: X={X_combined.shape}, y={y_combined.shape}")
print(f"Label distribution: {dict(zip(*np.unique(y_combined, return_counts=True)))}")
print(f"Subjects: {len(np.unique(subject_ids_combined))}")

# --- Save combined file as float32 ---
np.savez_compressed(output_path, raw_X=X_combined, labels=y_combined, subject_ids=subject_ids_combined)
print(f"💾 Saved combined file to {output_path}")