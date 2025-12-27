import os
import numpy as np
from collections import defaultdict

# =====================
# Paths
# =====================
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))        # cnn_model/
DL_INS = os.path.dirname(os.path.dirname(SCRIPT_DIR))         # dl_ins/

IN_PATH = os.path.join(
    DL_INS,
    "dl_ins_results",
    "combined_embeddings.npz"
)

OUT_PATH = os.path.join(
    DL_INS,
    "dl_ins_results",
    "combined_subject_features.npz"
)

# =====================
# Load embeddings
# =====================
data = np.load(IN_PATH)
X = data["X_emb"]          # (num_epochs, 64)
y = data["y"]              # (num_epochs,)
subjects = data["subject_ids"]

print("Loaded epoch embeddings:", X.shape)

# =====================
# Aggregate per subject
# =====================
subject_dict = defaultdict(list)

for xi, sid in zip(X, subjects):
    subject_dict[sid].append(xi)

X_sub, y_sub, sub_ids = [], [], []

for sid, feats in subject_dict.items():
    feats = np.vstack(feats)     # (epochs_per_subject, 64)

    # Mean + Std pooling → 128-D
    subject_vector = np.hstack([
        feats.mean(axis=0),
        feats.std(axis=0)
    ])

    X_sub.append(subject_vector)
    y_sub.append(y[subjects == sid][0])
    sub_ids.append(sid)

X_sub = np.array(X_sub)   # (num_subjects, 128)
y_sub = np.array(y_sub)
sub_ids = np.array(sub_ids)

# =====================
# Save
# =====================
np.savez(
    OUT_PATH,
    X=X_sub,
    y=y_sub,
    subject_ids=sub_ids
)

print("Saved subject-level features:", X_sub.shape)
print("Subjects:", len(sub_ids))