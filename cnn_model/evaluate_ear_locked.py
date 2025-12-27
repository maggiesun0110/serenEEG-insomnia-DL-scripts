import os
import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report, roc_auc_score
)

from model import BaseEEGCNN  

# ----------------------------
# Config
# ----------------------------
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

EESM19_NPZ = "/Volumes/ORICO/SerenEEG/dl_ins/dl_ins_results/eesm19_raw.npz"
CKPT_PATH = "/Volumes/ORICO/SerenEEG/dl_ins/dl_ins_scripts/cnn_model/checkpoints/sereneeg_cnn_v1.pth"


# EESM19 is healthy 
POS_LABEL = 0  
BATCH_SIZE = 256

# ----------------------------
# Helpers
# ----------------------------
@torch.no_grad()
def predict_logits(model, X, batch_size=256):
    model.eval()
    logits_all = []
    n = len(X)
    for i in range(0, n, batch_size):
        xb = torch.tensor(X[i:i+batch_size], dtype=torch.float32, device=DEVICE)
        logits = model(xb).squeeze(-1)  # shape (B,)
        logits_all.append(logits.detach().cpu().numpy())
    return np.concatenate(logits_all, axis=0)

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def subject_majority_vote(probs, subject_ids, threshold=0.5):
    """Aggregate epoch predictions to subject predictions (majority vote)."""
    subj_preds = {}
    subj_true = {}
    subjects = np.unique(subject_ids)

    for s in subjects:
        idx = np.where(subject_ids == s)[0]
        p = probs[idx]
        pred_epochs = (p >= threshold).astype(int)
        pred_subj = int(np.mean(pred_epochs) >= 0.5)  # majority
        subj_preds[s] = pred_subj
        # assume all epochs for subject share same label:
        subj_true[s] = int(np.round(np.mean(y[idx])))

    y_true_subj = np.array([subj_true[s] for s in subjects])
    y_pred_subj = np.array([subj_preds[s] for s in subjects])
    return subjects, y_true_subj, y_pred_subj

# ----------------------------
# Load data
# ----------------------------
data = np.load(EESM19_NPZ, allow_pickle=True)
X = data["raw_X"]           #(N, 3, 6000)
y = data["labels"]          #(N,)
subject_ids = data["subject_ids"]  #(N,)

print("Loaded EESM19:", X.shape, y.shape, subject_ids.shape)
print("Unique subjects:", len(np.unique(subject_ids)))

# ----------------------------
# Load locked model
# ----------------------------
ckpt = torch.load(CKPT_PATH, map_location=DEVICE)

model = BaseEEGCNN(num_classes=2).to(DEVICE)

# model outputs 1 logit, so state dict is usually just model_state_dict
model.load_state_dict(ckpt["model_state_dict"])
model.eval()

# threshold used during training (if saved)
threshold = ckpt.get("threshold", 0.5)
print("Using threshold:", threshold)

# ----------------------------
# Predict
# ----------------------------
logits = predict_logits(model, X, batch_size=BATCH_SIZE)
probs = sigmoid(logits)

# Binary predictions
y_pred = (probs >= threshold).astype(int)

# ----------------------------
# Metrics (epoch-level)
# ----------------------------
acc = accuracy_score(y, y_pred)
prec_macro = precision_score(y, y_pred, average="macro", zero_division=0)
rec_macro = recall_score(y, y_pred, average="macro", zero_division=0)
f1_macro = f1_score(y, y_pred, average="macro", zero_division=0)

prec_pos = precision_score(y, y_pred, pos_label=POS_LABEL, zero_division=0)
rec_pos = recall_score(y, y_pred, pos_label=POS_LABEL, zero_division=0)
f1_pos = f1_score(y, y_pred, pos_label=POS_LABEL, zero_division=0)

cm = confusion_matrix(y, y_pred)

print("\n=== Epoch-level Metrics ===")
print("Accuracy:", acc)
print("Precision (macro):", prec_macro)
print("Recall (macro):", rec_macro)
print("F1 (macro):", f1_macro)
print(f"Precision (pos={POS_LABEL}):", prec_pos)
print(f"Recall (pos={POS_LABEL}):", rec_pos)
print(f"F1 (pos={POS_LABEL}):", f1_pos)
print("\nConfusion matrix:\n", cm)
print("\nClassification report:\n", classification_report(y, y_pred, digits=4, zero_division=0))

# AUC only valid if both classes appear
if len(np.unique(y)) == 2:
    try:
        auc = roc_auc_score(y, probs)
        print("ROC-AUC:", auc)
    except Exception as e:
        print("ROC-AUC could not be computed:", e)
else:
    print("ROC-AUC skipped (only one class in y).")

# ----------------------------
# Metrics (subject-level)
# ----------------------------
# NOTE: EESM19 is healthy only, so subject-level metrics will be trivial unless you mix insomnia subjects too.
subjects = np.unique(subject_ids)
print("\n=== Subject-level (majority vote) ===")
print("Subjects:", len(subjects))

# If y contains only 1s, accuracy is still computable but precision/recall for other class is undefined.
# We'll still print the confusion matrix for sanity.
# (If you later combine with insomnia subjects, this becomes very meaningful.)
subj_pred = {}
subj_true = {}
for s in subjects:
    idx = np.where(subject_ids == s)[0]
    pred_s = int(np.mean((probs[idx] >= threshold).astype(int)) >= 0.5)
    true_s = int(np.round(np.mean(y[idx])))
    subj_pred[s] = pred_s
    subj_true[s] = true_s

y_true_subj = np.array([subj_true[s] for s in subjects])
y_pred_subj = np.array([subj_pred[s] for s in subjects])

print("Subject accuracy:", accuracy_score(y_true_subj, y_pred_subj))
print("Subject confusion matrix:\n", confusion_matrix(y_true_subj, y_pred_subj))