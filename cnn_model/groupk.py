import os
import numpy as np
import torch
import pandas as pd
import matplotlib.pyplot as plt

from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import GroupKFold
from sklearn.metrics import (
    precision_recall_fscore_support,
    confusion_matrix,
    accuracy_score
)

from train import load_raw
from model import BaseEEGCNN

# ---------------- CONFIG ----------------
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
N_SPLITS = 5
THRESHOLD = 0.82

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
CHECKPOINT_PATH = os.path.join(BASE_DIR, "checkpoints", "sereneeg_cnn_v1.pth")

# ---------------- DATASET ----------------
class EEGDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.long)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

# ---------------- LOAD MODEL ----------------
checkpoint = torch.load(CHECKPOINT_PATH, map_location=DEVICE)

# ---------------- LOAD DATA ----------------
raw_X, labels, subject_ids = load_raw("../dl_ins_results/combined_raw.npz")
subject_ids = np.array(subject_ids).astype(str)

gkf = GroupKFold(n_splits=N_SPLITS)

rows = []

# ---------------- CV LOOP ----------------
for fold, (_, test_idx) in enumerate(
    gkf.split(raw_X, labels, groups=subject_ids)
):
    print(f"\n===== Fold {fold+1}/{N_SPLITS} =====")

    X_test = raw_X[test_idx]
    y_test = labels[test_idx]

    test_loader = DataLoader(
        EEGDataset(X_test, y_test),
        batch_size=32,
        shuffle=False
    )

    model = BaseEEGCNN(num_classes=2).to(DEVICE)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    all_preds, all_labels = [], []

    with torch.no_grad():
        for Xb, yb in test_loader:
            Xb = Xb.to(DEVICE)
            probs = torch.sigmoid(model(Xb)).squeeze(1)
            preds = (probs > THRESHOLD).long()

            all_preds.append(preds.cpu().numpy())
            all_labels.append(yb.numpy())

    all_preds = np.concatenate(all_preds)
    all_labels = np.concatenate(all_labels)

    # ---- Metrics ----
    acc = accuracy_score(all_labels, all_preds)

    precision, recall, f1, _ = precision_recall_fscore_support(
        all_labels, all_preds,
        labels=[0, 1],
        zero_division=0
    )

    macro_precision = precision.mean()
    macro_recall = recall.mean()
    macro_f1 = f1.mean()

    rows.append([
        fold + 1,
        acc,
        precision[0], recall[0], f1[0],
        precision[1], recall[1], f1[1],
        macro_precision, macro_recall, macro_f1
    ])

# ---------------- TABLE ----------------
columns = [
    "Fold", "Accuracy",
    "P_Ins", "R_Ins", "F1_Ins",
    "P_H", "R_H", "F1_H",
    "Macro_P", "Macro_R", "Macro_F1"
]

df = pd.DataFrame(rows, columns=columns)

# Add mean + std
mean_row = ["Mean"] + df.iloc[:, 1:].mean().tolist()
std_row  = ["Std"]  + df.iloc[:, 1:].std().tolist()

df_summary = pd.concat(
    [df, pd.DataFrame([mean_row, std_row], columns=columns)],
    ignore_index=True
)

print("\n===== GROUPKFOLD RESULTS =====")
print(df_summary.round(4))

# ---------------- PLOT TABLE ----------------
fig, ax = plt.subplots(figsize=(14, 4))
ax.axis("off")

table = ax.table(
    cellText=df_summary.round(4).values,
    colLabels=df_summary.columns,
    loc="center",
    cellLoc="center"
)

table.auto_set_font_size(False)
table.set_fontsize(9)
table.scale(1.1, 1.5)

plt.title("Subject-wise 5-Fold GroupKFold Evaluation (SerenEEG-CNN-v1)")
plt.tight_layout()
plt.show()