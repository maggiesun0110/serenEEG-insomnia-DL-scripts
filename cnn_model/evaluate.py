import torch
import numpy as np
from sklearn.metrics import precision_recall_fscore_support, confusion_matrix
from torch.utils.data import DataLoader
from train import load_raw, split_subjects
from model import BaseEEGCNN   
import os

class EEGDataset(torch.utils.data.Dataset):
    def __init__(self, X, y):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.long)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]
    
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
CHECKPOINT_PATH = os.path.join(BASE_DIR, "checkpoints", "sereneeg_cnn_v1.pth")

checkpoint = torch.load(CHECKPOINT_PATH, map_location=device)

THRESHOLD = checkpoint["threshold"]
print("Using fixed threshold:", THRESHOLD)

#model
model = BaseEEGCNN(num_classes=2).to(device)
model.load_state_dict(checkpoint["model_state_dict"])
model.eval() 

#test/split
raw_X, labels, subject_ids = load_raw("../dl_ins_results/combined_raw.npz")

_, _, X_test, y_test = split_subjects(
    raw_X, labels, subject_ids, test_size=0.2
)

test_dataset = EEGDataset(X_test, y_test)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

all_preds = []
all_labels = []

with torch.no_grad():
    for Xb, yb in test_loader:
        Xb = Xb.to(device)

        logits = model(Xb)
        probs = torch.sigmoid(logits).squeeze(1)
        preds = (probs > THRESHOLD).long()

        all_preds.append(preds.cpu().numpy())
        all_labels.append(yb.numpy())

all_preds = np.concatenate(all_preds)
all_labels = np.concatenate(all_labels)

precision, recall, f1, _ = precision_recall_fscore_support(
    all_labels,
    all_preds,
    labels=[0, 1],
    zero_division=0
)

macro_f1 = f1.mean()
cm = confusion_matrix(all_labels, all_preds)

print("Precision:", precision)
print("Recall:", recall)
print("F1:", f1)
print("Macro F1:", macro_f1)
print("Confusion Matrix:\n", cm)