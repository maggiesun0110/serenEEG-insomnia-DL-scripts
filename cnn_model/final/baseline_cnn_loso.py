import os
import sys

# Fix import path
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
CNN_MODEL_DIR = os.path.dirname(SCRIPT_DIR)
sys.path.append(CNN_MODEL_DIR)

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset
from sklearn.model_selection import LeaveOneGroupOut
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report, confusion_matrix

from dataset import EEGDataset
from model import BaseEEGCNN

# config
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
BATCH_SIZE = 32
EPOCHS = 15
LR = 1e-3
THRESHOLD = 0.82

DATA_PATH = "../dl_ins_results/combined_raw.npz"

#data
dataset = EEGDataset(DATA_PATH)

X = dataset.X
y = dataset.y
groups = dataset.subject_ids

logo = LeaveOneGroupOut()

subject_preds = []
subject_true = []

print("Running CNN Subject-Level LOSO\n")

#loso
for fold, (train_idx, test_idx) in enumerate(logo.split(X, y, groups), 1):

    print(f"\n===== Fold {fold} =====")

    train_subset = Subset(dataset, train_idx)
    test_subset  = Subset(dataset, test_idx)

    train_loader = DataLoader(train_subset, batch_size=BATCH_SIZE, shuffle=True)
    test_loader  = DataLoader(test_subset, batch_size=BATCH_SIZE, shuffle=False)

    model = BaseEEGCNN().to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    criterion = nn.BCEWithLogitsLoss()

    # train
    model.train()
    for epoch in range(EPOCHS):
        epoch_loss = 0.0
        num_batches = 0

        for Xb, yb, _ in train_loader:
            Xb = Xb.to(DEVICE)
            yb = yb.float().to(DEVICE)

            optimizer.zero_grad()
            logits = model(Xb).squeeze(1)
            loss = criterion(logits, yb)
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()
            num_batches += 1

        epoch_loss /= num_batches 

        print(f"Epoch {epoch+1}/{EPOCHS} - Loss: {epoch_loss:.4f}")

    #test
    model.eval()
    fold_probs = []
    fold_labels = []
    fold_subjects = []

    with torch.no_grad():
        for Xb, yb, sid in test_loader:
            Xb = Xb.to(DEVICE)
            probs = torch.sigmoid(model(Xb)).squeeze(1)

            fold_probs.append(probs.cpu().numpy())
            fold_labels.append(yb.numpy())
            fold_subjects.append(sid)

    fold_probs = np.concatenate(fold_probs)
    fold_labels = np.concatenate(fold_labels)
    fold_subjects = np.concatenate(fold_subjects)

    #per subject aggregation
    unique_subjects = np.unique(fold_subjects)

    for subj in unique_subjects:
        mask = fold_subjects == subj
        subj_probs = fold_probs[mask]
        subj_label = fold_labels[mask][0]

        subj_mean_prob = np.mean(subj_probs)
        subj_pred = int(subj_mean_prob >= THRESHOLD)

        subject_preds.append(subj_pred)
        subject_true.append(subj_label)

    print(f"Subject {subj} - True={subj_label}, Pred={subj_pred}")

#metrics
subject_preds = np.array(subject_preds)
subject_true = np.array(subject_true)

print("\n" + "="*60)
print("CNN Subject-Level LOSO Results")
print("Accuracy:", accuracy_score(subject_true, subject_preds))
print("Precision:", precision_score(subject_true, subject_preds, zero_division=0))
print("Recall:", recall_score(subject_true, subject_preds, zero_division=0))
print("F1:", f1_score(subject_true, subject_preds, zero_division=0))
print("Confusion Matrix:\n", confusion_matrix(subject_true, subject_preds))
print("="*60)

print("\nClassification Report")
print(classification_report(subject_true, subject_preds, digits=4))
print("="*60)