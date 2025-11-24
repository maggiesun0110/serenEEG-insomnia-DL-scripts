import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import precision_recall_fscore_support, confusion_matrix
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from torch.utils.tensorboard import SummaryWriter
import datetime

# 1. Load EEG data
def load_raw(npz_path):
    data = np.load(npz_path, allow_pickle=True)

    try:
        raw_X = data['raw_X']
        labels = data['labels']
        subject_ids = data['subject_ids']
    except KeyError:
        possible_dict = data[data.files[0]].item()
        raw_X = possible_dict['raw_X']
        labels = possible_dict['labels']
        subject_ids = possible_dict['subject_ids']

    print("Loaded:", raw_X.shape, labels.shape, subject_ids.shape)
    return raw_X, labels, subject_ids

# 2. Subject-wise split (no leakage)
def split_subjects(X, y, subj_ids, test_size=0.2):
    subj_ids = np.array([str(s) for s in subj_ids])
    unique = np.unique(subj_ids)

    train_subj, test_subj = train_test_split(unique, test_size=test_size, random_state=42)

    train_mask = np.isin(subj_ids, train_subj)
    test_mask  = np.isin(subj_ids, test_subj)

    return X[train_mask], y[train_mask], X[test_mask], y[test_mask]

# 3. data
class EEGDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.long)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

# 4. Simple CNN
class BaseEEGCNN(nn.Module):
    def __init__(self, num_classes=2):
        super().__init__()
        self.conv1 = nn.Conv1d(3, 16, kernel_size=64, stride=16)
        self.pool1 = nn.MaxPool1d(4)

        self.conv2 = nn.Conv1d(16, 32, 3)
        self.pool2 = nn.MaxPool1d(4)

        self.relu = nn.ReLU()
        self.fc1 = nn.Linear(32 * 22, 64)
        self.fc2 = nn.Linear(64, num_classes)

    def forward(self, x):
        x = self.pool1(self.relu(self.conv1(x)))
        x = self.pool2(self.relu(self.conv2(x)))
        x = x.view(x.size(0), -1)
        x = self.relu(self.fc1(x))
        return self.fc2(x)

# 5. Minimal training loop
def train_minimal(model, train_loader, test_loader, device, epochs=10, lr=1e-3):
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    #NEW CLEAN TENSORBOARD RUN
    run_name = f"baseline_with_precrecf1_{datetime.datetime.now().strftime('%Y%m%d-%H%M%S')}"
    writer = SummaryWriter(log_dir=f"runs/{run_name}")

    for epoch in range(epochs):
        model.train()
        total_loss = 0

        for Xb, yb in train_loader:
            Xb, yb = Xb.to(device), yb.to(device)
            optimizer.zero_grad()
            out = model(Xb)
            loss = criterion(out, yb)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        avg_loss = total_loss / len(train_loader)

        # ---------- Evaluate ----------
        model.eval()
        all_preds = []
        all_labels = []
        with torch.no_grad():
            for Xb, yb in test_loader:
                Xb, yb = Xb.to(device), yb.to(device)
                out = model(Xb)
                _, pred = torch.max(out, 1)
                all_preds.append(pred.cpu().numpy())
                all_labels.append(yb.cpu().numpy())

        all_preds = np.concatenate(all_preds)
        all_labels = np.concatenate(all_labels)
        test_acc = (all_preds == all_labels).mean()
        precision, recall, f1, _ = precision_recall_fscore_support(all_labels, all_preds, labels = np.unique(all_labels), zero_division=0)

        cm = confusion_matrix(all_labels, all_preds)
        # TensorBoard logs
        writer.add_scalar("Loss/train", avg_loss, epoch)
        writer.add_scalar("Accuracy/test", test_acc, epoch)
        writer.add_scalar("Precision/test", np.mean(precision), epoch)
        writer.add_scalar("Recall/test", np.mean(recall), epoch)
        writer.add_scalar("F1_macro/test", np.mean(f1), epoch)

        print(f"Epoch {epoch+1}/{epochs} | Loss: {avg_loss:.4f} | Test Acc: {test_acc:.4f}")
        print(f"  Precision per class: {precision}")
        print(f"  Recall per class:    {recall}")
        print(f"  F1 per class:        {f1}")
        print(f"  Confusion matrix:\n{cm}")

    writer.close()

# 6. Main
if __name__ == "__main__":
    npz_file = "../dl_ins_results/combined_raw.npz"
    raw_X, labels, subject_ids = load_raw(npz_file)

    X_train, y_train, X_test, y_test = split_subjects(raw_X, labels, subject_ids)

    train_dataset = EEGDataset(X_train, y_train)
    test_dataset  = EEGDataset(X_test, y_test)

    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    test_loader  = DataLoader(test_dataset, batch_size=32, shuffle=False)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Device:", device)

    model = BaseEEGCNN(num_classes=len(np.unique(labels))).to(device)

    train_minimal(model, train_loader, test_loader, device, epochs=10, lr=1e-3)

    print("Done!")