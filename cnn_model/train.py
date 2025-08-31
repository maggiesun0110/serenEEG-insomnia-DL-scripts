
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

# -------------------------------
# 1. Load EEG data
# -------------------------------
def load_combined_raw(npz_path):
    data = np.load(npz_path, allow_pickle=True)
    print(f"Keys in the npz file: {data.files}")
    
    try:
        raw_X = data['raw_X']
        labels = data['labels']
        subject_ids = data['subject_ids']
    except KeyError:
        if len(data.files) == 1:
            possible_dict = data[data.files[0]].item()
            raw_X = possible_dict['raw_X']
            labels = possible_dict['labels']
            subject_ids = possible_dict['subject_ids']
        else:
            raise KeyError("Required keys not found in npz file.")
    
    print(f"Data loaded successfully! Shapes:")
    print(f"raw_X: {raw_X.shape}, labels: {labels.shape}, subject_ids: {subject_ids.shape}")
    return raw_X, labels, subject_ids

# -------------------------------
# 2. Dataset Split with leakage check
# -------------------------------
def split_subjects(raw_X, labels, subject_ids, test_size=0.2, random_state=42):
    subject_ids = np.array([str(s) for s in subject_ids])
    unique_subjects = np.unique(subject_ids)
    
    train_subjects, test_subjects = train_test_split(unique_subjects, test_size=test_size, random_state=random_state)
    
    # Masks
    train_mask = np.isin(subject_ids, train_subjects)
    test_mask  = np.isin(subject_ids, test_subjects)
    
    # Apply masks
    X_train, y_train = raw_X[train_mask], labels[train_mask]
    X_test, y_test   = raw_X[test_mask], labels[test_mask]
    
    train_subject_ids = subject_ids[train_mask]
    test_subject_ids  = subject_ids[test_mask]
    
    # Sanity check for leakage
    overlap = np.intersect1d(train_subject_ids, test_subject_ids)
    if len(overlap) > 0:
        print("WARNING: Subject leakage detected!", overlap)
    else:
        print("No subject overlap between train and test sets.")
    
    print(f"Train shape: {X_train.shape}, Test shape: {X_test.shape}")
    return X_train, y_train, X_test, y_test, train_subject_ids, test_subject_ids

# -------------------------------
# 3. Dataset & DataLoader
# -------------------------------
class EEGDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.long)
    
    def __len__(self):
        return len(self.X)
    
    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

# -------------------------------
# 4. CNN Model with weight initialization
# -------------------------------
class RobustEEGCNN(nn.Module):
    def __init__(self, num_classes=2):
        super(RobustEEGCNN, self).__init__()
        self.conv1 = nn.Conv1d(3, 32, kernel_size=64, stride=16)
        self.bn1   = nn.BatchNorm1d(32)
        self.pool1 = nn.MaxPool1d(4)
        self.dropout1 = nn.Dropout(0.3)
        
        self.conv2 = nn.Conv1d(32, 64, 3)
        self.bn2   = nn.BatchNorm1d(64)
        self.pool2 = nn.MaxPool1d(4)
        self.dropout2 = nn.Dropout(0.3)
        
        self.fc1 = nn.Linear(64*22, 128)
        self.dropout3 = nn.Dropout(0.5)
        self.fc2 = nn.Linear(128, num_classes)
        self.relu = nn.ReLU()
        
        # Weight initialization
        self._initialize_weights()
    
    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv1d) or isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight)  # He initialization
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
    
    def forward(self, x):
        x = self.pool1(self.relu(self.bn1(self.conv1(x))))
        x = self.dropout1(x)
        x = self.pool2(self.relu(self.bn2(self.conv2(x))))
        x = self.dropout2(x)
        x = x.view(x.size(0), -1)
        x = self.relu(self.fc1(x))
        x = self.dropout3(x)
        x = self.fc2(x)
        return x

# -------------------------------
# 5b. Training & loss + accuracy tracking with per-subject plots (class weighting)
# -------------------------------
from sklearn.utils.class_weight import compute_class_weight

def train_model(model, train_loader, test_loader, test_subject_ids, y_train, device, epochs=10, lr=1e-3):
    # Compute class weights
    classes = np.unique(y_train)
    class_weights = compute_class_weight(class_weight='balanced', classes=classes, y=y_train)
    class_weights = torch.tensor(class_weights, dtype=torch.float32).to(device)
    print(f"Class weights: {class_weights}")

    criterion = nn.CrossEntropyLoss(weight=class_weights)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    
    train_losses = []
    test_accuracies = []
    per_subject_acc_history = {}

    unique_subjects = np.unique(test_subject_ids)
    for subj in unique_subjects:
        per_subject_acc_history[subj] = []

    for epoch in range(epochs):
        # ----------------- Training -----------------
        model.train()
        train_loss = 0
        for X_batch, y_batch in train_loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            optimizer.zero_grad()
            output = model(X_batch)
            loss = criterion(output, y_batch)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
        avg_train_loss = train_loss / len(train_loader)
        train_losses.append(avg_train_loss)
        
        # ----------------- Evaluation -----------------
        model.eval()
        all_preds, all_labels = [], []
        with torch.no_grad():
            for X_batch, y_batch in test_loader:
                X_batch, y_batch = X_batch.to(device), y_batch.to(device)
                outputs = model(X_batch)
                _, predicted = torch.max(outputs, 1)
                all_preds.append(predicted.cpu().numpy())
                all_labels.append(y_batch.cpu().numpy())
        all_preds = np.concatenate(all_preds)
        all_labels = np.concatenate(all_labels)
        
        overall_acc = (all_preds == all_labels).mean()
        test_accuracies.append(overall_acc)
        
        # Per-subject accuracy
        subject_accs = {}
        for subj in unique_subjects:
            mask = test_subject_ids == subj
            acc = (all_preds[mask] == all_labels[mask]).mean()
            subject_accs[subj] = acc
            per_subject_acc_history[subj].append(acc)
        
        print(f"Epoch {epoch+1}/{epochs}: Train Loss={avg_train_loss:.4f}, Overall Test Acc={overall_acc:.4f}")
        print(f"Per-subject test accuracy (first 5 subjects): {dict(list(subject_accs.items())[:5])}")
    
    # ----------------- Plotting -----------------
    epochs_range = range(1, epochs+1)
    
    plt.figure(figsize=(16,5))
    
    # Training loss
    plt.subplot(1,3,1)
    plt.plot(epochs_range, train_losses, marker='o', color='blue')
    plt.xlabel("Epoch")
    plt.ylabel("Train Loss")
    plt.title("Training Loss Curve")
    plt.grid(True)
    
    # Overall test accuracy
    plt.subplot(1,3,2)
    plt.plot(epochs_range, test_accuracies, marker='o', color='green')
    plt.xlabel("Epoch")
    plt.ylabel("Test Accuracy")
    plt.title("Overall Test Accuracy Curve")
    plt.grid(True)
    
    # Per-subject accuracy
    plt.subplot(1,3,3)
    for subj in unique_subjects:
        plt.plot(epochs_range, per_subject_acc_history[subj], label=f"Subj {subj}")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.title("Per-Subject Test Accuracy")
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=8)
    plt.grid(True)
    
    plt.tight_layout()
    plt.show()

# -------------------------------
# 6. Main
# -------------------------------
if __name__ == "__main__":
    npz_file = "../dl_ins_results/combined_raw.npz"
    raw_X, labels, subject_ids = load_combined_raw(npz_file)
    
    X_train, y_train, X_test, y_test, train_subject_ids, test_subject_ids = split_subjects(raw_X, labels, subject_ids)
    
    train_dataset = EEGDataset(X_train, y_train)
    test_dataset  = EEGDataset(X_test, y_test)
    
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    test_loader  = DataLoader(test_dataset, batch_size=32, shuffle=False)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    model = RobustEEGCNN(num_classes=len(np.unique(labels))).to(device)

    train_model(model, train_loader, test_loader, test_subject_ids, y_train, device, epochs=10, lr=1e-3)
    print("Training complete!")