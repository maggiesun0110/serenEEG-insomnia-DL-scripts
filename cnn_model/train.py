import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import precision_score, recall_score, classification_report
import matplotlib.pyplot as plt
from torch.utils.tensorboard import SummaryWriter
import datetime

# -------------------------------
# 1. Load & Normalize EEG data
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

    # Normalize each channel to zero mean and unit variance
    raw_X = (raw_X - np.mean(raw_X, axis=-1, keepdims=True)) / (np.std(raw_X, axis=-1, keepdims=True) + 1e-8)

    print(f"Data loaded successfully! Shapes:")
    print(f"raw_X: {raw_X.shape}, labels: {labels.shape}, subject_ids: {subject_ids.shape}")
    return raw_X, labels, subject_ids

# -------------------------------
# 2. Split dataset without leakage
# -------------------------------
def split_subjects(raw_X, labels, subject_ids, test_size=0.2, random_state=42):
    subject_ids = np.array([str(s) for s in subject_ids])
    unique_subjects = np.unique(subject_ids)

    train_subjects, test_subjects = train_test_split(unique_subjects, test_size=test_size, random_state=random_state)
    train_mask = np.isin(subject_ids, train_subjects)
    test_mask = np.isin(subject_ids, test_subjects)

    X_train, y_train = raw_X[train_mask], labels[train_mask]
    X_test, y_test = raw_X[test_mask], labels[test_mask]
    train_subject_ids = subject_ids[train_mask]
    test_subject_ids = subject_ids[test_mask]

    overlap = np.intersect1d(train_subject_ids, test_subject_ids)
    if len(overlap) > 0:
        print("WARNING: Subject leakage detected!", overlap)
    else:
        print("No subject overlap between train and test sets.")

    print(f"Train shape: {X_train.shape}, Test shape: {X_test.shape}")
    return X_train, y_train, X_test, y_test, train_subject_ids, test_subject_ids

# -------------------------------
# 3. Dataset class
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
# 4. Improved CNN Model
# -------------------------------
class RobustEEGCNN(nn.Module):
    def __init__(self, num_classes=2):
        super(RobustEEGCNN, self).__init__()
        self.conv1 = nn.Conv1d(3, 32, kernel_size=64, stride=16)
        self.bn1 = nn.BatchNorm1d(32)
        self.pool1 = nn.MaxPool1d(4)
        self.dropout1 = nn.Dropout(0.25)

        self.conv2 = nn.Conv1d(32, 64, kernel_size=3)
        self.bn2 = nn.BatchNorm1d(64)
        self.pool2 = nn.MaxPool1d(4)
        self.dropout2 = nn.Dropout(0.25)

        # Added third convolution block for deeper temporal understanding
        self.conv3 = nn.Conv1d(64, 128, kernel_size=3)
        self.bn3 = nn.BatchNorm1d(128)
        self.pool3 = nn.MaxPool1d(2)
        self.dropout3 = nn.Dropout(0.3)

        # Adaptive pooling avoids fixed feature map size
        self.global_pool = nn.AdaptiveAvgPool1d(1)

        self.fc1 = nn.Linear(128, 128)
        self.fc2 = nn.Linear(128, num_classes)
        self.relu = nn.ReLU()

        self._initialize_weights()

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, (nn.Conv1d, nn.Linear)):
                nn.init.kaiming_normal_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x):
        x = self.pool1(self.relu(self.bn1(self.conv1(x))))
        x = self.dropout1(x)
        x = self.pool2(self.relu(self.bn2(self.conv2(x))))
        x = self.dropout2(x)
        x = self.pool3(self.relu(self.bn3(self.conv3(x))))
        x = self.dropout3(x)
        x = self.global_pool(x).squeeze(-1)
        x = self.relu(self.fc1(x))
        x = self.fc2(x)
        return x

# -------------------------------
# 5. Training function
# -------------------------------
def train_model(model, train_loader, test_loader, test_subject_ids, y_train, device, epochs=60, lr=1e-3):
    classes = np.unique(y_train)
    class_weights = compute_class_weight(class_weight='balanced', classes=classes, y=y_train)
    class_weights = torch.tensor(class_weights, dtype=torch.float32).to(device)
    print(f"Class weights: {class_weights}")

    criterion = nn.CrossEntropyLoss(weight=class_weights, label_smoothing=0.1)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=3)

    writer = SummaryWriter(log_dir=f"runs/eeg_{datetime.datetime.now().strftime('%Y%m%d-%H%M%S')}")
    print("Training started...")

    best_acc = 0
    patience, patience_counter = 10, 0
    train_losses, test_accuracies = [], []
    unique_subjects = np.unique(test_subject_ids)

    for epoch in range(epochs):
        model.train()
        total_loss = 0
        for X_batch, y_batch in train_loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            optimizer.zero_grad()
            outputs = model(X_batch)
            loss = criterion(outputs, y_batch)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=2.0)
            optimizer.step()
            total_loss += loss.item()

        avg_loss = total_loss / len(train_loader)
        train_losses.append(avg_loss)

        # ---------- Evaluation ----------
        model.eval()
        all_preds, all_labels = [], []
        with torch.no_grad():
            for X_batch, y_batch in test_loader:
                X_batch, y_batch = X_batch.to(device), y_batch.to(device)
                outputs = model(X_batch)
                _, preds = torch.max(outputs, 1)
                all_preds.append(preds.cpu().numpy())
                all_labels.append(y_batch.cpu().numpy())

        all_preds = np.concatenate(all_preds)
        all_labels = np.concatenate(all_labels)
        overall_acc = (all_preds == all_labels).mean()
        test_accuracies.append(overall_acc)
        precision = precision_score(all_labels, all_preds, average='weighted', zero_division=0)
        recall = recall_score(all_labels, all_preds, average='weighted', zero_division=0)

        scheduler.step(overall_acc)
        writer.add_scalar("Loss/train", avg_loss, epoch)
        writer.add_scalar("Accuracy/test", overall_acc, epoch)

        print(f"Epoch {epoch+1}/{epochs} | Loss: {avg_loss:.4f} | Acc: {overall_acc:.4f} | Prec: {precision:.4f} | Rec: {recall:.4f}")

        if overall_acc > best_acc:
            best_acc = overall_acc
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print("Early stopping triggered.")
                break

    print("\nFinal Classification Report:")
    print(classification_report(all_labels, all_preds))

    plt.figure(figsize=(10,4))
    plt.subplot(1,2,1)
    plt.plot(train_losses, label='Train Loss')
    plt.legend(); plt.grid(); plt.title("Training Loss")

    plt.subplot(1,2,2)
    plt.plot(test_accuracies, label='Test Accuracy', color='green')
    plt.legend(); plt.grid(); plt.title("Test Accuracy")
    plt.show()
    writer.close()

# -------------------------------
# 6. Main
# -------------------------------
if __name__ == "__main__":
    npz_file = "../dl_ins_results/combined_raw.npz"
    raw_X, labels, subject_ids = load_combined_raw(npz_file)

    X_train, y_train, X_test, y_test, train_subject_ids, test_subject_ids = split_subjects(raw_X, labels, subject_ids)
    train_dataset = EEGDataset(X_train, y_train)
    test_dataset = EEGDataset(X_test, y_test)
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    model = RobustEEGCNN(num_classes=len(np.unique(labels))).to(device)

    train_model(model, train_loader, test_loader, test_subject_ids, y_train, device)