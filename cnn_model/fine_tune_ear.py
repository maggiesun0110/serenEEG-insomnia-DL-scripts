import os
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

from model import BaseEEGCNN

# =====================
# Paths (ROBUST)
# =====================
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))          # cnn_model/
DL_INS_SCRIPTS = os.path.dirname(SCRIPT_DIR)                     # dl_ins_scripts/
DL_INS = os.path.dirname(DL_INS_SCRIPTS)                         # dl_ins/

DATA_PATH = os.path.join(
    DL_INS,
    "dl_ins_results",
    "eesm19_raw.npz"
)

BASE_CKPT = os.path.join(
    SCRIPT_DIR,
    "checkpoints",
    "sereneeg_cnn_v1.pth"
)

OUT_CKPT = os.path.join(
    SCRIPT_DIR,
    "checkpoints",
    "cnn_ft_eesm19.pth"
)

# =====================
# Training config
# =====================
BATCH_SIZE = 32
EPOCHS = 8
LR = 1e-4
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

torch.manual_seed(42)
np.random.seed(42)

print("Using device:", DEVICE)

# =====================
# Dataset
# =====================
class EEGDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.float32)

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


# =====================
# Load EESM19 data
# =====================
data = np.load(DATA_PATH)
X = data["raw_X"]          # (N, 3, 6000)
y = data["labels"]         # all 1s (healthy domain)

print("Loaded EESM19:", X.shape, y.shape)
# Remove bad epochs (NaN or Inf)
mask = np.isfinite(X).all(axis=(1, 2))
X = X[mask]
y = y[mask]

print("After cleaning:", X.shape, y.shape)

dataset = EEGDataset(X, y)
loader = DataLoader(
    dataset,
    batch_size=BATCH_SIZE,
    shuffle=True,
    drop_last=True
)

# =====================
# Load base CNN
# =====================
model = BaseEEGCNN(num_classes=2).to(DEVICE)

ckpt = torch.load(BASE_CKPT, map_location=DEVICE)

# handle both raw state_dict or full checkpoint
if isinstance(ckpt, dict) and "model_state_dict" in ckpt:
    model.load_state_dict(ckpt["model_state_dict"])
else:
    model.load_state_dict(ckpt)

print("Loaded base CNN checkpoint.")

# =====================
# Freeze feature extractor
# =====================
for name, param in model.named_parameters():
    if not name.startswith("fc"):
        param.requires_grad = False

print("\nTrainable parameters:")
for name, param in model.named_parameters():
    if param.requires_grad:
        print("  ", name)

# =====================
# Training setup
# =====================
criterion = nn.BCEWithLogitsLoss()

optimizer = torch.optim.Adam(
    filter(lambda p: p.requires_grad, model.parameters()),
    lr=LR
)

# =====================
# Fine-tuning loop
# =====================
model.train()

for epoch in range(EPOCHS):
    total_loss = 0.0

    for Xb, yb in loader:
        Xb = Xb.to(DEVICE)
        yb = yb.to(DEVICE).unsqueeze(1)  # (B, 1)

        optimizer.zero_grad()
        logits = model(Xb)
        loss = criterion(logits, yb)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    avg_loss = total_loss / len(loader)
    print(f"Epoch {epoch+1}/{EPOCHS} | Loss: {avg_loss:.6f}")

# =====================
# Save fine-tuned model
# =====================
torch.save(
    {
        "model_state_dict": model.state_dict(),
        "source": "sereneeg_cnn_v1",
        "target_domain": "EESM19 ear-EEG",
        "epochs": EPOCHS,
        "lr": LR
    },
    OUT_CKPT
)

print("\nSaved fine-tuned model →", OUT_CKPT)