import os
import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset
from model import BaseEEGCNN

# =====================
# Paths
# =====================
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DL_INS = os.path.dirname(os.path.dirname(SCRIPT_DIR))

DATA_PATH = os.path.join(DL_INS, "dl_ins_results", "combined_raw.npz")
CKPT_PATH = os.path.join(SCRIPT_DIR, "checkpoints", "cnn_ft_eesm19.pth")
OUT_PATH  = os.path.join(DL_INS, "dl_ins_results", "combined_embeddings.npz")

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
BATCH_SIZE = 64

# =====================
# Load data
# =====================
data = np.load(DATA_PATH)
X = torch.tensor(data["raw_X"], dtype=torch.float32)
y = data["labels"]
subjects = data["subject_ids"]

loader = DataLoader(TensorDataset(X), batch_size=BATCH_SIZE, shuffle=False)

# =====================
# Load model
# =====================
model = BaseEEGCNN().to(DEVICE)
ckpt = torch.load(CKPT_PATH, map_location=DEVICE)
model.load_state_dict(ckpt["model_state_dict"])
model.eval()

# =====================
# Extract embeddings
# =====================
embeddings = []

with torch.no_grad():
    for (xb,) in loader:
        xb = xb.to(DEVICE)
        feats = model.forward_features(xb)
        embeddings.append(feats.cpu().numpy())

embeddings = np.vstack(embeddings)

np.savez(
    OUT_PATH,
    X_emb=embeddings,
    y=y,
    subject_ids=subjects
)

print("Saved insomnia embeddings:", embeddings.shape)