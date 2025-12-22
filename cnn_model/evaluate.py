import torch
from model import BaseEEGCNN
from config import MODEL_CHECKPOINT, DECISION_THRESHOLD
import device

checkpoint = torch.load(MODEL_CHECKPOINT, map_location=device)

model = BaseEEGCNN()
model.load_state_dict(checkpoint["model_state_dict"])
model.to(device)
model.eval()

threshold = checkpoint["threshold"]

probs = torch.sigmoid(out).squeeze(1)
preds = (probs > threshold).long()