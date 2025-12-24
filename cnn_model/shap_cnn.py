import os
import shap
import torch
import numpy as np
import matplotlib.pyplot as plt

from train import load_raw, split_subjects
from model import BaseEEGCNN

# -----------------------------
# Reproducibility
# -----------------------------
torch.manual_seed(42)
np.random.seed(42)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", DEVICE)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
CHECKPOINT_PATH = os.path.join(BASE_DIR, "checkpoints", "sereneeg_cnn_v1.pth")

checkpoint = torch.load(CHECKPOINT_PATH, map_location=DEVICE)
THRESHOLD = checkpoint.get("threshold", 0.82)
print("Using fixed threshold:", THRESHOLD)

# -----------------------------
# Load data (subject-wise split)
# -----------------------------
raw_X, labels, subject_ids = load_raw("../dl_ins_results/combined_raw.npz")
X_train, y_train, X_test, y_test = split_subjects(raw_X, labels, subject_ids)

# -----------------------------
# Load locked CNN
# -----------------------------
model = BaseEEGCNN(num_classes=2).to(DEVICE)
model.load_state_dict(checkpoint["model_state_dict"])
model.eval()

# -----------------------------
# Background + explanation samples
# -----------------------------
background_idx = np.random.choice(len(X_train), size=50, replace=False)
background = torch.tensor(X_train[background_idx], dtype=torch.float32).to(DEVICE)

X_explain = torch.tensor(X_test[:20], dtype=torch.float32).to(DEVICE)

print("Background shape:", background.shape)  # (50, 3, 6000)
print("Explain shape:", X_explain.shape)      # (20, 3, 6000)

# -----------------------------
# SHAP computation
# -----------------------------
explainer = shap.GradientExplainer(model, background)
shap_values = explainer.shap_values(X_explain)

print("SHAP computed successfully.")

# shap can return:
# - list (per output) of arrays
# - single array
if isinstance(shap_values, list):
    shap_vals = shap_values[0]
else:
    shap_vals = shap_values

shap_vals = np.array(shap_vals)
print("Raw SHAP shape:", shap_vals.shape)

# -----------------------------
# Make shap_mean_samples be (channels, time)
# -----------------------------
# Common cases:
# A) (samples, channels, time)
# B) (channels, time, 1) or (channels, time)  <-- your case
# C) (samples, channels, time, 1)

if shap_vals.ndim == 4:
    # (samples, channels, time, outputs) -> drop outputs
    if shap_vals.shape[-1] == 1:
        shap_vals = shap_vals[..., 0]
    # now (samples, channels, time)
    shap_mean_samples = np.mean(np.abs(shap_vals), axis=0)  # (channels, time)

elif shap_vals.ndim == 3:
    # Either (samples, channels, time) OR (channels, time, 1)
    if shap_vals.shape[0] == X_explain.shape[0]:
        # (samples, channels, time)
        shap_mean_samples = np.mean(np.abs(shap_vals), axis=0)  # (channels, time)
    else:
        # assume (channels, time, outputs) or (channels, time, 1)
        if shap_vals.shape[-1] == 1:
            shap_vals = shap_vals[..., 0]  # (channels, time)
        shap_mean_samples = np.abs(shap_vals)  # already (channels, time)

elif shap_vals.ndim == 2:
    # (channels, time) (rare but possible)
    shap_mean_samples = np.abs(shap_vals)

else:
    raise ValueError(f"Unexpected SHAP shape: {shap_vals.shape}")

print("shap_mean_samples shape (should be channels x time):", shap_mean_samples.shape)

# Sanity check: channels should be 3
if shap_mean_samples.shape[0] != 3:
    raise ValueError(
        f"Expected 3 channels, got {shap_mean_samples.shape[0]}. "
        f"Check SHAP axis ordering: shap_mean_samples.shape={shap_mean_samples.shape}"
    )

# -----------------------------
# Channel importance (channels,)
# -----------------------------
channel_importance = shap_mean_samples.mean(axis=1)  # mean over time -> (channels,)
channel_importance = channel_importance / (channel_importance.sum() + 1e-12)  # normalize

channel_names = ["F4-A1", "C4-A1", "O2-A1"]

print("Final channel importance shape:", channel_importance.shape)

for ch, val in zip(channel_names, channel_importance):
    print(f"{ch}: {val:.6f}")

# -----------------------------
# Bar plot (paper-friendly)
# -----------------------------
plt.figure(figsize=(6, 4))
plt.bar(channel_names, channel_importance)
plt.ylabel("Normalized Mean |SHAP|")
plt.xlabel("EEG Channel")
plt.title("Channel-level SHAP Importance (CNN)")
plt.tight_layout()
plt.savefig(os.path.join(BASE_DIR, "shap_channel_importance_cnn.png"), dpi=300, bbox_inches="tight")
plt.show()

# Save channel importance + metadata safely
np.savez(
    os.path.join(BASE_DIR, "shap_channel_importance.npz"),
    channels=np.array(channel_names),
    importance=channel_importance,
)

# -----------------------------
# Temporal heatmaps
# -----------------------------
plt.figure(figsize=(14, 6))
for i, ch in enumerate(channel_names):
    plt.subplot(3, 1, i + 1)
    plt.imshow(
        shap_mean_samples[i][None, :],
        aspect="auto",
        cmap="hot"
    )
    plt.colorbar(label="Mean |SHAP|")
    plt.ylabel(ch)
    plt.xlabel("Time (samples)")

plt.suptitle("Channel-wise Temporal SHAP Heatmaps (CNN)")
plt.tight_layout()
plt.savefig(os.path.join(BASE_DIR, "shap_heatmaps_cnn.png"), dpi=300, bbox_inches="tight")
plt.show()

# Save raw SHAP values
np.save(os.path.join(BASE_DIR, "shap_values.npy"), shap_vals)

print("SHAP plots + values saved.")