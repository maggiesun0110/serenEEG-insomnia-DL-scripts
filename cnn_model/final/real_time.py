import torch
import torch.nn as nn
import numpy as np
import time
import joblib
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from model import BaseEEGCNN

device = torch.device("cpu")

cnn_model = BaseEEGCNN()

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
checkpoint_path = os.path.join(BASE_DIR, "checkpoints", "sereneeg_cnn_v1.pth")
rf_path = os.path.join(BASE_DIR, "results", "rf_cnn_embeddings.joblib")

checkpoint = torch.load(checkpoint_path, map_location=device)

cnn_model.load_state_dict(checkpoint["model_state_dict"])
cnn_model.to(device)
cnn_model.eval()

rf_model = joblib.load(rf_path)

# 3. CREATE DUMMY INPUT

# One 30s EEG window: (batch, channels, samples)
dummy_epoch = torch.randn(1, 3, 6000).to(device)

# 4. BENCHMARK FUNCTION

def benchmark(fn, runs=200, warmup=20):
    # Warmup
    for _ in range(warmup):
        fn()

    start = time.perf_counter()
    for _ in range(runs):
        fn()
    end = time.perf_counter()

    avg_ms = (end - start) / runs * 1000
    return avg_ms


# 5. CNN EMBEDDING TIMING

def run_cnn():
    with torch.no_grad():
        _ = cnn_model.forward_features(dummy_epoch)

cnn_latency = benchmark(run_cnn)
print(f"CNN embedding latency per 30s window: {cnn_latency:.4f} ms")


# 6. RF TIMING

dummy_features = np.random.randn(1, 128)

def run_rf():
    _ = rf_model.predict(dummy_features)

rf_latency = benchmark(run_rf, runs=1000)
print(f"RF latency: {rf_latency:.6f} ms")


# 7. FULL PIPELINE TIMING

def run_full():
    with torch.no_grad():
        emb = cnn_model.forward_features(dummy_epoch)

    emb_np = emb.numpy()

    # simulate aggregation (mean + std)
    mean_emb = emb_np
    std_emb = np.zeros_like(emb_np)

    subject_features = np.concatenate([mean_emb, std_emb], axis=1)

    _ = rf_model.predict(subject_features)

full_latency = benchmark(run_full)
print(f"Full pipeline latency: {full_latency:.4f} ms")


# 8. REAL-TIME FACTOR

# Each window represents 30 seconds = 30000 ms
real_time_factor = 30000 / full_latency

print(f"Speed relative to real-time: {real_time_factor:.1f}x faster than real-time")