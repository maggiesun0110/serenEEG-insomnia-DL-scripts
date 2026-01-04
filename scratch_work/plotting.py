import matplotlib.pyplot as plt

# Logged losses
epochs = list(range(1, 9))
losses = [
    0.123869,
    0.018380,
    0.006466,
    0.003767,
    0.002129,
    0.001504,
    0.001115,
    0.000970
]

plt.figure(figsize=(6, 4))
plt.plot(epochs, losses, marker='o', linewidth=2)

plt.xlabel("Epoch")
plt.ylabel("Binary Cross-Entropy Loss")
plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig("cnn_finetune_loss_eesm19.png", dpi=300)
plt.show()