import torch
import numpy as np
from torch.utils.data import Dataset

class EEGDataset(Dataset):
    """
    SerenEEG Dataset for CNNs using combined_raw.npz.
    
    File must contain:
        - 'raw_X': EEG data, shape (n_epochs, 3, n_samples)
        - 'labels': labels, shape (n_epochs,)
        - 'subject_ids': array of subject IDs (n_epochs,)
    """

    def __init__(self, npz_path, transform=None):
        data = np.load(npz_path)
        self.X = data["raw_X"]   # (n_epochs, channels=3, samples)
        self.y = data["labels"]
        self.subject_ids = data["subject_ids"]
        self.transform = transform

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        X = self.X[idx]  # (channels, samples)
        y = self.y[idx]
        sid = self.subject_ids[idx]

        # Convert to tensors
        X = torch.tensor(X, dtype=torch.float32)  # shape: (3, n_samples)
        y = torch.tensor(y, dtype=torch.long)

        if self.transform:
            X = self.transform(X)

        return X, y, sid   # return subject_id too for per-subject eval if needed