import os
import h5py
import numpy as np

def load_h5(filepath):
    with h5py.File(filepath, 'r') as f:
        X = np.array(f['X'])
        y = np.array(f['y'])
    return X, y

def combine_h5(folder, output_file="combined.h5"):
    X_list, y_list = [], []
    channel_orders = []

    for file in sorted(os.listdir(folder)):
        if file.endswith(".h5"):
            path = os.path.join(folder, file)
            X, y = load_h5(path)

            print(f"{file}: X={X.shape}, y={y.shape}")

            if X.size == 0 or y.size == 0:
                print(f"⚠️ Skipping empty file: {file}")
                continue

            X_list.append(X)
            y_list.append(y)

            # Record channel/feature dimensions
            channel_orders.append(X.shape[1:])

    if not X_list:
        raise ValueError("❌ No valid data found in folder.")

    # Check consistency of channels/features
    if len(set(channel_orders)) != 1:
        print("⚠️ Inconsistent channel/features across datasets:")
        print(channel_orders)
        raise ValueError("Mismatch in channel/features order")

    # Concatenate
    X_all = np.concatenate(X_list, axis=0)
    y_all = np.concatenate(y_list, axis=0)

    # Save combined
    with h5py.File(output_file, "w") as f:
        f.create_dataset("X", data=X_all)
        f.create_dataset("y", data=y_all)

    print(f"\n✅ Combined shape: X={X_all.shape}, y={y_all.shape}")
    print("Label distribution:", {label: np.sum(y_all == label) for label in np.unique(y_all)})

    return X_all, y_all

if __name__ == "__main__":
    combine_h5("/Users/maggiesun/Downloads/research/SerenEEG/dl_ins/dl_ins_results/h5_data")