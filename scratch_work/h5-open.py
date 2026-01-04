import numpy as np

def load_combined_raw(npz_path):
    """
    Safely load 'raw_X', 'labels', and 'subject_ids' from a .npz file.
    Returns a tuple: (raw_X, labels, subject_ids)
    """
    try:
        data = np.load(npz_path, allow_pickle=True)
    except FileNotFoundError:
        raise FileNotFoundError(f"File not found: {npz_path}")
    
    print(f"Keys in the npz file: {data.files}")
    
    # Try loading directly
    keys_to_find = ['raw_X', 'labels', 'subject_ids']
    loaded = {}
    for key in keys_to_find:
        if key in data:
            loaded[key] = data[key]
    
    # If any key is missing, maybe it's stored inside a dict
    missing_keys = [k for k in keys_to_find if k not in loaded]
    if missing_keys:
        # Try checking if there is only one item which is a dict
        if len(data.files) == 1:
            first_key = data.files[0]
            possible_dict = data[first_key].item() if hasattr(data[first_key], "item") else None
            if isinstance(possible_dict, dict):
                for k in missing_keys:
                    if k in possible_dict:
                        loaded[k] = possible_dict[k]
    
    # Final check
    for k in keys_to_find:
        if k not in loaded:
            raise KeyError(f"Could not find key '{k}' in the npz file or inside a dict.")
    
    return loaded['raw_X'], loaded['labels'], loaded['subject_ids']


if __name__ == "__main__":
    npz_file = "../dl_ins_results/combined_raw.npz"  # replace with your path
    raw_X, labels, subject_ids = load_combined_raw(npz_file)
    print("Data loaded successfully!")
    print(f"raw_X shape: {raw_X.shape}")
    print(f"labels shape: {labels.shape}")
    print(f"subject_ids shape: {subject_ids.shape}")