import mne
import os

BASE_PATH = "/Users/maggiesun/downloads/research/serenEEG/data/CapSleep"
files_to_check = ["ins7 (1).edf", "ins8 (1).edf", "ins9 (1).edf"]

for fname in files_to_check:
    subj_path = os.path.join(BASE_PATH, fname)
    print(f"\n--- {fname} ---")
    try:
        raw = mne.io.read_raw_edf(subj_path, preload=False, verbose="ERROR")
        print("Channels:", raw.ch_names)
        print("Sampling frequency:", raw.info.get("sfreq"))
        print("Duration (s):", raw.n_times / raw.info.get("sfreq", 1))
    except Exception as e:
        print("Failed to load:", e)