import os
import mne
from collections import Counter, defaultdict

BASE_PATH = "/Volumes/ORICO/SerenEEG/data/EESM19_tmp"

SUBJECTS = [f"sub-{i:03d}" for i in range(4, 21)]
TARGET_SUFFIX = "_task-sleep_acq-earEEG_eeg.set"

channel_counts = Counter()
file_count = 0
per_file_missing = defaultdict(list)

for subj in SUBJECTS:
    subj_dir = os.path.join(BASE_PATH, subj)
    if not os.path.isdir(subj_dir):
        print("Missing subject folder:", subj)
        continue

    for root, _, files in os.walk(subj_dir):
        for fn in files:
            if not fn.endswith(TARGET_SUFFIX):
                continue

            path = os.path.join(root, fn)
            raw = mne.io.read_raw_eeglab(path, preload=False, verbose=False)

            file_count += 1
            chs = list(raw.ch_names)

            for ch in chs:
                channel_counts[ch] += 1

            # track missing of common candidates
            candidates = ["ELA","ELB","ELC","ELE","ELI","ELT","ERA","ERB","ERC","ERE","ERI","ERT","EOGr"]
            missing = [c for c in candidates if c not in chs]
            if missing:
                per_file_missing[tuple(missing)].append(path)

print("\n=== SUMMARY ===")
print("Total earEEG sleep files scanned:", file_count)

print("\nChannel presence (count / files):")
for ch, c in channel_counts.most_common():
    print(f"{ch:5s}: {c}/{file_count}")

print("\nMost common missing-sets (top 10):")
for miss, paths in sorted(per_file_missing.items(), key=lambda x: len(x[1]), reverse=True)[:10]:
    print(f"Missing {miss} -> {len(paths)} files")