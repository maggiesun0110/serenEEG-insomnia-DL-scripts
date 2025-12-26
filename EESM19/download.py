import os
import requests
from pathlib import Path

# ===== CONFIG =====
DATASET_ID = "ds005185"
VERSION = "1.0.2"

BASE_URL = f"https://openneuro.org/crn/datasets/{DATASET_ID}/versions/{VERSION}/files"

DEST_ROOT = Path("/Volumes/ORICO/SerenEEG/data/EESM19")

START_SUB = 4
END_SUB = 20

TIMEOUT = 30
# ==================

def download_file(url, out_path):
    """Download url to out_path if not already present."""
    if out_path.exists():
        print(f"[SKIP] {out_path.name} (exists)")
        return

    print(f"[DOWNLOAD] {out_path.name}")
    try:
        r = requests.get(url, stream=True, timeout=TIMEOUT)
    except Exception as e:
        print(f"  ⚠ failed request: {e}")
        return

    if r.status_code != 200:
        print(f"  ⚠ not found: {url}")
        return

    out_path.parent.mkdir(parents=True, exist_ok=True)

    with open(out_path, "wb") as f:
        for chunk in r.iter_content(chunk_size=8192):
            if chunk:
                f.write(chunk)

    print(f"  ✓ saved")

print("\n📥 Downloading EESM19 ear-EEG sleep data (Python only)\n")

for sid in range(START_SUB, END_SUB + 1):
    sub = f"sub-{sid:03d}"
    print(f"\n🔹 {sub}")

    # Try session numbers 001 → 020
    for ses in range(1, 21):
        ses_str = f"ses-{ses:03d}"

        # For both .set and .fdt file parts
        for ext in ["set", "fdt"]:
            fname = f"{sub}_{ses_str}_task-sleep_acq-earEEG_eeg.{ext}"

            url = f"{BASE_URL}/{sub}:{fname}"
            out_path = DEST_ROOT / sub / "eeg" / fname

            download_file(url, out_path)

print("\n✅ Done (any ear-EEG sleep files were downloaded).")