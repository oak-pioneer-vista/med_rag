"""Download the MTSamples (Medical Transcriptions) dataset from Kaggle.

Requires:
  - kaggle CLI installed (`pip install kaggle`)
  - Kaggle API credentials at ~/.kaggle/kaggle.json

Usage:
  python scripts/download_mtsamples.py
"""

import subprocess
import sys
from pathlib import Path

DATASET = "tboyle10/medicaltranscriptions"
DATA_DIR = Path(__file__).resolve().parent.parent / "data" / "kaggle" / "mtsamples"


def main():
    DATA_DIR.mkdir(parents=True, exist_ok=True)

    print(f"Downloading {DATASET} → {DATA_DIR}")
    result = subprocess.run(
        [
            "kaggle",
            "datasets",
            "download",
            "-d",
            DATASET,
            "-p",
            str(DATA_DIR),
            "--unzip",
        ],
        capture_output=True,
        text=True,
    )

    if result.returncode != 0:
        print(f"Error: {result.stderr.strip()}", file=sys.stderr)
        sys.exit(1)

    files = list(DATA_DIR.iterdir())
    print(f"Done. {len(files)} file(s) in {DATA_DIR}:")
    for f in sorted(files):
        print(f"  {f.name}  ({f.stat().st_size / 1024:.1f} KB)")


if __name__ == "__main__":
    main()
