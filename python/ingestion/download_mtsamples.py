"""Download the MTSamples (Medical Transcriptions) dataset from GCS.

Requires:
  - gcloud CLI authenticated with access to gs://med_rag
  - (`gcloud auth login` or a service account with storage.objects.get)

Usage:
  python python/ingestion/download_mtsamples.py
"""

import subprocess
import sys
from pathlib import Path

GCS_URI = "gs://med_rag/datasets/mtsamples.csv"
DATA_DIR = Path(__file__).resolve().parent.parent.parent / "data" / "kaggle" / "mtsamples"
DEST = DATA_DIR / "mtsamples.csv"


def main():
    DATA_DIR.mkdir(parents=True, exist_ok=True)

    print(f"Downloading {GCS_URI} → {DEST}")
    result = subprocess.run(
        ["gsutil", "cp", GCS_URI, str(DEST)],
        capture_output=True,
        text=True,
    )

    if result.returncode != 0:
        print(f"Error: {result.stderr.strip()}", file=sys.stderr)
        sys.exit(1)

    size_kb = DEST.stat().st_size / 1024
    print(f"Done. {DEST.name} ({size_kb:.1f} KB)")


if __name__ == "__main__":
    main()
