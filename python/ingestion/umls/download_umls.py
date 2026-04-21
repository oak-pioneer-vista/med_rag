"""Download the UMLS 2025AB Metathesaurus Full release from GCS.

Requires:
  - gcloud CLI authenticated with access to gs://med_rag
  - (`gcloud auth login` or a service account with storage.objects.get)

Usage:
  python python/ingestion/umls/download_umls.py
"""

import subprocess
import sys
from pathlib import Path

GCS_URI = "gs://med_rag/datasets/umls-2025AB-metathesaurus-full(1).zip"
DATA_DIR = Path(__file__).resolve().parent.parent.parent.parent / "data" / "umls"
DEST = DATA_DIR / "umls-2025AB-metathesaurus-full.zip"


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

    size_mb = DEST.stat().st_size / (1024 * 1024)
    print(f"Done. {DEST.name} ({size_mb:.1f} MB)")


if __name__ == "__main__":
    main()
