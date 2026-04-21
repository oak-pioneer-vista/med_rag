"""Download the UMLS SPECIALIST Lexicon LRABR table from GCS.

LRABR is a purpose-built abbreviation/acronym list maintained by NLM's
Lexical Systems Group. We pull it because it is substantially cleaner
than deriving the same mapping from MRCONSO's AB/ACR atoms -- those are
dominated by NCI chemo-protocol names and HGNC gene symbols, which
resolve clinical notes to wrong senses (e.g. BP -> bleomycin/cisplatin
protocol, CPK -> PIK3C2A gene).

File format: pipe-delimited, UTF-8, one entry per line:

    EUI1|abbreviation|type|EUI2|expansion|

where `type` is `acronym` or `abbreviation` and the EUIs point into the
SPECIALIST Lexicon's inflected-form table. build_abbreviations.py only
needs columns 2 and 5.

Upstream source (public, no UMLS auth required):
  https://data.lhncbc.nlm.nih.gov/public/lsg/lexicon/2026/release/LEX_DOC/LRABR

Mirrored to gs://med_rag/datasets/ so this downloader stays consistent
with download_umls.py / download_mtsamples.py (single-auth path via
gcloud rather than mixing HTTP and GCS fetches).

Requires:
  - gcloud CLI authenticated with access to gs://med_rag
  - (`gcloud auth login` or a service account with storage.objects.get)

Usage:
  python python/ingestion/umls/download_specialist_lexicon.py
"""

import subprocess
import sys
from pathlib import Path

GCS_URI = "gs://med_rag/datasets/umls-2026-lrabr.txt"
DATA_DIR = Path(__file__).resolve().parent.parent.parent.parent / "data" / "umls"
DEST = DATA_DIR / "LRABR"


def main():
    DATA_DIR.mkdir(parents=True, exist_ok=True)

    print(f"Downloading {GCS_URI} -> {DEST}")
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
