#!/usr/bin/env bash
# Unzip UMLS metathesaurus and convert RRF files to Neo4j admin-import CSVs.
#
# Prereq:
#   - data/umls/umls-2025AB-metathesaurus-full.zip  (download_umls.py)

set -euo pipefail

cd "$(dirname "$0")/.."

ZIP="data/umls/umls-2025AB-metathesaurus-full.zip"
EXTRACT_DIR="data/umls"
META_DIR="$EXTRACT_DIR/2025AB/META"
OUT_DIR="data/neo4j_import"

if [[ ! -f "$ZIP" ]]; then
    echo "missing $ZIP — run python/ingestion/download_umls.py first" >&2
    exit 1
fi

if [[ ! -f "$META_DIR/MRCONSO.RRF" ]]; then
    echo "==> unzipping $ZIP -> $EXTRACT_DIR"
    python3 -c "import zipfile, sys; zipfile.ZipFile(sys.argv[1]).extractall(sys.argv[2])" \
        "$ZIP" "$EXTRACT_DIR"
else
    echo "==> META already extracted at $META_DIR, skipping unzip"
fi

echo "==> converting RRF -> CSV (out=$OUT_DIR)"
python3 python/ingestion/umls_to_neo4j_csv.py \
    --meta "$META_DIR" \
    --out  "$OUT_DIR" \
    --english-only --drop-suppressed
