#!/usr/bin/env bash
# Bulk-load UMLS CSVs into the Neo4j container.
#
# Prereqs:
#   - docker compose up -d neo4j  (container defined, can be running or stopped)
#   - data/neo4j_import/ populated by python/ingestion/umls/umls_to_neo4j_csv.py
#   - docker-compose.yml mounts ./data/neo4j_import -> /import in the neo4j service
#
# Runs `neo4j-admin database import full` against the 'neo4j' database,
# overwriting any existing contents. Neo4j must be stopped during import.

set -euo pipefail

cd "$(dirname "$0")/.."

IMPORT_DIR="data/neo4j_import"
SERVICE="neo4j"
DB="neo4j"

required_files=(
    concepts.csv
    atoms.csv
    semantic_types.csv
    sources.csv
    concept_atom.csv
    concept_semtype.csv
    concept_relates.csv
    concept_parent.csv
    concept_definition.csv
)

for f in "${required_files[@]}"; do
    if [[ ! -f "$IMPORT_DIR/$f" ]]; then
        echo "missing $IMPORT_DIR/$f — run python/ingestion/umls/umls_to_neo4j_csv.py first" >&2
        exit 1
    fi
done

echo "==> stopping $SERVICE (release lock on /data)"
docker compose stop "$SERVICE" >/dev/null

echo "==> running neo4j-admin database import"
docker compose run --rm --no-deps --entrypoint neo4j-admin "$SERVICE" \
    database import full "$DB" \
    --overwrite-destination \
    --threads=32 \
    --high-parallel-io=on \
    --nodes=/import/concepts.csv \
    --nodes=/import/atoms.csv \
    --nodes=/import/semantic_types.csv \
    --nodes=/import/sources.csv \
    --relationships=/import/concept_atom.csv \
    --relationships=/import/concept_semtype.csv \
    --relationships=/import/concept_relates.csv \
    --relationships=/import/concept_parent.csv \
    --relationships=/import/concept_definition.csv

echo "==> starting $SERVICE"
docker compose up -d "$SERVICE"

NEO4J_PASSWORD="${NEO4J_PASSWORD:-medragpass}"

bash "$(dirname "$0")/create_neo4j_indices.sh"

echo "done. Browser: http://localhost:7474 (neo4j / $NEO4J_PASSWORD)"
