#!/usr/bin/env bash
# Bulk-load UMLS CSVs into the Neo4j container.
#
# Prereqs:
#   - docker compose up -d neo4j  (container defined, can be running or stopped)
#   - data/neo4j_import/ populated by scripts/umls_to_neo4j_csv.py
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
    semantic_types.csv
    sources.csv
    concept_semtype.csv
    concept_relates.csv
    concept_parent.csv
    concept_definition.csv
)

for f in "${required_files[@]}"; do
    if [[ ! -f "$IMPORT_DIR/$f" ]]; then
        echo "missing $IMPORT_DIR/$f — run scripts/umls_to_neo4j_csv.py first" >&2
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
    --nodes=/import/semantic_types.csv \
    --nodes=/import/sources.csv \
    --relationships=/import/concept_semtype.csv \
    --relationships=/import/concept_relates.csv \
    --relationships=/import/concept_parent.csv \
    --relationships=/import/concept_definition.csv

echo "==> starting $SERVICE"
docker compose up -d "$SERVICE"

NEO4J_PASSWORD="${NEO4J_PASSWORD:-medragpass}"
CONTAINER="med_rag-${SERVICE}-1"

echo "==> waiting for bolt"
until docker exec "$CONTAINER" cypher-shell -u neo4j -p "$NEO4J_PASSWORD" "RETURN 1" >/dev/null 2>&1; do
    sleep 2
done

echo "==> creating constraints and indexes"
docker exec "$CONTAINER" cypher-shell -u neo4j -p "$NEO4J_PASSWORD" <<'CYPHER'
CREATE CONSTRAINT concept_cui_unique IF NOT EXISTS FOR (c:Concept) REQUIRE c.cui IS UNIQUE;
CREATE CONSTRAINT semtype_tui_unique IF NOT EXISTS FOR (s:SemanticType) REQUIRE s.tui IS UNIQUE;
CREATE CONSTRAINT source_sab_unique  IF NOT EXISTS FOR (s:Source) REQUIRE s.sab IS UNIQUE;
CREATE INDEX concept_name IF NOT EXISTS FOR (c:Concept) ON (c.name);
CREATE FULLTEXT INDEX concept_name_fts IF NOT EXISTS FOR (c:Concept) ON EACH [c.name];
CYPHER

echo "done. Browser: http://localhost:7474 (neo4j / $NEO4J_PASSWORD)"
