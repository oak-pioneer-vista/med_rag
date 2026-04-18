#!/usr/bin/env bash
# Bulk-load UMLS CSVs into the Neo4j container.
#
# Prereqs:
#   - docker compose up -d neo4j  (container defined, can be running or stopped)
#   - data/neo4j_import/ populated by python/ingestion/umls_to_neo4j_csv.py
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
        echo "missing $IMPORT_DIR/$f — run python/ingestion/umls_to_neo4j_csv.py first" >&2
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
CONTAINER="med_rag-${SERVICE}-1"

echo "==> waiting for bolt"
until docker exec "$CONTAINER" cypher-shell -u neo4j -p "$NEO4J_PASSWORD" "RETURN 1" >/dev/null 2>&1; do
    sleep 2
done

echo "==> creating constraints and indexes"
docker exec -i "$CONTAINER" cypher-shell -u neo4j -p "$NEO4J_PASSWORD" <<'CYPHER'
CREATE CONSTRAINT concept_cui_unique IF NOT EXISTS FOR (c:Concept) REQUIRE c.cui IS UNIQUE;
CREATE CONSTRAINT atom_aui_unique    IF NOT EXISTS FOR (a:Atom)    REQUIRE a.aui IS UNIQUE;
CREATE CONSTRAINT semtype_tui_unique IF NOT EXISTS FOR (s:SemanticType) REQUIRE s.tui IS UNIQUE;
CREATE CONSTRAINT source_sab_unique  IF NOT EXISTS FOR (s:Source) REQUIRE s.sab IS UNIQUE;
CREATE INDEX concept_name IF NOT EXISTS FOR (c:Concept) ON (c.name);
CREATE INDEX atom_sab_code IF NOT EXISTS FOR (a:Atom) ON (a.sab, a.code);
CREATE FULLTEXT INDEX concept_name_fts IF NOT EXISTS FOR (c:Concept) ON EACH [c.name];
CREATE FULLTEXT INDEX atom_str_fts IF NOT EXISTS FOR (a:Atom) ON EACH [a.str];
CYPHER

echo "==> applying tier labels (ClinicalCore / ClinicalSupport / ClinicalDiscipline / Peripheral)"
docker exec -i "$CONTAINER" cypher-shell -u neo4j -p "$NEO4J_PASSWORD" <<'CYPHER'
MATCH (c:Concept)-[:HAS_SEMTYPE]->(s:SemanticType)
WHERE s.name IN ['Disease or Syndrome','Pharmacologic Substance',
                 'Therapeutic or Preventive Procedure',
                 'Body Part, Organ, or Organ Component']
WITH DISTINCT c
CALL { WITH c SET c:ClinicalCore } IN TRANSACTIONS OF 50000 ROWS;

MATCH (c:Concept)-[:HAS_SEMTYPE]->(s:SemanticType)
WHERE s.name IN ['Medical Device','Laboratory or Test Result','Finding','Food']
WITH DISTINCT c
CALL { WITH c SET c:ClinicalSupport } IN TRANSACTIONS OF 50000 ROWS;

// Provider / discipline axis: who practices medicine and where.
// Covers specialties (Cardiology), provider groups (Physician), and
// care organizations (Hospital, Professional Society).
MATCH (c:Concept)-[:HAS_SEMTYPE]->(s:SemanticType)
WHERE s.name IN ['Biomedical Occupation or Discipline',
                 'Occupation or Discipline',
                 'Professional or Occupational Group',
                 'Health Care Related Organization',
                 'Organization',
                 'Professional Society',
                 'Self-help or Relief Organization']
WITH DISTINCT c
CALL { WITH c SET c:ClinicalDiscipline } IN TRANSACTIONS OF 50000 ROWS;

MATCH (c:Concept)
WHERE NOT c:ClinicalCore AND NOT c:ClinicalSupport AND NOT c:ClinicalDiscipline
CALL { WITH c SET c:Peripheral } IN TRANSACTIONS OF 50000 ROWS;
CYPHER

echo "done. Browser: http://localhost:7474 (neo4j / $NEO4J_PASSWORD)"
