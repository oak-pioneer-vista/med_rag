#!/usr/bin/env bash
# Create/ensure Neo4j constraints, indexes, full-text indexes, and the
# semantic-type-derived tier labels on the UMLS graph.
#
# Idempotent — safe to re-run. Assumes the import has already happened
# (see scripts/load_neo4j.sh) and that the neo4j service is up.

set -euo pipefail

SERVICE="neo4j"
CONTAINER="med_rag-${SERVICE}-1"
NEO4J_PASSWORD="${NEO4J_PASSWORD:-medragpass}"

echo "==> waiting for bolt on $CONTAINER"
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

echo "done."
