# med_rag — Claude guidance

## Neo4j query patterns

The graph has ~3.3M `:Concept` nodes. A RANGE index on `Concept.name` exists but is defeated by `toLower()` or `CONTAINS`. Fulltext indexes `concept_name_fts` and `atom_str_fts` exist for substring / case-insensitive lookup.

Prefer:
- **Known exact name** → `MATCH (c:Concept {name:'risperidone'})`
- **Substring / fuzzy / case-insensitive** → `CALL db.index.fulltext.queryNodes('concept_name_fts', 'risperidone') YIELD node`
- **Atoms** → use `atom_str_fts`

Avoid:
- `WHERE toLower(c.name) CONTAINS '...'` on `:Concept` — full label scan, ~30s.
- OR across two endpoints (`a.name CONTAINS x OR b.name CONTAINS x`) — anchor on one side, then traverse.

Run `SHOW INDEXES` before writing exploratory Cypher when unsure.

## Graph shape notes

- `may_treat` is stored as `rela` on generic `:RELATES` edges (source: MED-RT, ~15k edges). It is not its own relationship type.
- MED-RT `may_treat` edges are materialized in **both directions** in the graph.
- Running Neo4j container name: `med_rag-neo4j-1`. Auth: `neo4j / medragpass` (see `docker-compose.yml`).
