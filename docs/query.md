# Neo4j queries

`may_treat` is stored as a `rela` property on generic `:RELATES` edges (source: MED-RT). Use the `concept_name_fts` fulltext index to anchor on a drug name, then traverse — a direct `toLower(name) CONTAINS ...` scans all ~3.3M `:Concept` nodes.

```cypher
CALL db.index.fulltext.queryNodes('concept_name_fts', 'risperidone') YIELD node
WITH node WHERE node.name = 'risperidone'
MATCH (node)-[r:RELATES {rela:'may_treat'}]-(t:Concept)
RETURN node.name AS drug, t.name AS target, r.sab AS sab
ORDER BY target;
```

Run from the host:

```bash
docker exec med_rag-neo4j-1 cypher-shell -u neo4j -p medragpass "<query>"
```
