# med_rag

Medical RAG (Retrieval-Augmented Generation) pipeline built on the MTSamples medical transcription dataset, backed by Qdrant for dense retrieval and Neo4j for a UMLS-derived clinical knowledge graph.

## What's in here

- `python/ingestion/` — dataset downloaders, MTSamples parser, UMLS → Neo4j CSV converter, and the MedTE/TEI embedding pipeline.
- `scripts/` — shell wrappers for the UMLS prepare + bulk-load steps.
- `docker-compose.yml` — Qdrant, Neo4j, and the MedTE text-embeddings-inference (TEI) service on the host GPU.

## Docs

- [**Ingestion pipeline**](docs/ingestion.md) — prerequisites, setup, and the step-by-step commands to build the Qdrant section-embedding collection and the Neo4j UMLS graph.
- [**Learning notes**](docs/learning_notes.md) — non-obvious things learned while building this: short-query degradation on sentence-level encoders, TEI vs. in-process transformers trade-offs, pooling gotchas.
- [**Neo4j queries**](docs/query.md) — sample Cypher queries against the UMLS graph (e.g. `may_treat` lookups via the `concept_name_fts` fulltext index).
