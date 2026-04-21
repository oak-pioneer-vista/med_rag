# med_rag

Medical RAG (Retrieval-Augmented Generation) pipeline built on the MTSamples medical transcription dataset, backed by Qdrant for dense retrieval and Neo4j for a UMLS-derived clinical knowledge graph.

## What's in here

- `python/ingestion/mtsamples/` — MTSamples downloader, section parser, MedTE/TEI embedding pipeline, Stanza NER, and the Note/Section/Entity Neo4j loader.
- `python/ingestion/umls/` — UMLS downloader and the RRF → Neo4j admin-import CSV converter.
- `scripts/` — shell wrappers for the UMLS prepare + bulk-load steps.
- `docker-compose.yml` — Qdrant, Neo4j, and the MedTE text-embeddings-inference (TEI) service on the host GPU.

## Docs

- [**Ingestion pipeline**](docs/ingestion.md) — prerequisites, setup, and the step-by-step commands to build the Qdrant section-embedding collection and the Neo4j UMLS graph.
- [**Learning notes**](docs/learning_notes.md) — non-obvious things learned while building this: short-query degradation on sentence-level encoders, TEI vs. in-process transformers trade-offs, pooling gotchas.
- [**Neo4j queries**](docs/query.md) — sample Cypher queries against the UMLS graph (e.g. `may_treat` lookups via the `concept_name_fts` fulltext index).
