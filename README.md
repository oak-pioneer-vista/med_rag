# med_rag

Medical RAG (Retrieval-Augmented Generation) pipeline built on the MTSamples medical transcription dataset and Qdrant vector database.

## Prerequisites

- Python 3.10+
- Docker & Docker Compose
- [Kaggle API credentials](https://www.kaggle.com/docs/api) at `~/.kaggle/kaggle.json`

## Setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Usage

### 1. Download the dataset

```bash
python python/ingestion/download_mtsamples.py
```

Downloads the [MTSamples](https://www.kaggle.com/datasets/tboyle10/medicaltranscriptions) dataset into `data/mtsamples/`.

### 2. Start Qdrant and Neo4j

```bash
docker compose up -d
```

- Qdrant on `localhost:6333` (REST) and `localhost:6334` (gRPC).
- Neo4j on `localhost:7474` (Browser) and `localhost:7687` (Bolt). Default credentials: `neo4j` / `medragpass` (override via `NEO4J_USER` / `NEO4J_PASSWORD`).

Data for both is persisted in Docker volumes. Verify the Neo4j connection with:

```bash
python python/neo4j_smoke_test.py
```

### 3. Ingest into Qdrant

```bash
python python/ingestion/ingest_mtsamples.py
```

Embeds medical transcriptions using `all-MiniLM-L6-v2` and upserts them into a `mtsamples` Qdrant collection with metadata (specialty, description, keywords).

### 4. Ingest UMLS into Neo4j

Two-step pipeline: convert RRF files to admin-import CSVs, then bulk-load.

```bash
# Convert (parallel; defaults to CPU count). ~70s on 32 cores for UMLS 2025AB.
python python/ingestion/umls_to_neo4j_csv.py \
    --meta data/datasets/umls-2025AB-metathesaurus-full1/2025AB/META \
    --out  data/neo4j_import \
    --english-only --drop-suppressed --workers 32

# Bulk-load. Stops neo4j, runs neo4j-admin import, restarts. ~25s.
bash scripts/load_neo4j.sh
```

Result: ~3.3M `Concept` nodes, ~9M `Atom` nodes (one per MRCONSO row, carrying source code/TTY/string), and ~93.7M relationships (`HAS_ATOM`, `IS_A`, `RELATES`, `HAS_SEMTYPE`, `DEFINED_BY`) plus `SemanticType` and `Source` nodes. See the docstring in `python/ingestion/umls_to_neo4j_csv.py` for the graph model.

`load_neo4j.sh` also creates indexes/constraints (`Concept.cui`, `Atom.aui`, `SemanticType.tui`, `Source.sab` uniqueness; `(Atom.sab, Atom.code)` range; `Concept.name` and `Atom.str` fulltext) and applies four semantic-type-derived labels to Concept nodes for query routing:

- `:ClinicalCore` — Disease/Drug/Procedure/Anatomy (~735K nodes)
- `:ClinicalSupport` — Device/Lab/Finding/Food (~418K nodes)
- `:ClinicalDiscipline` — specialties, provider groups, care organizations (~13K nodes)
- `:Peripheral` — everything else (~2.14M nodes)

A pre-built snapshot of the imported store is at `gs://med_rag/neo4j_processed/neo4j_data.tar.zst` (~900 MB) — restore by extracting into the `med_rag_neo4j_data` Docker volume with neo4j stopped.
