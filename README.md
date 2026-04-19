# med_rag

Medical RAG (Retrieval-Augmented Generation) pipeline built on the MTSamples medical transcription dataset and Qdrant vector database.

## Prerequisites

- Python 3.10+
- Docker Engine 20.10+ with the Compose v2 plugin (`docker compose`, not the legacy `docker-compose`). On Ubuntu, install from Docker's official apt repo:
  ```bash
  sudo install -m 0755 -d /etc/apt/keyrings
  sudo curl -fsSL https://download.docker.com/linux/ubuntu/gpg -o /etc/apt/keyrings/docker.asc
  sudo chmod a+r /etc/apt/keyrings/docker.asc
  echo "deb [arch=$(dpkg --print-architecture) signed-by=/etc/apt/keyrings/docker.asc] https://download.docker.com/linux/ubuntu $(. /etc/os-release && echo $VERSION_CODENAME) stable" \
      | sudo tee /etc/apt/sources.list.d/docker.list > /dev/null
  sudo apt-get update
  sudo apt-get install -y docker-ce docker-ce-cli containerd.io docker-buildx-plugin docker-compose-plugin
  sudo usermod -aG docker $USER   # log out/in (or `newgrp docker`) for group to take effect
  ```
- `gcloud` / `gsutil` authenticated with read access to `gs://med_rag/datasets/`
- **(Recommended) NVIDIA GPU + CUDA** for the MedTE sentence encoder used in `python/ingestion/embed_sections.py`. The encoder auto-selects `cuda` when available and falls back to CPU otherwise, but CPU embedding the full MTSamples corpus is impractically slow. Verified on an NVIDIA L4 (24 GB) with driver 580.x. Install a matching CUDA build of PyTorch, e.g.:
  ```bash
  pip install torch --index-url https://download.pytorch.org/whl/cu121
  python -c "import torch; print(torch.cuda.is_available(), torch.cuda.get_device_name(0) if torch.cuda.is_available() else '')"
  ```

## Setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Usage

### 1. Download the datasets

```bash
# MTSamples medical transcriptions -> data/kaggle/mtsamples.csv (~16 MB)
python python/ingestion/download_mtsamples.py

# UMLS 2025AB Metathesaurus Full -> data/umls/umls-2025AB-metathesaurus-full.zip (~5.3 GB)
python python/ingestion/download_umls.py
```

Both pull from `gs://med_rag/datasets/`.

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

Two-step pipeline: unzip + convert RRF files to admin-import CSVs, then bulk-load. Both steps are parallelized — the RRF-to-CSV pass uses one worker per CPU core (`multiprocessing` in `umls_to_neo4j_csv.py`), and `neo4j-admin database import` runs with `--threads=32 --high-parallel-io=on`.

```bash
# Unzip data/umls/*.zip and convert to CSVs in data/neo4j_import/. ~90s on 32 cores.
bash scripts/prepare_umls.sh

# Bulk-load with 32 import threads + parallel I/O. Stops neo4j, runs neo4j-admin import, restarts. ~25s.
bash scripts/load_neo4j.sh
```

Result: ~3.3M `Concept` nodes, ~9M `Atom` nodes (one per MRCONSO row, carrying source code/TTY/string), and ~93.7M relationships (`HAS_ATOM`, `IS_A`, `RELATES`, `HAS_SEMTYPE`, `DEFINED_BY`) plus `SemanticType` and `Source` nodes. See the docstring in `python/ingestion/umls_to_neo4j_csv.py` for the graph model.

`load_neo4j.sh` also creates indexes/constraints (`Concept.cui`, `Atom.aui`, `SemanticType.tui`, `Source.sab` uniqueness; `(Atom.sab, Atom.code)` range; `Concept.name` and `Atom.str` fulltext) and applies four semantic-type-derived labels to Concept nodes for query routing:

- `:ClinicalCore` — Disease/Drug/Procedure/Anatomy (~735K nodes)
- `:ClinicalSupport` — Device/Lab/Finding/Food (~418K nodes)
- `:ClinicalDiscipline` — specialties, provider groups, care organizations (~13K nodes)
- `:Peripheral` — everything else (~2.14M nodes)

A pre-built snapshot of the imported store is at `gs://med_rag/neo4j_processed/neo4j_data.tar.zst` (~900 MB) — restore by extracting into the `med_rag_neo4j_data` Docker volume with neo4j stopped.
