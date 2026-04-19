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
- **(Recommended) NVIDIA GPU + CUDA** for the MedTE TEI container (see `docker-compose.yml`). Embedding runs server-side inside TEI — the Python pipeline only POSTs text — so no host-side PyTorch install is required. Verified on an NVIDIA L4 (24 GB) with driver 580.x.

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

### 2. Start Qdrant, Neo4j, and the MedTE embedding service

```bash
docker compose up -d
```

- Qdrant on `localhost:6333` (REST) and `localhost:6334` (gRPC).
- Neo4j on `localhost:7474` (Browser) and `localhost:7687` (Bolt). Default credentials: `neo4j` / `medragpass` (override via `NEO4J_USER` / `NEO4J_PASSWORD`).
- MedTE (HuggingFace [text-embeddings-inference](https://github.com/huggingface/text-embeddings-inference) serving `MohammadKhodadad/MedTE-cl15-step-8000`) on `localhost:8080`, configured with `--pooling=mean` since MedTE (a GTE-family BertModel) was contrastively pretrained on mean-pooled + L2-normalized vectors — TEI's default CLS pooling would ignore that training geometry. Embed text with `POST /embed`: `curl -s localhost:8080/embed -H 'content-type: application/json' -d '{"inputs":"acute myocardial infarction"}'`. The image tag `89-latest` targets NVIDIA compute capability 8.9 (L4/Ada); swap it if your GPU has a different CC.

The MedTE container requires the [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html). If `docker run --gpus all` errors out, run `sudo nvidia-ctk runtime configure --runtime=docker && sudo systemctl restart docker` to register the nvidia runtime.

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

Row counts for UMLS 2025AB after `--english-only --drop-suppressed` (raw RRF → admin-import CSV; self-loops and refs to dropped concepts also removed):

| RRF | Raw lines | Output CSV | Rows |
|---|---:|---|---:|
| MRCONSO.RRF | 17,390,109 | `concepts.csv` (unique CUIs) | 3,303,277 |
| MRCONSO.RRF | (same) | `atoms.csv` | 9,026,723 |
| MRCONSO.RRF | (same) | `concept_atom.csv` | 9,026,723 |
| MRSTY.RRF | 3,834,110 | `semantic_types.csv` | 127 |
| MRSTY.RRF | (same) | `concept_semtype.csv` | 3,645,587 |
| MRREL.RRF | 63,494,934 | `concept_relates.csv` | 40,124,828 |
| MRHIER.RRF | 40,573,034 | `concept_parent.csv` | 40,418,715 |
| MRDEF.RRF | 479,151 | `concept_definition.csv` | 464,510 |
| MRSAB.RRF | 197 | `sources.csv` | 192 |

Imported into Neo4j: ~3.3M `Concept` nodes, ~9M `Atom` nodes (one per MRCONSO row, carrying source code/TTY/string), and ~93.7M relationships (`HAS_ATOM`, `IS_A`, `RELATES`, `HAS_SEMTYPE`, `DEFINED_BY`) plus `SemanticType` and `Source` nodes. See the docstring in `python/ingestion/umls_to_neo4j_csv.py` for the graph model.

`load_neo4j.sh` also creates indexes/constraints (`Concept.cui`, `Atom.aui`, `SemanticType.tui`, `Source.sab` uniqueness; `(Atom.sab, Atom.code)` range; `Concept.name` and `Atom.str` fulltext) and applies four semantic-type-derived labels to Concept nodes for query routing:

- `:ClinicalCore` — Disease/Drug/Procedure/Anatomy (~735K nodes)
- `:ClinicalSupport` — Device/Lab/Finding/Food (~418K nodes)
- `:ClinicalDiscipline` — specialties, provider groups, care organizations (~13K nodes)
- `:Peripheral` — everything else (~2.14M nodes)

A pre-built snapshot of the imported store is at `gs://med_rag/neo4j_processed/neo4j_data.tar.zst` (~900 MB) — restore by extracting into the `med_rag_neo4j_data` Docker volume with neo4j stopped.

### 5. Parse MTSamples into per-doc JSON

Splits each transcription on ALL-CAPS `HEADING:` tokens and emits one `MTSampleDoc` JSON per row (one file per source row) into `data/mtsamples_docs/`. These intermediates are what the embedding step in `python/ingestion/embed_sections.py` consumes.

```bash
# Build the heading allowlist from the raw CSV (1,803 headings, filtered by MIN_DOCS/MAX_WORDS).
python python/ingestion/extract_mt_headings.py

# Fan out per-row parse across dask workers (default 16). ~few seconds on 32 cores.
python python/ingestion/parse_mtsamples.py [--workers 16]
```

Result: 4,966 per-doc JSONs under `data/mtsamples_docs/{doc_id:04d}.json`. Each doc carries `doc_id`, `specialty`, `specialty_cui`, `doctype_cui` (explicit mapping or heuristic rule), `sample_name`, `keywords`, and a list of `Section` records (`chunk_id`, `section_type`, `text`, plus placeholders for `embedding` and `entities` populated downstream). `doctype_cui` is set on 3,583/4,966 docs (specialties without an explicit mapping in `data/doctype_cui.json` and no matching heading rule get `""`).

### 6. Embed section text into Qdrant

Windows each `Section` into sentence packs of ≤350 tokens with 15% overlap, embeds every window by POSTing batches of up to 64 texts to the MedTE TEI service on `localhost:8080`, and upserts the vectors into the `mtsamples_sections` Qdrant collection. Parallelism is via `dask.bag.map_partitions` — each of the default 16 worker processes owns its own HTTP session and local tokenizer (used only for counting tokens during packing; the GPU-resident TEI container does all the encoding).

```bash
# Requires qdrant + medte services from `docker compose up -d`.
python python/ingestion/embed_sections.py [--workers 16] [--recreate]
```

Point ids are `uuid5(chunk_id)` so re-runs are idempotent. Each Qdrant point's payload carries `chunk_id`, `parent_chunk_id`, `window_index`/`window_count`, `doc_id`, `section_type`, `section_cui`, `specialty`, `specialty_cui`, `doctype_cui`, `sample_name`, `keywords`, and the windowed `text`. Collection dim and distance (cosine) are inferred from TEI's `/embed` response on startup. End state after a full run: 4,966 docs → **43,244** section-window points (768-d cosine).

#### Retrieval sanity check

Short concept query `"lymphoblastic Leukemia"` against the collection, top-50:

- **26 unique docs** among the 50 section hits.
- Top 4 @ score **0.656**: `Lymphoblastic Leukemia - Consult` appearing under four specialties (Hematology-Oncology, General Medicine, Cardiovascular/Pulmonary, Consult-H&P) — MTSamples cross-files the same note under multiple specialty tags, so all four surface together on `HISTORY OF PRESENT ILLNESS`. The same four docs also cluster on `CHIEF COMPLAINT` (0.65), `ASSESSMENT` (0.648), and `LABORATORY DATA` (0.472), consuming a large share of the top-50 slots.
- Remaining hits are clinically adjacent: `Antibiotic Therapy Consult` (0.644), `Aplastic Anemia Followup` (0.603), `Non-Hodgkin lymphoma Followup` (0.58), `MediPort Placement` / `Removal of Venous Port` (central-line care for chemotherapy), `Thrombocytopenia - Consult`, `Ommaya reservoir`, `Leiomyosarcoma`, etc. Two off-topic bleed-ins at ~0.48: `Prostatitis - Recheck` and a urology `REVIEW OF SYSTEMS`.

Takeaways: (1) MedTE + mean pooling retrieves a clinically coherent oncology neighborhood (leukemia → related heme/onc consults → chemo access devices) rather than keyword-matching only. (2) The two-word query produces much lower absolute scores (~0.66 peak) than a sentence-length query would — MedTE is tuned for sentence embeddings, so bare concept strings aren't ideal inputs. (3) MTSamples contains duplicate notes cross-filed under multiple specialty tags, so downstream retrieval should dedupe on a content hash (or on `sample_name` + `description`) before presenting results to the user, or the top-k will be padded with the same underlying note.
