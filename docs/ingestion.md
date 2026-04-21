# Ingestion pipeline

End-to-end steps for building the MTSamples section-embedding collection in Qdrant and the UMLS knowledge graph in Neo4j.

## Pipeline overview

1. **Parse MTSamples** — split each transcription by section headers (`CHIEF COMPLAINT:`, `HISTORY OF PRESENT ILLNESS:`, `ASSESSMENT AND PLAN:`, etc.). Section-aware chunking matters for clinical text — don't blindly split on 512 tokens.
2. **Chunk within sections** (e.g. 200–400 tokens, small overlap). Assign each a stable `chunk_id` (UUID or `{doc_id}:{section}:{i}`).
3. **Embed with MedTE and upsert to Qdrant** — put `chunk_id`, `doc_id`, `specialty`, and `section` in the payload.

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
python python/ingestion/mtsamples/download_mtsamples.py

# UMLS 2025AB Metathesaurus Full -> data/umls/umls-2025AB-metathesaurus-full.zip (~5.3 GB)
python python/ingestion/umls/download_umls.py
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
python python/ingestion/mtsamples/ingest_mtsamples.py
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

Imported into Neo4j: ~3.3M `Concept` nodes, ~9M `Atom` nodes (one per MRCONSO row, carrying source code/TTY/string), and ~93.7M relationships (`HAS_ATOM`, `IS_A`, `RELATES`, `HAS_SEMTYPE`, `DEFINED_BY`) plus `SemanticType` and `Source` nodes. See the docstring in `python/ingestion/umls/umls_to_neo4j_csv.py` for the graph model.

`load_neo4j.sh` invokes `scripts/create_neo4j_indices.sh` at the end — that's where the constraints/indexes/full-text indexes live and where the four semantic-type-derived labels below get applied to Concept nodes for query routing:

- `:ClinicalCore` — Disease/Drug/Procedure/Anatomy (~735K nodes)
- `:ClinicalSupport` — Device/Lab/Finding/Food (~418K nodes)
- `:ClinicalDiscipline` — specialties, provider groups, care organizations (~13K nodes)
- `:Peripheral` — everything else (~2.14M nodes)

A pre-built snapshot of the imported store is at `gs://med_rag/neo4j_processed/neo4j_data.tar.zst` (~900 MB) — restore by extracting into the `med_rag_neo4j_data` Docker volume with neo4j stopped.

### 5. Clean MTSamples (drop empty transcripts, dedupe cross-filings)

```bash
python python/ingestion/mtsamples/clean_mtsamples.py
```

MTSamples cross-files the same note under multiple `medical_specialty` tags with byte-identical `transcription`. This step drops rows with no transcription, collapses duplicate transcriptions (keeping first-row metadata, recording dropped specialty names in `alt_specialties`, and unioning keyword tokens), and writes the survivors to `data/mtsamples_clean.jsonl`. Expected: **4,966 raw rows → 2,357 cleaned records** (2,150 clusters collapsed, 2,609 dupe rows dropped).

### 6. Parse MTSamples into per-doc JSON

Splits each cleaned transcription on ALL-CAPS `HEADING:` tokens and emits one `MTSampleDoc` JSON per row into `data/mtsamples_docs/`. These intermediates are what the embedding step in `python/ingestion/mtsamples/embed_sections.py` consumes.

```bash
# Build the heading allowlist from the raw CSV (1,803 headings, filtered by MIN_DOCS/MAX_WORDS).
python python/ingestion/mtsamples/extract_mt_headings.py

# Fan out per-row parse across dask workers (default 16). ~few seconds on 32 cores.
python python/ingestion/mtsamples/parse_mtsamples.py [--workers 16]
```

Result: **2,357 per-doc JSONs** under `data/mtsamples_docs/{doc_id:04d}.json`. Each doc carries `doc_id`, `specialty`, `specialty_cui`, `doctype_cui` (explicit mapping or heuristic rule), `sample_name`, merged `keywords`, `alt_specialties` (list of `{specialty, specialty_cui, doctype_cui}` from the dropped cross-filings — every entry has at least one CUI populated), and a list of `Section` records (`chunk_id`, `section_type`, `text`, plus placeholders for `embedding` and `entities` populated downstream). `doctype_cui` is set on 1,634/2,357 docs (specialties without an explicit mapping in `data/doctype_cui.json` and no matching heading rule get `""`).

### 7. Build per-doc abbreviation maps (Schwartz-Hearst + override + LRABR + MedTE WSD)

Four-stage resolution with provenance tracking. Each doc gets `abbreviations`, `abbreviations_source` (`sh`/`override`/`lrabr`), and `abbreviations_score` (1.0 for S-H and override hits, cosine similarity for LRABR hits) written into its per-doc JSON.

```bash
# prereqs: MedTE TEI up, LRABR staged, parse_mtsamples already run
docker compose up -d medte
python python/ingestion/umls/download_specialist_lexicon.py
python python/ingestion/mtsamples/build_abbreviations.py [--min-score 0.3]
```

**1. Schwartz-Hearst** over joined section texts (`" . "` separator so the line-wise scan terminates at section boundaries). Highest confidence because the definition is evidence-in-doc.

**2. Curated clinical override** (`data/clinical_abbreviations_override.json`) — a ~60-entry hand-picked list for clinical-canon abbreviations (IV, BUN, CT, MRI, EKG, ABCD, AICD, PTT, PSA, …) whose canonical expansion is overwhelmingly unambiguous in clinical notes. Wins over LRABR; source tag `override`.

**3. LRABR gazetteer + MedTE WSD** for everything else — NLM's SPECIALIST Lexicon LRABR (~62K abbrevs, ~38K with multiple senses) is purpose-built and far cleaner than deriving AB/ACR atoms from MRCONSO (which is dominated by chemo-protocol shorthand and gene symbols). Every LRABR hit is gated by cosine similarity between the doc context and each candidate expansion via MedTE/TEI; winner must clear `--min-score` (default 0.3). This drops the obscure-single-sense-LRABR cases (e.g. `AMSA → amphotericin in solid-state administration`) in docs where the clinical sense isn't in LRABR, rather than injecting them blindly.

Expected yield on MTSamples: **~1,500/2,357 docs** (64%) with ≥1 pair. Typical run: **143 S-H, ~2,400 override, ~1,500 LRABR**; ~4,500 LRABR candidates skipped below threshold. Uniform threshold gate is load-bearing — without it, single-sense LRABR hits for `ABCD`/`AICD`/`PSA`/etc. would inject wrong-sense expansions in clinical docs.

### 8. Extract per-section entities with spans (Stanza i2b2 NER)

```bash
# GPU path (fastest): if host cuDNN lags torch's bundled cuDNN, prepend the venv's
# nvidia libs so torch's bundled cuDNN wins. Symptom without it: "cuDNN version
# incompatibility" at pipeline-load time.
VENV_LIB=.venv/lib/python3.10/site-packages
LD_LIBRARY_PATH="$VENV_LIB/nvidia/cudnn/lib:$VENV_LIB/nvidia/cublas/lib:$LD_LIBRARY_PATH" \
  python python/ingestion/mtsamples/extract_section_entities.py [--workers 8] [--batch 16]

# CPU fallback (~10x slower, no driver fuss):
python python/ingestion/mtsamples/extract_section_entities.py --cpu
```

Runs Stanza's `mimic/i2b2` NER over each Section's text (sections are the logical unit — not Qdrant-aligned windows) and fills each Section's `entities` list in the per-doc JSON. Each entity record carries three text views — `surface_text` (literal section-text slice), `recognized_text` (Stanza's output), `resolved_text` (with abbreviations from step 7 substituted in) — plus i2b2 type (`PROBLEM` / `TEST` / `TREATMENT`) and section-local `start_char` / `end_char`.

Parallelism uses `multiprocessing.Pool` with `spawn` start method so each worker gets a clean CUDA context; doc paths are sharded via **Longest-Processing-Time-first bin packing** on total section char-count (keeps the heaviest shard within ~4/3 of the mean, vs equal-doc-count sharding that lumped several 2K-token PROCEDURE notes together and stretched wall time by 30%+). Within each worker, sections are **length-sorted descending** before batching so Stanza's padding-to-longest-in-batch waste is near-zero on a corpus with p50=23 / p99=863 / max=2,819 MedTE tokens.

Tuned defaults (`--workers 8 --batch 16`) from a sweep on L4: 8 is the knee — 6 is within 2s, ≥12 regresses as CUDA context-switching across processes overtakes the marginal gain, 32 OOMs (each pipeline + activations is ~700 MB–1 GB on GPU). Batch size is nearly irrelevant with length-bucketing in place (16/64/1024 all within 2s). Expected on MTSamples: **~121K entities across ~18K non-empty sections in ~97s on L4**; ~16 min if you fall back to `--cpu`. ~4K entities (3.4%) get a `resolved_text` that differs from `recognized_text`.

This is distinct from `python/ingestion/mtsamples/extract_entities.py`, which targets Qdrant-aligned chunk windows for the retrieval pipeline. The two produce complementary artifacts: section entities live inside the per-doc JSON (coarse-grained, directly usable by the Neo4j loader's Section→Entity layer), while chunk entities live in `data/entities/chunk_entities.jsonl` and key off `chunk_id` for joins with Qdrant points.

### 9. Embed section text into Qdrant

Windows each `Section` into sentence packs of ≤200 tokens with 10% overlap, embeds every window by POSTing batches of up to 64 texts to the MedTE TEI service on `localhost:8080`, and upserts the vectors into the `mtsamples_sections` Qdrant collection. Parallelism is via `dask.bag.map_partitions` — each of the default 16 worker processes owns its own HTTP session and local tokenizer (used only for counting tokens during packing; the GPU-resident TEI container does all the encoding).

```bash
# Requires qdrant + medte services from `docker compose up -d`.
python python/ingestion/mtsamples/embed_sections.py [--workers 16] [--recreate]
```

Point ids are `uuid5(chunk_id)` so re-runs are idempotent. Each Qdrant point's payload carries `chunk_id`, `parent_chunk_id`, `window_index`/`window_count`, `doc_id`, `section_type`, `section_cui`, `specialty`, `specialty_cui`, `alt_specialties`, `doctype_cui`, `sample_name`, `keywords`, and the windowed `text`. Collection dim and distance (cosine) are inferred from TEI's `/embed` response on startup. The collection is created with `optimizers.indexing_threshold=100` — Qdrant's default (~20k per segment) would otherwise leave every segment unindexed at this scale, falling back to brute-force search on every query. End state after a full run: 2,357 docs → **22,824** section-window points (768-d cosine).

#### Retrieval sanity check

Instruction-style query `"Retrieve all patients diagnosed with Lymphoblastic Leukemia"` against the dedup'd collection, top-50:

- **37 unique docs** among the 50 section hits (up from 23 pre-dedupe — cross-filed copies no longer pad the ranking).
- **Rank 1 @ 0.688**: `Antibiotic Therapy Consult` (doc 2175) `FAMILY HISTORY` — enumerates leukemia diagnoses in relatives, so the lexical overlap outranks the actual index-case notes. This is the retrieval failure mode worth knowing about: a bi-encoder without instruction tuning can't tell "patient was diagnosed with X" from "patient's relative was diagnosed with X".
- **Rank 2 @ 0.613**: the target `Lymphoblastic Leukemia - Consult` (doc 2004) `ASSESSMENT`. The same note re-surfaces 6 more times in the top-50 on different sections (`CHIEF COMPLAINT`, three `HISTORY OF PRESENT ILLNESS` windows, `LABORATORY DATA`, `FAMILY HISTORY`) — legitimate multi-section retrieval of one doc rather than cross-filed duplicates.
- Remaining hits are clinically adjacent: `Removal of Venous Port` (0.599), `Aplastic Anemia Followup`, `Non-Hodgkin lymphoma Followup`, `Polycythemia Vera Followup`, `Leiomyosarcoma`, `MediPort Placement` / `Ommaya reservoir` (chemo access), `Astrocytoma`, `T-Cell Lymphoma Consult`, `Posttransplant Lymphoproliferative Disorder`, and a couple of general discharge summaries.

Takeaways: (1) MedTE + mean pooling retrieves a clinically coherent oncology neighborhood (leukemia → related heme/onc consults → chemo access devices). (2) Instruction-style query phrasing and the family-history-vs-index-case distinction both trip the encoder — the top hit being a family-history surface match is expected bi-encoder behavior; a reranker (cross-encoder) or an instruction-tuned embedder (`intfloat/e5-*`, BGE) would be needed to fix it. (3) Post-dedupe the top-k now surfaces genuinely distinct clinical neighbors instead of near-identical cross-filings — compare to the pre-dedupe run where the target doc occupied ~half of the top-50 as four cross-filed copies.

### 10. Create Qdrant payload indexes

```bash
python scripts/create_qdrant_indices.py
```

Idempotent. Ensures `indexing_threshold=100` (matches what `embed_sections.py` sets on create, so re-asserts the value against any pre-existing collection), and creates payload indexes on the fields we filter by:

- keyword: `parent_chunk_id`, `section_type`, `section_cui`, `specialty`, `specialty_cui`, `doctype_cui`, `sample_name`
- integer: `doc_id`, `window_index`, `window_count`

Without these, filters like `doc_id == 2306` or `specialty == "Nephrology"` are O(n) scans over 22K payloads on every query. Re-run whenever a new filterable field is introduced; existing indexes are a no-op.

### 11. Create Neo4j constraints, indexes, and tier labels

```bash
bash scripts/create_neo4j_indices.sh
```

`scripts/load_neo4j.sh` calls this automatically at the end of the bulk-import step, so you don't need to run it again after a fresh load. Run it standalone when you want to re-assert the schema without re-importing — e.g. after restoring from the `neo4j_data.tar.zst` snapshot. Creates uniqueness constraints (`Concept.cui`, `Atom.aui`, `SemanticType.tui`, `Source.sab`), a range index on `(Atom.sab, Atom.code)` and `Concept.name`, full-text indexes `concept_name_fts` and `atom_str_fts`, and applies the four `:ClinicalCore` / `:ClinicalSupport` / `:ClinicalDiscipline` / `:Peripheral` tier labels.
