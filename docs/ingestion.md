# Ingestion pipeline

End-to-end steps for building the MTSamples sentence-embedding collection in Qdrant and the UMLS knowledge graph in Neo4j.

## Pipeline overview

1. **Parse MTSamples** — split each transcription by section headers (`CHIEF COMPLAINT:`, `HISTORY OF PRESENT ILLNESS:`, `ASSESSMENT AND PLAN:`, etc.). Section-aware chunking matters for clinical text — don't blindly split on 512 tokens.
2. **Sentence-chunk within sections** (Stanza clinical tokenizer). Each sentence gets a stable `chunk_id` and carries the set of UMLS CUIs/TUIs mentioned in it.
3. **Embed each sentence with MedTE and upsert to Qdrant** — payload carries `chunk_id`, `section_chunk_id`, `doc_id`, `specialty`, `section_type`, CUIs, TUIs, and surface forms for index-only filtering before any vector search.

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

### 3. Ingest UMLS into Neo4j

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

### 4. Clean MTSamples (drop empty transcripts, dedupe cross-filings)

```bash
python python/ingestion/mtsamples/clean_mtsamples.py
```

MTSamples cross-files the same note under multiple `medical_specialty` tags with byte-identical `transcription`. This step drops rows with no transcription, collapses duplicate transcriptions (keeping first-row metadata, recording dropped specialty names in `alt_specialties`, and unioning keyword tokens), and writes the survivors to `data/mtsamples_clean.jsonl`. Expected: **4,966 raw rows → 2,357 cleaned records** (2,150 clusters collapsed, 2,609 dupe rows dropped).

### 5. Parse MTSamples into per-doc JSON

Splits each cleaned transcription on ALL-CAPS `HEADING:` tokens and emits one `MTSampleDoc` JSON per row into `data/mtsamples_docs/`. These intermediates are what the embedding step in `python/ingestion/mtsamples/embed_sections.py` consumes.

```bash
# Build the heading allowlist from the raw CSV (1,803 headings, filtered by MIN_DOCS/MAX_WORDS).
python python/ingestion/mtsamples/extract_mt_headings.py

# Fan out per-row parse across dask workers (default 16). ~few seconds on 32 cores.
python python/ingestion/mtsamples/parse_mtsamples.py [--workers 16]
```

Result: **2,357 per-doc JSONs** under `data/mtsamples_docs/{doc_id:04d}.json`. Each doc carries `doc_id`, `specialty`, `specialty_cui`, `doctype_cui` (explicit mapping or heuristic rule), `sample_name`, merged `keywords`, `alt_specialties` (list of `{specialty, specialty_cui, doctype_cui}` from the dropped cross-filings — every entry has at least one CUI populated), and a list of `Section` records (`chunk_id`, `section_type`, `text`, plus placeholders for `embedding` and `entities` populated downstream). `doctype_cui` is set on 1,634/2,357 docs (specialties without an explicit mapping in `data/doctype_cui.json` and no matching heading rule get `""`).

### 6. Build per-section abbreviation maps (Schwartz-Hearst + override + LRABR + MedTE WSD)

Three-stage resolution **per Section**, not per doc. Each Section gets `abbreviations`, `abbreviations_source` (`sh`/`override`/`lrabr`), and `abbreviations_score` (1.0 for S-H and override, cosine similarity for LRABR) written into the per-doc JSON. The same surface form can resolve to different expansions in different sections of the same doc — useful when one note mixes topics (e.g. `MS` could mean mitral stenosis in `PROCEDURE` and multiple sclerosis in `HPI` of the same doc; the section text is what disambiguates).

```bash
# prereqs: MedTE TEI up, LRABR staged, parse_mtsamples already run
docker compose up -d medte
python python/ingestion/umls/download_specialist_lexicon.py
python python/ingestion/mtsamples/build_abbreviations.py [--min-score 0.3]
```

**1. Schwartz-Hearst** over each section's text alone. Most clinical notes introduce abbreviations once at the top and reuse them across sections, so per-section S-H gives many fewer hits than the per-doc variant — but the hits that remain are still the highest-confidence signal because the definition is evidence-in-section.

**2. Curated clinical override** (`data/clinical_abbreviations_override.json`) — a ~60-entry hand-picked list for clinical-canon abbreviations (IV, BUN, CT, MRI, EKG, ABCD, AICD, PTT, PSA, …) whose canonical expansion is overwhelmingly unambiguous in clinical notes regardless of section context. Wins over LRABR; source tag `override`; no embedding check needed.

**3. LRABR gazetteer + MedTE WSD** for everything else — NLM's SPECIALIST Lexicon LRABR (~62K abbrevs, ~38K with multiple senses) is purpose-built and far cleaner than deriving AB/ACR atoms from MRCONSO (which is dominated by chemo-protocol shorthand and gene symbols). Every LRABR hit is gated by cosine similarity between **THIS SECTION's text** and each candidate expansion via MedTE/TEI; winner must clear `--min-score` (default 0.3). Per-section context is shorter than the joined-doc context, which is the point — when a doc spans multiple topics the WSD shouldn't average them.

Resolution stats on MTSamples: **3,634/18,336 sections (~20%)** carry ≥1 abbreviation; **135 S-H + 2,799 override + 3,327 LRABR** per-section assignments; ~3,861 LRABR candidates skipped below threshold. Embedding pass is small (~3,700 unique section contexts + ~14,400 unique expansions, ~8s on the local TEI service). Compared to the previous per-doc design (1,500 docs / 2,357 with ≥1 abbrev; 143 S-H, 2,392 override, 1,542 LRABR): S-H is ~unchanged (most parenthetical intros sit in the same section as their abbrev), override and LRABR roughly double because the dedup key is now per-section instead of per-doc — the same `IV` or `CT` appearing in three sections of one doc now resolves three times, with WSD scored against each section's local context.

### 7. Extract per-section entities with spans (Stanza i2b2 NER)

```bash
# GPU path (fastest): if host cuDNN lags torch's bundled cuDNN, prepend the venv's
# nvidia libs so torch's bundled cuDNN wins. Symptom without it: "cuDNN version
# incompatibility" at pipeline-load time.
VENV_LIB=.venv/lib/python3.10/site-packages
LD_LIBRARY_PATH="$VENV_LIB/nvidia/cudnn/lib:$VENV_LIB/nvidia/cublas/lib:$LD_LIBRARY_PATH" \
  python python/ingestion/mtsamples/extract_section_entities.py [--workers 8] [--batch 64]

# CPU fallback (~10x slower, no driver fuss):
python python/ingestion/mtsamples/extract_section_entities.py --cpu
```

Runs Stanza's `mimic/i2b2` NER over each Section's text (sections are the logical unit — not Qdrant-aligned windows) and fills each Section's `entities` list in the per-doc JSON. Each entity record carries `surface_text` (literal section-text slice), `recognized_text` (Stanza's output), i2b2 `type` (`PROBLEM` / `TEST` / `TREATMENT`), and section-local `start_char` / `end_char`.

**This step is deliberately idempotent and text-normalization-free.** All derived text fields (`resolved_text`, `expanded_text`) are computed in step 8 from these raw records, so iterating on normalization rules doesn't require re-running Stanza.

Parallelism uses `multiprocessing.Pool` with `spawn` start method so each worker gets a clean CUDA context; doc paths are sharded via **Longest-Processing-Time-first bin packing** on total section char-count (keeps the heaviest shard within ~4/3 of the mean, vs equal-doc-count sharding that lumped several 2K-token PROCEDURE notes together and stretched wall time by 30%+). Within each worker, sections are **length-sorted descending** before batching so Stanza's padding-to-longest-in-batch waste is near-zero on a corpus with p50=23 / p99=863 / max=2,819 MedTE tokens.

Tuned defaults (`--workers 8 --batch 64`) from two sweeps on L4:
- **Workers**: 8 is the knee. 6 within 2s; ≥12 regresses as CUDA context-switching across processes overtakes the marginal gain; 32 OOMs (each pipeline + activations is ~700 MB–1 GB on GPU).
- **Batch**: clean U-curve with a flat bottom at **32–128** (~91s); batch=8 pays a 17s dispatch-overhead tax, batch=512 pays ~2s as the length-sorted first batch gets too big. 64 sits in the middle of the minimum with activation headroom.

Expected on MTSamples: **~121K entities across ~18K non-empty sections in ~91s on L4**; ~16 min on `--cpu`.

This is distinct from `python/ingestion/mtsamples/extract_entities.py`, which targets Qdrant-aligned chunk windows for the retrieval pipeline. The two produce complementary artifacts: section entities live inside the per-doc JSON (coarse-grained, directly usable by the Neo4j loader's Section→Entity layer), while chunk entities live in `data/entities/chunk_entities.jsonl` and key off `chunk_id` for joins with Qdrant points.

### 8. Normalize per-section entities (strip articles + expand abbreviations)

```bash
python python/ingestion/mtsamples/normalize_section_entities.py
```

Derives two text fields per entity record from the `recognized_text` written by step 7:
- `resolved_text` — `recognized_text` with articles (`a`/`an`/`the`) removed and whitespace collapsed. Reserved as the NER canonical surface; future string-level normalization rules chain in here.
- `expanded_text` — `resolved_text` with known abbreviations substituted using the doc's `abbreviations` map (step 6), then article-stripped again in case the expansion introduced one. Use this for UMLS/Neo4j grounding and dense-retrieval keying, where the expanded form matches more Concept atoms.

Pure Python string work — no GPU, no heavy compute. **Typically finishes in ~2s over all 18K entities**, vs the ~91s cost of re-running NER. That's the payoff of keeping the layers separate: iterate on the stopword list, the token regex, or any new normalization rule in a fast edit-run-inspect loop without paying the NER tax each time. Expected on MTSamples: **~32K entities get an article stripped (26%)**, **~4K entities (3.4%)** get an abbrev expansion that changes `expanded_text` relative to `resolved_text`.

### 9. Link specialties to UMLS CUIs

```bash
python python/ingestion/mtsamples/link_specialty_to_cui.py
```

Resolves both `doc['specialty']` and every `doc['alt_specialties'][i]['specialty']` to UMLS CUIs in a single pass. Each unique specialty string (there are ~40 across the corpus, including cross-filing-only ones like `Diets and Nutritions`) is looked up via exact `Atom.str_norm` match first, fulltext fallback on `concept_name_fts` for the rest. Overrides the CUIs seeded from `data/specialty_cui.json` during parse — with UMLS loaded, the graph is authoritative.

Expected: all 40 unique specialty strings resolve (~0.3s); ~1,400 alt_specialty entries get updated CUIs across ~1,270 docs.

### 10. Link section types to UMLS CUIs

```bash
python python/ingestion/mtsamples/link_sections_to_cui.py
```

For every Section's `section_type`, applies a small alias map (`HPI` → `history of present illness`, `PMH` → `past medical history`, `ROS` → `review of systems`, `HEENT` → `head ears eyes nose throat`, etc.) before the Neo4j lookup, so short-form and full-form headings resolve to the same CUI (critical for downstream grouping). Exact match + fulltext fallback. Writes `section_cui` on each section in the per-doc JSONs.

Expected: **~1,736/1,778 unique section_type values resolved (97.6%)**; `HPI` and `HISTORY OF PRESENT ILLNESS` both → `C0262512`. ~52s on a warm Neo4j.

### 11. Link entities to UMLS CUIs + TUIs (dask map-reduce)

```bash
python python/ingestion/mtsamples/link_entities_to_cui.py [--workers 16] [--batch 500]
```

Resolves each entity to a UMLS concept (CUI) *and* its semantic types (TUIs) — the biggest lookup workload in the pipeline, so the architecture is built around three separate parallelism stages:

1. **Dask map-reduce dedup** — `dask.bag` over per-doc JSON paths, N worker processes. Each partition worker reads its shard and emits `{entity_hash: expanded_text_lower}` (content-addressable `sha1(text)[:16]` keys). Worker dicts are union-merged into one global dedup map. ~121K entity mentions collapse to **~49K unique hashes** so Neo4j never sees a duplicate.

2. **Exact CUI pass** — single session, batched `UNWIND + MATCH` on `Atom.str_norm`. ~22% of hashes hit exact in ~5s.

3. **Fulltext CUI fallback** — residual ~38K hashes split across `--workers` (default 16) processes, each with its own Neo4j session, batched through `UNWIND + CALL { CALL db.index.fulltext.queryNodes(...) }`. Lucene special-char escape + alphanumeric guard pre-filter. Finishes in ~3 min wall on a local Neo4j, vs. ~10 min single-threaded or ~15 min per-query.

4. **TUI pass** — unique CUIs (~27K) batched via `UNWIND + MATCH (c)-[:HAS_SEMTYPE]->(st)`, collecting `st.tui` + `st.name` lists. One concept can have multiple semantic types (e.g. CD4 → `[T192, T129, T116]` = Receptor + Immunologic Factor + Protein). ~2.5s.

5. **Write-back** — every entity mention gets stamped with its resolved payload by `entity_hash` lookup.

Fields written per entity:

| field | type | purpose |
|---|---|---|
| `entity_hash` | `str` (16 hex) | Deterministic dedup key for joins, safe for use as a graph-node ID downstream. |
| `cui` | `str` | UMLS Concept Unique Identifier, or `""`. |
| `cui_name` | `str` | Matched `Concept.name` (for audit). |
| `cui_match` | `str` | `"exact"` / `"fulltext/<score>"` / `""`. |
| `tuis` | `list[str]` | Semantic Type Unique Identifiers (e.g. `["T047"]` = Disease or Syndrome). |
| `tui_names` | `list[str]` | Human-readable TUI names. |

Expected on MTSamples: **~119K of 121K entities linked (~98%)**, 100% of linked entities carry ≥1 TUI. Top semantic types across the corpus: **T033 Finding (16.6K), T061 Therapeutic/Preventive Procedure (13.8K), T047 Disease or Syndrome (12.6K), T074 Medical Device (9.9K), T121 Pharmacologic Substance (8.9K), T184 Sign or Symptom (8.3K)** — i.e. the distribution a clinical corpus should produce.

### 12. Snapshot the lexical entity→CUI map for later comparison

```bash
python python/ingestion/mtsamples/export_entity_cui_lexical.py
```

Step 11 resolves entities lexically — exact `Atom.str_norm` match plus a Lucene fulltext fallback on `concept_name_fts`. To make it easy to A/B against a future **semantic** variant (e.g. BioLORD-2023 nearest-neighbor over Concept name embeddings), this step dumps one JSONL line per unique `expanded_text` with the CUI that the lexical pipeline picked:

```json
{"text": "acute myocardial infarction", "cui": "C0155626"}
```

Unresolved entities are preserved with `cui: ""` so the snapshot is a complete coverage map, not just the hits. Expected on MTSamples: **49,390 unique entities → 48,102 linked / 1,288 unlinked** written to `data/entity_cui_lexical.jsonl` in ~1s. Re-run the same export after swapping in a semantic linker and diff the two files to see exactly which entities move.

### 13. Sentence-chunk sections + tag each sentence with linked entity sets

```bash
python python/ingestion/mtsamples/chunk_sentences.py [--workers 8] [--batch 128]
```

For every Section, splits the text into sentences using **Stanza's `mimic` clinical tokenizer** (same tokenizer used by step 7 for NER — handles medical abbreviations like `q.i.d.`, `2.5 mg`, `Dr.`, and does better on MTSamples' comma-pseudo-paragraph style than a bare regex). For each sentence, scans for the presence of any linked entity's surface form (word-boundary, case-insensitive) via a single compiled alternation regex per doc. Matched entities contribute to three per-sentence sets:

- `cuis` — sorted list of UMLS CUIs covered in this sentence
- `tuis` — sorted list of semantic TUIs in this sentence
- `surface_forms` — sorted list of matched entity surface forms (lowercased)

Written back as `section.sentences = [{text, cuis, tuis, surface_forms}, ...]`.

Parallelism is `mp.Pool` with `spawn` start method (same pattern as step 7). Each worker loads Stanza once in its initializer and processes its shard's sections in batches of `--batch` via `nlp.bulk_process`. Per-doc surface-form indices are compiled once per doc and reused across all sections of that doc, so a sentence is scanned against all candidate surface forms in one pass.

Expected on MTSamples: **~83K sentences across 18K sections in ~20s wall on L4 (8 workers, batch=128)**; ~74% of sentences carry ≥1 CUI hit. Stanza catches ~800 more sentence boundaries than the naive regex, mostly in operative-note comma-joined clauses.

### 14. Embed sentence-level chunks into Qdrant

```bash
docker compose up -d medte qdrant
python python/ingestion/mtsamples/embed_sentences.py [--workers 16] [--batch 32] [--recreate]
```

Embeds each sentence chunk (produced by step 13) via MedTE/TEI and upserts one Qdrant point per sentence into the `mtsamples_sentences` collection. Point id is `uuid5(chunk_id)` so re-runs are idempotent. Parallelism is `mp.Pool` — each worker owns an HTTP session + Qdrant client; TEI does server-side dynamic batching across concurrent worker requests.

**Payload carries full provenance** (all fields payload-indexed): `chunk_id`, `section_chunk_id`, `doc_id`, `section_type`, `section_cui`, `specialty`, `specialty_cui`, `alt_specialty_cuis`, `doctype_cui`, `cuis`, `tuis`, `surface_forms`, `text`. Enables index-only filters like *"sentences in General Medicine notes, HPI section, mentioning C0001175"* before any vector search.

**Tuned defaults** (`--workers 16 --batch 32`) from a sweep on L4:

| workers | batch | wall | rate |
|---:|---:|---:|---:|
| 16 | 16 | 17.9s | 4,658/s |
| **16** | **32** | **15.5s** | **5,380/s** |
| 16 | 64 | — | HTTP 429 (TEI queue saturated) |
| 8  | 64 | 20.9s | 3,984/s |
| 32 | 64 | — | HTTP 429 |

16×64 and 32×64 overwhelm TEI's concurrent-request queue; 8×64 starves the GPU. 16×32 is the knee. If you bump `--workers`, re-tune `--batch` so `workers × batch` stays under TEI's effective concurrency budget (the script does exponential backoff on 429 so transient saturation doesn't crash the job).

Expected on MTSamples: **~83K sentences embedded + upserted in ~16s wall**. Collection is configured with `cosine` distance and `indexing_threshold=100` so segments get indexed promptly (default would leave small segments unindexed, falling back to brute-force search).

### 15. Create Neo4j constraints, indexes, and tier labels

```bash
bash scripts/create_neo4j_indices.sh
```

`scripts/load_neo4j.sh` calls this automatically at the end of the bulk-import step, so you don't need to run it again after a fresh load. Run it standalone when you want to re-assert the schema without re-importing — e.g. after restoring from the `neo4j_data.tar.zst` snapshot. Creates uniqueness constraints (`Concept.cui`, `Atom.aui`, `SemanticType.tui`, `Source.sab`), a range index on `(Atom.sab, Atom.code)` and `Concept.name`, full-text indexes `concept_name_fts` and `atom_str_fts`, and applies the four `:ClinicalCore` / `:ClinicalSupport` / `:ClinicalDiscipline` / `:Peripheral` tier labels.

### 16. (Experimental) Build the BioLORD-2023 UMLS Concept index

Two interchangeable variants — both write to the same `umls_concepts_biolord` Qdrant collection with the same `uuid5(cui)` point ids, so one can pick up where the other left off via the shared `data/biolord_concept_index.state` resume file.

**Variant A — TEI-backed (default, matches the rest of the pipeline):**

```bash
docker compose up -d biolord qdrant neo4j
python python/ingestion/umls/build_biolord_concept_index.py [--workers 8] [--batch 64] [--resume]
```

Embeds every `Concept.name` via BioLORD-2023 (served by a second TEI container on `localhost:8081`, `--pooling=mean` — BioLORD is a sentence-transformers model trained with mean-pooled + L2-normalized vectors, same as MedTE). Payload: `{cui, name}`. Neo4j is streamed in `--page-size` (default 20,000) keyset-paginated pages on `cui ASC`; each page is **LPT-first size-balanced** across `--workers` mp.Pool processes (one biolord HTTP session + one Qdrant client per worker), and within each shard items are **length-sorted descending before batching** so padding-to-longest waste in the forward pass is near-zero (same pattern as step 7's Stanza NER).

Sweep on L4 (8 workers, 40 K-concept sample, LPT-balanced shards) picked **8 × batch 64 at ~2,000/s** as the knee — beats 16×64 because extra workers only add TEI-queue contention, beats batch 128 by <1%, beats batch 256 by ~2× (at 256 TEI's dynamic batcher can't re-pack effectively). The script has a 30-retry / 8 s-max-delay backoff on HTTP 429 so transient saturation never kills the run.

**Variant B — TEI-free, local GPU (faster for one-time full rebuilds):**

```bash
docker compose stop biolord        # free VRAM for torch
docker compose up -d qdrant neo4j
python python/ingestion/umls/build_biolord_concept_index_local.py [--batch 1024] [--resume] [--half]
```

Loads BioLORD directly via `sentence-transformers` on the host GPU, streams concepts page-by-page from Neo4j, length-sorts each page, runs a single chunked forward pass at `--batch`, and upserts to Qdrant over **gRPC** (port 6334). No HTTP/JSON round-trips between Python and the embedder — every vector stays on-device until the batch is done, which typically gives 2–3× the TEI-variant throughput on the same hardware. Requires the venv's CUDA torch (already present for step 7's Stanza GPU path; `python -c 'import torch; print(torch.cuda.is_available())'`). `--half` casts the model to fp16 for another ~2× at a tiny numeric delta (BioLORD was trained fp32).

Both variants persist the last successfully-upserted CUI to `data/biolord_concept_index.state` after every page, so `--resume` is safe to re-run across variants. One-time cost on MTSamples-scale UMLS (~3.3 M concepts × 768-d): variant A ~30 min, variant B ~10–15 min with `--half`. The collection is the search side of the comparison in step 17.

### 17. (Experimental) Link entities via BioLORD nearest-neighbor — diff against the lexical baseline

```bash
docker compose up -d biolord qdrant
python python/ingestion/mtsamples/link_entities_to_cui_biolord.py [--workers 16] [--batch 32] [--min-score 0.7]
```

Semantic counterpart to step 11. Dedups the same universe of unique `expanded_text` values across `data/mtsamples_docs/*.json`, embeds each via BioLORD on `:8081`, and does a top-1 nearest-neighbor search against `umls_concepts_biolord` (built in step 16). Writes `data/entity_cui_biolord.jsonl`:

```json
{"text": "subpectoral pocket", "cui": "C0229909", "cui_name": "Pectoral region structure", "score": 0.81}
```

The `text` + `cui` columns diff directly against `data/entity_cui_lexical.jsonl` (step 12), so the delta between the two linkers is a one-line `join`. Below-threshold hits (cosine < `--min-score`) are written with `cui: ""` so the snapshot stays a complete coverage map rather than only the matches.

**Why we expect BioLORD to win where lexical loses.** The lexical linker scores candidate Concepts by BM25-style token overlap on the entity string in isolation, so `"copious irrigation"` resolves to `Copious` (the one-word finding literally named that) instead of the `Irrigation` procedure, and `"subpectoral pocket"` resolves to `Periodontal Pocket` on the `pocket` token even though the sentence is about chest-wall anatomy. BioLORD-2023 is fine-tuned on UMLS synonym and definition contrastive pairs, so semantically related Concepts cluster in the embedding space and lexically-similar-but-semantically-distant ones do not — the nearest neighbor of `"subpectoral pocket"` sits in the pectoral-region cluster, not the dental one.

**Scope of v1.**
- Snapshot-only — does not write back into per-doc JSONs. Audit the diff before deciding whether to rewire the main pipeline (steps 13–14 still consume step 11's lexical CUIs in-place).
- Embeds each entity in isolation for apples-to-apples comparison with the lexical linker. A follow-up can pass the enclosing sentence as context for additional lift on polysemous mentions.
- TUIs are not re-fetched; derive them from the resolved CUIs via the same `HAS_SEMTYPE` pass as step 11 phase 4 if/when needed.
