# Learning notes

Running log of non-obvious things learned while building this pipeline.

## Short queries against a sentence-level encoder

**Issue.** MedTE (`MohammadKhodadad/MedTE-cl15-step-8000`) is a GTE-family BertModel contrastively pretrained on **sentence-length inputs** with mean pooling + L2 normalization. Sending a two-word query like `"lymphoblastic Leukemia"` uses it outside of its training distribution and noticeably degrades retrieval.

**Observation.** Same corpus (`mtsamples_sections`, 43,244 points), same encoder, two different queries:

| Query | Style | Peak cosine | Notes |
|---|---|---:|---|
| `"Retrieve all patients diagnosed with Lymphoblastic Leukemia"` | instruction-style sentence | **0.688** | top hit is a family-history false positive; index case at rank 3 |
| `"lymphoblastic Leukemia"` | bare concept, 2 words | **0.656** | coherent neighborhood, but low absolute scores |

Against a verbatim sentence copied from a source document, the same encoder peaks around **0.97** — exact-match retrieval. Peak score drops from ~0.97 → 0.66 just by shortening the query, even when the concept is identical to something in the corpus.

**Why.**
1. The encoder's contrastive objective shapes the embedding geometry specifically for sentence-length inputs. Short inputs land in a sparser, less discriminative region of the space.
2. After WordPiece tokenization, two words become ~4–6 subword tokens. Mean pooling over so few non-special tokens is dominated by tokenizer artifacts (casing, subword splits) rather than clinical semantics.
3. L2-normalized cosine over low-information vectors compresses into a narrow band — everything in the corpus becomes "moderately similar," so the ranking signal is weaker.

**Mitigations, in order of effort.**
- **Reformulate the query as a short declarative sentence** before embedding. `"lymphoblastic Leukemia"` → `"Patient diagnosed with lymphoblastic leukemia."` This is the cheapest fix and typically recovers most of the gap.
- **Hybrid retrieval**: pair dense retrieval with a lexical index (BM25 over the same section text) and fuse scores. Rescues exact-term matches that the dense encoder misses on short queries.
- **Instruction-tuned / asymmetric encoder** (e.g. `intfloat/e5-*` with its `query:` / `passage:` prefixes, BGE models): trained explicitly to embed short queries against longer passages, so the asymmetry is built in.
- **Cross-encoder reranker** on the top-100 bi-encoder hits. Doesn't fix the short-query embedding itself but largely cancels the problem at retrieval time, at the cost of latency.

**Takeaway.** When evaluating a retriever, don't just measure "can it find the document." Measure it across query lengths and phrasings — a model that looks great on sentence-length queries can collapse on the two- and three-word queries users actually type.

## TEI for batched inference (vs. in-process transformers)

**Context.** The first cut of `embed_sections.py` loaded MedTE in each dask worker via `transformers.AutoModel.from_pretrained(...)` and ran `model(**enc)` directly. Works, but has real problems at scale.

**Issues with the in-process approach.**
- **GPU memory contention.** With `num_workers=16`, every dask process tried to hold its own copy of the model on the GPU. On a 24 GB L4 this nearly OOMs immediately; even when it fits, you're paying the weights cost N times for no gain.
- **Model-load latency on every partition boundary.** `AutoModel.from_pretrained` takes seconds per worker on first call; repartitioning (or worker crashes) amortizes poorly.
- **Batching stops at the worker boundary.** Each worker batches only its own partition's windows, so the GPU sees many small batches instead of one large queue — SM utilization stays low.
- **Every client needs the pytorch + CUDA toolchain** installed on the host, which is a painful ask for collaborators.

**Switching to TEI (`ghcr.io/huggingface/text-embeddings-inference`).**
- Serves MedTE once from a single container holding the weights in fp16 on the GPU; scales via server-side concurrency (`max-concurrent-requests`, `max-batch-tokens`, `max-client-batch-size`).
- Workers become pure HTTP clients: no torch, no CUDA, no model-loading cost — just `requests.post("/embed", json={"inputs": [...]})`. The only local model artifact we still need is the **tokenizer**, kept for token-budget packing (TEI also exposes `/tokenize`, but local is faster and doesn't burn RTT).
- Dynamic server-side batching means small per-worker batches get merged into larger GPU batches across concurrent requests — throughput goes up, not down, when you add workers.

**Gotchas discovered the hard way.**
- **TEI's default pooling is `cls`.** For GTE-family checkpoints (including MedTE), training used **mean pooling + L2 normalize**. CLS pooling still produces usable vectors, but they don't reflect the geometry the encoder was optimized for. Fix: pass `--pooling=mean` in the compose command and verify via `curl localhost:8080/info` → `"pooling":"mean"` before ingesting. If you forget, you have to drop the Qdrant collection and re-embed.
- **`--max-client-batch-size` caps the per-request batch.** Match your client-side batch to this value so a single POST fills one server-side batch cleanly. Going over just splits server-side — but going well under leaves throughput on the table.
- **`auto_truncate=true`** is on by default. The server silently truncates oversized inputs to `max_input_length` (512 for MedTE). Fine for us (our packing targets 350 tokens and leaves headroom for specials), but worth knowing — the server won't error on oversized inputs.
- **CUDA compute-capability tag.** The image is built per compute capability (`89-latest` = 8.9 = L4/Ada). Wrong tag = cryptic startup failure. Change it if you move to A100 (8.0), H100 (9.0), etc.

**Takeaway.** For any fan-out embedding pipeline, front the model with TEI (or another dedicated inference server) and make the dask/worker layer a batching HTTP client. You get better GPU utilization, simpler client dependencies, and clean separation of "shape my data" from "run the model" — at the cost of one extra process to operate.

## Chunking: sentence-packed windows, 200 tokens, 10% overlap

**Strategy.** Each section is split into sentences, then greedily packed into windows of up to **200 tokens** (MedTE tokenizer, excluding specials), with ~**10% overlap** (20 tokens ≈ one trailing sentence) carried into the next window. See `_pack_sentences` in `python/ingestion/mtsamples/embed_sections.py`.

**Why sentence-aligned.** Splitting on sentence boundaries keeps each chunk semantically coherent, which matters for a sentence-trained encoder (see note above) — the alternative, fixed-length subword windows, routinely cuts mid-phrase and degrades embedding quality. Packing *multiple* sentences per window (vs. one-sentence-per-chunk) gives the encoder enough context to disambiguate pronouns and short clinical phrases, while staying well inside MedTE's 512-token limit.

**Why 200 tokens.** Tight enough that a chunk is usually a single coherent thought cluster (so retrieval hits the right passage, not a whole section), loose enough that one sentence almost never overflows. Leaves comfortable headroom below MedTE's 512-token max so we don't depend on TEI's `auto_truncate` silently lopping content.

**Why 10% overlap.** Cross-boundary context without duplicating too much. 20 tokens is roughly one short sentence — enough that a concept introduced at the end of one window is still present at the start of the next, so retrieval doesn't miss passages where the answer straddles a chunk boundary. More overlap inflates the vector store for diminishing gain; less risks boundary blind spots.

**Edge case.** A single sentence longer than 200 tokens forms its own window and gets truncated by TEI (`auto_truncate=true`). Rare in MTSamples but worth knowing — if truncation becomes material on another corpus, swap to a real sentence splitter (`syntok`, `nltk.sent_tokenize`) and/or fall back to token-window splitting within oversized sentences.

## Multi-concept query dilution

**Observation.** A comma-joined query mixing two unrelated findings retrieves worse than either finding on its own, because mean-pooled embeddings average the two concepts into a single centroid that belongs to neither.

**Worked example** against `mtsamples_sections` (MedTE, mean pool, cosine):

| Query | Rank-1 score | Rank-1 text |
|---|---:|---|
| `"shortness of breath, significant bradycardia"` | **0.7359** | "No shortness of breath." (negated — false positive) |
| `"significant bradycardia"` | **0.8456** | "Sinus bradycardia at 58 bpm, mild inferolateral ST abnormalities." |

Dropping the competing half of the query lifted peak score from 0.74 → 0.85 and pushed the entire top-5 into bradycardia-relevant territory. The multi-concept version had a **negated** top hit — the bradycardia signal wasn't strong enough to outrank "no shortness of breath" because it was diluted by the averaged SoB half.

**Why.** Mean pooling of `"A, B"` produces roughly `½(vec(A) + vec(B))`. If the corpus has strong matches for A alone or B alone, they sit closer to their own centroid than to the midpoint. Result: the two halves compete, and whichever has denser coverage in the corpus (here, "shortness of breath" — extremely common in ROS sections) wins on raw density while the other concept gets squeezed out.

**Implications.**
- For multi-concept retrieval, **issue one query per concept and fuse results** (RRF or max-score) rather than concatenating terms. Concatenation only works when the concepts reliably co-occur in target passages.
- This compounds the negation problem (see TODO below): not only can a negated mention outrank a positive one on a single-concept query, but query dilution makes it more likely by lowering the score the positive mentions need to beat.
- Also compounds short-query degradation (see top of file): short + multi-concept is the worst case, because you're both out of distribution *and* asking the encoder to average two weak vectors.

## Entity linking / grounding against UMLS in Neo4j

**What this is.** Taking free-text mentions inside clinical chunks (`"risperidone"`, `"coronary artery disease"`, `"IDDM"`) and mapping each to a canonical **UMLS CUI**, so downstream retrieval and reasoning can operate on stable concept IDs instead of surface strings. Dense retrieval alone doesn't do this — it finds *similar passages*, not *the same concept under a different name*.

**Why persist the RRFs into Neo4j instead of loading UMLS per-query.**
- The full Metathesaurus is ~17 GB of pipe-delimited text; loading it per process is a non-starter. One bulk import into Neo4j (`umls_to_neo4j_csv.py` + `neo4j-admin database import`) gives every client fast, indexed access.
- The graph shape — `Concept` ↔ `Atom` (synonyms across sources) ↔ `SemanticType`, plus `RELATES` / `IS_A` / `DEFINED_BY` — is exactly what grounding needs: one lookup can resolve a synonym to its CUI *and* walk to parents, siblings, definitions, and semantic types in the same query.
- Shared infrastructure: the same graph powers `may_treat` edges, hierarchy walks, and atom-level fuzzy lookup. Entity linking is one consumer among several, not a bespoke data store.

**How linking will work (lookup side).**
1. **Candidate generation** — for each mention string, hit Neo4j's fulltext indexes (`atom_str_fts` for synonyms, `concept_name_fts` for preferred names) to pull top-K candidates. These are the indexes documented in CLAUDE.md and exist specifically to make this step sub-100 ms.
2. **Disambiguation** — when multiple CUIs come back (the common case: `"discharge"` → hospital discharge vs. bodily discharge), score candidates with (a) semantic-type priors from the surrounding section (`DISCHARGE SUMMARY` → prefer event/procedure TUIs), (b) context similarity between the chunk embedding and each candidate's preferred-name embedding, and (c) source-reliability priors from `MRRANK` (SNOMEDCT_US > MTH > …).
3. **Write-back** — store resolved `(chunk_id, cui, mention_span, score)` as Qdrant payload metadata (for filtering/faceting) and optionally as `(:Chunk)-[:MENTIONS]->(:Concept)` edges in Neo4j (for graph-side joins like "all chunks mentioning any descendant of `C0010068` Coronary Artery Disease").

**Why this matters for retrieval quality.**
- **Lexical variation robustness.** `"myocardial infarction"`, `"MI"`, `"heart attack"`, `"STEMI"` all collapse to a single CUI. Dense retrieval handles *some* of this, but it breaks on acronyms, misspellings, and rare synonyms — exactly where a curated synonym table wins.
- **Hierarchy-aware recall.** Grounded CUIs let a query for `"diabetes mellitus"` expand through `IS_A` to hit `"type 2 diabetes"`, `"IDDM"`, `"gestational diabetes"`, etc. Pure dense retrieval often misses narrower terms whose surface form diverges from the query.
- **Negation / assertion are per-mention, not per-chunk.** Once mentions are spans with CUIs, an assertion classifier can tag each span individually — addressing the negation TODO below at the right granularity instead of trying to shape the whole-chunk embedding around polarity.
- **Cross-dataset joins.** The CUI is the lingua franca. MTSamples chunks, MED-RT `may_treat` edges, and any future corpus (PubMed, discharge summaries, claims) all resolve to the same node, so a query can pivot from "passage mentioning X" to "drugs that treat X" to "passages mentioning those drugs" in one graph.

**Gotchas / design notes.**
- **The RRF dump is big, and most of it is noise for our use case.** `umls_to_neo4j_csv.py` already applies `--english-only --drop-suppressed`; revisit filtering further (by SAB allowlist) if graph-write times become painful.
- **Atom explosion.** `Atom` nodes vastly outnumber `Concept` nodes. Fulltext search should run against `atom_str_fts` for recall and then collapse to the parent CUI via `HAS_ATOM`, not the other way around.
- **Case-insensitive lookup must use fulltext.** A RANGE index on `Concept.name` exists but is defeated by `toLower()` / `CONTAINS` — full label scan (~30 s on 3.3M nodes). See the query-pattern notes in `CLAUDE.md`.
- **Disambiguation is the hard part**, not candidate generation. Plan to evaluate against a held-out, hand-linked subset of MTSamples chunks before claiming a linker works.
- **Off-the-shelf linkers (scispaCy, QuickUMLS, MedCAT)** are worth evaluating as the first pass — they already handle tokenization, abbreviation expansion, and some disambiguation, and can emit CUIs that we then join against the graph. Building a linker from scratch on top of Neo4j lookups is a larger project than it looks.

## TODO: negation-aware embedding

**Status.** Not currently implemented. Revisit and fix.

**Prior discussion / design notes.** <https://claude.ai/chat/3f898821-0901-412b-9d22-7adb40d2f8f2>

**Problem.** Dense encoders like MedTE treat `"patient denies chest pain"` and `"patient reports chest pain"` as near-identical in vector space — negation cues (`no`, `denies`, `without`, `ruled out`, `negative for`) barely move the embedding. Retrieval for a query like `"chest pain"` will surface negated mentions alongside positive findings, which is clinically wrong.

**Why it matters here.** Clinical notes are dense with negated findings (ROS, pertinent negatives, ruled-out differentials). Any downstream use — cohort selection, QA, symptom matching — that can't distinguish assertion polarity will produce false positives.

**Options to consider when revisiting.**
- Assertion/negation tagging at chunk time (NegEx, medspaCy, or a fine-tuned BERT assertion classifier) and store polarity as payload metadata; filter or rerank at query time.
- Split or flag negated spans before embedding so the encoder sees only positive assertions, or embed negated spans into a separate namespace/field.
- Cross-encoder reranker trained on assertion-sensitive pairs over the top-k bi-encoder hits.
- Evaluate negation-aware clinical encoders (e.g. CORE-style, or models trained with hard-negative pairs that include polarity flips).
