# Learning notes

Running log of non-obvious things learned while building this pipeline.

## Short queries against a sentence-level encoder

**Issue.** MedTE (`MohammadKhodadad/MedTE-cl15-step-8000`) is a GTE-family BertModel contrastively pretrained on **sentence-length inputs** with mean pooling + L2 normalization. Sending a two-word query like `"lymphoblastic Leukemia"` uses it outside of its training distribution and noticeably degrades retrieval.

**Observation.** Same corpus (`mtsamples_sections`, 43,244 points), same encoder, three different queries:

| Query | Style | Peak cosine | Notes |
|---|---|---:|---|
| `"Left testicular swelling for one day.  Testicular Ultrasound"` | sentence (copied from a doc) | **0.969** | exact-match retrieval |
| `"Retrieve all patients diagnosed with Lymphoblastic Leukemia"` | instruction-style sentence | **0.688** | top hit is a family-history false positive; index case at rank 3 |
| `"lymphoblastic Leukemia"` | bare concept, 2 words | **0.656** | coherent neighborhood, but low absolute scores |

Peak score drops from 0.97 → 0.66 just by shortening the query, even when the concept is identical to something in the corpus.

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
