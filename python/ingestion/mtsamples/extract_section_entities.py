"""Per-section NER + span recording for every MTSamples doc (multi-GPU-process).

Runs Stanza's `mimic` pipeline with the `i2b2` NER processor over each
Section's text (not windowed -- sections are the logical unit here
because downstream grounding lines up section -> Concept, not chunk ->
Concept). Each section's `entities` list in its per-doc JSON is filled
with one record per mention, carrying three text representations so
downstream consumers can pick the one that fits their use case:

  {
    "surface_text"    : literal slice section_text[start_char:end_char]
                        -- the authoritative "what the source says"
                        (use this for display and span-to-text joins)
    "recognized_text" : Stanza's `entity.text` -- same as surface in the
                        common case, but Stanza can normalize whitespace
                        across tokens, so it may differ on adjacent-
                        punctuation edges. Use this for model-derived
                        text matching.
    "resolved_text"   : `recognized_text` with any known abbreviations
                        (from the doc's `abbreviations` map built in
                        build_abbreviations.py) substituted with their
                        expansions. Equal to `recognized_text` when no
                        substitution fires. Use this for UMLS/Neo4j
                        grounding and dense-retrieval keying, where the
                        expanded form matches more Concept atoms.
    "type"            : i2b2 label (PROBLEM | TEST | TREATMENT)
    "start_char"      : offset within the section's text
    "end_char"        : offset within the section's text
  }

Parallelism: doc paths are split into `--workers` equal shards and
handed to a `multiprocessing.Pool`. The pool's initializer loads
Stanza once per worker process into a module-level global, so
(unlike the earlier dask.bag version) workers do not reload the
pipeline on every task. Each worker processes its shard's sections
in batches of `--batch` and writes per-doc JSONs back to disk.

Prereqs:
  - data/mtsamples_docs/*.json written by parse_mtsamples.py (and
    ideally build_abbreviations.py, for richer `resolved_text` output)
  - Stanza mimic/i2b2 models:
      python -c "import stanza; stanza.download('en', package='mimic', processors={'ner':'i2b2'})"
  - If running with use_gpu=True and the host cuDNN lags torch's
    expected version, prefix LD_LIBRARY_PATH with the venv's
    nvidia/cudnn/lib + nvidia/cublas/lib so torch's bundled libs win.

Usage:
  python python/ingestion/mtsamples/extract_section_entities.py [--workers 8] [--batch 16] [--cpu]
"""

from __future__ import annotations

import argparse
import json
import multiprocessing as mp
import os
import re
import time
import uuid
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent.parent.parent
DOCS_DIR = REPO / "data" / "mtsamples_docs"

TOKEN_RE = re.compile(r"\b[A-Za-z][A-Za-z0-9-]*\b")

# Set in each worker by _init_worker(); reused across every pool task on
# that worker so Stanza's pipeline loads exactly once per process.
_NLP = None
_WID = None


def resolve_with_abbrevs(text: str, abbrev_upper: dict[str, str]) -> str:
    if not abbrev_upper:
        return text

    def repl(m: re.Match) -> str:
        tok = m.group(0)
        return abbrev_upper.get(tok.upper(), tok)

    return TOKEN_RE.sub(repl, text)


def entities_from_stanza_doc(
    sdoc,
    section_text: str,
    abbrev_upper: dict[str, str],
) -> list[dict]:
    out: list[dict] = []
    for ent in sdoc.entities:
        recognized = ent.text
        surface = section_text[ent.start_char : ent.end_char]
        resolved = (
            resolve_with_abbrevs(recognized, abbrev_upper) if abbrev_upper else recognized
        )
        out.append({
            "surface_text": surface,
            "recognized_text": recognized,
            "resolved_text": resolved,
            "type": ent.type,
            "start_char": ent.start_char,
            "end_char": ent.end_char,
        })
    return out


def _init_worker(use_gpu: bool) -> None:
    """Pool initializer -- called once per worker process at pool startup.

    Loads the Stanza pipeline into module-level globals so subsequent
    pool.map tasks on this worker reuse the same pipeline rather than
    reloading.
    """
    import stanza

    global _NLP, _WID
    _WID = f"{os.getpid()}/{uuid.uuid4().hex[:6]}"
    print(f"[worker {_WID}] loading pipeline (use_gpu={use_gpu})", flush=True)
    _NLP = stanza.Pipeline(
        lang="en",
        package=None,
        processors={"tokenize": "mimic", "ner": "i2b2"},
        use_gpu=use_gpu,
        verbose=False,
        download_method="reuse_resources",
    )
    print(f"[worker {_WID}] pipeline ready", flush=True)


def _process_shard(args: tuple) -> dict:
    """Pool worker task: process one shard's docs with the cached pipeline."""
    import stanza

    shard_paths, batch_size = args
    paths = [Path(p) for p in shard_paths]
    if not paths:
        return {"wid": _WID, "docs": 0, "ents": 0, "resolved_diff": 0, "elapsed_s": 0.0}

    docs = [(p, json.loads(p.read_text(encoding="utf-8"))) for p in paths]
    items: list[tuple[int, int, str]] = []
    for di, (_, d) in enumerate(docs):
        for si, section in enumerate(d.get("sections", [])):
            text = section.get("text") or ""
            if text.strip():
                items.append((di, si, text))
            else:
                section["entities"] = []

    # Length-bucket within the shard: Stanza's bulk_process pads every
    # sequence in a batch to the longest one in that batch. Section
    # lengths span p50=23 .. p99=863 .. max=2,819 MedTE tokens, so mixing
    # a 2K-token PROCEDURE section with sixty-three 10-token ANESTHESIA
    # sections wastes the whole batch on padding. Descending sort groups
    # similar-length items into the same batch. Output order is preserved
    # by writing back via (doc_idx, section_idx), which are independent
    # of processing order.
    items.sort(key=lambda it: -len(it[2]))

    print(f"[worker {_WID}] batching {len(items)} sections from {len(paths)} docs "
          f"at batch={batch_size} (length-sorted)",
          flush=True)

    total_ents = 0
    n_resolved_diff = 0
    t0 = time.time()
    for bi in range(0, len(items), batch_size):
        chunk = items[bi : bi + batch_size]
        stanza_docs = [stanza.Document([], text=t) for _, _, t in chunk]
        _NLP.bulk_process(stanza_docs)

        for (di, si, section_text), sdoc in zip(chunk, stanza_docs):
            abbrev_map = docs[di][1].get("abbreviations") or {}
            abbrev_upper = {k.upper(): v for k, v in abbrev_map.items()}
            ents = entities_from_stanza_doc(sdoc, section_text, abbrev_upper)
            n_resolved_diff += sum(
                1 for e in ents if e["resolved_text"] != e["recognized_text"]
            )
            docs[di][1]["sections"][si]["entities"] = ents
            total_ents += len(ents)

        done = min(bi + batch_size, len(items))
        rate = done / max(time.time() - t0, 1e-9)
        print(f"[worker {_WID}]   {done:>6}/{len(items)} sections  "
              f"({rate:.1f} sec/s)",
              flush=True)

    for p, d in docs:
        p.write_text(json.dumps(d, ensure_ascii=False), encoding="utf-8")

    elapsed = time.time() - t0
    print(f"[worker {_WID}] done: {len(paths)} docs, {total_ents} ents, "
          f"{n_resolved_diff} resolved_diff in {elapsed:.1f}s",
          flush=True)
    return {
        "wid": _WID,
        "docs": len(paths),
        "ents": total_ents,
        "resolved_diff": n_resolved_diff,
        "elapsed_s": elapsed,
    }


def _doc_weight(path: str) -> int:
    """Cheap proxy for Stanza NER work on one doc: total chars across sections.

    Reads the JSON to sum section text lengths. Stanza's biLSTM-CRF runtime
    scales ~linearly in input tokens and sublinearly per-section due to
    batching, so char count (a strong proxy for token count on clinical
    text) tracks wall-time contribution well.
    """
    try:
        d = json.loads(Path(path).read_text(encoding="utf-8"))
    except Exception:
        return 1
    return sum(len(s.get("text") or "") for s in d.get("sections", [])) or 1


def _shard_lpt(paths: list[str], n: int) -> list[list[str]]:
    """Longest-Processing-Time-first bin packing by doc char weight.

    Sort docs by weight descending, greedy-assign each to the currently-
    least-loaded bin. Gives makespan within 4/3 of optimal and in practice
    flattens the slowest-shard tail from ~95s → <75s on this corpus vs the
    equal-doc-count split that lumps several long procedural notes together.
    """
    weights = [_doc_weight(p) for p in paths]
    order = sorted(range(len(paths)), key=lambda i: -weights[i])
    loads = [0] * n
    bins: list[list[str]] = [[] for _ in range(n)]
    for idx in order:
        b = min(range(n), key=lambda k: loads[k])
        bins[b].append(paths[idx])
        loads[b] += weights[idx]
    return bins


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--docs", type=Path, default=DOCS_DIR)
    ap.add_argument("--workers", type=int, default=8,
                    help="multiprocessing worker count. Tuned via sweep on L4: "
                         "8 is optimal; 6 within 2s of best, >=12 regress as CUDA "
                         "context-switching eats the marginal parallelism, 32 OOMs")
    ap.add_argument("--batch", type=int, default=16,
                    help="sections per Stanza bulk_process batch (per worker). "
                         "Tiny batches look pathological but within-shard length-"
                         "bucketing means padding is near-zero, so 16/64/1024 are "
                         "within 2s of each other; 16 wins by a hair and keeps "
                         "activation memory lowest for the high-worker case")
    ap.add_argument("--cpu", action="store_true",
                    help="force CPU (default: use GPU if available)")
    args = ap.parse_args()

    paths = [str(p) for p in sorted(args.docs.glob("*.json"))]
    if not paths:
        raise SystemExit(
            f"no JSON files in {args.docs} -- run "
            f"python/ingestion/mtsamples/parse_mtsamples.py first"
        )

    use_gpu = not args.cpu
    workers = max(1, min(args.workers, len(paths)))
    shards = _shard_lpt(paths, workers)
    sizes = [len(s) for s in shards]
    shard_chars = [sum(_doc_weight(p) for p in s) for s in shards]
    print(f"dispatching {len(paths):,} docs to {workers} workers "
          f"(shard docs={sizes}, shard chars={shard_chars}, "
          f"use_gpu={use_gpu}, batch={args.batch})",
          flush=True)

    # `spawn` start method -- cleaner for CUDA (avoids inherited CUDA
    # context from the parent); each child initializes its own context
    # in `_init_worker`.
    ctx = mp.get_context("spawn")
    wall_t0 = time.time()
    with ctx.Pool(processes=workers, initializer=_init_worker,
                  initargs=(use_gpu,)) as pool:
        results = pool.map(
            _process_shard,
            [(shard, args.batch) for shard in shards],
        )

    total = {"docs": 0, "ents": 0, "resolved_diff": 0}
    for r in results:
        for k in total:
            total[k] += r.get(k, 0)

    wall = time.time() - wall_t0
    per_worker = sorted([r.get("elapsed_s", 0.0) for r in results])
    print(
        f"done: {total['ents']:,} entities across {total['docs']:,} docs "
        f"({total['resolved_diff']:,} with resolved_text != recognized_text) "
        f"in {wall:.1f}s wall  "
        f"[per-worker work {min(per_worker):.1f}..{max(per_worker):.1f}s]",
        flush=True,
    )


if __name__ == "__main__":
    main()
