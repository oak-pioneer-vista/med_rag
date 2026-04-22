"""Sentence-chunk every Section and tag each sentence with CUI/TUI/surface-form sets.

For every Section across data/mtsamples_docs/*.json: splits the section
text into sentences using Stanza's `mimic` clinical tokenizer (handles
medical abbreviations, operative-note comma-pseudo-paragraphs, and
MTSamples-style surface conventions better than a bare regex). For
each sentence, checks the presence of any linked entity's surface form
inside the sentence text (word-boundary, case-insensitive). Entities
have already been CUI/TUI-resolved by link_entities_to_cui.py; matched
entities contribute their identifiers to three per-sentence sets.

Output is written back into each Section as a new `sentences` list:

  {
    "chunk_id"     : stable sentence-level id "{section_chunk_id}:s{idx}",
                     e.g. "1003:procedure:s5". This is the join key used
                     by Neo4j (and any other downstream store) that wants
                     to address individual sentences independent of their
                     positional index.
    "text"         : sentence text,
    "cuis"         : sorted list of UMLS CUIs in this sentence,
    "tuis"         : sorted list of semantic TUIs in this sentence,
    "surface_forms": sorted list of entity surface forms present.
  }

Parallelism: mp.Pool with spawn start method, `--workers` processes.
Each worker loads the Stanza mimic tokenizer once in its initializer
(so it survives across pool tasks), then processes its shard's
sections in batches of `--batch` via `nlp.bulk_process`. Per-doc
surface-form indices are compiled once (single word-boundary
alternation regex per doc) so a sentence is scanned against all
candidate surface forms in one pass.

Prereqs:
  - data/mtsamples_docs/*.json with entities linked via
    link_entities_to_cui.py (needs `cui` and `tuis` populated)
  - Stanza mimic/i2b2 models:
      python -c "import stanza; stanza.download('en', package='mimic', processors={'ner':'i2b2'})"

Usage:
  python python/ingestion/mtsamples/chunk_sentences.py [--workers 8] [--batch 128] [--cpu]
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

# Filter out extremely short surface forms (single characters) from the
# surface-index -- they're noise-prone.
MIN_SURFACE_LEN = 2


def build_doc_surface_index(doc: dict) -> dict[str, tuple[str, tuple[str, ...]]]:
    """Return {surface_lower: (cui, tuis_tuple)} merged across the doc's entities."""
    idx: dict[str, tuple[str, set[str]]] = {}
    for sec in doc.get("sections", []):
        for e in sec.get("entities") or []:
            cui = e.get("cui") or ""
            if not cui:
                continue
            surface = (e.get("surface_text") or "").strip()
            if len(surface) < MIN_SURFACE_LEN:
                continue
            surface_lower = surface.lower()
            tuis = set(e.get("tuis") or [])
            if surface_lower in idx:
                existing_cui, existing_tuis = idx[surface_lower]
                idx[surface_lower] = (existing_cui, existing_tuis | tuis)
            else:
                idx[surface_lower] = (cui, set(tuis))
    return {k: (cui, tuple(sorted(tuis))) for k, (cui, tuis) in idx.items()}


def compile_surface_pattern(surfaces: list[str]) -> re.Pattern | None:
    """Single word-boundary alternation; longer surfaces preferred (max-munch)."""
    if not surfaces:
        return None
    surfaces_sorted = sorted(surfaces, key=len, reverse=True)
    return re.compile(
        r"\b(?:" + "|".join(re.escape(s) for s in surfaces_sorted) + r")\b",
        re.IGNORECASE,
    )


def annotate_sentence(
    sent_text: str,
    idx: dict,
    pattern: re.Pattern | None,
    chunk_id: str,
) -> dict:
    cuis: set[str] = set()
    tuis: set[str] = set()
    surfaces: set[str] = set()
    if pattern:
        for m in pattern.finditer(sent_text):
            surface_lower = m.group(0).lower()
            hit = idx.get(surface_lower)
            if hit:
                cui, tui_tuple = hit
                cuis.add(cui)
                tuis.update(tui_tuple)
                surfaces.add(surface_lower)
    return {
        "chunk_id": chunk_id,
        "text": sent_text,
        "cuis": sorted(cuis),
        "tuis": sorted(tuis),
        "surface_forms": sorted(surfaces),
    }


# ---------- mp.Pool worker state ----------

_NLP = None
_WID = None


def _init_worker(use_gpu: bool) -> None:
    import stanza

    global _NLP, _WID
    _WID = f"{os.getpid()}/{uuid.uuid4().hex[:6]}"
    print(f"[worker {_WID}] loading stanza mimic tokenizer (use_gpu={use_gpu})",
          flush=True)
    # tokenize-only pipeline (no NER/pos/lemma) -- sentence boundaries only.
    _NLP = stanza.Pipeline(
        lang="en",
        package=None,
        processors={"tokenize": "mimic"},
        use_gpu=use_gpu,
        verbose=False,
        download_method="reuse_resources",
    )
    print(f"[worker {_WID}] pipeline ready", flush=True)


def _process_shard(args: tuple) -> dict:
    """Pool worker: tokenize sections in this shard, annotate and write back."""
    import stanza

    shard_paths, batch_size = args
    paths = [Path(p) for p in shard_paths]
    if not paths:
        return {"wid": _WID, "docs": 0, "sections": 0, "sentences": 0,
                "sentences_with_hit": 0, "elapsed_s": 0.0}

    t0 = time.time()
    docs = [(p, json.loads(p.read_text(encoding="utf-8"))) for p in paths]

    # Build per-doc surface index + compiled pattern once per doc, reuse
    # across all sections of that doc.
    doc_indices: dict[int, tuple[dict, re.Pattern | None]] = {}
    for di, (_, d) in enumerate(docs):
        idx = build_doc_surface_index(d)
        pat = compile_surface_pattern(list(idx.keys())) if idx else None
        doc_indices[di] = (idx, pat)

    # Flatten non-empty sections for bulk_process batching.
    items: list[tuple[int, int, str]] = []
    for di, (_, d) in enumerate(docs):
        for si, sec in enumerate(d.get("sections", [])):
            t = sec.get("text") or ""
            if t.strip():
                items.append((di, si, t))
            else:
                sec["sentences"] = []

    # Length-sort descending to keep padding waste low across a batch.
    items.sort(key=lambda it: -len(it[2]))

    n_sections = len(items) + sum(
        1 for _, d in docs for sec in d.get("sections", []) if not (sec.get("text") or "").strip()
    )
    print(f"[worker {_WID}] batching {len(items)} non-empty sections "
          f"from {len(paths)} docs (batch={batch_size})",
          flush=True)

    n_sentences = 0
    n_sentences_hit = 0
    for bi in range(0, len(items), batch_size):
        chunk = items[bi : bi + batch_size]
        stanza_docs = [stanza.Document([], text=t) for _, _, t in chunk]
        _NLP.bulk_process(stanza_docs)
        for (di, si, _), sdoc in zip(chunk, stanza_docs):
            idx, pat = doc_indices[di]
            section = docs[di][1]["sections"][si]
            section_chunk_id = section.get("chunk_id", "")
            sents_out: list[dict] = []
            sent_idx = 0
            for sent in sdoc.sentences:
                text = sent.text.strip() if sent.text else ""
                if not text:
                    continue
                chunk_id = f"{section_chunk_id}:s{sent_idx}"
                rec = annotate_sentence(text, idx, pat, chunk_id)
                sents_out.append(rec)
                sent_idx += 1
                n_sentences += 1
                if rec["cuis"]:
                    n_sentences_hit += 1
            section["sentences"] = sents_out

    for p, d in docs:
        p.write_text(json.dumps(d, ensure_ascii=False), encoding="utf-8")

    elapsed = time.time() - t0
    print(f"[worker {_WID}] done: {len(paths)} docs, {n_sections} sections, "
          f"{n_sentences} sentences ({n_sentences_hit} with >=1 CUI hit) "
          f"in {elapsed:.1f}s",
          flush=True)
    return {"wid": _WID, "docs": len(paths), "sections": n_sections,
            "sentences": n_sentences, "sentences_with_hit": n_sentences_hit,
            "elapsed_s": elapsed}


def _shard(seq: list, n: int) -> list[list]:
    k, m = divmod(len(seq), n)
    out, i = [], 0
    for shard_idx in range(n):
        size = k + (1 if shard_idx < m else 0)
        out.append(seq[i : i + size])
        i += size
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--docs", type=Path, default=DOCS_DIR)
    ap.add_argument("--workers", type=int, default=8,
                    help="pool worker count (each loads Stanza mimic once)")
    ap.add_argument("--batch", type=int, default=128,
                    help="sections per Stanza bulk_process batch (per worker)")
    ap.add_argument("--cpu", action="store_true",
                    help="force CPU (default: GPU if available). Stanza "
                         "tokenize alone is fast on CPU; the main cost is "
                         "pipeline load, which is one-shot per worker.")
    args = ap.parse_args()

    paths = [str(p) for p in sorted(args.docs.glob("*.json"))]
    if not paths:
        raise SystemExit(f"no JSON files in {args.docs}")

    use_gpu = not args.cpu
    workers = max(1, min(args.workers, len(paths)))
    shards = _shard(paths, workers)
    sizes = [len(s) for s in shards]
    print(f"dispatching {len(paths):,} docs across {workers} workers "
          f"(shard docs={sizes}, use_gpu={use_gpu}, batch={args.batch})",
          flush=True)

    ctx = mp.get_context("spawn")
    t0 = time.time()
    with ctx.Pool(processes=workers, initializer=_init_worker,
                  initargs=(use_gpu,)) as pool:
        results = pool.map(
            _process_shard,
            [(shard, args.batch) for shard in shards],
        )

    total = {"docs": 0, "sections": 0, "sentences": 0, "sentences_with_hit": 0}
    for r in results:
        for k in total:
            total[k] += r.get(k, 0)
    per_worker = sorted(r.get("elapsed_s", 0.0) for r in results)
    wall = time.time() - t0
    print(
        f"done: {total['docs']:,} docs, {total['sections']:,} sections, "
        f"{total['sentences']:,} sentences "
        f"({total['sentences_with_hit']:,} with >=1 CUI hit = "
        f"{100*total['sentences_with_hit']/max(total['sentences'],1):.1f}%) "
        f"in {wall:.1f}s wall  "
        f"[per-worker work {min(per_worker):.1f}..{max(per_worker):.1f}s]",
        flush=True,
    )


if __name__ == "__main__":
    main()
