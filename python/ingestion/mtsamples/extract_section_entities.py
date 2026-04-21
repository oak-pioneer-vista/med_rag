"""Per-section NER + span recording for every MTSamples doc (dask parallel).

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

Parallelism: docs are sharded across `--workers` dask worker processes;
each worker owns a Stanza pipeline on the GPU, processes all sections
from its docs in batches of `--batch`, and writes per-doc JSONs back
to disk. Each entity's character offsets are section-local (Stanza
returns them relative to the input Document's text).

Prereqs:
  - data/mtsamples_docs/*.json written by parse_mtsamples.py (and
    ideally build_abbreviations.py, for richer `resolved_text` output)
  - Stanza mimic/i2b2 models:
      python -c "import stanza; stanza.download('en', package='mimic', processors={'ner':'i2b2'})"
  - If running with use_gpu=True and the host cuDNN lags torch's
    expected version, prefix LD_LIBRARY_PATH with the venv's
    nvidia/cudnn/lib + nvidia/cublas/lib so torch's bundled libs win.

Usage:
  python python/ingestion/mtsamples/extract_section_entities.py [--workers 6] [--batch 1024] [--cpu]
"""

from __future__ import annotations

import argparse
import json
import os
import re
import time
import uuid
from functools import partial
from pathlib import Path
from typing import Iterable

import dask.bag as db

REPO = Path(__file__).resolve().parent.parent.parent.parent
DOCS_DIR = REPO / "data" / "mtsamples_docs"

# Token boundary pattern for abbreviation substitution. Alphanumerics +
# intra-word hyphens (e.g. C-spine) match; surrounding punctuation
# (commas, parens) does not.
TOKEN_RE = re.compile(r"\b[A-Za-z][A-Za-z0-9-]*\b")


def resolve_with_abbrevs(text: str, abbrev_upper: dict[str, str]) -> str:
    """Token-wise substitute known abbreviations with their expansions."""
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


def _process_partition(
    partition: Iterable[str],
    *,
    batch_size: int,
    use_gpu: bool,
) -> list[dict]:
    """Dask worker entry point.

    Loads Stanza once per worker (skipping mimic's lemma/pos/depparse
    processors, which the recent Stanza releases fail to load for the
    mimic model), then processes every section in this partition's docs
    in batches of `batch_size`. Writes each per-doc JSON back in place
    so the driver has no post-processing work to do.
    """
    import stanza

    wid = f"{os.getpid()}/{uuid.uuid4().hex[:6]}"
    paths = [Path(p) for p in partition]
    if not paths:
        return [{"wid": wid, "docs": 0, "ents": 0, "resolved_diff": 0, "elapsed_s": 0.0}]

    print(f"[worker {wid}] loading pipeline for {len(paths)} docs (use_gpu={use_gpu})",
          flush=True)
    nlp = stanza.Pipeline(
        lang="en",
        package=None,
        processors={"tokenize": "mimic", "ner": "i2b2"},
        use_gpu=use_gpu,
        verbose=False,
        download_method="reuse_resources",
    )

    # Load all docs for this partition; flatten non-empty sections.
    docs = [(p, json.loads(p.read_text(encoding="utf-8"))) for p in paths]
    items: list[tuple[int, int, str]] = []
    for di, (_, d) in enumerate(docs):
        for si, section in enumerate(d.get("sections", [])):
            text = section.get("text") or ""
            if text.strip():
                items.append((di, si, text))
            else:
                section["entities"] = []

    print(f"[worker {wid}] batching {len(items)} sections at batch={batch_size}",
          flush=True)

    total_ents = 0
    n_resolved_diff = 0
    t0 = time.time()
    for bi in range(0, len(items), batch_size):
        chunk = items[bi : bi + batch_size]
        stanza_docs = [stanza.Document([], text=t) for _, _, t in chunk]
        nlp.bulk_process(stanza_docs)

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
        print(f"[worker {wid}]   {done:>6}/{len(items)} sections  "
              f"({rate:.1f} sec/s)",
              flush=True)

    for p, d in docs:
        p.write_text(json.dumps(d, ensure_ascii=False), encoding="utf-8")

    elapsed = time.time() - t0
    print(f"[worker {wid}] done: {len(paths)} docs, {total_ents} ents, "
          f"{n_resolved_diff} resolved_diff in {elapsed:.1f}s",
          flush=True)
    return [{
        "wid": wid,
        "docs": len(paths),
        "ents": total_ents,
        "resolved_diff": n_resolved_diff,
        "elapsed_s": elapsed,
    }]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--docs", type=Path, default=DOCS_DIR)
    ap.add_argument("--workers", type=int, default=6,
                    help="dask worker processes (each loads one Stanza pipeline; "
                         "on L4, 6 fits in VRAM with headroom)")
    ap.add_argument("--batch", type=int, default=1024,
                    help="sections per Stanza bulk_process batch (per worker)")
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
    print(f"dispatching {len(paths):,} docs across {args.workers} dask workers "
          f"(use_gpu={use_gpu}, batch={args.batch})",
          flush=True)

    bag = db.from_sequence(paths, npartitions=args.workers)
    worker_fn = partial(_process_partition, batch_size=args.batch, use_gpu=use_gpu)
    results = bag.map_partitions(worker_fn).compute(
        scheduler="processes", num_workers=args.workers
    )

    total = {"docs": 0, "ents": 0, "resolved_diff": 0}
    for r in results:
        for k in total:
            total[k] += r.get(k, 0)

    wall_t0 = min((r.get("elapsed_s", 0) for r in results), default=0.0)
    print(
        f"done: {total['ents']:,} entities across {total['docs']:,} docs "
        f"({total['resolved_diff']:,} with resolved_text != recognized_text)",
        flush=True,
    )


if __name__ == "__main__":
    main()
