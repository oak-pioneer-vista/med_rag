"""Snapshot the lexical entity->CUI map produced by step 11.

Reads data/mtsamples_docs/*.json and writes one JSONL line per unique
`expanded_text` (lowercased) with the `cui` assigned by the lexical
pipeline (exact `Atom.str_norm` match + Lucene fulltext fallback on
`concept_name_fts`):

    {"text": "<expanded_text_lower>", "cui": "<cui or ''>"}

Baseline for diffing against future semantic-linking variants (e.g.
SapBERT / MedTE nearest-neighbor over Concept embeddings): swap the
linker, re-emit, diff to see which entities move.

Usage:
  python python/ingestion/mtsamples/export_entity_cui_lexical.py
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent.parent.parent
DOCS_DIR = REPO / "data" / "mtsamples_docs"
OUT_PATH = REPO / "data" / "entity_cui_lexical.jsonl"


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--docs", type=Path, default=DOCS_DIR)
    ap.add_argument("--out", type=Path, default=OUT_PATH)
    args = ap.parse_args()

    paths = sorted(args.docs.glob("*.json"))
    if not paths:
        raise SystemExit(f"no JSON files in {args.docs}")

    t0 = time.time()
    unique: dict[str, str] = {}
    conflicts = 0
    for p in paths:
        d = json.loads(p.read_text(encoding="utf-8"))
        for sec in d.get("sections", []):
            for e in sec.get("entities") or []:
                t = (e.get("expanded_text") or "").strip().lower()
                if not t:
                    continue
                cui = e.get("cui") or ""
                prev = unique.get(t)
                if prev is None:
                    unique[t] = cui
                elif prev != cui:
                    conflicts += 1

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w", encoding="utf-8") as f:
        for t in sorted(unique):
            f.write(json.dumps({"text": t, "cui": unique[t]}, ensure_ascii=False) + "\n")

    linked = sum(1 for v in unique.values() if v)
    print(
        f"wrote {len(unique):,} unique entities "
        f"({linked:,} linked, {len(unique) - linked:,} unlinked; "
        f"{conflicts} cross-doc text->cui conflicts) "
        f"to {args.out.relative_to(REPO)} in {time.time() - t0:.1f}s",
        flush=True,
    )


if __name__ == "__main__":
    main()
