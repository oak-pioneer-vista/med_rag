"""Build per-doc abbreviation maps via Schwartz-Hearst, write back into per-doc JSON.

For each parsed MTSampleDoc under data/mtsamples_docs/, joins all section
texts and runs the Schwartz-Hearst algorithm to recover `{abbrev: long_form}`
pairs, then writes the map back into the same JSON file under a new
top-level `abbreviations` field.

Joining sections is load-bearing: S-H finds "long form (ABBR)" / "ABBR
(long form)" patterns, and the introduction can sit in a different
section from later mentions ("congestive heart failure (CHF)" in HPI,
"CHF exacerbation" in A/P). The `" . "` separator terminates sentences
so S-H's line-wise scan doesn't run definitions across section boundaries.

Prereqs:
  - data/mtsamples_docs/*.json written by parse_mtsamples.py

Usage:
  python python/ingestion/mtsamples/build_abbreviations.py
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from abbreviations import schwartz_hearst

REPO = Path(__file__).resolve().parent.parent.parent.parent
DOCS_DIR = REPO / "data" / "mtsamples_docs"


def doc_abbrev_map(doc: dict) -> dict[str, str]:
    full = " . ".join(
        s.get("text", "").strip() for s in doc.get("sections", []) if s.get("text")
    )
    if not full:
        return {}
    return schwartz_hearst.extract_abbreviation_definition_pairs(
        doc_text=full, most_common_definition=True
    )


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--docs", type=Path, default=DOCS_DIR, help=f"per-doc JSON dir (default: {DOCS_DIR})")
    args = ap.parse_args()

    paths = sorted(args.docs.glob("*.json"))
    if not paths:
        raise SystemExit(
            f"no JSON files in {args.docs} -- run python/ingestion/mtsamples/parse_mtsamples.py first"
        )

    n_with_abbrev = 0
    total_pairs = 0
    for p in paths:
        doc = json.loads(p.read_text(encoding="utf-8"))
        abbrev = doc_abbrev_map(doc)
        doc["abbreviations"] = abbrev
        p.write_text(json.dumps(doc, ensure_ascii=False), encoding="utf-8")
        if abbrev:
            n_with_abbrev += 1
            total_pairs += len(abbrev)

    print(
        f"updated {len(paths)} docs  ({n_with_abbrev} with >=1 abbreviation, "
        f"{total_pairs} total pairs)"
    )


if __name__ == "__main__":
    main()
