"""Clean raw MTSamples CSV: drop missing transcriptions, dedupe cross-filings.

MTSamples ships the same note under multiple `medical_specialty` rows (e.g.
"Lumbar Discogram" filed under 5 specialties) with byte-identical
`transcription`. Embedding all copies wastes Qdrant points and pads top-k
with trivial dupes, so we collapse duplicates here before the parse step.
For every cluster, the first row's metadata wins; the dropped rows'
specialty names are recorded in `alt_specialties`, and keyword tokens are
unioned (case-insensitive) so downstream specialty/keyword filters still
match cross-filings.

Outputs:
  - data/mtsamples_clean.jsonl  (one JSON object per surviving row)

Usage:
  python python/ingestion/mtsamples/clean_mtsamples.py
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

REPO = Path(__file__).resolve().parent.parent.parent.parent
SRC_CSV = REPO / "data" / "kaggle" / "mtsamples" / "mtsamples.csv"
OUT_JSONL = REPO / "data" / "mtsamples_clean.jsonl"


def _merge_keyword_tokens(a: str, b: str) -> str:
    """Union comma-split tokens from two keyword strings, case-insensitive dedupe.

    MTSamples keyword strings in a dupe cluster typically differ only by a
    leading specialty token (e.g. "surgery, ..." vs "gastroenterology, ...")
    on otherwise identical content -- union keeps the cross-filing tokens
    without duplicating the clinical vocabulary.
    """
    seen: dict[str, str] = {}
    for part in a.split(",") + b.split(","):
        tok = part.strip()
        if not tok:
            continue
        key = tok.lower()
        if key not in seen:
            seen[key] = tok
    return ", ".join(seen.values())


def dedupe_by_transcription(df: pd.DataFrame) -> pd.DataFrame:
    """Collapse byte-identical transcriptions across medical_specialty filings."""
    first_idx: dict[str, int] = {}
    alts_by_first: dict[int, list[str]] = {}
    kw_by_first: dict[int, str] = {}
    drop_idx: list[int] = []

    for i, row in df.iterrows():
        t = row["transcription"]
        if t in first_idx:
            fi = first_idx[t]
            spec = str(row.get("medical_specialty") or "").strip()
            alts_by_first.setdefault(fi, []).append(spec)
            base_kw = kw_by_first.get(fi, str(df.at[fi, "keywords"] or "").strip())
            kw_by_first[fi] = _merge_keyword_tokens(
                base_kw, str(row.get("keywords") or "").strip()
            )
            drop_idx.append(int(i))
        else:
            first_idx[t] = int(i)

    out = df.drop(index=drop_idx).reset_index(drop=True)

    # Map (new row index in `out`) -> original-first index in `df`.
    # `first_idx.values()` is ordered by first-occurrence, same order `drop`
    # preserves, so enumerating `out` lines up 1:1 with sorted kept indices.
    kept_sorted = sorted(first_idx.values())
    for new_i, old_i in enumerate(kept_sorted):
        if old_i in kw_by_first:
            out.at[new_i, "keywords"] = kw_by_first[old_i]

    alt_col: list[list[str]] = [alts_by_first.get(old_i, []) for old_i in kept_sorted]
    out["alt_specialties"] = alt_col

    n_clusters = len(alts_by_first)
    n_dropped = len(drop_idx)
    print(
        f"dedupe: {len(df)} -> {len(out)} rows "
        f"({n_clusters} clusters collapsed, {n_dropped} dupe rows dropped)"
    )
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--src", type=Path, default=SRC_CSV, help=f"input CSV (default: {SRC_CSV})")
    ap.add_argument("--out", type=Path, default=OUT_JSONL, help=f"output JSONL (default: {OUT_JSONL})")
    args = ap.parse_args()

    if not args.src.exists():
        raise SystemExit(
            f"missing {args.src} -- run python/ingestion/mtsamples/download_mtsamples.py first"
        )

    df = pd.read_csv(args.src).dropna(subset=["transcription"]).fillna("").reset_index(drop=True)
    print(f"read {len(df)} rows with non-null transcription from {args.src.name}")
    df = dedupe_by_transcription(df)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w", encoding="utf-8") as f:
        for rec in df.to_dict(orient="records"):
            f.write(json.dumps(rec, ensure_ascii=False))
            f.write("\n")
    print(f"wrote {len(df)} cleaned records to {args.out}")


if __name__ == "__main__":
    main()
