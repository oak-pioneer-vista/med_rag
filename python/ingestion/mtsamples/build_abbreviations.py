"""Build per-doc abbreviation maps: Schwartz-Hearst + clinical override + LRABR + MedTE WSD.

Four-stage per doc:

1. **Schwartz-Hearst** over the joined section texts. S-H requires the
   "long form (ABBR)" introduction to exist somewhere in the doc, so it
   is the highest-confidence signal ("evidence-in-doc").

2. **Curated clinical override** (data/clinical_abbreviations_override.json):
   a small hand-picked list of abbreviations whose clinical sense is
   overwhelmingly canonical in clinical notes (IV = intravenous,
   BUN = blood urea nitrogen, CT = computed tomography, ABCD = airway/
   breathing/circulation/disability, ...). These bypass LRABR entirely
   -- we trust the override more than the gazetteer for this short list.

3. **LRABR gazetteer + MedTE WSD** for everything the override didn't
   cover. ALL-CAPS candidate tokens that S-H missed are looked up in
   NLM's SPECIALIST Lexicon LRABR. **Every** LRABR hit -- whether
   single-sense or ambiguous -- is gated by **MedTE cosine similarity**
   between the doc context and each candidate expansion, subject to a
   minimum absolute similarity (`--min-score`, default 0.3). This is
   load-bearing: many LRABR entries are obscure research/pharmacology
   senses (e.g. some single-sense entries for common abbrevs point at
   niche research terms) where the clinical canonical sense isn't in
   LRABR at all. The threshold lets the embedding decide whether the
   LRABR sense fits the doc's context.

Output on each per-doc JSON:

  - `abbreviations`        : flat {abbrev: expansion}
  - `abbreviations_source` : parallel {abbrev: "sh" | "override" | "lrabr"}
  - `abbreviations_score`  : parallel {abbrev: float} -- 1.0 for S-H
                             and override hits, cosine similarity for
                             all LRABR hits

Prereqs:
  - data/mtsamples_docs/*.json written by parse_mtsamples.py
  - data/umls/LRABR pulled by python/ingestion/umls/download_specialist_lexicon.py
  - MedTE TEI service running (docker compose up -d medte)

Usage:
  python python/ingestion/mtsamples/build_abbreviations.py [--min-score 0.3]
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
from pathlib import Path

import numpy as np
import requests
from abbreviations import schwartz_hearst

REPO = Path(__file__).resolve().parent.parent.parent.parent
DOCS_DIR = REPO / "data" / "mtsamples_docs"
LRABR_PATH = REPO / "data" / "umls" / "LRABR"
OVERRIDE_PATH = REPO / "data" / "clinical_abbreviations_override.json"
TEI_URL_DEFAULT = os.environ.get("TEI_URL", "http://localhost:8080")

# 2+ char all-caps candidate, optionally mixed with digits.
ABBREV_TOKEN_RE = re.compile(r"\b[A-Z][A-Z0-9]+\b")

TEI_BATCH = 64  # matches compose --max-client-batch-size


def load_override(path: Path) -> dict[str, str]:
    """Flatten the category-grouped override file into {UPPER_ABBREV: expansion}."""
    raw = json.loads(path.read_text(encoding="utf-8"))
    flat: dict[str, str] = {}
    for key, val in raw.items():
        if key.startswith("_") or not isinstance(val, dict):
            continue
        for abbrev, expansion in val.items():
            flat[abbrev.upper()] = expansion
    return flat


def parse_lrabr(path: Path) -> dict[str, list[str]]:
    """Return {uppercase_abbrev: [unique_expansions]} from LRABR.

    LRABR format: EUI1|abbreviation|type|EUI2|expansion|
    """
    grouped: dict[str, set[str]] = {}
    with path.open(encoding="utf-8", errors="replace") as f:
        for line in f:
            parts = line.rstrip("\n").split("|")
            if len(parts) < 5:
                continue
            abbrev = parts[1].strip()
            expansion = parts[4].strip()
            if not abbrev or not expansion:
                continue
            grouped.setdefault(abbrev.upper(), set()).add(expansion)
    return {k: sorted(v) for k, v in grouped.items()}


def _sh_pairs(doc: dict) -> dict[str, str]:
    full = " . ".join(
        s.get("text", "").strip() for s in doc.get("sections", []) if s.get("text")
    )
    if not full:
        return {}
    return schwartz_hearst.extract_abbreviation_definition_pairs(
        doc_text=full, most_common_definition=True
    )


def _doc_text(doc: dict) -> str:
    """Join all section texts -- context used for embedding-based WSD."""
    return " ".join(
        s.get("text", "") for s in doc.get("sections", []) if s.get("text")
    ).strip()


def embed_batched(texts: list[str], tei_url: str) -> np.ndarray:
    """POST /embed in chunks of TEI_BATCH, L2-normalize, return (n, d) array."""
    if not texts:
        return np.zeros((0, 0), dtype=np.float32)
    out: list[np.ndarray] = []
    session = requests.Session()
    for i in range(0, len(texts), TEI_BATCH):
        chunk = texts[i : i + TEI_BATCH]
        r = session.post(
            f"{tei_url}/embed",
            json={"inputs": chunk, "truncate": True},
            timeout=300,
        )
        r.raise_for_status()
        out.append(np.asarray(r.json(), dtype=np.float32))
    vecs = np.concatenate(out, axis=0)
    norms = np.linalg.norm(vecs, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return vecs / norms


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--docs", type=Path, default=DOCS_DIR)
    ap.add_argument("--lrabr", type=Path, default=LRABR_PATH)
    ap.add_argument("--override", type=Path, default=OVERRIDE_PATH)
    ap.add_argument("--tei-url", default=TEI_URL_DEFAULT, help=f"(default: {TEI_URL_DEFAULT})")
    ap.add_argument("--min-score", type=float, default=0.3,
                    help="minimum cosine similarity for LRABR picks, single or ambiguous (default: 0.3)")
    ap.add_argument("--no-gazetteer", action="store_true", help="skip LRABR pass (S-H + override only)")
    ap.add_argument("--no-override", action="store_true", help="skip curated override pass")
    args = ap.parse_args()

    paths = sorted(args.docs.glob("*.json"))
    if not paths:
        raise SystemExit(
            f"no JSON files in {args.docs} -- run python/ingestion/mtsamples/parse_mtsamples.py first"
        )

    override: dict[str, str] = {}
    if not args.no_override:
        if not args.override.exists():
            raise SystemExit(f"missing {args.override}")
        override = load_override(args.override)
        print(f"loaded clinical override: {len(override):,} abbreviations")

    lrabr: dict[str, list[str]] = {}
    if not args.no_gazetteer:
        if not args.lrabr.exists():
            raise SystemExit(
                f"missing {args.lrabr} -- run "
                f"python/ingestion/umls/download_specialist_lexicon.py first"
            )
        lrabr = parse_lrabr(args.lrabr)
        n_multi = sum(1 for v in lrabr.values() if len(v) > 1)
        print(f"loaded LRABR: {len(lrabr):,} abbrevs ({n_multi:,} ambiguous)")

    # ---- Pass 1: load docs, run S-H, queue every LRABR candidate occurrence.
    docs: list[tuple[Path, dict]] = []
    # (doc_idx, tok, up) for each LRABR candidate occurrence
    work: list[tuple[int, str, str]] = []

    for p in paths:
        doc = json.loads(p.read_text(encoding="utf-8"))
        sh = _sh_pairs(doc)
        doc["abbreviations"] = dict(sh)
        doc["abbreviations_source"] = {k: "sh" for k in sh}
        doc["abbreviations_score"] = {k: 1.0 for k in sh}
        sh_upper = {k.upper() for k in sh}

        seen_upper: set[str] = set()
        doc_idx = len(docs)
        for section in doc.get("sections", []):
            text = section.get("text") or ""
            for m in ABBREV_TOKEN_RE.finditer(text):
                tok = m.group(0)
                up = tok.upper()
                if up in seen_upper or up in sh_upper:
                    continue
                seen_upper.add(up)
                # Curated override wins over LRABR for truly canonical
                # clinical abbreviations. We trust the override more than
                # the gazetteer here, so no embedding check.
                if up in override:
                    doc["abbreviations"][tok] = override[up]
                    doc["abbreviations_source"][tok] = "override"
                    doc["abbreviations_score"][tok] = 1.0
                elif up in lrabr:
                    work.append((doc_idx, tok, up))
        docs.append((p, doc))

    # ---- Pass 2: batch-embed doc contexts and unique candidate expansions.
    print(f"LRABR candidate occurrences queued: {len(work):,}")

    if work:
        # Unique docs needing embedding (by index into `docs`)
        needed_doc_idxs = sorted({w[0] for w in work})
        doc_texts = [_doc_text(docs[i][1]) for i in needed_doc_idxs]
        doc_idx_to_row = {di: row for row, di in enumerate(needed_doc_idxs)}

        # Unique expansions across all ambiguous abbrevs encountered
        unique_expansions: dict[str, int] = {}
        expansion_texts: list[str] = []
        for _, _, up in work:
            for exp in lrabr[up]:
                if exp not in unique_expansions:
                    unique_expansions[exp] = len(expansion_texts)
                    expansion_texts.append(exp)

        print(
            f"embedding {len(doc_texts):,} doc contexts and "
            f"{len(expansion_texts):,} expansions via {args.tei_url} ..."
        )
        try:
            doc_vecs = embed_batched(doc_texts, args.tei_url)
            exp_vecs = embed_batched(expansion_texts, args.tei_url)
        except requests.RequestException as e:
            print(f"error: TEI request failed ({e}). "
                  f"Is medte running? `docker compose up -d medte`",
                  file=sys.stderr)
            sys.exit(1)

        # ---- Pass 3: resolve each ambiguous occurrence by cosine sim.
        n_resolved = 0
        n_below_threshold = 0
        for doc_idx, tok, up in work:
            exps = lrabr[up]
            dv = doc_vecs[doc_idx_to_row[doc_idx]]
            ev = exp_vecs[np.array([unique_expansions[e] for e in exps])]
            sims = ev @ dv
            best = int(np.argmax(sims))
            score = float(sims[best])
            if score < args.min_score:
                n_below_threshold += 1
                continue
            docs[doc_idx][1]["abbreviations"][tok] = exps[best]
            docs[doc_idx][1]["abbreviations_source"][tok] = "lrabr"
            docs[doc_idx][1]["abbreviations_score"][tok] = round(score, 4)
            n_resolved += 1
        print(
            f"resolved {n_resolved:,} by cosine, "
            f"skipped {n_below_threshold:,} below --min-score={args.min_score}"
        )

    # ---- Write back.
    n_with_abbrev = 0
    counts = {"sh": 0, "override": 0, "lrabr": 0}
    for p, doc in docs:
        p.write_text(json.dumps(doc, ensure_ascii=False), encoding="utf-8")
        if doc["abbreviations"]:
            n_with_abbrev += 1
            for s in doc["abbreviations_source"].values():
                if s in counts:
                    counts[s] += 1
    print(
        f"updated {len(docs)} docs  ({n_with_abbrev} with >=1 abbreviation; "
        f"{counts['sh']} S-H, {counts['override']} override, {counts['lrabr']} LRABR)"
    )


if __name__ == "__main__":
    main()
