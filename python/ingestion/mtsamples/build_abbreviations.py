"""Build per-section abbreviation maps: Schwartz-Hearst + clinical override + LRABR + MedTE WSD.

Resolution happens at the call site (per Section), not aggregated to
the doc level. The same surface form can resolve to different
expansions in different sections of the same doc -- useful when one
note mixes topics (e.g. `MS` could mean mitral stenosis in PROCEDURE
and multiple sclerosis in HPI of the same doc, and the section text
is what disambiguates).

Three-stage per Section:

1. **Schwartz-Hearst** over THIS section's text alone. S-H requires
   the "long form (ABBR)" introduction inside the same window, so
   per-section S-H gives many fewer hits than the per-doc variant
   (most clinical notes introduce abbreviations once at the top, then
   reuse them across sections); the hits that remain are still the
   highest-confidence signal because the definition is evidence-in-
   section.

2. **Curated clinical override** (data/clinical_abbreviations_override.json):
   a small hand-picked list of abbreviations whose clinical sense is
   overwhelmingly canonical regardless of section context (IV =
   intravenous, BUN = blood urea nitrogen, ABCD = airway/breathing/
   circulation/disability, ...). These bypass LRABR and don't need a
   context vector.

3. **LRABR gazetteer + MedTE WSD** for everything the override didn't
   cover. ALL-CAPS candidate tokens that S-H missed are looked up in
   NLM's SPECIALIST Lexicon LRABR. **Every** LRABR hit (single-sense
   or ambiguous) is gated by MedTE cosine similarity between THIS
   SECTION's text and each candidate expansion, subject to a minimum
   absolute similarity (`--min-score`, default 0.3). Per-section
   context is shorter than the joined-doc context, which is the point
   -- when a doc spans multiple topics, the WSD shouldn't average
   them.

Output on each section:

  - sections[i].abbreviations        : flat {abbrev: expansion}
  - sections[i].abbreviations_source : parallel {abbrev: "sh"|"override"|"lrabr"}
  - sections[i].abbreviations_score  : parallel {abbrev: float} -- 1.0 for
                                       S-H and override, cosine for LRABR

Any legacy doc-level `abbreviations*` fields written by an older
version of this script are removed on write-back.

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
import time
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


def _section_text(section: dict) -> str:
    return (section.get("text") or "").strip()


def _sh_pairs_section(text: str) -> dict[str, str]:
    """Schwartz-Hearst over a single section's text."""
    if not text:
        return {}
    return schwartz_hearst.extract_abbreviation_definition_pairs(
        doc_text=text, most_common_definition=True
    )


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
        if (i // TEI_BATCH) and (i // TEI_BATCH) % 10 == 0:
            print(f"  embedded {min(i + TEI_BATCH, len(texts))}/{len(texts)}",
                  flush=True)
    vecs = np.concatenate(out, axis=0)
    norms = np.linalg.norm(vecs, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return vecs / norms


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--docs", type=Path, default=DOCS_DIR)
    ap.add_argument("--lrabr", type=Path, default=LRABR_PATH)
    ap.add_argument("--override", type=Path, default=OVERRIDE_PATH)
    ap.add_argument("--tei-url", default=TEI_URL_DEFAULT,
                    help=f"(default: {TEI_URL_DEFAULT})")
    ap.add_argument("--min-score", type=float, default=0.3,
                    help="minimum cosine similarity for LRABR picks "
                         "(default: 0.3)")
    ap.add_argument("--no-gazetteer", action="store_true",
                    help="skip LRABR pass (S-H + override only)")
    ap.add_argument("--no-override", action="store_true",
                    help="skip curated override pass")
    args = ap.parse_args()

    paths = sorted(args.docs.glob("*.json"))
    if not paths:
        raise SystemExit(
            f"no JSON files in {args.docs} -- run "
            f"python/ingestion/mtsamples/parse_mtsamples.py first"
        )

    override: dict[str, str] = {}
    if not args.no_override:
        if not args.override.exists():
            raise SystemExit(f"missing {args.override}")
        override = load_override(args.override)
        print(f"loaded clinical override: {len(override):,} abbreviations",
              flush=True)

    lrabr: dict[str, list[str]] = {}
    if not args.no_gazetteer:
        if not args.lrabr.exists():
            raise SystemExit(
                f"missing {args.lrabr} -- run "
                f"python/ingestion/umls/download_specialist_lexicon.py first"
            )
        lrabr = parse_lrabr(args.lrabr)
        n_multi = sum(1 for v in lrabr.values() if len(v) > 1)
        print(f"loaded LRABR: {len(lrabr):,} abbrevs ({n_multi:,} ambiguous)",
              flush=True)

    # ---- Pass 1: per-section S-H + override; queue LRABR work per section.
    docs: list[tuple[Path, dict]] = []
    # Each work item: (doc_idx, sec_idx, tok, up)
    work: list[tuple[int, int, str, str]] = []
    n_sections_total = 0
    n_sections_with_text = 0
    t0 = time.time()

    for p in paths:
        doc = json.loads(p.read_text(encoding="utf-8"))
        # Drop legacy doc-level fields written by older versions.
        for k in ("abbreviations", "abbreviations_source", "abbreviations_score"):
            doc.pop(k, None)

        di = len(docs)
        for si, section in enumerate(doc.get("sections", [])):
            n_sections_total += 1
            text = _section_text(section)
            sh = _sh_pairs_section(text) if text else {}
            section["abbreviations"] = dict(sh)
            section["abbreviations_source"] = {k: "sh" for k in sh}
            section["abbreviations_score"] = {k: 1.0 for k in sh}
            if not text:
                continue
            n_sections_with_text += 1

            sh_upper = {k.upper() for k in sh}
            seen_upper: set[str] = set()
            for m in ABBREV_TOKEN_RE.finditer(text):
                tok = m.group(0)
                up = tok.upper()
                if up in seen_upper or up in sh_upper:
                    continue
                seen_upper.add(up)
                if up in override:
                    section["abbreviations"][tok] = override[up]
                    section["abbreviations_source"][tok] = "override"
                    section["abbreviations_score"][tok] = 1.0
                elif up in lrabr:
                    work.append((di, si, tok, up))
        docs.append((p, doc))
        if len(docs) % 500 == 0:
            print(f"  scanned {len(docs)}/{len(paths)} docs "
                  f"({n_sections_total} sections, {len(work)} LRABR queue depth)",
                  flush=True)

    print(f"pass 1 done in {time.time()-t0:.1f}s: "
          f"{n_sections_total} sections ({n_sections_with_text} with text), "
          f"{len(work)} LRABR candidate occurrences queued",
          flush=True)

    # ---- Pass 2: batch-embed section contexts and unique candidate expansions.
    if work:
        # Unique (doc_idx, sec_idx) needing context embedding.
        needed_keys = sorted({(di, si) for (di, si, _, _) in work})
        section_texts: list[str] = []
        section_key_to_row: dict[tuple[int, int], int] = {}
        for row, (di, si) in enumerate(needed_keys):
            section_key_to_row[(di, si)] = row
            sec = docs[di][1]["sections"][si]
            section_texts.append(_section_text(sec))

        # Unique expansions across all candidates.
        unique_expansions: dict[str, int] = {}
        expansion_texts: list[str] = []
        for _, _, _, up in work:
            for exp in lrabr[up]:
                if exp not in unique_expansions:
                    unique_expansions[exp] = len(expansion_texts)
                    expansion_texts.append(exp)

        print(
            f"embedding {len(section_texts):,} section contexts and "
            f"{len(expansion_texts):,} expansions via {args.tei_url} ...",
            flush=True,
        )
        t1 = time.time()
        try:
            sec_vecs = embed_batched(section_texts, args.tei_url)
            exp_vecs = embed_batched(expansion_texts, args.tei_url)
        except requests.RequestException as e:
            print(f"error: TEI request failed ({e}). "
                  f"Is medte running? `docker compose up -d medte`",
                  file=sys.stderr)
            sys.exit(1)
        print(f"  embeddings done in {time.time()-t1:.1f}s", flush=True)

        # ---- Pass 3: resolve each LRABR occurrence by cosine sim against
        # its parent section's context vector.
        n_resolved = 0
        n_below_threshold = 0
        for di, si, tok, up in work:
            exps = lrabr[up]
            sv = sec_vecs[section_key_to_row[(di, si)]]
            ev = exp_vecs[np.array([unique_expansions[e] for e in exps])]
            sims = ev @ sv
            best = int(np.argmax(sims))
            score = float(sims[best])
            if score < args.min_score:
                n_below_threshold += 1
                continue
            sec = docs[di][1]["sections"][si]
            sec["abbreviations"][tok] = exps[best]
            sec["abbreviations_source"][tok] = "lrabr"
            sec["abbreviations_score"][tok] = round(score, 4)
            n_resolved += 1
        print(
            f"resolved {n_resolved:,} by cosine, "
            f"skipped {n_below_threshold:,} below --min-score={args.min_score}",
            flush=True,
        )

    # ---- Write back. Tally section-level stats.
    n_sections_with_abbrev = 0
    counts = {"sh": 0, "override": 0, "lrabr": 0}
    for p, doc in docs:
        for sec in doc.get("sections", []):
            sec_map = sec.get("abbreviations") or {}
            if sec_map:
                n_sections_with_abbrev += 1
                for s in (sec.get("abbreviations_source") or {}).values():
                    if s in counts:
                        counts[s] += 1
        p.write_text(json.dumps(doc, ensure_ascii=False), encoding="utf-8")

    print(
        f"updated {len(docs)} docs / {n_sections_total:,} sections "
        f"({n_sections_with_abbrev:,} sections with >=1 abbreviation; "
        f"{counts['sh']} S-H, {counts['override']} override, {counts['lrabr']} LRABR)",
        flush=True,
    )


if __name__ == "__main__":
    main()
