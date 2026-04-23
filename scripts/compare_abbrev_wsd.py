"""Side-by-side comparison of MedTE vs BioLORD for LRABR WSD (step 6, stage 3).

Replicates `build_abbreviations.py`'s Pass 1 (S-H + override + LRABR
candidate detection) to enumerate every (section, abbreviation,
candidates) tuple the production linker would disambiguate, then runs
the WSD vote with BOTH encoders against the same section-context and
candidate-expansion strings. Does NOT modify per-doc JSONs.

Compares:
  - overall agreement on multi-sense cases
  - threshold acceptance (medte: --min-score-medte, bio: --min-score-biolord)
  - sample disagreements where both encoders accept but pick different
    expansions (the only interesting bucket for an A/B judgment)

Usage:
  python scripts/compare_abbrev_wsd.py \
      [--min-score-medte 0.3] [--min-score-biolord 0.7]

Notes:
  - MedTE default threshold 0.3 is from build_abbreviations.py; BioLORD's
    higher threshold reflects its tighter cosine distribution for
    concept-pair similarity. Comparing "any winner" is more informative
    than comparing "winner above threshold" for measuring encoder
    preference on the WSD task itself.
"""

from __future__ import annotations

import argparse
import json
import random
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "python"))

from ingestion.mtsamples.build_abbreviations import (  # noqa: E402
    ABBREV_TOKEN_RE,
    DOCS_DIR,
    LRABR_PATH,
    OVERRIDE_PATH,
    _section_text,
    _sh_pairs_section,
    embed_batched,
    load_override,
    parse_lrabr,
)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--docs", type=Path, default=DOCS_DIR)
    ap.add_argument("--lrabr", type=Path, default=LRABR_PATH)
    ap.add_argument("--override", type=Path, default=OVERRIDE_PATH)
    ap.add_argument("--medte-url", default="http://localhost:8080")
    ap.add_argument("--biolord-url", default="http://localhost:8081")
    ap.add_argument("--min-score-medte", type=float, default=0.3)
    ap.add_argument("--min-score-biolord", type=float, default=0.7)
    ap.add_argument("--sample-disagreements", type=int, default=12)
    args = ap.parse_args()

    override = load_override(args.override)
    lrabr = parse_lrabr(args.lrabr)
    n_multi_in_lrabr = sum(1 for v in lrabr.values() if len(v) > 1)
    print(f"loaded: override={len(override):,}  lrabr={len(lrabr):,} "
          f"(multi-sense {n_multi_in_lrabr:,})",
          flush=True)

    # ---- Pass 1: enumerate LRABR candidate occurrences per section ----
    paths = sorted(args.docs.glob("*.json"))
    if not paths:
        raise SystemExit(f"no JSON files in {args.docs}")

    work: list[tuple[int, int, str, str, str, list[str]]] = []
    # (doc_id, section_idx, surface_tok, surface_upper, section_text, candidate_expansions)
    t0 = time.time()
    for p in paths:
        doc = json.loads(p.read_text(encoding="utf-8"))
        doc_id = int(doc.get("doc_id", 0))
        for si, sec in enumerate(doc.get("sections", [])):
            text = _section_text(sec)
            if not text:
                continue
            sh_keys_upper = {k.upper() for k in _sh_pairs_section(text)}
            seen = set()
            for m in ABBREV_TOKEN_RE.finditer(text):
                tok = m.group(0)
                up = tok.upper()
                if up in seen or up in sh_keys_upper or up in override:
                    continue
                seen.add(up)
                if up in lrabr:
                    work.append((doc_id, si, tok, up, text, lrabr[up]))
    print(f"pass 1 done in {time.time() - t0:.1f}s: "
          f"{len(work):,} LRABR candidate occurrences across all sections",
          flush=True)

    # ---- Dedup section contexts and expansions for batched embedding ----
    section_texts: list[str] = []
    sec_row_of: dict[tuple[int, int], int] = {}
    for (doc_id, si, _, _, text, _) in work:
        k = (doc_id, si)
        if k not in sec_row_of:
            sec_row_of[k] = len(section_texts)
            section_texts.append(text)

    exp_idx: dict[str, int] = {}
    for _, _, _, _, _, exps in work:
        for e in exps:
            if e not in exp_idx:
                exp_idx[e] = len(exp_idx)
    expansion_texts = list(exp_idx.keys())

    print(f"unique section contexts: {len(section_texts):,}  "
          f"unique expansions: {len(expansion_texts):,}",
          flush=True)

    # ---- Embed with both encoders ----
    def embed_pair(url: str) -> tuple[np.ndarray, np.ndarray]:
        print(f"  embedding via {url} ...", flush=True)
        t1 = time.time()
        sv = embed_batched(section_texts, url)
        ev = embed_batched(expansion_texts, url)
        print(f"    done in {time.time() - t1:.1f}s "
              f"(sec {sv.shape}, exp {ev.shape})",
              flush=True)
        return sv, ev

    med_sec, med_exp = embed_pair(args.medte_url)
    bio_sec, bio_exp = embed_pair(args.biolord_url)

    # ---- Dual WSD pass ----
    rows = []  # per multi-sense work item
    for (doc_id, si, tok, up, text, exps) in work:
        sec_row = sec_row_of[(doc_id, si)]
        exp_rows = np.array([exp_idx[e] for e in exps])
        med_sims = med_exp[exp_rows] @ med_sec[sec_row]
        bio_sims = bio_exp[exp_rows] @ bio_sec[sec_row]
        med_best = int(np.argmax(med_sims))
        bio_best = int(np.argmax(bio_sims))
        if len(exps) > 1:
            rows.append({
                "doc_id": doc_id, "si": si, "surface": up,
                "candidates": exps,
                "med_pick": exps[med_best], "med_score": float(med_sims[med_best]),
                "bio_pick": exps[bio_best], "bio_score": float(bio_sims[bio_best]),
                "text": text,
            })

    print(f"\n=== WSD on multi-sense LRABR hits: {len(rows):,} occurrences ===")

    same_winner = sum(1 for r in rows if r["med_pick"] == r["bio_pick"])
    diff_winner = len(rows) - same_winner
    print(f"  same winner:      {same_winner:,}  ({100 * same_winner / max(len(rows), 1):.1f}%)")
    print(f"  different winner: {diff_winner:,}  ({100 * diff_winner / max(len(rows), 1):.1f}%)")

    med_ok = [r for r in rows if r["med_score"] >= args.min_score_medte]
    bio_ok = [r for r in rows if r["bio_score"] >= args.min_score_biolord]
    both_ok = [r for r in rows if r["med_score"] >= args.min_score_medte
                                 and r["bio_score"] >= args.min_score_biolord]
    print(f"\n=== threshold acceptance "
          f"(med>={args.min_score_medte}, bio>={args.min_score_biolord}) ===")
    print(f"  medte would accept:   {len(med_ok):,}")
    print(f"  biolord would accept: {len(bio_ok):,}")
    print(f"  both accept:          {len(both_ok):,}")

    dis_both_ok = [r for r in both_ok if r["med_pick"] != r["bio_pick"]]
    print(f"\n=== both above threshold, different winner: {len(dis_both_ok):,} ===")
    print("(these are the cases where the two encoders substantively disagree)\n")

    # De-duplicate by (surface, med_pick, bio_pick) for variety.
    seen = set()
    uniq = []
    for r in dis_both_ok:
        key = (r["surface"], r["med_pick"], r["bio_pick"])
        if key in seen:
            continue
        seen.add(key)
        uniq.append(r)
    print(f"unique (surface, medte_pick, biolord_pick) disagreement triples: {len(uniq):,}")

    random.seed(42)
    sample_rows = random.sample(uniq, min(args.sample_disagreements, len(uniq)))
    for r in sample_rows:
        print(f"\n  [{r['surface']}]   doc_id={r['doc_id']}  section_idx={r['si']}")
        cands_show = ", ".join(r["candidates"][:6]) + ("..." if len(r["candidates"]) > 6 else "")
        print(f"    LRABR candidates ({len(r['candidates'])}): {cands_show}")
        print(f"    medte  -> {r['med_pick']!r:45s}  score={r['med_score']:.3f}")
        print(f"    biolord-> {r['bio_pick']!r:45s}  score={r['bio_score']:.3f}")
        print(f"    section text: {r['text'][:180]!r}...")


if __name__ == "__main__":
    main()
