"""Normalize per-section entities: article-strip + abbrev-expand.

Reads per-doc JSONs from data/mtsamples_docs/, and for each entity record
in each Section's `entities` list, derives two post-NER text fields from
the entity's already-populated `recognized_text`:

  resolved_text : recognized_text with articles (a/an/the) removed and
                  whitespace collapsed. Reserved as the "NER canonical
                  surface" -- future string-level normalization (casing,
                  dash/slash handling) would chain in here.

  expanded_text : resolved_text with known abbreviations substituted
                  using THIS section's `abbreviations` map (from
                  build_abbreviations.py, which resolves abbrevs at
                  the section level), then article-stripped again in
                  case the expansion introduced one. Use this for
                  UMLS/Neo4j grounding and dense-retrieval keying.

This step is split from extract_section_entities.py on purpose: Stanza
NER takes ~91s on L4 GPU (~16 min on CPU) and its output is deterministic
from the section text + model. Every iteration on normalization rules
(stopword list, token regex, new abbrev handling) would otherwise re-pay
that NER cost. By keeping NER idempotent and all text-level derivations
here, iterating on this layer is pure Python string work over the
already-written entity records -- typically a few seconds end-to-end.

Prereqs:
  - data/mtsamples_docs/*.json with entities populated by
    extract_section_entities.py (surface_text, recognized_text, type,
    start_char, end_char)
  - Optionally, sections[i]['abbreviations'] populated by
    build_abbreviations.py (without it, expanded_text == resolved_text
    everywhere)

Usage:
  python python/ingestion/mtsamples/normalize_section_entities.py
"""

from __future__ import annotations

import argparse
import json
import re
import time
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent.parent.parent
DOCS_DIR = REPO / "data" / "mtsamples_docs"

TOKEN_RE = re.compile(r"\b[A-Za-z][A-Za-z0-9-]*\b")

# Stripped from resolved_text and expanded_text so downstream dense-retrieval
# / UMLS-atom-string joins aren't thrown off by leading determiners Stanza
# picked up as part of the NP chunk (e.g. "a BMI", "an EGD scope",
# "the incision").
ARTICLE_RE = re.compile(r"\b(?:a|an|the)\b", re.IGNORECASE)
WS_RE = re.compile(r"\s+")


def strip_articles(text: str) -> str:
    """Remove 'a', 'an', 'the' (whole-word, case-insensitive) and collapse whitespace."""
    out = ARTICLE_RE.sub("", text)
    out = WS_RE.sub(" ", out).strip()
    return out or text  # don't return empty string if the entity IS just an article


def resolve_with_abbrevs(text: str, abbrev_upper: dict[str, str]) -> str:
    """Token-wise substitute known abbreviations with their expansions."""
    if not abbrev_upper:
        return text

    def repl(m: re.Match) -> str:
        tok = m.group(0)
        return abbrev_upper.get(tok.upper(), tok)

    return TOKEN_RE.sub(repl, text)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--docs", type=Path, default=DOCS_DIR)
    args = ap.parse_args()

    paths = sorted(args.docs.glob("*.json"))
    if not paths:
        raise SystemExit(
            f"no JSON files in {args.docs} -- run "
            f"python/ingestion/mtsamples/extract_section_entities.py first"
        )

    print(f"normalizing entities across {len(paths):,} docs", flush=True)

    t0 = time.time()
    total_ents = 0
    n_stripped = 0  # resolved_text differs from recognized_text
    n_expanded = 0  # expanded_text differs from resolved_text
    n_skipped_missing_recognized = 0
    n_sections_with_abbrev = 0

    for p in paths:
        d = json.loads(p.read_text(encoding="utf-8"))

        for sec in d.get("sections", []):
            sec_abbrev_map = sec.get("abbreviations") or {}
            if sec_abbrev_map:
                n_sections_with_abbrev += 1
            sec_abbrev_upper = {k.upper(): v for k, v in sec_abbrev_map.items()}

            for ent in sec.get("entities") or []:
                recognized = ent.get("recognized_text")
                if recognized is None:
                    n_skipped_missing_recognized += 1
                    continue
                resolved = strip_articles(recognized)
                expanded = resolve_with_abbrevs(resolved, sec_abbrev_upper)
                expanded = strip_articles(expanded)
                ent["resolved_text"] = resolved
                ent["expanded_text"] = expanded
                total_ents += 1
                if resolved != recognized:
                    n_stripped += 1
                if expanded != resolved:
                    n_expanded += 1

        p.write_text(json.dumps(d, ensure_ascii=False), encoding="utf-8")

    elapsed = time.time() - t0
    print(
        f"done: {total_ents:,} entities across {len(paths):,} docs  "
        f"({n_stripped:,} article-stripped; {n_expanded:,} abbrev-expanded; "
        f"{n_sections_with_abbrev:,} sections had an abbreviations map)  "
        f"in {elapsed:.1f}s",
        flush=True,
    )
    if n_skipped_missing_recognized:
        print(
            f"warning: {n_skipped_missing_recognized} entities had no "
            f"recognized_text -- re-run extract_section_entities.py?",
            flush=True,
        )


if __name__ == "__main__":
    main()
