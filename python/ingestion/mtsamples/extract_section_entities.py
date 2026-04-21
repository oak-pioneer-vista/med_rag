"""Per-section NER + span recording for every MTSamples doc.

Runs Stanza's `mimic` pipeline with the `i2b2` NER processor over each
Section's text (not windowed -- sections are the logical unit here
because downstream grounding lines up section -> Concept, not chunk ->
Concept). Each section's `entities` list in its per-doc JSON is filled
with one record per mention:

  {
    "text"          : surface form as it appears in the section,
    "type"          : i2b2 label (PROBLEM | TEST | TREATMENT),
    "start_char"    : offset within the section's text,
    "end_char"      : offset within the section's text,
    "resolved_text" : optional -- present when abbreviation substitution
                      (from the doc's `abbreviations` map built in
                      build_abbreviations.py) changed the surface form
                      (e.g. "CHF exacerbation" ->
                      "congestive heart failure exacerbation")
  }

Sections are batched through Stanza for GPU throughput. Each Document
in a batch carries its own text, so character offsets returned by
Stanza are already section-local -- no window math required, unlike
extract_entities.py which aligns to Qdrant windows.

Prereqs:
  - data/mtsamples_docs/*.json written by parse_mtsamples.py (and
    ideally build_abbreviations.py, for richer `resolved_text` output)
  - Stanza mimic/i2b2 models:
      python -c "import stanza; stanza.download('en', package='mimic', processors={'ner':'i2b2'})"

Usage:
  python python/ingestion/mtsamples/extract_section_entities.py [--batch 128] [--cpu]
"""

from __future__ import annotations

import argparse
import json
import re
import time
from pathlib import Path

import stanza

REPO = Path(__file__).resolve().parent.parent.parent.parent
DOCS_DIR = REPO / "data" / "mtsamples_docs"

# Token boundary pattern for abbreviation substitution. Alphanumerics +
# intra-word hyphens (e.g. C-spine) match; surrounding punctuation
# (commas, parens) does not.
TOKEN_RE = re.compile(r"\b[A-Za-z][A-Za-z0-9-]*\b")


def build_pipeline(use_gpu: bool) -> stanza.Pipeline:
    # package=None + explicit processors dict skips lemma/pos/depparse,
    # which matters because loading the mimic lemma model under newer
    # Stanza tries to read a config key that is missing in that model.
    return stanza.Pipeline(
        lang="en",
        package=None,
        processors={"tokenize": "mimic", "ner": "i2b2"},
        use_gpu=use_gpu,
        verbose=False,
        download_method="reuse_resources",
    )


def resolve_with_abbrevs(text: str, abbrev_upper: dict[str, str]) -> str:
    """Token-wise substitute known abbreviations with their expansions.

    Case-insensitive match against the uppercase-keyed map. If no tokens
    match, returns the original string unchanged (caller uses that to
    decide whether to record `resolved_text`).
    """
    if not abbrev_upper:
        return text

    def repl(m: re.Match) -> str:
        tok = m.group(0)
        return abbrev_upper.get(tok.upper(), tok)

    return TOKEN_RE.sub(repl, text)


def entities_from_stanza_doc(doc: stanza.Document) -> list[dict]:
    """Extract entity records from a Stanza Document."""
    return [
        {
            "text": ent.text,
            "type": ent.type,
            "start_char": ent.start_char,
            "end_char": ent.end_char,
        }
        for ent in doc.entities
    ]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--docs", type=Path, default=DOCS_DIR)
    ap.add_argument("--batch", type=int, default=128,
                    help="sections per Stanza bulk_process batch")
    ap.add_argument("--cpu", action="store_true", help="force CPU (default: use GPU if available)")
    args = ap.parse_args()

    paths = sorted(args.docs.glob("*.json"))
    if not paths:
        raise SystemExit(
            f"no JSON files in {args.docs} -- run "
            f"python/ingestion/mtsamples/parse_mtsamples.py first"
        )

    print(f"loading Stanza mimic/i2b2 pipeline (use_gpu={not args.cpu})...")
    nlp = build_pipeline(use_gpu=not args.cpu)

    # Load all docs, flatten every (doc_idx, section_idx, text) where text is non-empty.
    docs: list[tuple[Path, dict]] = [
        (p, json.loads(p.read_text(encoding="utf-8"))) for p in paths
    ]
    items: list[tuple[int, int, str]] = []
    for di, (_, d) in enumerate(docs):
        for si, section in enumerate(d.get("sections", [])):
            text = section.get("text") or ""
            if text.strip():
                items.append((di, si, text))
            else:
                section["entities"] = []

    n_sections = len(items)
    print(f"processing {n_sections:,} non-empty sections across {len(paths):,} docs")

    total_ents = 0
    n_with_resolved = 0
    t0 = time.time()
    for bi in range(0, n_sections, args.batch):
        chunk = items[bi : bi + args.batch]
        stanza_docs = [stanza.Document([], text=t) for _, _, t in chunk]
        nlp.bulk_process(stanza_docs)

        for (di, si, _), sdoc in zip(chunk, stanza_docs):
            ents = entities_from_stanza_doc(sdoc)

            abbrev_map = docs[di][1].get("abbreviations") or {}
            abbrev_upper = {k.upper(): v for k, v in abbrev_map.items()}
            for ent in ents:
                resolved = resolve_with_abbrevs(ent["text"], abbrev_upper)
                if resolved and resolved != ent["text"]:
                    ent["resolved_text"] = resolved
                    n_with_resolved += 1

            docs[di][1]["sections"][si]["entities"] = ents
            total_ents += len(ents)

        done = min(bi + args.batch, n_sections)
        print(f"  {done:>6}/{n_sections} sections  "
              f"({done / max(time.time() - t0, 1e-9):.1f} sec/s)")

    # Write per-doc JSONs back in place.
    for p, d in docs:
        p.write_text(json.dumps(d, ensure_ascii=False), encoding="utf-8")

    print(
        f"done: {total_ents:,} entities across {n_sections:,} sections "
        f"({n_with_resolved:,} with abbreviation-resolved text) "
        f"in {time.time() - t0:.1f}s"
    )


if __name__ == "__main__":
    main()
