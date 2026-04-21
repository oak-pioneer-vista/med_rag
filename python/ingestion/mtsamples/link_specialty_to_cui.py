"""Link every MTSamples specialty string (primary + alt_specialties) to a UMLS CUI.

Reads data/mtsamples_docs/*.json. For each doc, collects:
  - doc['specialty']  (primary specialty of the note)
  - doc['alt_specialties'][i]['specialty']  (cross-filing specialties
    from the dedupe step; one entry per collapsed duplicate transcript)

All unique specialty strings across both surfaces are resolved in a
single Neo4j pass (exact `Atom.str_norm` match + fulltext fallback on
`concept_name_fts`), then written back to `specialty_cui` on the doc
and on every `alt_specialties[i]`. The lookup overrides whatever was
seeded during parse_mtsamples.py from data/specialty_cui.json --
with UMLS loaded, the graph is the source of truth.

Prereqs:
  - data/mtsamples_docs/*.json written by parse_mtsamples.py
  - UMLS loaded into Neo4j (scripts/load_neo4j.sh)

Usage:
  python python/ingestion/mtsamples/link_specialty_to_cui.py
"""

from __future__ import annotations

import argparse
import json
import os
import re
import time
from collections import Counter
from pathlib import Path

from neo4j import GraphDatabase

# Lucene query-parser special chars. We escape them before sending the
# query to the concept_name_fts index so inputs like "Allergy / Immunology"
# or "CT: head" don't blow up the parser.
_LUCENE_SPECIAL = re.compile(r'([+\-!(){}\[\]\^"~*?:\\/&|])')

def _escape_lucene(q: str) -> str:
    return _LUCENE_SPECIAL.sub(r"\\\1", q)

REPO = Path(__file__).resolve().parent.parent.parent.parent
DOCS_DIR = REPO / "data" / "mtsamples_docs"

NEO4J_URI = os.environ.get("NEO4J_URI", "bolt://localhost:7687")
NEO4J_USER = os.environ.get("NEO4J_USER", "neo4j")
NEO4J_PASSWORD = os.environ.get("NEO4J_PASSWORD", "medragpass")

FULLTEXT_SCORE_MIN = 5.0


def lookup_cuis(session, texts: list[str]) -> dict[str, tuple[str, str]]:
    """Batch-lookup: return {input_text_lowered: (cui, name)} for exact hits.

    Uses Atom.str_norm (RANGE indexed, lowercased) + Concept.HAS_ATOM to
    resolve in one UNWIND+MATCH. For texts with multiple hits we keep
    the first (arbitrary but deterministic within a session).
    """
    out: dict[str, tuple[str, str]] = {}
    if not texts:
        return out
    records = session.run(
        "UNWIND $texts AS t "
        "MATCH (c:Concept)-[:HAS_ATOM]->(a:Atom) "
        "WHERE a.str_norm = t "
        "WITH t, c "
        "RETURN t AS text, c.cui AS cui, c.name AS name",
        texts=texts,
    ).data()
    for r in records:
        t = r["text"]
        if t not in out:
            out[t] = (r["cui"], r["name"])
    return out


def lookup_fulltext(session, text: str) -> tuple[str, str, float] | None:
    """Single fulltext query against concept_name_fts, top hit only."""
    if not re.search(r"[A-Za-z0-9]", text):
        return None
    try:
        rec = session.run(
            "CALL db.index.fulltext.queryNodes('concept_name_fts', $q) "
            "YIELD node, score "
            "RETURN node.cui AS cui, node.name AS name, score "
            "ORDER BY score DESC LIMIT 1",
            q=_escape_lucene(text),
        ).single()
    except Exception:
        return None
    if rec and rec["score"] >= FULLTEXT_SCORE_MIN:
        return rec["cui"], rec["name"], rec["score"]
    return None


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--docs", type=Path, default=DOCS_DIR)
    args = ap.parse_args()

    paths = sorted(args.docs.glob("*.json"))
    if not paths:
        raise SystemExit(f"no JSON files in {args.docs}")

    # Collect unique specialty strings from both the primary `specialty`
    # field and every entry in `alt_specialties`. One lookup table
    # serves both surfaces.
    primary_counts: Counter[str] = Counter()
    alt_counts: Counter[str] = Counter()
    for p in paths:
        d = json.loads(p.read_text(encoding="utf-8"))
        s = (d.get("specialty") or "").strip()
        if s:
            primary_counts[s] += 1
        for alt in d.get("alt_specialties") or []:
            a = (alt.get("specialty") or "").strip()
            if a:
                alt_counts[a] += 1

    unique_specs = sorted(set(primary_counts) | set(alt_counts))
    print(f"collecting {len(unique_specs)} unique specialty strings "
          f"({len(primary_counts)} distinct as primary, "
          f"{len(alt_counts)} distinct in alt_specialties) "
          f"across {len(paths):,} docs",
          flush=True)

    t0 = time.time()
    with GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASSWORD)) as driver:
        with driver.session() as session:
            # Pass 1: exact atom_str_norm match
            texts_lower = [s.lower() for s in unique_specs]
            exact = lookup_cuis(session, texts_lower)

            # Pass 2: fulltext fallback for anything missing
            resolved: dict[str, tuple[str, str, str]] = {}  # spec -> (cui, name, source)
            for spec in unique_specs:
                hit = exact.get(spec.lower())
                if hit:
                    resolved[spec] = (hit[0], hit[1], "exact")
                    continue
                ft = lookup_fulltext(session, spec)
                if ft:
                    resolved[spec] = (ft[0], ft[1], f"fulltext/{ft[2]:.1f}")

    # Report mapping. Show primary-count + alt-count per specialty so you
    # can see which strings only appear as cross-filings.
    print(f"\nresolved {len(resolved)}/{len(unique_specs)} specialty strings in "
          f"{time.time()-t0:.1f}s\n",
          flush=True)
    for spec in unique_specs:
        hit = resolved.get(spec)
        n_prim = primary_counts.get(spec, 0)
        n_alt = alt_counts.get(spec, 0)
        tag = f"(primary: {n_prim:>3}, alt: {n_alt:>3})"
        if hit:
            cui, name, src = hit
            print(f"  {spec!r:<42} {tag}  -> {cui}  {name!r}  [{src}]")
        else:
            print(f"  {spec!r:<42} {tag}  -> <no match>")

    # Write back: primary specialty_cui on the doc AND specialty_cui on
    # each alt_specialties[i]. Touch every doc once.
    n_primary_updated = 0
    n_alt_updated = 0
    n_docs_written = 0
    for p in paths:
        d = json.loads(p.read_text(encoding="utf-8"))
        dirty = False

        s = (d.get("specialty") or "").strip()
        hit = resolved.get(s)
        new_cui = hit[0] if hit else ""
        if new_cui != d.get("specialty_cui", ""):
            d["specialty_cui"] = new_cui
            n_primary_updated += 1
            dirty = True

        for alt in d.get("alt_specialties") or []:
            a = (alt.get("specialty") or "").strip()
            alt_hit = resolved.get(a)
            alt_new = alt_hit[0] if alt_hit else ""
            if alt_new != alt.get("specialty_cui", ""):
                alt["specialty_cui"] = alt_new
                n_alt_updated += 1
                dirty = True

        if dirty:
            p.write_text(json.dumps(d, ensure_ascii=False), encoding="utf-8")
            n_docs_written += 1

    print(f"\nwrote back: {n_primary_updated} primary specialty_cui changes, "
          f"{n_alt_updated} alt_specialties[i].specialty_cui changes "
          f"across {n_docs_written} docs",
          flush=True)


if __name__ == "__main__":
    main()
