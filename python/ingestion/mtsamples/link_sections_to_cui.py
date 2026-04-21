"""Link each Section's `section_type` to a UMLS CUI via Neo4j.

Section headings in MTSamples come in both short and spelled-out forms
in the same corpus ("HPI" and "HISTORY OF PRESENT ILLNESS"), and they
must map to the same UMLS concept for downstream grouping. A small
alias table normalizes short forms (HPI/PMH/PSH/ROS/HEENT/CC/...) to
their canonical full phrase before the Neo4j lookup, so both variants
resolve to the same `section_cui`.

Matching: exact Atom.str_norm (lowercased) first; fulltext fallback on
concept_name_fts above a score threshold.

Prereqs:
  - data/mtsamples_docs/*.json with sections from parse_mtsamples.py
  - UMLS loaded into Neo4j (scripts/load_neo4j.sh)

Usage:
  python python/ingestion/mtsamples/link_sections_to_cui.py
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

_LUCENE_SPECIAL = re.compile(r'([+\-!(){}\[\]\^"~*?:\\/&|])')

def _escape_lucene(q: str) -> str:
    return _LUCENE_SPECIAL.sub(r"\\\1", q)

REPO = Path(__file__).resolve().parent.parent.parent.parent
DOCS_DIR = REPO / "data" / "mtsamples_docs"

NEO4J_URI = os.environ.get("NEO4J_URI", "bolt://localhost:7687")
NEO4J_USER = os.environ.get("NEO4J_USER", "neo4j")
NEO4J_PASSWORD = os.environ.get("NEO4J_PASSWORD", "medragpass")

FULLTEXT_SCORE_MIN = 4.0

# Short-form -> canonical phrase. Applied before the Neo4j lookup so
# different surface forms for the same concept resolve to the same CUI.
# Keys are uppercased for case-insensitive match against section_type.
SECTION_ALIASES: dict[str, str] = {
    "HPI": "history of present illness",
    "PMH": "past medical history",
    "PSH": "past surgical history",
    "ROS": "review of systems",
    "FH": "family history",
    "SH": "social history",
    "CC": "chief complaint",
    "HEENT": "head ears eyes nose throat",
    "GEN": "general appearance",
    "PE": "physical examination",
    "EXAM": "physical examination",
    "LABS": "laboratory data",
    "MEDS": "medications",
    "DX": "diagnosis",
    "TX": "treatment",
    "IMPRESSION": "assessment",  # many notes use IMPRESSION for the A of SOAP
    "PLAN": "plan of treatment",
    "ABD": "abdomen",
    "EXT": "extremities",
    "HX": "history",
    "NEURO": "neurological examination",
    "CV": "cardiovascular examination",
    "RESP": "respiratory examination",
}


def normalize_section_type(st: str) -> str:
    return SECTION_ALIASES.get(st.upper(), st)


def lookup_cuis_exact(session, texts: list[str]) -> dict[str, tuple[str, str]]:
    """Batch exact atom_str_norm lookup. Returns {lowered_text: (cui, name)}."""
    out: dict[str, tuple[str, str]] = {}
    if not texts:
        return out
    records = session.run(
        "UNWIND $texts AS t "
        "MATCH (c:Concept)-[:HAS_ATOM]->(a:Atom) "
        "WHERE a.str_norm = t "
        "RETURN t AS text, c.cui AS cui, c.name AS name",
        texts=texts,
    ).data()
    for r in records:
        t = r["text"]
        if t not in out:
            out[t] = (r["cui"], r["name"])
    return out


def lookup_fulltext(session, text: str) -> tuple[str, str, float] | None:
    # Skip queries that have no searchable tokens after escaping; Lucene's
    # classic parser rejects empty/all-operator inputs.
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

    # Gather unique (raw section_type, normalized lookup text) pairs
    st_counts: Counter[str] = Counter()
    st_to_norm: dict[str, str] = {}
    for p in paths:
        d = json.loads(p.read_text(encoding="utf-8"))
        for sec in d.get("sections", []):
            st = (sec.get("section_type") or "").strip()
            if not st:
                continue
            st_counts[st] += 1
            st_to_norm[st] = normalize_section_type(st)

    print(f"collecting {len(st_counts)} unique section_type values "
          f"across {len(paths):,} docs "
          f"(aliases applied to {sum(1 for k, v in st_to_norm.items() if k.upper() in SECTION_ALIASES)} of them)",
          flush=True)

    t0 = time.time()
    resolved: dict[str, tuple[str, str, str]] = {}  # raw_st -> (cui, name, source)

    with GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASSWORD)) as driver:
        with driver.session() as session:
            # Pass 1: batch exact match on normalized forms
            norm_texts = sorted({v.lower() for v in st_to_norm.values()})
            exact = lookup_cuis_exact(session, norm_texts)

            # Pass 2: fulltext fallback
            for raw_st, norm in st_to_norm.items():
                hit = exact.get(norm.lower())
                if hit:
                    resolved[raw_st] = (hit[0], hit[1], "exact")
                    continue
                ft = lookup_fulltext(session, norm)
                if ft:
                    resolved[raw_st] = (ft[0], ft[1], f"fulltext/{ft[2]:.1f}")

    print(f"\nresolved {len(resolved)}/{len(st_counts)} section_type values "
          f"in {time.time()-t0:.1f}s\n",
          flush=True)

    # Show top 25 by frequency
    print("top 25 by frequency:")
    for st, n in st_counts.most_common(25):
        hit = resolved.get(st)
        norm_shown = f"  (via {st_to_norm[st]!r})" if st_to_norm[st] != st else ""
        if hit:
            cui, name, src = hit
            print(f"  {st!r:<40}{norm_shown:<50} x{n:<4}  -> {cui}  {name!r}  [{src}]")
        else:
            print(f"  {st!r:<40}{norm_shown:<50} x{n:<4}  -> <no match>")

    # Verify HPI ≡ HISTORY OF PRESENT ILLNESS
    hpi_cui = resolved.get("HPI", (None,))[0]
    hopi_cui = resolved.get("HISTORY OF PRESENT ILLNESS", (None,))[0]
    if hpi_cui and hopi_cui:
        ok = "OK" if hpi_cui == hopi_cui else "MISMATCH"
        print(f"\nHPI vs HISTORY OF PRESENT ILLNESS: {hpi_cui} vs {hopi_cui}  [{ok}]")

    # Write back
    n_sections_updated = 0
    for p in paths:
        d = json.loads(p.read_text(encoding="utf-8"))
        dirty = False
        for sec in d.get("sections", []):
            st = (sec.get("section_type") or "").strip()
            hit = resolved.get(st)
            new_cui = hit[0] if hit else ""
            if new_cui != sec.get("section_cui", ""):
                sec["section_cui"] = new_cui
                dirty = True
                n_sections_updated += 1
        if dirty:
            p.write_text(json.dumps(d, ensure_ascii=False), encoding="utf-8")

    print(f"\nwrote back: {n_sections_updated} sections updated",
          flush=True)


if __name__ == "__main__":
    main()
