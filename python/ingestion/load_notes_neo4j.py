"""Materialize MTSamples notes, sections, and entity mentions as a Neo4j graph.

Target shape (Atom + Concept come from the UMLS import; this script only
adds the clinical-note layer on top):

    (:Note {doc_id, sample_name, specialty, doctype_cui})
        -[:IN_SPECIALTY]-> (:Concept {cui})                   -- specialty_cui
        -[:HAS_SECTION]-> (:Section {chunk_id, section_type})
                              -[:OF_TYPE]-> (:Concept {cui})  -- section_cui
                              -[:HAS_MENTION {start_char, end_char,
                                              raw_text, expanded}]->
                          (:Entity {text, type})
                              -[:LINKS_TO]-> (:Atom {str_norm})

`Entity.text` is the canonical, *expansion-resolved* form. When the Stanza
extraction step recorded an `expansions` dict on a mention, each abbreviation
token is substituted with its S-H long form before the lookup, so a mention
like "CHF exacerbation" becomes Entity `text = "congestive heart failure
exacerbation"` and its str_norm probe goes against the UMLS Atom index with
the expanded string. Mentions whose canonical form has no Atom hit still
materialize an :Entity node but without a `:LINKS_TO` edge -- the graph
keeps coverage gaps visible instead of silently dropping them.

The Stanza output is emitted per Qdrant-aligned window (chunk_id ending in
`#<n>` for multi-window sections). This loader re-groups by
`parent_chunk_id` so a Section is the graph-level unit; window-level
:Section nodes would not round-trip back to the section-local `section_cui`
from the parse step.

Prereqs:
  - UMLS Neo4j loaded (scripts/load_neo4j.sh)
  - scripts/create_neo4j_indices.sh run (so atom_str_norm is populated and
    note/section/entity constraints exist)
  - extract_entities.py has produced data/entities/chunk_entities.jsonl

Usage:
    python python/ingestion/load_notes_neo4j.py \\
        [--entities PATH] [--uri bolt://localhost:7687] [--limit N]
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Iterable

from neo4j import GraphDatabase

REPO = Path(__file__).resolve().parent.parent.parent
DOCS_DIR = REPO / "data" / "mtsamples_docs"
ENTITIES_DEFAULT = REPO / "data" / "entities" / "chunk_entities.jsonl"

NEO4J_URI = "bolt://localhost:7687"
NEO4J_AUTH = ("neo4j", "medragpass")


def _apply_expansions(text: str, expansions: dict[str, str] | None) -> str:
    """Replace each abbreviation token in `text` with its long form.

    Whole-word boundary so "CT" in "CT scan" expands but "ct" inside
    "fact" stays put. re.escape handles S-H keys that include punctuation
    ("HbA1c", "T-cell" if ever produced).
    """
    if not expansions:
        return text
    out = text
    for tok, long_form in expansions.items():
        out = re.sub(rf"\b{re.escape(tok)}\b", long_form, out)
    return out


def _norm_for_atom(text: str) -> str:
    """Mirror the normalization applied to Atom.str_norm in the UMLS graph.

    create_neo4j_indices.sh sets `a.str_norm = toLower(a.str)`, so an index
    lookup needs the lookup key normalized the same way. Strip surrounding
    whitespace and lowercase; anything more aggressive (e.g. punctuation
    collapse) would drift from the Atom keys and silently miss.
    """
    return text.strip().lower()


def _load_doc_meta(doc_id: int) -> dict:
    """Load the parsed doc JSON for section_cui + specialty_cui lookups."""
    # parse_mtsamples wrote files with zero-padded ids; mirror the padding.
    path = DOCS_DIR / f"{int(doc_id):04d}.json"
    with path.open(encoding="utf-8") as f:
        return json.load(f)


def _iter_entity_records(path: Path) -> Iterable[dict]:
    with path.open(encoding="utf-8") as f:
        for line in f:
            yield json.loads(line)


def _build_doc_payload(doc_id: int, records: list[dict]) -> dict:
    """Shape one UPSERT_CYPHER payload from all window records of a doc.

    Groups mentions by parent_chunk_id (= Section). Deduplicates within a
    section by (raw_text, type, start_char, end_char) so identical rows
    from overlapping windows don't land as duplicate HAS_MENTION edges;
    offsets that differ across overlap windows still produce separate
    edges because they're window-relative. TODO: promote section-relative
    offsets in extract_entities output for exact cross-window dedupe.
    """
    doc_meta = _load_doc_meta(doc_id)
    section_cui_by_chunk = {
        s["chunk_id"]: (s.get("section_cui") or "")
        for s in doc_meta.get("sections", [])
    }

    note = {
        "doc_id": int(doc_id),
        "sample_name": doc_meta.get("sample_name", "") or "",
        "specialty": doc_meta.get("specialty", "") or "",
        "specialty_cui": doc_meta.get("specialty_cui", "") or "",
        "doctype_cui": doc_meta.get("doctype_cui", "") or "",
    }

    sections: dict[str, dict] = {}
    seen: set[tuple] = set()
    for r in records:
        sec_id = r["parent_chunk_id"]
        sec = sections.setdefault(
            sec_id,
            {
                "chunk_id": sec_id,
                "section_type": r.get("section_type", "") or "",
                "section_cui": section_cui_by_chunk.get(sec_id, "") or "",
                "mentions": [],
            },
        )
        for e in r.get("entities", []):
            key = (sec_id, e["text"], e["type"], e["start_char"], e["end_char"])
            if key in seen:
                continue
            seen.add(key)
            expansions = e.get("expansions") or {}
            canonical = _apply_expansions(e["text"], expansions)
            sec["mentions"].append({
                "raw_text": e["text"],
                "canonical_text": canonical,
                "type": e["type"],
                "start_char": int(e["start_char"]),
                "end_char": int(e["end_char"]),
                "str_norm": _norm_for_atom(canonical),
                "expanded": bool(expansions),
            })

    return {"note": note, "sections": list(sections.values())}


# One Cypher per doc: MERGE the Note + its specialty link, then UNWIND
# sections (Section + OF_TYPE), then UNWIND mentions (Entity + HAS_MENTION
# + optional LINKS_TO). Subqueries guard on empty cui/str_norm so missing
# metadata or out-of-vocabulary mentions skip their link without exploding
# the Entity count with empty-key merges.
UPSERT_CYPHER = """
MERGE (n:Note {doc_id: $note.doc_id})
SET n.sample_name = $note.sample_name,
    n.specialty   = $note.specialty,
    n.doctype_cui = $note.doctype_cui
WITH n
CALL {
    WITH n
    WITH n WHERE $note.specialty_cui <> ''
    MERGE (sc:Concept {cui: $note.specialty_cui})
    MERGE (n)-[:IN_SPECIALTY]->(sc)
}
WITH n
UNWIND $sections AS sec
    MERGE (s:Section {chunk_id: sec.chunk_id})
    SET s.section_type = sec.section_type,
        s.doc_id       = $note.doc_id
    MERGE (n)-[:HAS_SECTION]->(s)
    WITH s, sec
    CALL {
        WITH s, sec
        WITH s, sec WHERE sec.section_cui <> ''
        MERGE (tc:Concept {cui: sec.section_cui})
        MERGE (s)-[:OF_TYPE]->(tc)
    }
    WITH s, sec
    UNWIND sec.mentions AS m
        MERGE (e:Entity {text: m.canonical_text, type: m.type})
        MERGE (s)-[hm:HAS_MENTION {start_char: m.start_char,
                                    end_char:   m.end_char}]->(e)
        SET hm.raw_text = m.raw_text,
            hm.expanded = m.expanded
        WITH e, m
        CALL {
            WITH e, m
            WITH e, m WHERE m.str_norm <> ''
            OPTIONAL MATCH (a:Atom {str_norm: m.str_norm})
            FOREACH (_ IN CASE WHEN a IS NOT NULL THEN [1] ELSE [] END |
                MERGE (e)-[:LINKS_TO]->(a)
            )
        }
"""


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--entities", type=str, default=str(ENTITIES_DEFAULT),
                    help="path to the per-window entities JSONL")
    ap.add_argument("--uri", type=str, default=NEO4J_URI)
    ap.add_argument("--limit", type=int, default=None,
                    help="process only the first N docs (debug/smoke)")
    args = ap.parse_args()

    path = Path(args.entities)
    print(f"loading entity records from {path}")

    driver = GraphDatabase.driver(args.uri, auth=NEO4J_AUTH)
    # extract_entities writes records in doc_id order (it sorts the docs
    # path list before dispatch), so same-doc records are contiguous in the
    # shard-concatenated JSONL. Stream and flush each doc as its run ends.
    n_docs = 0
    n_mentions = 0
    n_linked = 0
    current_doc: int | None = None
    buf: list[dict] = []

    def _flush(session) -> None:
        nonlocal n_docs, n_mentions, n_linked, current_doc, buf
        if current_doc is None:
            return
        payload = _build_doc_payload(current_doc, buf)
        if payload["sections"]:
            session.run(UPSERT_CYPHER, **payload).consume()
            n_docs += 1
            for sec in payload["sections"]:
                n_mentions += len(sec["mentions"])
                n_linked += sum(1 for m in sec["mentions"] if m["str_norm"])

    with driver.session() as session:
        for rec in _iter_entity_records(path):
            did = int(rec["doc_id"])
            if did != current_doc:
                _flush(session)
                if args.limit is not None and n_docs >= args.limit:
                    current_doc = None
                    break
                current_doc = did
                buf = []
            buf.append(rec)
        _flush(session)

    driver.close()
    print(
        f"merged {n_docs} notes, {n_mentions} mentions, "
        f"{n_linked} with a str_norm lookup attempted"
    )


if __name__ == "__main__":
    main()
