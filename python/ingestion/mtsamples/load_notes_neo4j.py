"""Materialize MTSamples notes, sections, entities as Neo4j nodes on top of UMLS.

Reads the per-doc JSONs at data/mtsamples_docs/*.json (written by the
full mtsamples pipeline: parse -> abbreviations -> extract_section_entities
-> normalize -> link_* -> chunk_sentences) and materializes the
clinical-note layer of the graph so it can be queried alongside the
pre-loaded UMLS Concepts/Atoms/SemanticTypes.

Graph shape (UMLS nodes are pre-existing):

    (:Note {doc_id, sample_name, specialty, doctype_cui})
        -[:IN_SPECIALTY]-> (:Concept {cui})            -- specialty_cui
        -[:IN_ALT_SPECIALTY]-> (:Concept {cui})        -- each alt_specialties[i]
        -[:HAS_SECTION]-> (:Section {chunk_id, section_type, doc_id})
                              -[:OF_TYPE]-> (:Concept {cui})  -- section_cui
                              -[:HAS_MENTION {start_char, end_char,
                                              surface_text, recognized_text,
                                              resolved_text, expanded_text,
                                              type, cui_match}]->
                          (:Entity {entity_hash, text, type})
                              -[:REFERS_TO]-> (:Concept {cui})  -- entity's resolved cui

Parallelism: dask.bag.map_partitions, N workers (each with its own
Neo4j session). Each worker shards its docs and runs an UNWIND-style
upsert per doc.

Prereqs:
  - UMLS Neo4j loaded (scripts/load_neo4j.sh) -- :Concept, :Atom,
    :SemanticType must exist.
  - data/mtsamples_docs/*.json populated by the pipeline through step
    12 (link_entities_to_cui.py) at minimum. Step 13 (chunk_sentences)
    is optional; this loader only needs doc-level metadata, sections,
    and section-level entities.

Usage:
  python python/ingestion/mtsamples/load_notes_neo4j.py [--workers 8] [--drop]
"""

from __future__ import annotations

import argparse
import json
import multiprocessing as mp
import os
import time
import uuid
from pathlib import Path

from neo4j import GraphDatabase

REPO = Path(__file__).resolve().parent.parent.parent.parent
DOCS_DIR = REPO / "data" / "mtsamples_docs"

NEO4J_URI = os.environ.get("NEO4J_URI", "bolt://localhost:7687")
NEO4J_USER = os.environ.get("NEO4J_USER", "neo4j")
NEO4J_PASSWORD = os.environ.get("NEO4J_PASSWORD", "medragpass")


# One upsert per doc. Subqueries guard every Concept merge on non-empty
# CUI so we never create empty-key Concept nodes alongside the real UMLS
# ones. HAS_MENTION uses (start_char, end_char) to dedupe re-runs --
# identical spans within a section collapse onto a single edge.
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
CALL {
    WITH n
    UNWIND $note.alt_cuis AS alt_cui
    WITH n, alt_cui WHERE alt_cui <> ''
    MERGE (ac:Concept {cui: alt_cui})
    MERGE (n)-[:IN_ALT_SPECIALTY]->(ac)
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
    UNWIND sec.entities AS m
        MERGE (e:Entity {entity_hash: m.entity_hash})
        SET e.text = m.resolved_text,
            e.type = m.type
        MERGE (s)-[hm:HAS_MENTION {start_char: m.start_char,
                                    end_char:   m.end_char}]->(e)
        SET hm.surface_text    = m.surface_text,
            hm.recognized_text = m.recognized_text,
            hm.resolved_text   = m.resolved_text,
            hm.expanded_text   = m.expanded_text,
            hm.type            = m.type,
            hm.cui_match       = m.cui_match
        WITH e, m
        CALL {
            WITH e, m
            WITH e, m WHERE m.cui <> ''
            MERGE (c:Concept {cui: m.cui})
            MERGE (e)-[:REFERS_TO]->(c)
        }
"""


def _build_payload(d: dict) -> dict | None:
    """Shape one doc's UPSERT payload. Returns None if the doc has no sections."""
    note = {
        "doc_id": int(d["doc_id"]),
        "sample_name": (d.get("sample_name") or "").strip(),
        "specialty": (d.get("specialty") or "").strip(),
        "specialty_cui": d.get("specialty_cui") or "",
        "doctype_cui": d.get("doctype_cui") or "",
        "alt_cuis": [
            (alt.get("specialty_cui") or "").strip()
            for alt in (d.get("alt_specialties") or [])
            if (alt.get("specialty_cui") or "").strip()
        ],
    }

    sections: list[dict] = []
    for sec in d.get("sections", []):
        chunk_id = (sec.get("chunk_id") or "").strip()
        if not chunk_id:
            continue
        entities: list[dict] = []
        for e in sec.get("entities") or []:
            h = (e.get("entity_hash") or "").strip()
            if not h:
                continue
            entities.append({
                "entity_hash": h,
                "surface_text": e.get("surface_text", "") or "",
                "recognized_text": e.get("recognized_text", "") or "",
                "resolved_text": e.get("resolved_text", "") or "",
                "expanded_text": e.get("expanded_text", "") or "",
                "type": e.get("type", "") or "",
                "start_char": int(e.get("start_char", 0)),
                "end_char": int(e.get("end_char", 0)),
                "cui": e.get("cui") or "",
                "cui_match": e.get("cui_match", "") or "",
            })
        sections.append({
            "chunk_id": chunk_id,
            "section_type": sec.get("section_type", "") or "",
            "section_cui": sec.get("section_cui") or "",
            "entities": entities,
        })
    if not sections:
        return None
    return {"note": note, "sections": sections}


# ---------- mp.Pool worker state ----------

_DRIVER = None
_SESSION = None
_WID = None


def _init_worker():
    global _DRIVER, _SESSION, _WID
    _DRIVER = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASSWORD))
    _SESSION = _DRIVER.session()
    _WID = f"{os.getpid()}/{uuid.uuid4().hex[:6]}"
    print(f"[worker {_WID}] neo4j session ready", flush=True)


def _process_shard(args: tuple) -> dict:
    shard_paths = args
    paths = [Path(p) for p in shard_paths]
    if not paths:
        return {"wid": _WID, "docs": 0, "sections": 0, "mentions": 0,
                "refers_to": 0, "elapsed_s": 0.0}

    t0 = time.time()
    n_docs = 0
    n_sections = 0
    n_mentions = 0
    n_refers = 0

    # Neo4j's managed transactions auto-retry on TransientError (deadlock
    # on a shared :Concept MERGE, lock wait timeout, etc.). With N workers
    # all MERGE-ing popular CUIs like C0013227 (Pharmaceutical Preparations)
    # deadlocks happen in normal operation; execute_write is the idiomatic
    # retry wrapper.
    def _tx_upsert(tx, payload):
        tx.run(UPSERT_CYPHER, **payload).consume()

    for p in paths:
        try:
            d = json.loads(p.read_text(encoding="utf-8"))
        except Exception:
            continue
        payload = _build_payload(d)
        if payload is None:
            continue
        _SESSION.execute_write(_tx_upsert, payload)
        n_docs += 1
        n_sections += len(payload["sections"])
        for sec in payload["sections"]:
            n_mentions += len(sec["entities"])
            n_refers += sum(1 for e in sec["entities"] if e["cui"])

    elapsed = time.time() - t0
    print(f"[worker {_WID}] done: {n_docs} notes, {n_sections} sections, "
          f"{n_mentions} mentions, {n_refers} with cui  in {elapsed:.1f}s",
          flush=True)
    return {"wid": _WID, "docs": n_docs, "sections": n_sections,
            "mentions": n_mentions, "refers_to": n_refers,
            "elapsed_s": elapsed}


def _shard(seq: list, n: int) -> list[list]:
    k, m = divmod(len(seq), n)
    out, i = [], 0
    for shard_idx in range(n):
        size = k + (1 if shard_idx < m else 0)
        out.append(seq[i : i + size])
        i += size
    return out


def _drop_existing_notes_layer(uri: str, user: str, password: str) -> None:
    """Delete every :Note, :Section, :Entity (and their edges) but leave the
    UMLS layer (:Concept, :Atom, :SemanticType, :Source) untouched. Also
    drop the stale (text, type) uniqueness constraint on :Entity that
    was created by the old schema -- under the entity_hash-keyed schema
    two different hashes can share the same resolved text + type, which
    that constraint would incorrectly reject."""
    print("dropping existing :Note/:Section/:Entity layer...", flush=True)
    with GraphDatabase.driver(uri, auth=(user, password)) as driver:
        with driver.session() as session:
            # Batch-detach-delete via APOC to avoid tx size blowups.
            for label in ("Entity", "Section", "Note"):
                t0 = time.time()
                n = session.run(
                    f"MATCH (x:{label}) RETURN count(x) AS n"
                ).single()["n"]
                if not n:
                    print(f"  :{label}  0 existing", flush=True)
                    continue
                res = session.run(
                    "CALL apoc.periodic.iterate("
                    f"'MATCH (x:{label}) RETURN x', "
                    "'DETACH DELETE x', "
                    "{batchSize:5000, parallel:false}) "
                    "YIELD batches, total RETURN batches, total"
                ).single()
                print(f"  :{label}  deleted {res['total']} ({res['batches']} batches) "
                      f"in {time.time()-t0:.1f}s",
                      flush=True)

            # Stale constraint from the old (text, type)-keyed schema.
            stale_constraints = [
                ("entity_text_type_unique", "DROP CONSTRAINT entity_text_type_unique IF EXISTS"),
            ]
            for name, cypher in stale_constraints:
                session.run(cypher).consume()
                print(f"  dropped constraint {name} (if existed)", flush=True)

            # Make the new access key uniqueness-enforced.
            session.run(
                "CREATE CONSTRAINT entity_hash_unique IF NOT EXISTS "
                "FOR (e:Entity) REQUIRE e.entity_hash IS UNIQUE"
            ).consume()
            print("  ensured constraint entity_hash_unique", flush=True)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--docs", type=Path, default=DOCS_DIR)
    ap.add_argument("--workers", type=int, default=8,
                    help="mp.Pool worker count (each owns a Neo4j session)")
    ap.add_argument("--drop", action="store_true",
                    help="drop existing :Note/:Section/:Entity before load")
    args = ap.parse_args()

    paths = [str(p) for p in sorted(args.docs.glob("*.json"))]
    if not paths:
        raise SystemExit(f"no JSON files in {args.docs}")

    if args.drop:
        _drop_existing_notes_layer(NEO4J_URI, NEO4J_USER, NEO4J_PASSWORD)

    workers = max(1, min(args.workers, len(paths)))
    shards = _shard(paths, workers)
    sizes = [len(s) for s in shards]
    print(f"dispatching {len(paths):,} docs across {workers} workers "
          f"(shard docs={sizes})",
          flush=True)

    ctx = mp.get_context("spawn")
    wall_t0 = time.time()
    with ctx.Pool(processes=workers, initializer=_init_worker) as pool:
        results = pool.map(_process_shard, [shard for shard in shards])

    total = {"docs": 0, "sections": 0, "mentions": 0, "refers_to": 0}
    for r in results:
        for k in total:
            total[k] += r.get(k, 0)
    per_worker = sorted(r.get("elapsed_s", 0.0) for r in results)
    wall = time.time() - wall_t0
    print(
        f"done: {total['docs']:,} notes, {total['sections']:,} sections, "
        f"{total['mentions']:,} mentions ({total['refers_to']:,} REFERS_TO edges) "
        f"in {wall:.1f}s wall  "
        f"[per-worker work {min(per_worker):.1f}..{max(per_worker):.1f}s]",
        flush=True,
    )


if __name__ == "__main__":
    main()
