"""Link each per-section entity to UMLS CUI + semantic types (TUIs).

Dask map-reduce design so Neo4j only sees unique entities:

1. **Map-reduce dedup** (dask.bag, N workers): each worker reads its
   shard of per-doc JSONs in parallel, emits `{entity_hash: expanded_text}`
   dicts. `entity_hash = sha1(expanded_text_lower)[:16]` -- a short,
   deterministic, content-addressable key. Per-worker dicts are then
   union-merged into one global dedup map.

2. **Exact CUI pass**: unique `(hash, text)` pairs batched through a
   single `UNWIND + MATCH` against `Atom.str_norm`.

3. **Fulltext CUI fallback**: residual hashes split across `--workers`
   worker processes, each with its own Neo4j session, batched through
   `UNWIND + CALL { fulltext }`.

4. **TUI pass**: unique CUIs from (2)+(3) batched via UNWIND + MATCH
   on `HAS_SEMTYPE` -> `SemanticType`. A concept may have multiple
   semantic types so results are lists.

5. **Write-back**: every entity mention gets its hash stamped and the
   resolved `cui`, `cui_name`, `cui_match`, `tuis`, `tui_names` fields
   populated by looking up its hash in the final `{hash: payload}` map.

Fields written per entity:
  entity_hash  : sha1(expanded_text_lower)[:16]
  cui          : UMLS CUI string, or "" if no match
  cui_name     : matched Concept.name
  cui_match    : "exact" | "fulltext/<score>" | ""
  tuis         : list[str] of TUIs (T047, T121, ...) for the matched Concept
  tui_names    : list[str] of TUI human-readable names

Prereqs:
  - data/mtsamples_docs/*.json normalized by normalize_section_entities.py
  - UMLS loaded into Neo4j (scripts/load_neo4j.sh)

Usage:
  python python/ingestion/mtsamples/link_entities_to_cui.py [--workers 16] [--batch 500]
"""

from __future__ import annotations

import argparse
import hashlib
import json
import multiprocessing as mp
import os
import re
import time
from pathlib import Path

import dask.bag as db
from neo4j import GraphDatabase

REPO = Path(__file__).resolve().parent.parent.parent.parent
DOCS_DIR = REPO / "data" / "mtsamples_docs"

NEO4J_URI = os.environ.get("NEO4J_URI", "bolt://localhost:7687")
NEO4J_USER = os.environ.get("NEO4J_USER", "neo4j")
NEO4J_PASSWORD = os.environ.get("NEO4J_PASSWORD", "medragpass")

EXACT_BATCH_DEFAULT = 2000
FULLTEXT_BATCH_DEFAULT = 500
TUI_BATCH = 2000
FULLTEXT_SCORE_MIN = 5.0

_LUCENE_SPECIAL = re.compile(r'([+\-!(){}\[\]\^"~*?:\\/&|])')
_ALNUM_RE = re.compile(r"[A-Za-z0-9]")


def _escape_lucene(q: str) -> str:
    return _LUCENE_SPECIAL.sub(r"\\\1", q)


def entity_hash(text: str) -> str:
    """Deterministic 16-char hash key for an (already-lowercased) entity string."""
    return hashlib.sha1(text.encode("utf-8")).hexdigest()[:16]


# ---------- Map-reduce dedup (dask) ----------

def _partition_unique(paths_iter) -> list[dict[str, str]]:
    """Dask map-partitions worker: return list of one dict per partition.

    Emits {hash: expanded_text_lower} for every non-empty expanded_text
    across the paths in this partition. Within-partition dedup uses the
    hash as the key so we never allocate duplicate strings in memory.
    """
    seen: dict[str, str] = {}
    for p in paths_iter:
        try:
            d = json.loads(Path(p).read_text(encoding="utf-8"))
        except Exception:
            continue
        for sec in d.get("sections", []):
            for e in sec.get("entities") or []:
                t = (e.get("expanded_text") or "").strip().lower()
                if not t:
                    continue
                h = entity_hash(t)
                if h not in seen:
                    seen[h] = t
    return [seen]


def map_reduce_unique_entities(paths: list[str], workers: int) -> dict[str, str]:
    """Returns {hash: text} across the whole corpus, via dask map-reduce."""
    bag = db.from_sequence(paths, npartitions=workers)
    parts = bag.map_partitions(_partition_unique).compute(
        scheduler="processes", num_workers=workers
    )
    unique: dict[str, str] = {}
    for d in parts:
        unique.update(d)
    return unique


# ---------- Neo4j lookups ----------

def batch_exact(session, items: list[dict]) -> dict[str, tuple[str, str]]:
    """items: [{'hash': h, 'text': t}, ...]. Returns {hash: (cui, name)}."""
    out: dict[str, tuple[str, str]] = {}
    if not items:
        return out
    records = session.run(
        "UNWIND $items AS row "
        "MATCH (c:Concept)-[:HAS_ATOM]->(a:Atom) "
        "WHERE a.str_norm = row.text "
        "RETURN row.hash AS hash, c.cui AS cui, c.name AS name",
        items=items,
    ).data()
    for r in records:
        if r["hash"] not in out:
            out[r["hash"]] = (r["cui"], r["name"])
    return out


def batch_fulltext(session, items: list[dict], min_score: float) -> dict[str, tuple[str, str, float]]:
    """items: [{'hash': h, 'q': escaped_text}, ...]. Returns {hash: (cui, name, score)}."""
    out: dict[str, tuple[str, str, float]] = {}
    if not items:
        return out
    try:
        records = session.run(
            "UNWIND $items AS row "
            "CALL { "
            "  WITH row "
            "  CALL db.index.fulltext.queryNodes('concept_name_fts', row.q) "
            "    YIELD node, score "
            "  WITH node, score ORDER BY score DESC LIMIT 1 "
            "  RETURN node.cui AS cui, node.name AS name, score "
            "} "
            "RETURN row.hash AS hash, cui, name, score",
            items=items,
        ).data()
    except Exception:
        # Fall back to per-item on batch failure (one bad query otherwise
        # kills the whole batch).
        for row in items:
            try:
                rec = session.run(
                    "CALL db.index.fulltext.queryNodes('concept_name_fts', $q) "
                    "YIELD node, score "
                    "RETURN node.cui AS cui, node.name AS name, score "
                    "ORDER BY score DESC LIMIT 1",
                    q=row["q"],
                ).single()
                if rec and rec["score"] >= min_score:
                    out[row["hash"]] = (rec["cui"], rec["name"], rec["score"])
            except Exception:
                continue
        return out
    for r in records:
        if r["score"] is not None and r["score"] >= min_score:
            out[r["hash"]] = (r["cui"], r["name"], r["score"])
    return out


def batch_tuis(session, cuis: list[str]) -> dict[str, tuple[list[str], list[str]]]:
    """Return {cui: (tuis, tui_names)} for each CUI with HAS_SEMTYPE edges.

    A concept may have multiple semantic types; collect both TUIs and
    their human-readable names. Concepts with zero semantic types
    (rare but possible) simply don't appear in the result.
    """
    out: dict[str, tuple[list[str], list[str]]] = {}
    if not cuis:
        return out
    records = session.run(
        "UNWIND $cuis AS cui "
        "MATCH (c:Concept {cui: cui})-[:HAS_SEMTYPE]->(st:SemanticType) "
        "WITH cui, collect(DISTINCT st.tui) AS tuis, "
        "     collect(DISTINCT st.name) AS tui_names "
        "RETURN cui, tuis, tui_names",
        cuis=cuis,
    ).data()
    for r in records:
        out[r["cui"]] = (r["tuis"], r["tui_names"])
    return out


# ---------- Parallel fulltext workers ----------

_DRIVER = None
_SESSION = None
_WID = None


def _init_worker():
    import uuid
    global _DRIVER, _SESSION, _WID
    _DRIVER = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASSWORD))
    _SESSION = _DRIVER.session()
    _WID = f"{os.getpid()}/{uuid.uuid4().hex[:6]}"
    print(f"[worker {_WID}] neo4j session ready", flush=True)


def _fulltext_shard(args: tuple) -> list[tuple[str, tuple[str, str, float]]]:
    shard_items, batch_size, min_score = args
    t0 = time.time()
    out: list[tuple[str, tuple[str, str, float]]] = []
    for i in range(0, len(shard_items), batch_size):
        chunk = shard_items[i : i + batch_size]
        hits = batch_fulltext(_SESSION, chunk, min_score)
        for h, v in hits.items():
            out.append((h, v))
    print(f"[worker {_WID}] done: {len(out)} fulltext hits out of "
          f"{len(shard_items)} items in {time.time()-t0:.1f}s",
          flush=True)
    return out


def _shard(seq: list, n: int) -> list[list]:
    k, m = divmod(len(seq), n)
    out, i = [], 0
    for shard_idx in range(n):
        size = k + (1 if shard_idx < m else 0)
        out.append(seq[i : i + size])
        i += size
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--docs", type=Path, default=DOCS_DIR)
    ap.add_argument("--workers", type=int, default=16,
                    help="process count for BOTH the dask map-reduce dedup "
                         "and the fulltext fallback (each owns a Neo4j session "
                         "during the fulltext phase)")
    ap.add_argument("--batch", type=int, default=FULLTEXT_BATCH_DEFAULT,
                    help="items per UNWIND+CALL fulltext batch (per worker)")
    ap.add_argument("--exact-batch", type=int, default=EXACT_BATCH_DEFAULT)
    ap.add_argument("--no-fulltext", action="store_true",
                    help="skip fulltext fallback")
    args = ap.parse_args()

    paths = [str(p) for p in sorted(args.docs.glob("*.json"))]
    if not paths:
        raise SystemExit(f"no JSON files in {args.docs}")

    # ---- Phase 1: map-reduce dedup via dask ----
    print(f"map-reduce dedup across {len(paths):,} docs on {args.workers} workers...",
          flush=True)
    t0 = time.time()
    unique = map_reduce_unique_entities(paths, args.workers)
    print(f"  -> {len(unique):,} unique entity hashes "
          f"in {time.time()-t0:.1f}s",
          flush=True)

    # Prepare batched inputs keyed by hash.
    exact_items = [{"hash": h, "text": t} for h, t in unique.items()]

    # ---- Phase 2: exact CUI pass ----
    resolved: dict[str, tuple[str, str, str]] = {}  # hash -> (cui, name, source)
    t0 = time.time()
    with GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASSWORD)) as driver:
        with driver.session() as session:
            for i in range(0, len(exact_items), args.exact_batch):
                chunk = exact_items[i : i + args.exact_batch]
                hits = batch_exact(session, chunk)
                for h, (cui, name) in hits.items():
                    resolved[h] = (cui, name, "exact")
    n_exact = len(resolved)
    print(f"exact pass: {n_exact:,}/{len(unique):,} hashes hit "
          f"({100*n_exact/len(unique):.1f}%) in {time.time()-t0:.1f}s",
          flush=True)

    # ---- Phase 3: parallel fulltext fallback ----
    if not args.no_fulltext:
        residual = [
            {"hash": h, "q": _escape_lucene(t)}
            for h, t in unique.items()
            if h not in resolved and _ALNUM_RE.search(t)
        ]
        workers = max(1, min(args.workers, len(residual)))
        shards = _shard(residual, workers)
        print(f"fulltext fallback: {len(residual):,} hashes -> "
              f"{workers} workers (shard sizes {[len(s) for s in shards]}, "
              f"batch {args.batch})",
              flush=True)

        t0 = time.time()
        ctx = mp.get_context("spawn")
        with ctx.Pool(processes=workers, initializer=_init_worker) as pool:
            shard_results = pool.map(
                _fulltext_shard,
                [(s, args.batch, FULLTEXT_SCORE_MIN) for s in shards],
            )
        n_ft = 0
        for sr in shard_results:
            for h, (cui, name, score) in sr:
                resolved[h] = (cui, name, f"fulltext/{score:.1f}")
                n_ft += 1
        print(f"fulltext pass: {n_ft:,} additional hits in {time.time()-t0:.1f}s",
              flush=True)

    total = len(resolved)
    print(f"\ncombined: {total:,}/{len(unique):,} hashes resolved "
          f"({100*total/len(unique):.1f}%)",
          flush=True)

    # ---- Phase 4: TUI lookup ----
    unique_cuis = sorted({v[0] for v in resolved.values()})
    print(f"fetching TUIs for {len(unique_cuis):,} unique CUIs...", flush=True)
    t0 = time.time()
    cui_to_tuis: dict[str, tuple[list[str], list[str]]] = {}
    with GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASSWORD)) as driver:
        with driver.session() as session:
            for i in range(0, len(unique_cuis), TUI_BATCH):
                chunk = unique_cuis[i : i + TUI_BATCH]
                cui_to_tuis.update(batch_tuis(session, chunk))
    print(f"  -> TUIs for {len(cui_to_tuis):,}/{len(unique_cuis):,} CUIs "
          f"({100*len(cui_to_tuis)/max(len(unique_cuis),1):.1f}%) "
          f"in {time.time()-t0:.1f}s",
          flush=True)

    # ---- Phase 5: write-back ----
    t0 = time.time()
    n_ents_seen = 0
    n_ents_linked = 0
    n_ents_with_tui = 0
    for p in paths:
        d = json.loads(Path(p).read_text(encoding="utf-8"))
        for sec in d.get("sections", []):
            for e in sec.get("entities") or []:
                t = (e.get("expanded_text") or "").strip().lower()
                n_ents_seen += 1
                if not t:
                    # stamp empty hash/fields for consistency
                    e["entity_hash"] = ""
                    e["cui"] = ""
                    e["cui_name"] = ""
                    e["cui_match"] = ""
                    e["tuis"] = []
                    e["tui_names"] = []
                    continue
                h = entity_hash(t)
                e["entity_hash"] = h
                hit = resolved.get(h)
                if hit:
                    cui, name, src = hit
                    e["cui"] = cui
                    e["cui_name"] = name
                    e["cui_match"] = src
                    tuis, tui_names = cui_to_tuis.get(cui, ([], []))
                    e["tuis"] = tuis
                    e["tui_names"] = tui_names
                    n_ents_linked += 1
                    if tuis:
                        n_ents_with_tui += 1
                else:
                    e["cui"] = ""
                    e["cui_name"] = ""
                    e["cui_match"] = ""
                    e["tuis"] = []
                    e["tui_names"] = []
        Path(p).write_text(json.dumps(d, ensure_ascii=False), encoding="utf-8")

    print(
        f"wrote back: {n_ents_linked:,}/{n_ents_seen:,} entities linked "
        f"({100*n_ents_linked/max(n_ents_seen,1):.1f}%); "
        f"{n_ents_with_tui:,} have >=1 TUI "
        f"in {time.time()-t0:.1f}s",
        flush=True,
    )


if __name__ == "__main__":
    main()
