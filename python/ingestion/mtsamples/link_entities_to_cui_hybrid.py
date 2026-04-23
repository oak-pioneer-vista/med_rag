"""Hybrid entity -> CUI linker: exact Atom.str_norm + BioLORD NN fallback.

Production-shape upgrade to step 11's lexical path, informed by the
step-12 vs step-18 diff:

  * Phase A -- exact Atom.str_norm via Neo4j. Strictly the best path
    on abbreviations and short polysemous surfaces (e.g. `tah/bso`,
    `djd`, `ca`, `hjr`, `stylet`) because BioLORD's embedding space
    doesn't co-locate abbreviations with their expansions.
  * Phase B -- BioLORD-2023 nearest-neighbor via `umls_concepts_biolord`
    on the residual ~38K hashes. Beats Lucene fulltext on descriptive
    phrases whose token overlap misleads BM25 (e.g. "copious irrigation"
    -> `Therapeutic Irrigation`, "subpectoral pocket" -> pectoral region
    instead of periodontal pocket).
  * Phase C -- TUI enrichment via Neo4j, same as step 11 phase 4.

Output: `data/entity_cui_hybrid.jsonl`, one JSONL line per unique
`expanded_text`:

    {"text": ..., "cui": ..., "cui_name": ..., "cui_match": "exact"|"biolord/<score>"|"",
     "score": <1.0 for exact, cosine for biolord, 0.0 for unresolved>,
     "tuis": [...], "tui_names": [...]}

The `text` + `cui` columns diff directly against
`data/entity_cui_lexical.jsonl` (step 12) and
`data/entity_cui_biolord.jsonl` (step 17), so the three-way delta
between linkers is one join away.

Snapshot-only by design -- does NOT write back into per-doc JSONs.
Once the hybrid snapshot is audited against the lexical baseline,
the same resolution table can be applied in a subsequent write-back
pass to upgrade step 11's per-section `entity.cui` values in-place.

Prereqs:
  - step 16 (`build_biolord_concept_index.py`) populated
    `umls_concepts_biolord`.
  - `data/mtsamples_docs/*.json` with entities from steps 7-8.
  - `docker compose up -d biolord qdrant neo4j`.

Usage:
  python python/ingestion/mtsamples/link_entities_to_cui_hybrid.py \
      [--workers 8] [--batch 64] [--min-score 0.7]
"""

from __future__ import annotations

import argparse
import hashlib
import json
import multiprocessing as mp
import os
import time
import uuid
from pathlib import Path

import numpy as np
import requests
from neo4j import GraphDatabase
from qdrant_client import QdrantClient
from qdrant_client.models import QueryRequest

REPO = Path(__file__).resolve().parent.parent.parent.parent
DOCS_DIR = REPO / "data" / "mtsamples_docs"
OUT_PATH = REPO / "data" / "entity_cui_hybrid.jsonl"

BIOLORD_URL = os.environ.get("BIOLORD_URL", "http://localhost:8081")
QDRANT_URL = os.environ.get("QDRANT_URL", "http://localhost:6333")
QDRANT_GRPC_PORT = int(os.environ.get("QDRANT_GRPC_PORT", "6334"))
NEO4J_URI = os.environ.get("NEO4J_URI", "bolt://localhost:7687")
NEO4J_USER = os.environ.get("NEO4J_USER", "neo4j")
NEO4J_PASSWORD = os.environ.get("NEO4J_PASSWORD", "medragpass")

COLLECTION = "umls_concepts_biolord"
EXACT_BATCH = 2000
TUI_BATCH = 2000


def entity_hash(text: str) -> str:
    return hashlib.sha1(text.encode("utf-8")).hexdigest()[:16]


def collect_unique_entities(docs_dir: Path) -> dict[str, str]:
    """{sha1-hash: expanded_text_lower} across all per-doc JSONs."""
    unique: dict[str, str] = {}
    for p in sorted(docs_dir.glob("*.json")):
        try:
            d = json.loads(p.read_text(encoding="utf-8"))
        except Exception:
            continue
        for sec in d.get("sections", []):
            for e in sec.get("entities") or []:
                t = (e.get("expanded_text") or "").strip().lower()
                if not t:
                    continue
                unique.setdefault(entity_hash(t), t)
    return unique


# ---------- Phase A: exact Atom.str_norm ----------

def batch_exact(session, items: list[dict]) -> dict[str, tuple[str, str]]:
    """items: [{'hash': h, 'text': t_lower}, ...]. Returns {hash: (cui, name)}."""
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


def batch_tuis(session, cuis: list[str]) -> dict[str, tuple[list[str], list[str]]]:
    """{cui: (tuis, tui_names)} for each CUI with HAS_SEMTYPE edges."""
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


# ---------- Phase B: BioLORD NN (mp.Pool worker) ----------

_SESSION: requests.Session | None = None
_QDRANT: QdrantClient | None = None
_BIOLORD_URL: str = BIOLORD_URL
_WID: str | None = None


def _init_worker(biolord_url: str, qdrant_url: str, qdrant_grpc_port: int) -> None:
    """Each worker: one biolord HTTP session + one Qdrant gRPC channel.
    gRPC (HTTP/2 persistent) scales far better than REST when workers
    concurrently issue query_batch_points."""
    global _SESSION, _QDRANT, _BIOLORD_URL, _WID
    _SESSION = requests.Session()
    _QDRANT = QdrantClient(
        url=qdrant_url, prefer_grpc=True, grpc_port=qdrant_grpc_port, timeout=300,
    )
    _BIOLORD_URL = biolord_url
    _WID = f"{os.getpid()}/{uuid.uuid4().hex[:6]}"
    print(f"[worker {_WID}] biolord+qdrant ready", flush=True)


def _embed_batch(texts: list[str], max_retries: int = 30, max_delay: float = 8.0) -> np.ndarray:
    """POST to biolord /embed with capped exp-backoff on 429."""
    delay = 0.5
    last_r = None
    for _ in range(max_retries):
        last_r = _SESSION.post(
            f"{_BIOLORD_URL}/embed",
            json={"inputs": texts, "truncate": True},
            timeout=300,
        )
        if last_r.status_code == 429:
            time.sleep(delay)
            delay = min(delay * 2, max_delay)
            continue
        last_r.raise_for_status()
        return np.asarray(last_r.json(), dtype=np.float32)
    last_r.raise_for_status()
    return np.asarray(last_r.json(), dtype=np.float32)


def _biolord_shard(args: tuple) -> list[tuple[str, str, str, float]]:
    """Process one shard: embed in batches, batch-query Qdrant top-1.
    Returns [(hash, cui, name, score), ...]; cui='' if score < min_score."""
    shard, batch_size, min_score = args
    if not shard:
        return []
    t0 = time.time()
    out: list[tuple[str, str, str, float]] = []
    for bi in range(0, len(shard), batch_size):
        chunk = shard[bi : bi + batch_size]
        texts = [t for _, t in chunk]
        vecs = _embed_batch(texts)
        reqs = [
            QueryRequest(query=v.tolist(), limit=1, with_payload=True)
            for v in vecs
        ]
        responses = _QDRANT.query_batch_points(
            collection_name=COLLECTION, requests=reqs,
        )
        for (h, _text), resp in zip(chunk, responses):
            hits = resp.points
            if hits and hits[0].score >= min_score:
                top = hits[0]
                out.append((
                    h,
                    top.payload.get("cui", "") or "",
                    top.payload.get("name", "") or "",
                    float(top.score),
                ))
            else:
                out.append((
                    h, "", "",
                    float(hits[0].score) if hits else 0.0,
                ))
    print(f"[worker {_WID}] done: {len(shard)} entities in "
          f"{time.time() - t0:.1f}s",
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
    ap.add_argument("--out", type=Path, default=OUT_PATH)
    ap.add_argument("--workers", type=int, default=8,
                    help="mp.Pool worker count for the BioLORD phase")
    ap.add_argument("--batch", type=int, default=64,
                    help="entities per biolord /embed request per worker")
    ap.add_argument("--min-score", type=float, default=0.7,
                    help="BioLORD cosine cutoff for accepting a semantic match")
    ap.add_argument("--no-tuis", action="store_true",
                    help="skip the TUI enrichment pass")
    ap.add_argument("--biolord-url", default=BIOLORD_URL)
    ap.add_argument("--qdrant-url", default=QDRANT_URL)
    ap.add_argument("--qdrant-grpc-port", type=int, default=QDRANT_GRPC_PORT)
    args = ap.parse_args()

    # ---- dedup ----
    print("collecting unique entities...", flush=True)
    t0 = time.time()
    unique = collect_unique_entities(args.docs)
    print(f"  {len(unique):,} unique entities in {time.time() - t0:.1f}s",
          flush=True)
    if not unique:
        raise SystemExit("no entities found; run steps 7-8 first")

    # hash -> (cui, name, match_kind, score)
    resolved: dict[str, tuple[str, str, str, float]] = {}

    # ---- Phase A: exact Atom.str_norm ----
    exact_items = [{"hash": h, "text": t} for h, t in unique.items()]
    print(f"phase A: exact Atom.str_norm pass on {len(exact_items):,} hashes...",
          flush=True)
    t0 = time.time()
    with GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASSWORD)) as driver:
        with driver.session() as session:
            for i in range(0, len(exact_items), EXACT_BATCH):
                chunk = exact_items[i : i + EXACT_BATCH]
                for h, (cui, name) in batch_exact(session, chunk).items():
                    resolved[h] = (cui, name, "exact", 1.0)
    n_exact = len(resolved)
    print(f"  -> {n_exact:,}/{len(exact_items):,} exact hits "
          f"({100 * n_exact / max(len(exact_items), 1):.1f}%) "
          f"in {time.time() - t0:.1f}s",
          flush=True)

    # ---- Phase B: BioLORD NN fallback on residual ----
    residual = [(h, t) for h, t in unique.items() if h not in resolved]
    print(f"phase B: BioLORD NN on residual {len(residual):,} hashes "
          f"({args.workers} workers x batch {args.batch}, "
          f"min-score {args.min_score})...",
          flush=True)
    n_bio = 0
    if residual:
        shards = _shard(residual, min(args.workers, len(residual)))
        t0 = time.time()
        ctx = mp.get_context("spawn")
        with ctx.Pool(
            processes=len(shards),
            initializer=_init_worker,
            initargs=(args.biolord_url, args.qdrant_url, args.qdrant_grpc_port),
        ) as pool:
            shard_results = pool.map(
                _biolord_shard,
                [(s, args.batch, args.min_score) for s in shards],
            )
        for sr in shard_results:
            for h, cui, name, score in sr:
                if cui:
                    resolved[h] = (cui, name, f"biolord/{score:.3f}", score)
                    n_bio += 1
        print(f"  -> {n_bio:,}/{len(residual):,} BioLORD hits "
              f">= {args.min_score} in {time.time() - t0:.1f}s",
              flush=True)

    # ---- Phase C: TUI lookup ----
    cui_to_tuis: dict[str, tuple[list[str], list[str]]] = {}
    if not args.no_tuis:
        unique_cuis = sorted({v[0] for v in resolved.values() if v[0]})
        print(f"phase C: TUI lookup on {len(unique_cuis):,} unique CUIs...",
              flush=True)
        t0 = time.time()
        with GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASSWORD)) as driver:
            with driver.session() as session:
                for i in range(0, len(unique_cuis), TUI_BATCH):
                    cui_to_tuis.update(
                        batch_tuis(session, unique_cuis[i : i + TUI_BATCH])
                    )
        print(f"  -> TUIs for {len(cui_to_tuis):,}/{len(unique_cuis):,} CUIs "
              f"in {time.time() - t0:.1f}s",
              flush=True)

    # ---- Phase D: snapshot ----
    args.out.parent.mkdir(parents=True, exist_ok=True)
    items_sorted = sorted(unique.items(), key=lambda kv: kv[1])
    linked = 0
    t0 = time.time()
    with args.out.open("w", encoding="utf-8") as f:
        for h, text in items_sorted:
            rec = resolved.get(h)
            if rec:
                cui, name, match_kind, score = rec
                tuis, tui_names = cui_to_tuis.get(cui, ([], []))
                out_rec = {
                    "text": text,
                    "cui": cui,
                    "cui_name": name,
                    "cui_match": match_kind,
                    "score": round(score, 4),
                }
                if tuis:
                    out_rec["tuis"] = tuis
                    out_rec["tui_names"] = tui_names
                linked += 1
            else:
                out_rec = {
                    "text": text, "cui": "", "cui_name": "",
                    "cui_match": "", "score": 0.0,
                }
            f.write(json.dumps(out_rec, ensure_ascii=False) + "\n")

    print(
        f"wrote {len(unique):,} unique entities "
        f"({linked:,} linked: {n_exact:,} exact + {n_bio:,} biolord; "
        f"{len(unique) - linked:,} unlinked) "
        f"to {args.out.relative_to(REPO)} in {time.time() - t0:.1f}s",
        flush=True,
    )


if __name__ == "__main__":
    main()
