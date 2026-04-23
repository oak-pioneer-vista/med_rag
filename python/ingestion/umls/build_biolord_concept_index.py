"""Embed every UMLS `Concept.name` via BioLORD-2023 (TEI) into Qdrant.

Produces the `umls_concepts_biolord` Qdrant collection used by
`link_entities_to_cui_biolord.py` as a dense-retrieval alternative to
the lexical linker in step 11. One point per Concept, 768-d, cosine.

Payload:
  cui  : UMLS CUI
  name : the Concept.name that was embedded

Point id: uuid5(cui) for idempotent re-runs.

Design:
- Keyset-paginate `:Concept` on `cui ASC` in `--page-size` chunks so we
  never hold the whole ~3.3M-row result set in a single Neo4j cursor.
- Per chunk: shard across `--workers` mp.Pool workers (each owns one
  HTTP session to biolord TEI + one Qdrant client), embed in batches,
  upsert to Qdrant.
- Persist the last successfully-upserted CUI to a state file after
  each chunk. `--resume` continues from that CUI so a long run can be
  interrupted and restarted cheaply.

Prereqs:
  docker compose up -d biolord qdrant neo4j
  UMLS loaded (steps 2-3).

Usage:
  python python/ingestion/umls/build_biolord_concept_index.py \
      [--workers 8] [--batch 128] [--page-size 20000] [--resume] [--recreate]
"""

from __future__ import annotations

import argparse
import multiprocessing as mp
import os
import time
import uuid
from pathlib import Path

import numpy as np
import requests
from neo4j import GraphDatabase
from qdrant_client import QdrantClient
from qdrant_client.http.models import (
    Distance,
    OptimizersConfigDiff,
    PayloadSchemaType,
    PointStruct,
    VectorParams,
)

REPO = Path(__file__).resolve().parent.parent.parent.parent
STATE_FILE = REPO / "data" / "biolord_concept_index.state"

BIOLORD_URL = os.environ.get("BIOLORD_URL", "http://localhost:8081")
QDRANT_URL = os.environ.get("QDRANT_URL", "http://localhost:6333")
QDRANT_GRPC_PORT = int(os.environ.get("QDRANT_GRPC_PORT", "6334"))
NEO4J_URI = os.environ.get("NEO4J_URI", "bolt://localhost:7687")
NEO4J_USER = os.environ.get("NEO4J_USER", "neo4j")
NEO4J_PASSWORD = os.environ.get("NEO4J_PASSWORD", "medragpass")

COLLECTION = "umls_concepts_biolord"
_PID_NS = uuid.UUID("b1010d55-2023-4e1f-9c3b-9a5b7e1f2a44")


def _point_id(cui: str) -> str:
    return str(uuid.uuid5(_PID_NS, cui))


def _tei_dim(url: str) -> int:
    r = requests.post(f"{url}/embed",
                      json={"inputs": ["dim probe"], "truncate": True},
                      timeout=60)
    r.raise_for_status()
    return len(r.json()[0])


def _ensure_collection(client: QdrantClient, dim: int, recreate: bool) -> None:
    if recreate and client.collection_exists(COLLECTION):
        client.delete_collection(COLLECTION)
        print(f"dropped existing collection {COLLECTION!r}", flush=True)
    if not client.collection_exists(COLLECTION):
        client.create_collection(
            collection_name=COLLECTION,
            vectors_config=VectorParams(size=dim, distance=Distance.COSINE),
            optimizers_config=OptimizersConfigDiff(indexing_threshold=100),
        )
        try:
            client.create_payload_index(
                collection_name=COLLECTION,
                field_name="cui",
                field_schema=PayloadSchemaType.KEYWORD,
            )
        except Exception:
            pass
        print(f"created collection {COLLECTION!r} dim={dim}", flush=True)


# ---------- mp.Pool worker state ----------

_SESSION: requests.Session | None = None
_QDRANT: QdrantClient | None = None
_TEI_URL: str = BIOLORD_URL
_WID: str | None = None


def _init_worker(tei_url: str, qdrant_url: str) -> None:
    global _SESSION, _QDRANT, _TEI_URL, _WID
    _SESSION = requests.Session()
    _QDRANT = QdrantClient(
        url=qdrant_url, prefer_grpc=True, grpc_port=QDRANT_GRPC_PORT, timeout=300,
    )
    _TEI_URL = tei_url
    _WID = f"{os.getpid()}/{uuid.uuid4().hex[:6]}"
    print(f"[worker {_WID}] biolord+qdrant sessions ready", flush=True)


def _embed_batch(texts: list[str], max_retries: int = 30, max_delay: float = 8.0) -> np.ndarray:
    """POST to TEI /embed; ride out HTTP 429 with capped exponential backoff.

    TEI's batching queue can saturate when many workers push concurrently
    (especially early in a run before the dynamic batcher settles). 429 is
    transient by design, so we retry up to `max_retries` times with delay
    capped at `max_delay` — total worst-case wait ~max_retries * max_delay.
    One worker's exception kills the whole mp.Pool.map, so being generous
    on the retry side is much cheaper than restarting a partial run.
    """
    delay = 0.5
    last_r = None
    for _ in range(max_retries):
        last_r = _SESSION.post(
            f"{_TEI_URL}/embed",
            json={"inputs": texts, "truncate": True},
            timeout=600,
        )
        if last_r.status_code == 429:
            time.sleep(delay)
            delay = min(delay * 2, max_delay)
            continue
        last_r.raise_for_status()
        return np.asarray(last_r.json(), dtype=np.float32)
    last_r.raise_for_status()
    return np.asarray(last_r.json(), dtype=np.float32)


def _process_shard(args: tuple) -> int:
    """Length-sort the shard descending before batching so each client
    batch bundles items of similar length -- minimizes padding-to-longest
    cost in TEI's forward pass and keeps batch sizes honest when a few
    very long Concept names would otherwise inflate a mixed batch's
    wall time. Same idea as step 7's within-worker sort before Stanza.
    """
    shard, batch_size = args
    if not shard:
        return 0
    shard = sorted(shard, key=lambda it: -len(it[1]))
    total = 0
    for bi in range(0, len(shard), batch_size):
        chunk = shard[bi : bi + batch_size]
        texts = [n for _, n in chunk]
        vecs = _embed_batch(texts)
        points = [
            PointStruct(
                id=_point_id(cui),
                vector=vec.tolist(),
                payload={"cui": cui, "name": name},
            )
            for (cui, name), vec in zip(chunk, vecs)
        ]
        _QDRANT.upsert(collection_name=COLLECTION, points=points, wait=False)
        total += len(points)
    return total


def _shard(seq: list[tuple[str, str]], n: int) -> list[list[tuple[str, str]]]:
    """Longest-Processing-Time-first bin pack on name char-length.

    Sorts items by name length descending, then greedily assigns each to
    the currently shortest bin. Same pattern as `extract_section_entities`
    -- keeps heaviest shards within ~4/3 of the mean so the slowest worker
    doesn't stretch wall time when the fixed page contains a tail of
    very long Concept names (drug protocols, long UMLS descriptors).
    """
    bins: list[list[tuple[str, str]]] = [[] for _ in range(n)]
    bin_len = [0] * n
    for item in sorted(seq, key=lambda it: -len(it[1])):
        i = min(range(n), key=lambda k: bin_len[k])
        bins[i].append(item)
        bin_len[i] += len(item[1])
    return bins


def _fetch_page(driver, after_cui: str, page_size: int) -> list[tuple[str, str]]:
    with driver.session() as session:
        rows = session.run(
            "MATCH (c:Concept) "
            "WHERE c.cui > $last AND c.name IS NOT NULL AND c.name <> '' "
            "RETURN c.cui AS cui, c.name AS name "
            "ORDER BY c.cui LIMIT $lim",
            last=after_cui, lim=page_size,
        ).data()
    return [(r["cui"], r["name"]) for r in rows]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--workers", type=int, default=16,
                    help="mp.Pool worker count; each owns an HTTP session "
                         "to biolord TEI and a Qdrant client")
    ap.add_argument("--batch", type=int, default=32,
                    help="names per TEI /embed request per worker. Must stay "
                         "<= biolord TEI's --max-client-batch-size (64 in "
                         "docker-compose). 16 workers x batch 32 matches the "
                         "sweet spot tuned for embed_sentences.py: below "
                         "TEI's concurrency budget (no HTTP 429) while still "
                         "saturating the GPU.")
    ap.add_argument("--page-size", type=int, default=20000,
                    help="neo4j keyset page size; one dispatch of "
                         "workers x batch-sized requests per page")
    ap.add_argument("--resume", action="store_true",
                    help="resume after the last CUI recorded in the state file")
    ap.add_argument("--recreate", action="store_true",
                    help="drop and recreate the Qdrant collection")
    ap.add_argument("--biolord-url", default=BIOLORD_URL)
    ap.add_argument("--qdrant-url", default=QDRANT_URL)
    args = ap.parse_args()

    print(f"probing biolord TEI at {args.biolord_url}...", flush=True)
    dim = _tei_dim(args.biolord_url)
    print(f"  biolord vector dim = {dim}", flush=True)

    qc = QdrantClient(
        url=args.qdrant_url, prefer_grpc=True, grpc_port=QDRANT_GRPC_PORT, timeout=300,
    )
    _ensure_collection(qc, dim, args.recreate)

    after_cui = ""
    if args.resume and STATE_FILE.exists():
        after_cui = STATE_FILE.read_text().strip()
        print(f"resuming after CUI {after_cui!r}", flush=True)

    driver = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASSWORD))

    ctx = mp.get_context("spawn")
    pool = ctx.Pool(
        processes=args.workers,
        initializer=_init_worker,
        initargs=(args.biolord_url, args.qdrant_url),
    )

    t0 = time.time()
    done = 0
    try:
        while True:
            page = _fetch_page(driver, after_cui, args.page_size)
            if not page:
                break
            shards = _shard(page, args.workers)
            shard_counts = pool.map(
                _process_shard,
                [(s, args.batch) for s in shards],
            )
            done += sum(shard_counts)
            after_cui = page[-1][0]
            STATE_FILE.parent.mkdir(parents=True, exist_ok=True)
            STATE_FILE.write_text(after_cui)
            rate = done / max(time.time() - t0, 1e-9)
            print(f"  page done: cumulative {done:,} concepts  "
                  f"({rate:.0f}/s; last cui {after_cui})",
                  flush=True)
    finally:
        pool.close()
        pool.join()
        driver.close()

    wall = time.time() - t0
    info = qc.get_collection(COLLECTION)
    print(
        f"done: {done:,} concepts embedded + upserted in {wall:.1f}s "
        f"({done / max(wall, 1e-9):.0f}/s). "
        f"qdrant {COLLECTION} state: {info.points_count} points, "
        f"status={info.status}",
        flush=True,
    )


if __name__ == "__main__":
    main()
