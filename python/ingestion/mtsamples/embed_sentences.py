"""Embed every sentence-level chunk via MedTE/TEI and upsert to Qdrant.

Reads data/mtsamples_docs/*.json (sentences must already be chunked by
chunk_sentences.py so each sentence has a stable chunk_id) and pushes
one Qdrant point per sentence to the `mtsamples_sentences` collection.
Each point's payload carries the full provenance chain so filters like
"sentences in General Medicine docs in the HPI section that mention
CUI C0001175" resolve index-only.

Parallelism is mp.Pool with spawn; each worker owns its own HTTP
session (to TEI) and its own Qdrant client. TEI does dynamic server-side
batching across concurrent requests, so more clients -> more work in
flight on the GPU even when per-request batch is modest. Per-request
batch is capped by TEI's `--max-client-batch-size` (64 in the current
compose), so the sweep ranges over 16..64.

Qdrant collection:
  name           : mtsamples_sentences
  vector_size    : discovered from the first TEI response (768 for MedTE)
  distance       : cosine (MedTE = mean-pooled + L2-normalized)
  indexing_threshold: 100  (match embed_sections.py -- prevents the
                           default ~20k-per-segment trap where queries
                           fall back to brute force)

Point payload (all fields indexable):
  chunk_id             : sentence-level id, e.g. "1003:procedure:s5"
  section_chunk_id     : "{doc_id}:{slug}"
  doc_id               : int
  section_type         : str
  section_cui          : str
  specialty            : str
  specialty_cui        : str
  alt_specialty_cuis   : list[str]
  doctype_cui          : str
  cuis                 : list[str]      -- linked CUIs in the sentence
  tuis                 : list[str]      -- semantic TUIs
  surface_forms        : list[str]      -- matched entity surfaces (lowercased)
  text                 : sentence text (for rehydration without a separate doc lookup)

Prereqs:
  - docker compose up -d medte qdrant
  - data/mtsamples_docs/*.json with sentences from chunk_sentences.py

Usage:
  python python/ingestion/mtsamples/embed_sentences.py [--workers 16] [--batch 32]
"""

from __future__ import annotations

import argparse
import json
import multiprocessing as mp
import os
import time
import uuid
from pathlib import Path

import numpy as np
import requests
from qdrant_client import QdrantClient
from qdrant_client.http.models import (
    Distance,
    OptimizersConfigDiff,
    PayloadSchemaType,
    PointStruct,
    VectorParams,
)

REPO = Path(__file__).resolve().parent.parent.parent.parent
DOCS_DIR = REPO / "data" / "mtsamples_docs"

TEI_URL = os.environ.get("TEI_URL", "http://localhost:8080")
QDRANT_URL = os.environ.get("QDRANT_URL", "http://localhost:6333")
COLLECTION = "mtsamples_sentences"

# Stable-ids namespace so chunk_id -> point id is deterministic across runs.
_PID_NS = uuid.UUID("f2b3e841-3c5e-4e1f-a5c3-5a9a2b0e7f9d")


def _point_id(chunk_id: str) -> str:
    return str(uuid.uuid5(_PID_NS, chunk_id))


def _tei_dim(tei_url: str) -> int:
    """Probe TEI for vector dim using a trivial /embed call."""
    r = requests.post(
        f"{tei_url}/embed", json={"inputs": ["dim probe"], "truncate": True},
        timeout=30,
    )
    r.raise_for_status()
    vec = r.json()[0]
    return len(vec)


def collect_sentences(docs_dir: Path) -> list[dict]:
    """Flatten per-doc JSONs into one list of sentence payload dicts."""
    out: list[dict] = []
    for p in sorted(docs_dir.glob("*.json")):
        d = json.loads(p.read_text(encoding="utf-8"))
        doc_id = int(d.get("doc_id", 0))
        specialty = (d.get("specialty") or "").strip()
        specialty_cui = d.get("specialty_cui") or ""
        doctype_cui = d.get("doctype_cui") or ""
        alt_cuis = [
            (a.get("specialty_cui") or "").strip()
            for a in (d.get("alt_specialties") or [])
            if (a.get("specialty_cui") or "").strip()
        ]
        for sec in d.get("sections", []):
            sec_chunk = (sec.get("chunk_id") or "").strip()
            if not sec_chunk:
                continue
            section_type = sec.get("section_type", "") or ""
            section_cui = sec.get("section_cui") or ""
            for sent in sec.get("sentences") or []:
                text = (sent.get("text") or "").strip()
                sent_chunk = (sent.get("chunk_id") or "").strip()
                if not text or not sent_chunk:
                    continue
                out.append({
                    "chunk_id": sent_chunk,
                    "section_chunk_id": sec_chunk,
                    "doc_id": doc_id,
                    "section_type": section_type,
                    "section_cui": section_cui,
                    "specialty": specialty,
                    "specialty_cui": specialty_cui,
                    "alt_specialty_cuis": alt_cuis,
                    "doctype_cui": doctype_cui,
                    "cuis": list(sent.get("cuis") or []),
                    "tuis": list(sent.get("tuis") or []),
                    "surface_forms": list(sent.get("surface_forms") or []),
                    "text": text,
                })
    return out


def ensure_collection(client: QdrantClient, dim: int) -> None:
    """Create the collection if it doesn't exist; idempotent."""
    if client.collection_exists(COLLECTION):
        return
    client.create_collection(
        collection_name=COLLECTION,
        vectors_config=VectorParams(size=dim, distance=Distance.COSINE),
        optimizers_config=OptimizersConfigDiff(indexing_threshold=100),
    )
    print(f"created collection {COLLECTION!r} dim={dim}", flush=True)


def create_payload_indexes(client: QdrantClient) -> None:
    """Index the common filter fields so they don't scan at query time."""
    specs: list[tuple[str, PayloadSchemaType]] = [
        ("chunk_id", PayloadSchemaType.KEYWORD),
        ("section_chunk_id", PayloadSchemaType.KEYWORD),
        ("doc_id", PayloadSchemaType.INTEGER),
        ("section_type", PayloadSchemaType.KEYWORD),
        ("section_cui", PayloadSchemaType.KEYWORD),
        ("specialty", PayloadSchemaType.KEYWORD),
        ("specialty_cui", PayloadSchemaType.KEYWORD),
        ("alt_specialty_cuis", PayloadSchemaType.KEYWORD),
        ("doctype_cui", PayloadSchemaType.KEYWORD),
        ("cuis", PayloadSchemaType.KEYWORD),
        ("tuis", PayloadSchemaType.KEYWORD),
    ]
    for field, schema in specs:
        try:
            client.create_payload_index(
                collection_name=COLLECTION,
                field_name=field,
                field_schema=schema,
            )
        except Exception:
            pass  # Already exists -- Qdrant raises on duplicate create.


# ---------- mp.Pool worker state ----------

_SESSION: requests.Session | None = None
_QDRANT: QdrantClient | None = None
_TEI_URL: str = TEI_URL
_WID: str | None = None


def _init_worker(tei_url: str, qdrant_url: str) -> None:
    global _SESSION, _QDRANT, _TEI_URL, _WID
    _SESSION = requests.Session()
    _QDRANT = QdrantClient(url=qdrant_url, timeout=120)
    _TEI_URL = tei_url
    _WID = f"{os.getpid()}/{uuid.uuid4().hex[:6]}"
    print(f"[worker {_WID}] http+qdrant sessions ready", flush=True)


def _embed_batch(texts: list[str], max_retries: int = 5) -> np.ndarray:
    """POST to TEI /embed with exponential backoff on 429.

    At batch=64 with 16 workers TEI's queue saturates; at batch=32 with
    16 workers it runs clean. We still retry on 429 so a transient burst
    (e.g. one shard finishes slower and the others back up) doesn't kill
    the whole job.
    """
    delay = 0.25
    for attempt in range(max_retries):
        r = _SESSION.post(
            f"{_TEI_URL}/embed",
            json={"inputs": texts, "truncate": True},
            timeout=300,
        )
        if r.status_code == 429:
            time.sleep(delay)
            delay *= 2
            continue
        r.raise_for_status()
        return np.asarray(r.json(), dtype=np.float32)
    # Final attempt raises if still failing.
    r.raise_for_status()
    return np.asarray(r.json(), dtype=np.float32)


def _process_shard(args: tuple) -> dict:
    records, batch_size = args
    if not records:
        return {"wid": _WID, "sentences": 0, "elapsed_s": 0.0}
    t0 = time.time()
    total = 0
    for bi in range(0, len(records), batch_size):
        chunk = records[bi : bi + batch_size]
        texts = [r["text"] for r in chunk]
        vecs = _embed_batch(texts)
        points = []
        for rec, vec in zip(chunk, vecs):
            payload = {k: v for k, v in rec.items() if k != "text"}
            payload["text"] = rec["text"]
            points.append(PointStruct(
                id=_point_id(rec["chunk_id"]),
                vector=vec.tolist(),
                payload=payload,
            ))
        _QDRANT.upsert(collection_name=COLLECTION, points=points, wait=False)
        total += len(points)
        if total % (batch_size * 20) == 0:
            rate = total / max(time.time() - t0, 1e-9)
            print(f"[worker {_WID}]   {total:>6}/{len(records)} upserted  "
                  f"({rate:.0f}/s)",
                  flush=True)
    elapsed = time.time() - t0
    print(f"[worker {_WID}] done: {total} sentences in {elapsed:.1f}s "
          f"({total/max(elapsed,1e-9):.0f}/s)",
          flush=True)
    return {"wid": _WID, "sentences": total, "elapsed_s": elapsed}


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
                    help="mp.Pool worker count (TEI-bound; each worker owns "
                         "an HTTP session and a Qdrant client)")
    ap.add_argument("--batch", type=int, default=32,
                    help="sentences per TEI /embed request. Tuned via sweep: "
                         "32 wins at 16 workers (15.5s / 5380 sentences/s). "
                         "batch=64 with 16 workers saturates TEI's concurrent-"
                         "request queue (HTTP 429), and batch=16 pays extra "
                         "per-request overhead. If you change --workers, "
                         "re-tune this: roughly, workers * batch should stay "
                         "under TEI's effective concurrency budget.")
    ap.add_argument("--recreate", action="store_true",
                    help="drop the collection before embedding (full re-run)")
    ap.add_argument("--tei-url", default=TEI_URL)
    ap.add_argument("--qdrant-url", default=QDRANT_URL)
    args = ap.parse_args()

    print(f"probing TEI at {args.tei_url}...", flush=True)
    dim = _tei_dim(args.tei_url)
    print(f"  TEI vector dim = {dim}", flush=True)

    client = QdrantClient(url=args.qdrant_url, timeout=120)
    if args.recreate and client.collection_exists(COLLECTION):
        client.delete_collection(COLLECTION)
        print(f"dropped existing collection {COLLECTION!r}", flush=True)
    ensure_collection(client, dim)
    create_payload_indexes(client)

    print(f"collecting sentences from {args.docs}...", flush=True)
    records = collect_sentences(args.docs)
    print(f"  {len(records):,} non-empty sentences", flush=True)
    if not records:
        raise SystemExit("no sentences to embed -- run chunk_sentences.py first")

    workers = max(1, min(args.workers, len(records)))
    shards = _shard(records, workers)
    sizes = [len(s) for s in shards]
    print(f"dispatching {len(records):,} sentences across {workers} workers "
          f"(shards={sizes}, batch={args.batch})",
          flush=True)

    ctx = mp.get_context("spawn")
    wall_t0 = time.time()
    with ctx.Pool(processes=workers, initializer=_init_worker,
                  initargs=(args.tei_url, args.qdrant_url)) as pool:
        results = pool.map(
            _process_shard,
            [(shard, args.batch) for shard in shards],
        )

    total = sum(r["sentences"] for r in results)
    per_worker = sorted(r.get("elapsed_s", 0.0) for r in results)
    wall = time.time() - wall_t0
    print(
        f"done: {total:,} sentences embedded + upserted in {wall:.1f}s wall  "
        f"[per-worker {min(per_worker):.1f}..{max(per_worker):.1f}s, "
        f"aggregate {total/max(wall,1e-9):.0f}/s]",
        flush=True,
    )

    # Confirm Qdrant point count (may lag the upserts by a moment; force=true
    # bypasses any per-segment optimization state).
    info = client.get_collection(COLLECTION)
    print(f"qdrant {COLLECTION} state: {info.points_count} points, "
          f"status={info.status}",
          flush=True)


if __name__ == "__main__":
    main()
