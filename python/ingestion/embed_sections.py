"""Embed MTSamples Section text via TEI and upsert to Qdrant (dask parallel).

Reads the per-doc JSON files written by parse_mtsamples.py, packs each
Section into overlapping sentence windows, embeds them by calling the
HuggingFace text-embeddings-inference (TEI) service on :8080, and
upserts the vectors to a Qdrant collection. Parallelism is via
`dask.bag.map_partitions`: each worker owns an HTTP session and a local
tokenizer (used only for token counting during packing -- TEI handles
the actual encoding).

Usage:
  docker compose up -d qdrant medte
  python python/ingestion/embed_sections.py [--workers 16] [--batch 64]
"""

from __future__ import annotations

import argparse
import json
import re
import uuid
from pathlib import Path
from typing import Iterable

import dask.bag as db
import numpy as np
import requests
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, PointStruct, VectorParams
from transformers import AutoTokenizer

REPO = Path(__file__).resolve().parent.parent.parent
DOCS_DIR = REPO / "data" / "mtsamples_docs"

MODEL_ID = "MohammadKhodadad/MedTE-cl15-step-8000"
COLLECTION = "mtsamples_sections"
QDRANT_URL = "http://localhost:6333"
TEI_URL = "http://localhost:8080"
# Deterministic uuid5 namespace so re-runs upsert into the same point id.
POINT_NS = uuid.UUID("6f3c0c2a-6c2c-4c9a-b9ea-0ea0d3f8f5a1")

# Sentence-aware windowing: pack whole sentences up to MAX_TOKENS (per the
# MedTE tokenizer, excluding special tokens) with OVERLAP_TOKENS of trailing
# sentences shared between neighbors. 10% overlap preserves cross-boundary
# context without duplicating too much. 200-token windows keep each chunk
# tight around a coherent sentence cluster.
MAX_TOKENS = 200
OVERLAP_FRAC = 0.10
OVERLAP_TOKENS = int(MAX_TOKENS * OVERLAP_FRAC)

# TEI's --max-client-batch-size in docker-compose.yml is 64; match it so a
# single POST fills one server-side batch.
TEI_BATCH = 64
QDRANT_BATCH = 256


# Lightweight sentence splitter: break on .!? followed by whitespace and an
# uppercase/digit/opening-bracket start. Good enough for MTSamples narrative;
# swap for nltk/syntok if it proves insufficient on real data.
_SENT_RE = re.compile(r"(?<=[.!?])\s+(?=[A-Za-z0-9(\[])")


def _split_sentences(text: str) -> list[str]:
    text = " ".join(text.split())
    if not text:
        return []
    return [s for s in (p.strip() for p in _SENT_RE.split(text)) if s]


def _pack_sentences(
    text: str,
    tokenizer,
    max_tokens: int = MAX_TOKENS,
    overlap: int = OVERLAP_TOKENS,
) -> list[str]:
    """Greedy sentence packing with token-budget windows and trailing overlap.

    Walks sentences left-to-right, adding each whole sentence to the current
    window until the next one would push past `max_tokens`. The next window
    starts a few sentences earlier so ~`overlap` tokens are shared. A single
    sentence longer than `max_tokens` forms its own window and gets truncated
    by TEI (auto_truncate=true).
    """
    sentences = _split_sentences(text)
    if not sentences:
        return []
    counts = [len(tokenizer.encode(s, add_special_tokens=False)) for s in sentences]

    windows: list[str] = []
    start = 0
    n = len(sentences)
    while start < n:
        cur = 0
        end = start
        while end < n and cur + counts[end] <= max_tokens:
            cur += counts[end]
            end += 1
        if end == start:
            end = start + 1  # oversized lone sentence; TEI will truncate
        windows.append(" ".join(sentences[start:end]))
        if end >= n:
            break
        back = 0
        new_start = end
        while new_start > start + 1 and back < overlap:
            new_start -= 1
            back += counts[new_start]
        start = new_start
    return windows


def _tei_embed(session: requests.Session, texts: list[str], dim: int) -> np.ndarray:
    """POST texts to TEI /embed in chunks of TEI_BATCH; return (N, dim) float32."""
    if not texts:
        return np.zeros((0, dim), dtype=np.float32)
    out = np.empty((len(texts), dim), dtype=np.float32)
    for i in range(0, len(texts), TEI_BATCH):
        batch = texts[i : i + TEI_BATCH]
        r = session.post(f"{TEI_URL}/embed", json={"inputs": batch}, timeout=300)
        r.raise_for_status()
        out[i : i + len(batch)] = np.asarray(r.json(), dtype=np.float32)
    return out


def _read_doc(path: str) -> dict:
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def _chunk_doc(doc: dict, tokenizer) -> tuple[list[str], list[str], list[dict]]:
    """Return (texts, point_ids, payloads) for one parsed doc.

    Each non-empty section is sentence-split and packed into overlapping
    token-budget windows; every window becomes its own Qdrant point with
    `chunk_id` suffixed `#<window_index>` when a section yields multiple.
    """
    texts, ids, payloads = [], [], []
    for s in doc["sections"]:
        text = s.get("text", "").strip()
        if not text:
            continue
        windows = _pack_sentences(text, tokenizer)
        for idx, win_text in enumerate(windows):
            suffix = f"#{idx}" if len(windows) > 1 else ""
            chunk_id = s["chunk_id"] + suffix
            texts.append(win_text)
            ids.append(str(uuid.uuid5(POINT_NS, chunk_id)))
            payloads.append({
                "chunk_id": chunk_id,
                "parent_chunk_id": s["chunk_id"],
                "window_index": idx,
                "window_count": len(windows),
                "doc_id": s["doc_id"],
                "section_type": s["section_type"],
                "section_cui": s["section_cui"],
                "specialty": s["specialty"],
                "specialty_cui": s["specialty_cui"],
                "alt_specialties": doc.get("alt_specialties", []),
                "doctype_cui": doc["doctype_cui"],
                "sample_name": doc["sample_name"],
                "keywords": s["keywords"],
                "text": win_text,
            })
    return texts, ids, payloads


def _tei_info() -> dict:
    r = requests.get(f"{TEI_URL}/info", timeout=30)
    r.raise_for_status()
    return r.json()


def _probe_dim(session: requests.Session) -> int:
    r = session.post(f"{TEI_URL}/embed", json={"inputs": ["probe"]}, timeout=60)
    r.raise_for_status()
    return len(r.json()[0])


def _process_partition(paths: Iterable[str]) -> list[tuple[int, int]]:
    """Dask worker: tokenize + pack + embed via TEI + upsert every section.

    Returns a single-element list [(n_docs, n_sections)] so the driver can
    aggregate across partitions.
    """
    import os, time
    wid = os.getpid()
    t0 = time.time()
    paths = list(paths)
    print(f"[worker {wid}] starting on {len(paths)} docs", flush=True)

    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    session = requests.Session()
    dim = _probe_dim(session)
    client = QdrantClient(url=QDRANT_URL, timeout=60)

    all_texts: list[str] = []
    all_ids: list[str] = []
    all_payloads: list[dict] = []
    for path in paths:
        texts, ids, payloads = _chunk_doc(_read_doc(path), tokenizer)
        all_texts.extend(texts)
        all_ids.extend(ids)
        all_payloads.extend(payloads)

    if not all_texts:
        print(f"[worker {wid}] no sections, done in {time.time()-t0:.1f}s", flush=True)
        return [(len(paths), 0)]

    t_embed = time.time()
    embs = _tei_embed(session, all_texts, dim)
    t_embed = time.time() - t_embed

    points = [
        PointStruct(id=pid, vector=emb.tolist(), payload=pl)
        for pid, emb, pl in zip(all_ids, embs, all_payloads)
    ]
    for i in range(0, len(points), QDRANT_BATCH):
        client.upsert(collection_name=COLLECTION, points=points[i : i + QDRANT_BATCH])

    print(
        f"[worker {wid}] embedded {len(points)} via TEI in {t_embed:.1f}s "
        f"(total {time.time()-t0:.1f}s)",
        flush=True,
    )
    return [(len(paths), len(points))]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--workers", type=int, default=16, help="dask worker processes")
    ap.add_argument(
        "--recreate", action="store_true",
        help="drop and recreate the Qdrant collection before writing",
    )
    args = ap.parse_args()

    info = _tei_info()
    print(f"tei model={info.get('model_id')}  pooling={info.get('model_type', {}).get('embedding', {}).get('pooling')}")
    with requests.Session() as s:
        dim = _probe_dim(s)
    print(f"embedding dim={dim}")

    client = QdrantClient(url=QDRANT_URL, timeout=60)
    if args.recreate and client.collection_exists(COLLECTION):
        client.delete_collection(COLLECTION)
    if not client.collection_exists(COLLECTION):
        client.create_collection(
            collection_name=COLLECTION,
            vectors_config=VectorParams(size=dim, distance=Distance.COSINE),
        )
        print(f"created collection: {COLLECTION}")
    else:
        print(f"reusing collection: {COLLECTION}")

    paths = sorted(str(p) for p in DOCS_DIR.glob("*.json"))
    print(f"dispatching {len(paths)} docs across {args.workers} dask partitions")

    bag = db.from_sequence(paths, npartitions=args.workers)
    results = bag.map_partitions(_process_partition).compute(
        scheduler="processes", num_workers=args.workers
    )

    n_docs = sum(r[0] for r in results)
    n_points = sum(r[1] for r in results)
    coll_info = client.get_collection(COLLECTION)
    print(f"processed {n_docs} docs, upserted {n_points} section embeddings")
    print(f"collection '{COLLECTION}' points_count={coll_info.points_count}")


if __name__ == "__main__":
    main()
