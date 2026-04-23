"""Link entities to UMLS CUIs via BioLORD-2023 nearest-neighbor.

Semantic counterpart to `link_entities_to_cui.py` (step 11). Same input
universe — the unique `expanded_text` values across
`data/mtsamples_docs/*.json` — but swaps the lexical resolver (exact
`Atom.str_norm` + Lucene fulltext on `concept_name_fts`) for
BioLORD-2023 nearest-neighbor search over the
`umls_concepts_biolord` Qdrant collection built by
`python/ingestion/umls/build_biolord_concept_index.py`.

Why: the lexical linker mis-resolves mentions whose token overlap with
an unrelated Concept outranks the true concept (e.g. "subpectoral
pocket" -> Periodontal Pocket because of the shared "pocket" token;
"copious irrigation" -> Copious because the adjective matches a
one-word finding concept exactly). BioLORD-2023 is fine-tuned on UMLS
synonym/definition contrastive pairs, so semantically-related concepts
cluster in the embedding space and lexically-similar-but-semantically-
distant ones do not.

Output: `data/entity_cui_biolord.jsonl`, one JSONL line per unique
`expanded_text`:

    {"text": "...", "cui": "...", "cui_name": "...", "score": 0.87}

The `text` + `cui` columns are directly diffable against
`data/entity_cui_lexical.jsonl` (step 12) on the shared `text` key.

Scope notes:
- v1 embeds each entity in isolation (same input signal as the lexical
  linker) so the diff is apples-to-apples. A follow-up can pass the
  enclosing sentence as context for more lift on polysemous mentions.
- v1 is snapshot-only — it does NOT write back into per-doc JSONs. Run
  the diff first; decide whether to rewire the main pipeline to
  BioLORD afterwards.
- TUIs are not fetched here. Derive from the resolved CUIs via the
  same `HAS_SEMTYPE` pass as step 11 phase 4 if needed.

Prereqs:
  docker compose up -d biolord qdrant
  build_biolord_concept_index.py already populated `umls_concepts_biolord`.
  data/mtsamples_docs/*.json with entities written by steps 7-8.

Usage:
  python python/ingestion/mtsamples/link_entities_to_cui_biolord.py \
      [--workers 8] [--batch 128] [--min-score 0.7]
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
from qdrant_client import QdrantClient
from qdrant_client.models import QueryRequest

REPO = Path(__file__).resolve().parent.parent.parent.parent
DOCS_DIR = REPO / "data" / "mtsamples_docs"
OUT_PATH = REPO / "data" / "entity_cui_biolord.jsonl"

BIOLORD_URL = os.environ.get("BIOLORD_URL", "http://localhost:8081")
QDRANT_URL = os.environ.get("QDRANT_URL", "http://localhost:6333")
QDRANT_GRPC_PORT = int(os.environ.get("QDRANT_GRPC_PORT", "6334"))
COLLECTION = "umls_concepts_biolord"


def _entity_hash(text: str) -> str:
    return hashlib.sha1(text.encode("utf-8")).hexdigest()[:16]


def collect_unique_entities(docs_dir: Path) -> dict[str, str]:
    """{hash: expanded_text_lower} across all per-doc JSONs."""
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
                unique.setdefault(_entity_hash(t), t)
    return unique


# ---------- mp.Pool worker state ----------

_SESSION: requests.Session | None = None
_QDRANT: QdrantClient | None = None
_TEI_URL: str = BIOLORD_URL
_WID: str | None = None


def _init_worker(tei_url: str, qdrant_url: str, qdrant_grpc_port: int) -> None:
    """Each worker owns its own biolord HTTP session + Qdrant gRPC
    channel. gRPC (HTTP/2, persistent connection) scales far better
    than REST when 8 workers concurrently issue query_batch_points --
    REST's per-query HTTP connections get reset under load."""
    global _SESSION, _QDRANT, _TEI_URL, _WID
    _SESSION = requests.Session()
    _QDRANT = QdrantClient(
        url=qdrant_url,
        prefer_grpc=True,
        grpc_port=qdrant_grpc_port,
        timeout=300,
    )
    _TEI_URL = tei_url
    _WID = f"{os.getpid()}/{uuid.uuid4().hex[:6]}"
    print(f"[worker {_WID}] biolord+qdrant sessions ready", flush=True)


def _embed_batch(texts: list[str], max_retries: int = 30, max_delay: float = 8.0) -> np.ndarray:
    """POST to TEI /embed with capped exp-backoff on 429. See
    build_biolord_concept_index._embed_batch for rationale."""
    delay = 0.5
    last_r = None
    for _ in range(max_retries):
        last_r = _SESSION.post(
            f"{_TEI_URL}/embed",
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


def _process_shard(args: tuple) -> list[tuple[str, str, str, str, float]]:
    shard, batch_size, min_score = args
    if not shard:
        return []
    t0 = time.time()
    out: list[tuple[str, str, str, str, float]] = []
    for bi in range(0, len(shard), batch_size):
        chunk = shard[bi : bi + batch_size]
        texts = [t for _, t in chunk]
        vecs = _embed_batch(texts)
        reqs = [
            QueryRequest(query=vec.tolist(), limit=1, with_payload=True)
            for vec in vecs
        ]
        responses = _QDRANT.query_batch_points(
            collection_name=COLLECTION, requests=reqs
        )
        for (h, text), resp in zip(chunk, responses):
            hits = resp.points
            if hits and hits[0].score >= min_score:
                top = hits[0]
                out.append((
                    h, text,
                    top.payload.get("cui", "") or "",
                    top.payload.get("name", "") or "",
                    float(top.score),
                ))
            else:
                score = float(hits[0].score) if hits else 0.0
                out.append((h, text, "", "", score))
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
    ap.add_argument("--workers", type=int, default=16,
                    help="mp.Pool worker count; each owns an HTTP session "
                         "to biolord TEI and a Qdrant client")
    ap.add_argument("--batch", type=int, default=32,
                    help="entities per TEI /embed request per worker. Must "
                         "stay <= biolord TEI's --max-client-batch-size "
                         "(64 in docker-compose). 16 x 32 matches the "
                         "embed_sentences.py sweet spot -- below TEI's "
                         "concurrency budget (no HTTP 429) but saturates GPU.")
    ap.add_argument("--min-score", type=float, default=0.7,
                    help="cosine similarity cutoff; below this, cui='' so "
                         "unresolved mentions are preserved in the snapshot "
                         "(just like the lexical snapshot)")
    ap.add_argument("--biolord-url", default=BIOLORD_URL)
    ap.add_argument("--qdrant-url", default=QDRANT_URL)
    ap.add_argument("--qdrant-grpc-port", type=int, default=QDRANT_GRPC_PORT)
    args = ap.parse_args()

    print("collecting unique entities...", flush=True)
    t0 = time.time()
    unique = collect_unique_entities(args.docs)
    items = sorted(unique.items(), key=lambda kv: kv[1])
    print(f"  {len(items):,} unique entities in {time.time() - t0:.1f}s",
          flush=True)
    if not items:
        raise SystemExit("no entities found; run steps 7-8 first")

    workers = max(1, min(args.workers, len(items)))
    shards = _shard(items, workers)
    print(f"dispatching {len(items):,} entities across {workers} workers "
          f"(shards={[len(s) for s in shards]}, batch={args.batch}, "
          f"min-score={args.min_score})",
          flush=True)

    ctx = mp.get_context("spawn")
    wall_t0 = time.time()
    with ctx.Pool(
        processes=workers,
        initializer=_init_worker,
        initargs=(args.biolord_url, args.qdrant_url, args.qdrant_grpc_port),
    ) as pool:
        results = pool.map(
            _process_shard,
            [(s, args.batch, args.min_score) for s in shards],
        )

    flat = [row for shard_out in results for row in shard_out]
    flat.sort(key=lambda r: r[1])  # sort by text for stable diff ordering

    args.out.parent.mkdir(parents=True, exist_ok=True)
    linked = 0
    with args.out.open("w", encoding="utf-8") as f:
        for _h, text, cui, name, score in flat:
            if cui:
                linked += 1
            f.write(json.dumps(
                {"text": text, "cui": cui, "cui_name": name, "score": round(score, 4)},
                ensure_ascii=False,
            ) + "\n")

    wall = time.time() - wall_t0
    print(
        f"wrote {len(flat):,} unique entities "
        f"({linked:,} linked >= {args.min_score}, "
        f"{len(flat) - linked:,} unlinked) "
        f"to {args.out.relative_to(REPO)} in {wall:.1f}s wall",
        flush=True,
    )


if __name__ == "__main__":
    main()
