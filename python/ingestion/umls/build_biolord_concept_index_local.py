"""Build the `umls_concepts_biolord` index without TEI (local GPU).

Direct sentence-transformers loop: loads FremyCompany/BioLORD-2023 on
the host GPU, streams Concept rows keyset-paginated from Neo4j,
length-sorts each page (minimizes tokenizer padding waste), runs
chunked forward passes at `--batch` on-GPU, and upserts points to
Qdrant via the gRPC client.

Why it exists alongside `build_biolord_concept_index.py` (TEI on
:8081): at 8 workers x batch 64 the TEI path on L4 sustains ~2,000/s
end-to-end, bottlenecked on HTTP round-trips + JSON marshalling
rather than GPU compute. The local variant skips that path. Pick:

  - TEI variant      : simpler infra, no host PyTorch, matches the
                       pattern of the rest of the embedding pipeline.
  - Local variant    : ~2-3x faster for a one-time full rebuild
                       because every embedding stays on-device until
                       the batch is complete.

Prereqs:
  - host-side PyTorch with CUDA (already required by step 7's Stanza
    GPU path): `python -c 'import torch; print(torch.cuda.is_available())'`
  - qdrant up (6333 REST, 6334 gRPC)
  - Neo4j with UMLS loaded
  - Recommended: stop the biolord TEI container first to free VRAM
    (`docker compose stop biolord`) so torch owns the GPU.

Output (same as TEI variant, so the two are interchangeable):
  Qdrant collection `umls_concepts_biolord`, one point per Concept,
  payload {cui, name}, point id uuid5(cui).

Usage:
  python python/ingestion/umls/build_biolord_concept_index_local.py \
      [--batch 1024] [--page-size 50000] [--resume] [--half]
"""

from __future__ import annotations

import argparse
import os
import time
import uuid
from pathlib import Path

import torch
from neo4j import GraphDatabase
from qdrant_client import QdrantClient
from qdrant_client.http.models import (
    Distance,
    OptimizersConfigDiff,
    PayloadSchemaType,
    PointStruct,
    VectorParams,
)
from sentence_transformers import SentenceTransformer

REPO = Path(__file__).resolve().parent.parent.parent.parent
STATE_FILE = REPO / "data" / "biolord_concept_index.state"

QDRANT_URL = os.environ.get("QDRANT_URL", "http://localhost:6333")
QDRANT_GRPC_PORT = int(os.environ.get("QDRANT_GRPC_PORT", "6334"))
NEO4J_URI = os.environ.get("NEO4J_URI", "bolt://localhost:7687")
NEO4J_USER = os.environ.get("NEO4J_USER", "neo4j")
NEO4J_PASSWORD = os.environ.get("NEO4J_PASSWORD", "medragpass")

COLLECTION = "umls_concepts_biolord"
MODEL_ID = "FremyCompany/BioLORD-2023"
# Must match the TEI variant so re-runs overwrite in-place.
_PID_NS = uuid.UUID("b1010d55-2023-4e1f-9c3b-9a5b7e1f2a44")


def _point_id(cui: str) -> str:
    return str(uuid.uuid5(_PID_NS, cui))


def _fetch_page(driver, after_cui: str, page_size: int) -> list[tuple[str, str]]:
    """Keyset-paginated page of (cui, name) with cui > after_cui, ORDER BY cui."""
    with driver.session() as session:
        rows = session.run(
            "MATCH (c:Concept) "
            "WHERE c.cui > $last AND c.name IS NOT NULL AND c.name <> '' "
            "RETURN c.cui AS cui, c.name AS name "
            "ORDER BY c.cui LIMIT $lim",
            last=after_cui, lim=page_size,
        ).data()
    return [(r["cui"], r["name"]) for r in rows]


def _ensure_collection(client: QdrantClient, dim: int, recreate: bool) -> None:
    if recreate and client.collection_exists(COLLECTION):
        client.delete_collection(COLLECTION)
        print(f"dropped {COLLECTION!r}", flush=True)
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
        print(f"created {COLLECTION!r} dim={dim}", flush=True)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--batch", type=int, default=1024,
                    help="on-GPU forward-pass batch size. 1024 fits easily "
                         "on L4 (24 GB) for bert-base @ seq<=128. Push to "
                         "2048-4096 if you have more VRAM and the concept "
                         "name distribution is short.")
    ap.add_argument("--page-size", type=int, default=50000,
                    help="neo4j keyset page size; one encode+upsert per page, "
                         "then the state file is updated.")
    ap.add_argument("--resume", action="store_true",
                    help="continue after the last CUI in data/biolord_concept_index.state")
    ap.add_argument("--recreate", action="store_true")
    ap.add_argument("--half", action="store_true",
                    help="cast the model to fp16 (~2x forward-pass speed on L4 "
                         "at the cost of a small numeric delta in cosine "
                         "scores; BioLORD was trained fp32).")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--qdrant-url", default=QDRANT_URL)
    ap.add_argument("--qdrant-grpc-port", type=int, default=QDRANT_GRPC_PORT)
    args = ap.parse_args()

    if args.device == "cuda" and not torch.cuda.is_available():
        raise SystemExit("CUDA not available. Pass --device cpu or fix the install.")

    print(f"loading {MODEL_ID} on {args.device}{' (fp16)' if args.half else ''}...",
          flush=True)
    t0 = time.time()
    model = SentenceTransformer(MODEL_ID, device=args.device)
    if args.half and args.device == "cuda":
        model = model.half()
    dim = model.get_sentence_embedding_dimension()
    print(f"  model loaded, dim={dim} ({time.time() - t0:.1f}s)", flush=True)

    qc = QdrantClient(
        url=args.qdrant_url,
        prefer_grpc=True,
        grpc_port=args.qdrant_grpc_port,
        timeout=300,
    )
    _ensure_collection(qc, dim, args.recreate)

    after_cui = ""
    if args.resume and STATE_FILE.exists():
        after_cui = STATE_FILE.read_text().strip()
        print(f"resuming after CUI {after_cui!r}", flush=True)

    driver = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASSWORD))

    done = 0
    wall_t0 = time.time()
    try:
        while True:
            page = _fetch_page(driver, after_cui, args.page_size)
            if not page:
                break
            # Neo4j returns ORDER BY cui ASC; grab the max now because we're
            # about to re-sort by length for tokenizer padding efficiency.
            next_after = page[-1][0]

            # Length-sort descending so adjacent items in a batch have
            # similar length -- minimizes padding-to-longest waste in the
            # forward pass. Same principle as extract_section_entities.py.
            page.sort(key=lambda it: -len(it[1]))
            texts = [n for _, n in page]

            t0 = time.time()
            with torch.inference_mode():
                vecs = model.encode(
                    texts,
                    batch_size=args.batch,
                    convert_to_numpy=True,
                    normalize_embeddings=True,
                    show_progress_bar=False,
                )
            t_embed = time.time() - t0

            t0 = time.time()
            UPSERT_CHUNK = 2000
            for i in range(0, len(page), UPSERT_CHUNK):
                sub = page[i : i + UPSERT_CHUNK]
                sub_vecs = vecs[i : i + UPSERT_CHUNK]
                pts = [
                    PointStruct(
                        id=_point_id(cui),
                        vector=v.astype("float32").tolist(),
                        payload={"cui": cui, "name": name},
                    )
                    for (cui, name), v in zip(sub, sub_vecs)
                ]
                qc.upsert(collection_name=COLLECTION, points=pts, wait=False)
            t_upsert = time.time() - t0

            after_cui = next_after
            STATE_FILE.parent.mkdir(parents=True, exist_ok=True)
            STATE_FILE.write_text(after_cui)
            done += len(page)
            rate = len(page) / max(t_embed + t_upsert, 1e-9)
            print(f"  page done: cumulative {done:,}  "
                  f"(embed {t_embed:.1f}s + upsert {t_upsert:.1f}s; "
                  f"{rate:.0f}/s; last cui {after_cui})",
                  flush=True)
    finally:
        driver.close()

    wall = time.time() - wall_t0
    info = qc.get_collection(COLLECTION)
    print(
        f"done: {done:,} concepts in {wall:.1f}s "
        f"({done / max(wall, 1e-9):.0f}/s). collection {COLLECTION}: "
        f"{info.points_count} points, status={info.status}",
        flush=True,
    )


if __name__ == "__main__":
    main()
