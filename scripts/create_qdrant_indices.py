"""Set Qdrant payload indexes and indexing threshold on `mtsamples_sections`.

Idempotent. Run once after `embed_sections.py`, or re-run any time (adding a
new payload-index field, for example — existing indexes are a no-op).

What it does:
  * optimizer `indexing_threshold` -> 100, so HNSW kicks in promptly instead
    of waiting for the default ~20k-per-segment threshold (at ~22k points
    across 8 segments, segments otherwise stay unindexed and every search
    falls back to brute force).
  * keyword payload indexes on the fields we filter by (specialty,
    section_type, CUIs, identifiers) and integer indexes on the numeric
    window bookkeeping fields.

Usage:
  docker compose up -d qdrant
  python scripts/create_qdrant_indices.py
"""

from __future__ import annotations

from qdrant_client import QdrantClient
from qdrant_client.models import OptimizersConfigDiff, PayloadSchemaType

QDRANT_URL = "http://localhost:6333"
COLLECTION = "mtsamples_sections"
INDEXING_THRESHOLD = 100

KEYWORD_FIELDS = [
    "parent_chunk_id",
    "section_type",
    "section_cui",
    "specialty",
    "specialty_cui",
    "doctype_cui",
    "sample_name",
]
INTEGER_FIELDS = [
    "doc_id",
    "window_index",
    "window_count",
]


def main() -> None:
    client = QdrantClient(url=QDRANT_URL, timeout=60)
    if not client.collection_exists(COLLECTION):
        raise SystemExit(
            f"collection {COLLECTION!r} doesn't exist — "
            f"run python/ingestion/embed_sections.py first"
        )

    print(f"==> setting indexing_threshold={INDEXING_THRESHOLD} on {COLLECTION}")
    client.update_collection(
        collection_name=COLLECTION,
        optimizer_config=OptimizersConfigDiff(indexing_threshold=INDEXING_THRESHOLD),
    )

    for f in KEYWORD_FIELDS:
        print(f"==> create_payload_index(keyword)  {f}")
        client.create_payload_index(
            collection_name=COLLECTION,
            field_name=f,
            field_schema=PayloadSchemaType.KEYWORD,
            wait=True,
        )
    for f in INTEGER_FIELDS:
        print(f"==> create_payload_index(integer)  {f}")
        client.create_payload_index(
            collection_name=COLLECTION,
            field_name=f,
            field_schema=PayloadSchemaType.INTEGER,
            wait=True,
        )

    info = client.get_collection(COLLECTION)
    print()
    print(f"status={info.status}  points={info.points_count}  "
          f"indexed={info.indexed_vectors_count}  segments={info.segments_count}")
    print("payload schema:")
    for field, schema in (info.payload_schema or {}).items():
        print(f"  {field}: {schema.data_type}  (points={schema.points})")


if __name__ == "__main__":
    main()
