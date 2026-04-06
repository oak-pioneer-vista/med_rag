"""Ingest MTSamples CSV into Qdrant.

Reads the downloaded CSV, embeds the transcription text with
sentence-transformers, and upserts into a Qdrant collection.

Usage:
  docker compose up -d          # start Qdrant
  python scripts/ingest_mtsamples.py
"""

import pandas as pd
from pathlib import Path
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct
from sentence_transformers import SentenceTransformer

DATA_DIR = Path(__file__).resolve().parent.parent / "data" / "mtsamples"
COLLECTION = "mtsamples"
QDRANT_URL = "http://localhost:6333"
EMBED_MODEL = "all-MiniLM-L6-v2"
BATCH_SIZE = 64


def main():
    csv_path = DATA_DIR / "mtsamples.csv"
    if not csv_path.exists():
        raise FileNotFoundError(
            f"{csv_path} not found. Run scripts/download_mtsamples.py first."
        )

    df = pd.read_csv(csv_path)
    # Drop rows without transcription text
    df = df.dropna(subset=["transcription"]).reset_index(drop=True)

    print(f"Loaded {len(df)} records from {csv_path.name}")

    model = SentenceTransformer(EMBED_MODEL)
    dim = model.get_sentence_embedding_dimension()

    client = QdrantClient(url=QDRANT_URL)

    # Recreate collection
    if client.collection_exists(COLLECTION):
        client.delete_collection(COLLECTION)

    client.create_collection(
        collection_name=COLLECTION,
        vectors_config=VectorParams(size=dim, distance=Distance.COSINE),
    )

    texts = df["transcription"].tolist()

    for start in range(0, len(texts), BATCH_SIZE):
        batch_texts = texts[start : start + BATCH_SIZE]
        batch_df = df.iloc[start : start + BATCH_SIZE]
        embeddings = model.encode(batch_texts, show_progress_bar=False)

        points = [
            PointStruct(
                id=start + i,
                vector=emb.tolist(),
                payload={
                    "description": row.get("description", ""),
                    "medical_specialty": row.get("medical_specialty", ""),
                    "sample_name": row.get("sample_name", ""),
                    "transcription": row["transcription"],
                    "keywords": row.get("keywords", ""),
                },
            )
            for i, (emb, (_, row)) in enumerate(
                zip(embeddings, batch_df.iterrows())
            )
        ]

        client.upsert(collection_name=COLLECTION, points=points)
        print(f"  upserted {start + len(points)}/{len(texts)}")

    info = client.get_collection(COLLECTION)
    print(f"Done. Collection '{COLLECTION}' has {info.points_count} points.")


if __name__ == "__main__":
    main()
