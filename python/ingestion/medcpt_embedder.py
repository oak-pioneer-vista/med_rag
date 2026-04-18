"""MedCPT encoders for medical text retrieval.

MedCPT (NCBI) is a two-tower contrastive model trained on PubMed click
logs. It has two encoders, both BERT-base (768-dim, CLS-pooled):

  - Article Encoder (max 512 tokens) — for passages / document chunks.
  - Query Encoder   (max  64 tokens) — for user queries.

The two sides are not interchangeable: retrieval quality depends on
encoding docs with the article tower and queries with the query tower,
then comparing via dot product.

Usage:
    from python.ingestion.medcpt_embedder import ArticleEncoder, QueryEncoder

    articles = ArticleEncoder()
    doc_embs = articles.encode([chunk["text"] for chunk in chunks])

    queries = QueryEncoder()
    q_emb = queries.encode(["symptoms of high blood pressure"])

CLI smoke-test:
    python python/ingestion/medcpt_embedder.py --n 16
"""

from __future__ import annotations

import argparse
from typing import Iterable

import numpy as np
import torch
from transformers import AutoModel, AutoTokenizer

ARTICLE_MODEL = "ncbi/MedCPT-Article-Encoder"
QUERY_MODEL = "ncbi/MedCPT-Query-Encoder"
EMBED_DIM = 768


class MedCPTEncoder:
    def __init__(
        self,
        model_id: str,
        max_length: int,
        device: str | None = None,
        batch_size: int = 32,
    ) -> None:
        self.model_id = model_id
        self.max_length = max_length
        self.batch_size = batch_size
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")

        self.tokenizer = AutoTokenizer.from_pretrained(model_id)
        self.model = AutoModel.from_pretrained(model_id).to(self.device).eval()

    @torch.no_grad()
    def encode(
        self,
        texts: Iterable[str],
        batch_size: int | None = None,
    ) -> np.ndarray:
        """Encode texts to (n, 768) float32 array via CLS pooling."""
        texts = list(texts)
        if not texts:
            return np.zeros((0, EMBED_DIM), dtype=np.float32)

        bs = batch_size or self.batch_size
        out = np.empty((len(texts), EMBED_DIM), dtype=np.float32)
        for start in range(0, len(texts), bs):
            batch = texts[start : start + bs]
            enc = self.tokenizer(
                batch,
                truncation=True,
                padding=True,
                max_length=self.max_length,
                return_tensors="pt",
            ).to(self.device)
            hidden = self.model(**enc).last_hidden_state
            cls = hidden[:, 0, :].cpu().numpy().astype(np.float32)
            out[start : start + len(batch)] = cls
        return out


class ArticleEncoder(MedCPTEncoder):
    """For encoding document / passage text (e.g. Section.text)."""

    def __init__(self, device: str | None = None, batch_size: int = 32) -> None:
        super().__init__(ARTICLE_MODEL, max_length=512, device=device, batch_size=batch_size)


class QueryEncoder(MedCPTEncoder):
    """For encoding user queries. Kept short (<=64 tokens)."""

    def __init__(self, device: str | None = None, batch_size: int = 64) -> None:
        super().__init__(QUERY_MODEL, max_length=64, device=device, batch_size=batch_size)


def _smoke_test(n: int) -> None:
    """Encode the first n Section chunks and score a sample query against them."""
    import json
    from pathlib import Path

    chunks_path = Path(__file__).resolve().parent.parent.parent / "data" / "mtsamples_chunks.jsonl"
    chunks: list[dict] = []
    with chunks_path.open(encoding="utf-8") as f:
        for line in f:
            chunks.append(json.loads(line))
            if len(chunks) >= n:
                break
    print(f"loaded {len(chunks)} chunks from {chunks_path.name}")

    print("loading ArticleEncoder...")
    articles = ArticleEncoder()
    print(f"  device={articles.device}")
    doc_embs = articles.encode([c["text"] for c in chunks])
    print(f"  article embeddings: shape={doc_embs.shape} dtype={doc_embs.dtype}")

    print("loading QueryEncoder...")
    queries = QueryEncoder()
    query = "patient with allergic rhinitis and nasal symptoms"
    q_emb = queries.encode([query])
    print(f"  query embedding: shape={q_emb.shape}")

    scores = (doc_embs @ q_emb[0]).tolist()
    ranked = sorted(zip(scores, chunks), key=lambda x: -x[0])[:3]
    print(f"\ntop 3 chunks for query: {query!r}")
    for score, c in ranked:
        snippet = c["text"][:120].replace("\n", " ")
        print(f"  score={score:.2f}  {c['chunk_id']:>16}  {c['section_type']:<18}  {snippet!r}")


def main() -> None:
    ap = argparse.ArgumentParser(description="MedCPT embedder smoke test")
    ap.add_argument("--n", type=int, default=16, help="number of chunks to encode")
    args = ap.parse_args()
    _smoke_test(args.n)


if __name__ == "__main__":
    main()
