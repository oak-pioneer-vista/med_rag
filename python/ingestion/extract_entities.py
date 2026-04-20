"""Extract clinical entities from MTSamples chunks via Stanza i2b2 NER (dask parallel).

Reads the same per-doc JSON files as embed_sections.py and packs each Section
into the same overlapping sentence windows so emitted chunk_ids match the
points written to Qdrant. Each worker loads Stanza's mimic tokenizer + i2b2
NER once and writes a JSONL shard; the driver concatenates shards at the end.

Setup (one time -- models cache to ~/stanza_resources/):
    python -c "import stanza; stanza.download('en', package='mimic', processors={'ner':'i2b2'})"

Usage:
    python python/ingestion/extract_entities.py [--workers 4] [--out PATH]
"""

from __future__ import annotations

import argparse
import json
import os
import time
from functools import partial
from pathlib import Path
from typing import Iterable

import dask.bag as db
from transformers import AutoTokenizer

from embed_sections import (
    DOCS_DIR,
    MODEL_ID,
    _pack_sentences,
    _read_doc,
)

REPO = Path(__file__).resolve().parent.parent.parent
OUT_DEFAULT = REPO / "data" / "entities" / "chunk_entities.jsonl"

# ner_batch_size passed straight through to Stanza's NER processor; we send
# bulk batches of this size to keep the GPU fed.
NER_BATCH_SIZE = 128


def _iter_chunks(doc: dict, tokenizer) -> Iterable[tuple[str, dict]]:
    """Yield (chunk_id, stub) for each windowed chunk in a doc.

    Chunk IDs match embed_sections._chunk_doc so this output joins cleanly
    with the Qdrant collection by chunk_id.
    """
    for s in doc["sections"]:
        text = s.get("text", "").strip()
        if not text:
            continue
        windows = _pack_sentences(text, tokenizer)
        for idx, win_text in enumerate(windows):
            suffix = f"#{idx}" if len(windows) > 1 else ""
            chunk_id = s["chunk_id"] + suffix
            yield chunk_id, {
                "chunk_id": chunk_id,
                "parent_chunk_id": s["chunk_id"],
                "doc_id": s["doc_id"],
                "section_type": s["section_type"],
                "text": win_text,
            }


def _flush_batch(nlp, stubs: list[dict], texts: list[str], out) -> int:
    """Run one NER batch through the pipeline and write a JSONL line per chunk.

    Uses stanza.Document so Stanza treats the list as pre-formed batches and
    its internal ner_batch_size controls GPU batching rather than per-call
    overhead.
    """
    import stanza
    if not texts:
        return 0
    docs = [stanza.Document([], text=t) for t in texts]
    nlp(docs)
    total_ents = 0
    for stub, d in zip(stubs, docs):
        ents = [
            {
                "text": e.text,
                "type": e.type,
                "start_char": e.start_char,
                "end_char": e.end_char,
            }
            for e in d.ents
        ]
        out.write(json.dumps({**stub, "entities": ents}, ensure_ascii=False) + "\n")
        total_ents += len(ents)
    return total_ents


def _process_partition(paths: Iterable[str], out_dir: str) -> list[tuple]:
    """Dask worker: load Stanza once, NER every chunk in batches, write a JSONL shard.

    Each worker writes its own shard under `out_dir` to avoid append
    contention across processes; the driver concatenates them at the end.
    """
    import stanza

    wid = os.getpid()
    t0 = time.time()
    paths = list(paths)
    print(f"[worker {wid}] starting on {len(paths)} docs", flush=True)

    # Models already cached by main(); workers must not attempt to re-download
    # because concurrent downloads race on the same files.
    nlp = stanza.Pipeline(
        lang="en",
        package="mimic",
        processors={"tokenize": "mimic", "ner": "i2b2"},
        use_gpu=True,
        ner_batch_size=NER_BATCH_SIZE,
        verbose=False,
        download_method=None,
    )
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)

    shard = Path(out_dir) / f"part-{wid}.jsonl"
    shard.parent.mkdir(parents=True, exist_ok=True)

    n_chunks = 0
    n_ents = 0
    batch_stubs: list[dict] = []
    batch_texts: list[str] = []
    with shard.open("w", encoding="utf-8") as out:
        for path in paths:
            doc = _read_doc(path)
            for _, stub in _iter_chunks(doc, tokenizer):
                batch_stubs.append(stub)
                batch_texts.append(stub["text"])
                if len(batch_texts) >= NER_BATCH_SIZE:
                    n_ents += _flush_batch(nlp, batch_stubs, batch_texts, out)
                    n_chunks += len(batch_texts)
                    batch_stubs.clear()
                    batch_texts.clear()
        if batch_texts:
            n_ents += _flush_batch(nlp, batch_stubs, batch_texts, out)
            n_chunks += len(batch_texts)

    print(
        f"[worker {wid}] {n_chunks} chunks, {n_ents} entities in "
        f"{time.time() - t0:.1f}s -> {shard.name}",
        flush=True,
    )
    return [(len(paths), n_chunks, n_ents, str(shard))]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    # Default 4 workers: each holds its own Stanza pipeline on the GPU, so
    # parallelism is bounded by VRAM (L4 24GB) rather than CPU count.
    ap.add_argument("--workers", type=int, default=4, help="dask worker processes (each loads Stanza on GPU)")
    ap.add_argument("--out", type=str, default=str(OUT_DEFAULT),
                    help="path to final JSONL (shards concatenated here)")
    args = ap.parse_args()

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    shard_dir = out_path.parent / f"{out_path.stem}.shards"
    shard_dir.mkdir(parents=True, exist_ok=True)

    # Prime the model cache once in the driver so workers never race on
    # downloads. Cheap no-op if already cached.
    print("ensuring stanza models are cached (mimic + i2b2)...")
    import stanza
    stanza.download("en", package="mimic", processors={"ner": "i2b2"}, verbose=False)

    paths = sorted(str(p) for p in DOCS_DIR.glob("*.json"))
    print(f"dispatching {len(paths)} docs across {args.workers} dask partitions -> {out_path}")

    bag = db.from_sequence(paths, npartitions=args.workers)
    results = bag.map_partitions(
        partial(_process_partition, out_dir=str(shard_dir))
    ).compute(scheduler="processes", num_workers=args.workers)

    shard_paths = [Path(r[3]) for r in results]
    with out_path.open("w", encoding="utf-8") as w:
        for sp in shard_paths:
            if not sp.exists():
                continue
            with sp.open("r", encoding="utf-8") as f:
                for line in f:
                    w.write(line)
            sp.unlink()
    try:
        shard_dir.rmdir()
    except OSError:
        pass

    n_docs = sum(r[0] for r in results)
    n_chunks = sum(r[1] for r in results)
    n_ents = sum(r[2] for r in results)
    print(f"processed {n_docs} docs, {n_chunks} chunks, {n_ents} entities -> {out_path}")


if __name__ == "__main__":
    main()
