"""Sweep biolord embed+upsert throughput across batch sizes.

Fixed sample of `--sample-size` Concepts fetched from Neo4j (starting
after `--start-cui` so we don't fight the in-flight production run),
then for each batch size in `--batches` a fresh mp.Pool of `--workers`
runs through the same sample with LPT-first size-balanced shards.

Wall time + throughput per trial.

Upserts are idempotent (uuid5(cui)), so each trial overwrites the
previous trial's points in Qdrant -- no pollution of the production
collection.

Usage:
  python scripts/bench_biolord_batch.py \
      [--workers 8] [--batches 16,32,64,128,256] \
      [--sample-size 40000] [--start-cui C2000000]
"""

from __future__ import annotations

import argparse
import multiprocessing as mp
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "python"))

from ingestion.umls.build_biolord_concept_index import (  # noqa: E402
    BIOLORD_URL,
    COLLECTION,
    NEO4J_PASSWORD,
    NEO4J_URI,
    NEO4J_USER,
    QDRANT_URL,
    _fetch_page,
    _init_worker,
    _point_id,
    _process_shard,
    _shard,
)
from neo4j import GraphDatabase  # noqa: E402
from qdrant_client import QdrantClient  # noqa: E402


def _count_upserted(sample: list[tuple[str, str]], qdrant_url: str) -> int:
    """Count how many of the sample's CUIs landed in Qdrant when a trial
    was capped mid-run; used to extrapolate rate from partial work."""
    client = QdrantClient(url=qdrant_url, prefer_grpc=True, grpc_port=6334, timeout=120)
    ids = [_point_id(cui) for cui, _ in sample]
    hit = 0
    CHUNK = 500
    for i in range(0, len(ids), CHUNK):
        try:
            pts = client.retrieve(
                collection_name=COLLECTION,
                ids=ids[i : i + CHUNK],
                with_payload=False,
                with_vectors=False,
            )
            hit += len(pts)
        except Exception:
            break
    return hit


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--batches", default="16,32,64,128,256",
                    help="comma-separated batch sizes to sweep")
    ap.add_argument("--sample-size", type=int, default=40000)
    ap.add_argument("--start-cui", default="C2000000",
                    help="fetch sample-size concepts with cui > this; pick a "
                         "range past the production resume point so sweep "
                         "upserts don't overlap in-flight work")
    ap.add_argument("--biolord-url", default=BIOLORD_URL)
    ap.add_argument("--qdrant-url", default=QDRANT_URL)
    ap.add_argument("--time-cap", type=float, default=180.0,
                    help="per-trial wall-time cap in seconds. A trial that "
                         "hits the cap is aborted; rate is extrapolated from "
                         "the points upserted up to that point so slow/"
                         "pathological configs don't dominate sweep time.")
    args = ap.parse_args()

    batches = [int(b) for b in args.batches.split(",")]

    print(f"fetching {args.sample_size:,} concepts after {args.start_cui} "
          f"from neo4j...", flush=True)
    driver = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASSWORD))
    t0 = time.time()
    sample = _fetch_page(driver, args.start_cui, args.sample_size)
    driver.close()
    total_chars = sum(len(n) for _, n in sample)
    print(f"  {len(sample):,} concepts fetched in {time.time() - t0:.1f}s "
          f"(avg name len {total_chars / max(len(sample),1):.1f} chars)",
          flush=True)

    shards = _shard(sample, args.workers)
    sh_lens = [sum(len(n) for _, n in s) for s in shards]
    sh_counts = [len(s) for s in shards]
    print(f"LPT shards: counts {sh_counts} / char-lens {sh_lens} "
          f"(max/mean = {max(sh_lens) / (sum(sh_lens)/len(sh_lens)):.2f})",
          flush=True)

    print(f"\nsweep: workers={args.workers}  batches={batches}  "
          f"sample={len(sample):,}\n",
          flush=True)
    print(f"  {'batch':>6}  {'wall_s':>8}  {'rate_hz':>10}  note")
    print(f"  {'-'*6}  {'-'*8}  {'-'*10}  {'-'*20}")

    results = []
    for batch in batches:
        ctx = mp.get_context("spawn")
        pool = ctx.Pool(
            processes=args.workers,
            initializer=_init_worker,
            initargs=(args.biolord_url, args.qdrant_url),
        )
        t0 = time.time()
        async_result = pool.map_async(_process_shard, [(s, batch) for s in shards])
        note = ""
        try:
            counts = async_result.get(timeout=args.time_cap)
            wall = time.time() - t0
            total = sum(counts)
        except mp.TimeoutError:
            pool.terminate()
            wall = time.time() - t0
            total = _count_upserted(sample, args.qdrant_url)
            note = f"capped @ {args.time_cap:.0f}s (extrapolated)"
        finally:
            pool.close()
            pool.join()
        rate = total / max(wall, 1e-9)
        results.append((batch, wall, rate, total))
        print(f"  {batch:>6}  {wall:>8.2f}  {rate:>10.0f}  {note}", flush=True)

    best = max(results, key=lambda r: r[2])
    print(
        f"\nbest: batch={best[0]}  rate={best[2]:.0f}/s  "
        f"(wall {best[1]:.2f}s on {best[3]:,} concepts)",
        flush=True,
    )


if __name__ == "__main__":
    main()
