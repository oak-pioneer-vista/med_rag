"""Extract clinical entities from MTSamples chunks via Stanza i2b2 NER (dask parallel).

Reads the same per-doc JSON files as embed_sections.py and packs each Section
into the same overlapping sentence windows so emitted chunk_ids match the
points written to Qdrant. Each worker loads Stanza's mimic tokenizer + i2b2
NER once and writes a JSONL shard; the driver concatenates shards at the end.

Also runs the Schwartz-Hearst abbreviation algorithm over the *whole*
transcript (all section texts joined) once per doc and writes the resulting
{abbrev: long_form} map to a sibling JSONL sidecar keyed by doc_id. S-H needs
the full "long form (ABBR)" introduction to appear somewhere in the doc, so
running it per-chunk would miss pairs whose long form sits in a different
section from later mentions.

Setup (one time -- models cache to ~/stanza_resources/):
    python -c "import stanza; stanza.download('en', package='mimic', processors={'ner':'i2b2'})"

Usage:
    python python/ingestion/mtsamples/extract_entities.py [--workers 4] [--out PATH] [--abbrev-out PATH]
"""

from __future__ import annotations

import argparse
import json
import os
import time
import uuid
import re
from functools import partial
from pathlib import Path
from typing import Iterable

import dask.bag as db
from abbreviations import schwartz_hearst
from transformers import AutoTokenizer

from embed_sections import (
    DOCS_DIR,
    MAX_TOKENS,
    MODEL_ID,
    OVERLAP_TOKENS,
    _read_doc,
    _split_sentences,
)

REPO = Path(__file__).resolve().parent.parent.parent.parent
OUT_DEFAULT = REPO / "data" / "entities" / "chunk_entities.jsonl"
ABBREV_OUT_DEFAULT = REPO / "data" / "entities" / "doc_abbreviations.jsonl"

# ner_batch_size passed straight through to Stanza's NER processor; we send
# bulk batches of this size to keep the GPU fed.
NER_BATCH_SIZE = 128


def _section_windows(
    section_text: str, tokenizer
) -> tuple[str, list[tuple[str, int, int]]]:
    """Normalize a section and return its windows with char ranges.

    Returns (section_norm, [(win_text, start_char, end_char), ...]). Mirrors
    the sentence-packing algorithm in embed_sections._pack_sentences so the
    emitted window texts + chunk_ids match the Qdrant points exactly, but
    also exposes each window's char span inside `section_norm`. Feeding the
    whole normalized section to Stanza (instead of each window) lets NER
    see cross-sentence context; the char ranges then let us partition the
    resulting entities back into the per-window records that Qdrant joins on.
    """
    sentences = _split_sentences(section_text)
    if not sentences:
        return "", []
    counts = [len(tokenizer.encode(s, add_special_tokens=False)) for s in sentences]

    # Each sentence in `section_norm = " ".join(sentences)` starts at the
    # previous sentence's end + 1 (the single space separator).
    sent_start = [0]
    for s in sentences[:-1]:
        sent_start.append(sent_start[-1] + len(s) + 1)
    sent_end = [sent_start[i] + len(sentences[i]) for i in range(len(sentences))]
    section_norm = " ".join(sentences)

    windows: list[tuple[str, int, int]] = []
    start = 0
    n = len(sentences)
    while start < n:
        cur = 0
        end = start
        while end < n and cur + counts[end] <= MAX_TOKENS:
            cur += counts[end]
            end += 1
        if end == start:
            end = start + 1  # oversized lone sentence; TEI will truncate
        win_text = " ".join(sentences[start:end])
        windows.append((win_text, sent_start[start], sent_end[end - 1]))
        if end >= n:
            break
        back = 0
        new_start = end
        while new_start > start + 1 and back < OVERLAP_TOKENS:
            new_start -= 1
            back += counts[new_start]
        start = new_start
    return section_norm, windows


def _iter_sections(doc: dict, tokenizer) -> Iterable[dict]:
    """Yield one NER job per non-empty section.

    Each job bundles the full normalized section text (fed to Stanza as a
    single document) with its window stubs + char ranges so the worker can
    demux entities back into per-window records after the section-level NER
    pass. Chunk IDs match embed_sections._chunk_doc for the Qdrant join.
    """
    for s in doc["sections"]:
        text = s.get("text", "").strip()
        if not text:
            continue
        section_norm, win_tuples = _section_windows(text, tokenizer)
        if not win_tuples:
            continue
        stubs: list[dict] = []
        ranges: list[tuple[int, int]] = []
        for idx, (win_text, ws, we) in enumerate(win_tuples):
            suffix = f"#{idx}" if len(win_tuples) > 1 else ""
            stubs.append({
                "chunk_id": s["chunk_id"] + suffix,
                "parent_chunk_id": s["chunk_id"],
                "doc_id": s["doc_id"],
                "section_type": s["section_type"],
                "text": win_text,
            })
            ranges.append((ws, we))
        yield {"section_text": section_norm, "stubs": stubs, "ranges": ranges}


def _normalize_abbrev(text: str) -> str:
    """Canonical key for matching an entity token to an abbreviation.

    Stanza span text can arrive with trailing punctuation, parentheses, or
    stray whitespace (e.g. "BP.", "(COPD)"), and S-H keys preserve the exact
    casing from the source text. Strip boundary punctuation + whitespace and
    uppercase both sides so "BP", " bp ", "BP." and "(BP)" all collide.
    """
    return text.strip().strip(".,;:()[]{}'\"").strip().upper()


# Alnum runs only — splits hyphen/slash/apostrophe joins like "post-MI",
# "CT/MRI", "pt's" so embedded abbreviations surface as standalone tokens.
_TOKEN_RE = re.compile(r"[A-Za-z0-9]+")


def _entity_expansions(text: str, lookup: dict[str, str]) -> dict[str, str]:
    """Tokenize inside an entity phrase and return every token that hits the map.

    Stanza routinely returns multi-word spans ("CHF exacerbation", "CT of the
    chest", "post-MI status") where the abbreviation is embedded rather than
    the whole phrase. Whole-phrase lookup only fires on bare single-token
    spans and misses those. Returning a {token: long_form} dict keeps the
    common single-token case natural while supporting multi-hit phrases.
    Preserves the original token casing from the entity text so consumers can
    substitute in place; keys are the surface form, values the S-H long form.
    """
    if not lookup:
        return {}
    hits: dict[str, str] = {}
    for tok in _TOKEN_RE.findall(text):
        exp = lookup.get(_normalize_abbrev(tok))
        if exp and tok not in hits:
            hits[tok] = exp
    return hits


def _flush_batch(
    nlp,
    jobs: list[dict],
    lookups: list[dict[str, str]],
    out,
) -> tuple[int, int]:
    """Run NER over a batch of whole sections and demux entities into windows.

    Stanza receives the full normalized section text per job -- not the
    windowed chunks -- so the NER model sees cross-sentence context. After
    tagging, each entity's char span is checked against every window's
    (start, end) range in the section; spans fully contained in a window
    are emitted on that window's JSONL line with offsets rebased to the
    window text. An entity landing in the sentence-overlap region between
    two adjacent windows is attributed to both, which matches the overlap
    semantics of the embedded chunks.
    """
    import stanza
    if not jobs:
        return 0, 0
    section_docs = [stanza.Document([], text=j["section_text"]) for j in jobs]
    nlp(section_docs)
    total_windows = 0
    total_ents = 0
    for job, d, lookup in zip(jobs, section_docs, lookups):
        section_ents = list(d.ents)
        for stub, (ws, we) in zip(job["stubs"], job["ranges"]):
            win_ents = []
            for e in section_ents:
                if e.start_char < ws or e.end_char > we:
                    continue
                ent = {
                    "text": e.text,
                    "type": e.type,
                    "start_char": e.start_char - ws,
                    "end_char": e.end_char - ws,
                }
                expansions = _entity_expansions(e.text, lookup)
                if expansions:
                    ent["expansions"] = expansions
                win_ents.append(ent)
            out.write(json.dumps({**stub, "entities": win_ents}, ensure_ascii=False) + "\n")
            total_windows += 1
            total_ents += len(win_ents)
    return total_windows, total_ents


def _doc_abbrev_map(doc: dict) -> dict[str, str]:
    """Join all section texts and run Schwartz-Hearst over the whole transcript.

    S-H looks for "long form (ABBR)" / "ABBR (long form)" patterns, so the
    introduction and its re-uses can live in different sections; joining
    sections (separator chosen to terminate sentences so S-H's line-wise scan
    doesn't run definitions across section boundaries) recovers those.
    """
    full = " . ".join(
        s.get("text", "").strip() for s in doc.get("sections", []) if s.get("text")
    )
    if not full:
        return {}
    return schwartz_hearst.extract_abbreviation_definition_pairs(
        doc_text=full, most_common_definition=True
    )


def _process_partition(paths: Iterable[str], out_dir: str, abbrev_dir: str) -> list[tuple]:
    """Dask worker: load Stanza once, NER every chunk in batches, write a JSONL shard.

    Each worker writes its own shard under `out_dir` to avoid append
    contention across processes; the driver concatenates them at the end.
    A parallel per-doc abbreviation shard is written to `abbrev_dir`.
    """
    import stanza

    # Shard id must be unique per partition invocation, not per process:
    # dask's multiprocessing scheduler can reuse the same process for multiple
    # partitions, and an earlier version keyed the shard on os.getpid(),
    # causing later partitions to clobber earlier shards.
    shard_id = uuid.uuid4().hex[:8]
    wid = f"{os.getpid()}/{shard_id}"
    t0 = time.time()
    paths = list(paths)
    print(f"[worker {wid}] starting on {len(paths)} docs", flush=True)

    # Models already cached by main(); workers must not attempt to re-download
    # because concurrent downloads race on the same files.
    # package=None blocks Stanza from auto-loading mimic's full processor set
    # (pos, lemma, constituency, depparse, sentiment) when we only want
    # tokenization + NER. Dropping those cuts per-chunk latency ~10x.
    nlp = stanza.Pipeline(
        lang="en",
        package=None,
        processors={"tokenize": "mimic", "ner": "i2b2"},
        use_gpu=True,
        ner_batch_size=NER_BATCH_SIZE,
        verbose=False,
        download_method="reuse_resources",
    )
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)

    shard = Path(out_dir) / f"part-{shard_id}.jsonl"
    shard.parent.mkdir(parents=True, exist_ok=True)
    abbrev_shard = Path(abbrev_dir) / f"part-{shard_id}.jsonl"
    abbrev_shard.parent.mkdir(parents=True, exist_ok=True)

    n_chunks = 0
    n_ents = 0
    n_abbrev_pairs = 0
    batch_jobs: list[dict] = []
    batch_lookups: list[dict[str, str]] = []
    with shard.open("w", encoding="utf-8") as out, \
         abbrev_shard.open("w", encoding="utf-8") as abbrev_out:
        for path in paths:
            doc = _read_doc(path)
            abbrev_map = _doc_abbrev_map(doc)
            abbrev_out.write(json.dumps(
                {"doc_id": doc["doc_id"], "abbrev_map": abbrev_map},
                ensure_ascii=False,
            ) + "\n")
            n_abbrev_pairs += len(abbrev_map)
            # Precompute the normalized-key lookup once per doc so every
            # chunk in this doc can hit it in O(1) without re-normalizing.
            abbrev_lookup = {
                _normalize_abbrev(k): v for k, v in abbrev_map.items()
            }
            for job in _iter_sections(doc, tokenizer):
                batch_jobs.append(job)
                batch_lookups.append(abbrev_lookup)
                # NER_BATCH_SIZE now counts *sections* per Stanza call; each
                # section expands to 1+ windows internally, so the emitted
                # chunk count per flush is >= batch size.
                if len(batch_jobs) >= NER_BATCH_SIZE:
                    nw, ne = _flush_batch(nlp, batch_jobs, batch_lookups, out)
                    n_chunks += nw
                    n_ents += ne
                    batch_jobs.clear()
                    batch_lookups.clear()
        if batch_jobs:
            nw, ne = _flush_batch(nlp, batch_jobs, batch_lookups, out)
            n_chunks += nw
            n_ents += ne

    print(
        f"[worker {wid}] {n_chunks} chunks, {n_ents} entities, "
        f"{n_abbrev_pairs} abbrev pairs in "
        f"{time.time() - t0:.1f}s -> {shard.name}",
        flush=True,
    )
    return [(len(paths), n_chunks, n_ents, n_abbrev_pairs, str(shard), str(abbrev_shard))]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    # Default 4 workers: each holds its own Stanza pipeline on the GPU, so
    # parallelism is bounded by VRAM (L4 24GB) rather than CPU count.
    ap.add_argument("--workers", type=int, default=4, help="dask worker processes (each loads Stanza on GPU)")
    ap.add_argument("--out", type=str, default=str(OUT_DEFAULT),
                    help="path to final entities JSONL (shards concatenated here)")
    ap.add_argument("--abbrev-out", type=str, default=str(ABBREV_OUT_DEFAULT),
                    help="path to final per-doc Schwartz-Hearst abbreviation JSONL")
    args = ap.parse_args()

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    shard_dir = out_path.parent / f"{out_path.stem}.shards"
    shard_dir.mkdir(parents=True, exist_ok=True)

    abbrev_path = Path(args.abbrev_out)
    abbrev_path.parent.mkdir(parents=True, exist_ok=True)
    abbrev_shard_dir = abbrev_path.parent / f"{abbrev_path.stem}.shards"
    abbrev_shard_dir.mkdir(parents=True, exist_ok=True)

    # Prime the model cache once in the driver so workers never race on
    # downloads. Cheap no-op if already cached.
    print("ensuring stanza models are cached (mimic + i2b2)...")
    import stanza
    stanza.download("en", package="mimic", processors={"ner": "i2b2"}, verbose=False)

    paths = sorted(str(p) for p in DOCS_DIR.glob("*.json"))
    print(
        f"dispatching {len(paths)} docs across {args.workers} dask partitions "
        f"-> {out_path} (+ abbrevs -> {abbrev_path})"
    )

    bag = db.from_sequence(paths, npartitions=args.workers)
    results = bag.map_partitions(
        partial(_process_partition, out_dir=str(shard_dir), abbrev_dir=str(abbrev_shard_dir))
    ).compute(scheduler="processes", num_workers=args.workers)

    def _concat(final: Path, shards: list[Path], shard_root: Path) -> None:
        with final.open("w", encoding="utf-8") as w:
            for sp in shards:
                if not sp.exists():
                    continue
                with sp.open("r", encoding="utf-8") as f:
                    for line in f:
                        w.write(line)
                sp.unlink()
        try:
            shard_root.rmdir()
        except OSError:
            pass

    _concat(out_path, [Path(r[4]) for r in results], shard_dir)
    _concat(abbrev_path, [Path(r[5]) for r in results], abbrev_shard_dir)

    n_docs = sum(r[0] for r in results)
    n_chunks = sum(r[1] for r in results)
    n_ents = sum(r[2] for r in results)
    n_abbrev_pairs = sum(r[3] for r in results)
    print(
        f"processed {n_docs} docs, {n_chunks} chunks, {n_ents} entities -> {out_path}"
    )
    print(f"extracted {n_abbrev_pairs} abbrev pairs -> {abbrev_path}")


if __name__ == "__main__":
    main()
