"""Preprocess UMLS RRF files into CSVs for Neo4j bulk import.

Parallel version. Each RRF is split by byte range (aligned to newline
boundaries) into `--workers` chunks (default = CPU count). Workers write
per-chunk part files; the main process then either (a) does a streaming
dedup when assembling node files, or (b) straight-concatenates part files
for relationship files.

Graph model:

    (Concept      {cui, name})
    (SemanticType {tui, name, tree_number})
    (Source       {sab, name, version})

    (Concept)-[:HAS_SEMTYPE]->(SemanticType)
    (Concept)-[:RELATES {rel, rela, sab}]->(Concept)     # from MRREL
    (Concept)-[:IS_A {sab, rela}]->(Concept)              # from MRHIER
    (Concept)-[:DEFINED_BY {def}]->(Source)               # from MRDEF

Usage:
    python scripts/umls_to_neo4j_csv.py \\
        --meta data/datasets/umls-2025AB-metathesaurus-full1/2025AB/META \\
        --out  data/neo4j_import \\
        --english-only --drop-suppressed [--workers N]

Then bulk-load (Neo4j 5 syntax):
    neo4j-admin database import full \\
        --nodes=/import/concepts.csv \\
        --nodes=/import/semantic_types.csv \\
        --nodes=/import/sources.csv \\
        --relationships=/import/concept_semtype.csv \\
        --relationships=/import/concept_relates.csv \\
        --relationships=/import/concept_parent.csv \\
        --relationships=/import/concept_definition.csv \\
        neo4j
"""

import argparse
import csv
import multiprocessing as mp
import os
import shutil
import time
from pathlib import Path

MRCONSO_COLS = ["CUI", "LAT", "TS", "LUI", "STT", "SUI", "ISPREF", "AUI",
                "SAUI", "SCUI", "SDUI", "SAB", "TTY", "CODE", "STR", "SRL",
                "SUPPRESS", "CVF"]
MRREL_COLS = ["CUI1", "AUI1", "STYPE1", "REL", "CUI2", "AUI2", "STYPE2",
              "RELA", "RUI", "SRUI", "SAB", "SL", "RG", "DIR", "SUPPRESS",
              "CVF"]
MRSTY_COLS = ["CUI", "TUI", "STN", "STY", "ATUI", "CVF"]
MRDEF_COLS = ["CUI", "AUI", "ATUI", "SATUI", "SAB", "DEF", "SUPPRESS", "CVF"]
MRSAB_COLS = ["VCUI", "RCUI", "VSAB", "RSAB", "SON", "SF", "SVER", "VSTART",
              "VEND", "IMETA", "RMETA", "SLC", "SCC", "SRL", "TFR", "CFR",
              "CXTY", "TTYL", "ATNL", "LAT", "CENC", "CURVER", "SABIN",
              "SSN", "SCIT"]
MRHIER_COLS = ["CUI", "AUI", "CXN", "PAUI", "SAB", "RELA", "PTR", "HCD", "CVF"]

SUPPRESSED = {"O", "E", "Y"}

# Shared state populated in main() before forking each pool. Workers on
# Linux inherit these via fork(); no pickling is needed.
_concepts: set = set()
_sources: set = set()
_aui_to_cui: dict = {}


# ---------------------------------------------------------------------------
# Byte-range chunking
# ---------------------------------------------------------------------------

def split_offsets(path: Path, n: int) -> list[tuple[int, int]]:
    """Return n (start, end) byte offsets evenly covering the file.

    Workers adjust on their own to skip any partial leading line and to
    stop once their pre-read position exceeds `end` (see iter_rrf_range).
    """
    size = path.stat().st_size
    if size == 0 or n <= 1:
        return [(0, size)]
    step = size // n
    return [(i * step, size if i == n - 1 else (i + 1) * step)
            for i in range(n)]


def iter_rrf_range(path: Path, cols: list[str], start: int, end: int):
    """Yield (fields, col_index) for rows whose start byte is in [start, end].

    Standard MapReduce-style split: if start > 0, skip the (partial) first
    line — it belongs to the previous chunk. Then read while the
    before-readline position is <= end, which means the chunk that owns a
    line's starting byte reads the whole line even if it spills past `end`.
    """
    idx = {c: i for i, c in enumerate(cols)}
    with path.open("rb") as f:
        f.seek(start)
        if start > 0:
            f.readline()
        while True:
            pos = f.tell()
            if pos > end:
                break
            line = f.readline()
            if not line:
                break
            parts = line.decode("utf-8").rstrip("\n").split("|")
            yield parts, idx


# ---------------------------------------------------------------------------
# Per-chunk workers (module-level so fork inheritance sees globals)
# ---------------------------------------------------------------------------

def _chunk_concepts(args):
    meta, start, end, part, english_only, drop_suppressed = args
    seen = set()
    with part.open("w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        for fields, ix in iter_rrf_range(meta / "MRCONSO.RRF",
                                         MRCONSO_COLS, start, end):
            cui = fields[ix["CUI"]]
            if cui in seen:
                continue
            if english_only and fields[ix["LAT"]] != "ENG":
                continue
            if drop_suppressed and fields[ix["SUPPRESS"]] in SUPPRESSED:
                continue
            w.writerow([cui, fields[ix["STR"]], "Concept"])
            seen.add(cui)
    return part


def _chunk_semtypes(args):
    meta, start, end, node_part, rel_part = args
    local_tuis = set()
    with node_part.open("w", encoding="utf-8", newline="") as nf, \
            rel_part.open("w", encoding="utf-8", newline="") as rf:
        nw = csv.writer(nf)
        rw = csv.writer(rf)
        for fields, ix in iter_rrf_range(meta / "MRSTY.RRF",
                                         MRSTY_COLS, start, end):
            cui = fields[ix["CUI"]]
            tui = fields[ix["TUI"]]
            if cui not in _concepts:
                continue
            if tui not in local_tuis:
                local_tuis.add(tui)
                nw.writerow([tui, fields[ix["STY"]], fields[ix["STN"]],
                             "SemanticType"])
            rw.writerow([cui, tui, "HAS_SEMTYPE"])
    return node_part, rel_part


def _chunk_rels(args):
    meta, start, end, part, drop_suppressed = args
    with part.open("w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        for fields, ix in iter_rrf_range(meta / "MRREL.RRF",
                                         MRREL_COLS, start, end):
            c1 = fields[ix["CUI1"]]
            c2 = fields[ix["CUI2"]]
            if not c1 or not c2 or c1 == c2:
                continue
            if c1 not in _concepts or c2 not in _concepts:
                continue
            if drop_suppressed and fields[ix["SUPPRESS"]] in SUPPRESSED:
                continue
            w.writerow([c1, c2, fields[ix["REL"]], fields[ix["RELA"]],
                        fields[ix["SAB"]], "RELATES"])
    return part


def _chunk_aui_cui(args):
    """Partial AUI -> CUI map for a byte range of MRCONSO."""
    meta, start, end = args
    out = {}
    for fields, ix in iter_rrf_range(meta / "MRCONSO.RRF",
                                     MRCONSO_COLS, start, end):
        cui = fields[ix["CUI"]]
        if cui not in _concepts:
            continue
        out[fields[ix["AUI"]]] = cui
    return out


def _chunk_hier(args):
    meta, start, end, part = args
    with part.open("w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        for fields, ix in iter_rrf_range(meta / "MRHIER.RRF",
                                         MRHIER_COLS, start, end):
            paui = fields[ix["PAUI"]]
            if not paui:
                continue
            child = _aui_to_cui.get(fields[ix["AUI"]])
            parent = _aui_to_cui.get(paui)
            if not child or not parent or child == parent:
                continue
            w.writerow([child, parent, fields[ix["SAB"]],
                        fields[ix["RELA"]], "IS_A"])
    return part


def _chunk_defs(args):
    meta, start, end, part, drop_suppressed = args
    with part.open("w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        for fields, ix in iter_rrf_range(meta / "MRDEF.RRF",
                                         MRDEF_COLS, start, end):
            cui = fields[ix["CUI"]]
            sab = fields[ix["SAB"]]
            if cui not in _concepts or sab not in _sources:
                continue
            if drop_suppressed and fields[ix["SUPPRESS"]] in SUPPRESSED:
                continue
            w.writerow([cui, sab, fields[ix["DEF"]], "DEFINED_BY"])
    return part


# ---------------------------------------------------------------------------
# Part concat / dedup
# ---------------------------------------------------------------------------

def concat_parts(parts: list[Path], final: Path, header: list[str]) -> int:
    """Prepend header, then concat each part verbatim. Delete parts."""
    rows = 0
    with final.open("w", encoding="utf-8", newline="") as out:
        w = csv.writer(out)
        w.writerow(header)
        for p in parts:
            with p.open("r", encoding="utf-8", newline="") as f:
                for line in f:
                    out.write(line)
                    rows += 1
            p.unlink()
    return rows


def dedup_concat_parts(parts: list[Path], final: Path, header: list[str],
                       id_col: int = 0) -> tuple[int, set]:
    """Concat parts with streaming dedup on id_col. Returns (row_count, seen set)."""
    seen: set = set()
    rows = 0
    with final.open("w", encoding="utf-8", newline="") as out:
        w = csv.writer(out)
        w.writerow(header)
        for p in parts:
            with p.open("r", encoding="utf-8", newline="") as f:
                for row in csv.reader(f):
                    key = row[id_col]
                    if key in seen:
                        continue
                    seen.add(key)
                    w.writerow(row)
                    rows += 1
            p.unlink()
    return rows, seen


# ---------------------------------------------------------------------------
# Serial writer (MRSAB is small, not worth chunking)
# ---------------------------------------------------------------------------

def write_sources_serial(meta: Path, out: Path) -> set:
    dest = out / "sources.csv"
    seen = set()
    with dest.open("w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow(["sab:ID(Source)", "name", "version", ":LABEL"])
        with (meta / "MRSAB.RRF").open("r", encoding="utf-8") as rf:
            for line in rf:
                fields = line.rstrip("\n").split("|")
                sab = fields[3]  # RSAB
                if not sab or sab in seen:
                    continue
                w.writerow([sab, fields[4], fields[6], "Source"])  # SON, SVER
                seen.add(sab)
    return seen


# ---------------------------------------------------------------------------
# Phase runners
# ---------------------------------------------------------------------------

def run_pool(ctx, workers: int, fn, tasks):
    with ctx.Pool(workers) as p:
        return p.map(fn, tasks)


def phase_concepts(meta, out, parts_dir, workers, ctx, english_only, drop_suppressed):
    global _concepts
    t0 = time.monotonic()
    offsets = split_offsets(meta / "MRCONSO.RRF", workers)
    tasks = [(meta, s, e, parts_dir / f"concepts.{i}.csv",
              english_only, drop_suppressed)
             for i, (s, e) in enumerate(offsets)]
    parts = run_pool(ctx, workers, _chunk_concepts, tasks)
    rows, seen = dedup_concat_parts(
        parts, out / "concepts.csv",
        ["cui:ID(Concept)", "name", ":LABEL"])
    _concepts = seen
    print(f"concepts.csv            {rows:>12,} rows  "
          f"[{time.monotonic()-t0:.1f}s]")


def phase_semtypes(meta, out, parts_dir, workers, ctx):
    t0 = time.monotonic()
    offsets = split_offsets(meta / "MRSTY.RRF", workers)
    tasks = [(meta, s, e,
              parts_dir / f"semtypes_nodes.{i}.csv",
              parts_dir / f"semtypes_rels.{i}.csv")
             for i, (s, e) in enumerate(offsets)]
    results = run_pool(ctx, workers, _chunk_semtypes, tasks)
    node_parts = [r[0] for r in results]
    rel_parts = [r[1] for r in results]
    node_rows, _ = dedup_concat_parts(
        node_parts, out / "semantic_types.csv",
        ["tui:ID(SemanticType)", "name", "tree_number", ":LABEL"])
    rel_rows = concat_parts(
        rel_parts, out / "concept_semtype.csv",
        [":START_ID(Concept)", ":END_ID(SemanticType)", ":TYPE"])
    print(f"semantic_types.csv      {node_rows:>12,} rows  "
          f"[{time.monotonic()-t0:.1f}s]")
    print(f"concept_semtype.csv     {rel_rows:>12,} rows")


def phase_relationships(meta, out, parts_dir, workers, ctx, drop_suppressed):
    t0 = time.monotonic()
    offsets = split_offsets(meta / "MRREL.RRF", workers)
    tasks = [(meta, s, e, parts_dir / f"rels.{i}.csv", drop_suppressed)
             for i, (s, e) in enumerate(offsets)]
    parts = run_pool(ctx, workers, _chunk_rels, tasks)
    rows = concat_parts(
        parts, out / "concept_relates.csv",
        [":START_ID(Concept)", ":END_ID(Concept)",
         "rel", "rela", "sab", ":TYPE"])
    print(f"concept_relates.csv     {rows:>12,} rows  "
          f"[{time.monotonic()-t0:.1f}s]")


def phase_hierarchy(meta, out, parts_dir, workers, ctx):
    global _aui_to_cui
    t0 = time.monotonic()

    # 1) Build AUI -> CUI in parallel by chunking MRCONSO again, then merge.
    offsets = split_offsets(meta / "MRCONSO.RRF", workers)
    tasks = [(meta, s, e) for s, e in offsets]
    maps = run_pool(ctx, workers, _chunk_aui_cui, tasks)
    merged: dict = {}
    for m in maps:
        merged.update(m)
    _aui_to_cui = merged
    idx_t = time.monotonic() - t0
    print(f"  AUI -> CUI index: {len(_aui_to_cui):,} entries  [{idx_t:.1f}s]")

    # 2) Chunk MRHIER with the shared map.
    offsets = split_offsets(meta / "MRHIER.RRF", workers)
    tasks = [(meta, s, e, parts_dir / f"hier.{i}.csv")
             for i, (s, e) in enumerate(offsets)]
    parts = run_pool(ctx, workers, _chunk_hier, tasks)
    rows = concat_parts(
        parts, out / "concept_parent.csv",
        [":START_ID(Concept)", ":END_ID(Concept)",
         "sab", "rela", ":TYPE"])
    print(f"concept_parent.csv      {rows:>12,} rows  "
          f"[{time.monotonic()-t0:.1f}s total]")


def phase_definitions(meta, out, parts_dir, workers, ctx, drop_suppressed):
    t0 = time.monotonic()
    offsets = split_offsets(meta / "MRDEF.RRF", workers)
    tasks = [(meta, s, e, parts_dir / f"defs.{i}.csv", drop_suppressed)
             for i, (s, e) in enumerate(offsets)]
    parts = run_pool(ctx, workers, _chunk_defs, tasks)
    rows = concat_parts(
        parts, out / "concept_definition.csv",
        [":START_ID(Concept)", ":END_ID(Source)", "def", ":TYPE"])
    print(f"concept_definition.csv  {rows:>12,} rows  "
          f"[{time.monotonic()-t0:.1f}s]")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    global _sources
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--meta", type=Path, required=True,
                    help="UMLS META directory (contains MRCONSO.RRF etc.)")
    ap.add_argument("--out", type=Path, required=True,
                    help="output directory for CSVs")
    ap.add_argument("--english-only", action="store_true",
                    help="keep only LAT=ENG concepts")
    ap.add_argument("--drop-suppressed", action="store_true",
                    help="drop rows with SUPPRESS in {O,E,Y}")
    ap.add_argument("--workers", type=int,
                    default=os.cpu_count() or 4,
                    help="parallel workers per file (default: CPU count)")
    args = ap.parse_args()

    if not args.meta.is_dir():
        raise SystemExit(f"META dir not found: {args.meta}")
    for required in ("MRCONSO.RRF", "MRREL.RRF", "MRSTY.RRF",
                     "MRDEF.RRF", "MRSAB.RRF", "MRHIER.RRF"):
        if not (args.meta / required).is_file():
            raise SystemExit(f"missing {required} in {args.meta}")

    args.out.mkdir(parents=True, exist_ok=True)
    parts_dir = args.out / ".parts"
    if parts_dir.exists():
        shutil.rmtree(parts_dir)
    parts_dir.mkdir()

    ctx = mp.get_context("fork")
    workers = max(1, args.workers)
    t_total = time.monotonic()

    print(f"workers={workers} meta={args.meta} out={args.out}")

    # Phase 1: concepts — must come first; populates _concepts.
    phase_concepts(args.meta, args.out, parts_dir, workers, ctx,
                   args.english_only, args.drop_suppressed)

    # Phase 2: sources — small file, serial; populates _sources.
    _sources = write_sources_serial(args.meta, args.out)
    print(f"sources.csv             {len(_sources):>12,} rows")

    # Phase 3: semantic types.
    phase_semtypes(args.meta, args.out, parts_dir, workers, ctx)

    # Phase 4: relationships (MRREL).
    phase_relationships(args.meta, args.out, parts_dir, workers, ctx,
                        args.drop_suppressed)

    # Phase 5: hierarchy (builds AUI->CUI, then walks MRHIER).
    phase_hierarchy(args.meta, args.out, parts_dir, workers, ctx)

    # Phase 6: definitions (MRDEF).
    phase_definitions(args.meta, args.out, parts_dir, workers, ctx,
                      args.drop_suppressed)

    # Cleanup parts dir (should already be empty of part files).
    shutil.rmtree(parts_dir, ignore_errors=True)

    print(f"\nCSVs written to {args.out.resolve()}  "
          f"[total {time.monotonic()-t_total:.1f}s]")


if __name__ == "__main__":
    main()
