"""Parse cleaned MTSamples records into per-doc MTSampleDoc JSON files.

MTSamples encodes notes inline ("SUBJECTIVE:, ... MEDICATIONS: , ..."),
so we carve each document on ALL-CAPS "HEADING:" tokens, keeping only
headings that appear in the cleaned allowlist at data/mt_section_headings.txt
(produced by extract_mt_headings.py). Rows are streamed through a
`dask.bag` partitioned across worker processes; each worker loads config
once per partition and writes one intermediate JSON doc under
data/mtsamples_docs/ so downstream stages (embedding, NER, indexing)
can fan out per file.

Input is the cleaned+deduped JSONL produced by clean_mtsamples.py;
`doc_id` is the row position within that file.

Outputs:
  - data/mtsamples_docs/{doc_id:04d}.json  (one file per MTSampleDoc)

Usage:
  python python/ingestion/mtsamples/clean_mtsamples.py    # one-time prereq
  python python/ingestion/mtsamples/parse_mtsamples.py [--workers 16]
"""

import argparse
import json
import re
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Iterable

import dask.bag as db

REPO = Path(__file__).resolve().parent.parent.parent.parent
SRC_JSONL = REPO / "data" / "mtsamples_clean.jsonl"
HEADINGS_FILE = REPO / "data" / "mt_section_headings.txt"
SPECIALTY_CUI_FILE = REPO / "data" / "specialty_cui.json"
DOCTYPE_CUI_FILE = REPO / "data" / "doctype_cui.json"
OUT_DIR = REPO / "data" / "mtsamples_docs"

HEADING_RE = re.compile(r"\b([A-Z][A-Z0-9 ,/&'\-]{2,78}):")
MAX_WORDS = 8


@dataclass
class Section:
    chunk_id: str           # unique ID
    doc_id: int             # parent document
    section_type: str       # normalized heading ("HPI", "MEDICATIONS", etc.)
    section_cui: str        # UMLS CUI for this section type
    specialty: str          # from MTSamples metadata
    specialty_cui: str      # UMLS CUI for the specialty
    text: str               # section content
    keywords: str           # from MTSamples metadata
    embedding: list[float] = field(default_factory=list)   # populated during embedding step
    entities: list[dict] = field(default_factory=list)     # populated during NER step


@dataclass
class MTSampleDoc:
    doc_id: int
    description: str
    specialty: str
    specialty_cui: str
    doctype_cui: str
    sample_name: str
    keywords: str
    # Cross-filing: MTSamples ships many notes under multiple medical_specialty
    # tags with byte-identical `transcription`. We dedupe on transcription
    # content and record the dropped rows' specialty/CUI here so downstream
    # specialty filters still match cross-filed notes.
    alt_specialties: list[dict] = field(default_factory=list)
    sections: list[Section] = field(default_factory=list)


def load_specialty_cui() -> dict[str, str]:
    raw = json.loads(SPECIALTY_CUI_FILE.read_text(encoding="utf-8"))
    return {k: v for k, v in raw.items() if not k.startswith("_")}


def load_doctype_config() -> tuple[dict[str, str], list[dict]]:
    raw = json.loads(DOCTYPE_CUI_FILE.read_text(encoding="utf-8"))
    explicit = raw.get("explicit_by_specialty", {})
    rules = [
        {"cui": r["cui"], "any_of": set(r["any_of"]), "min_hits": int(r.get("min_hits", 1))}
        for r in raw.get("rules", [])
    ]
    return explicit, rules


def infer_doctype_cui(
    specialty: str,
    section_types: set[str],
    explicit: dict[str, str],
    rules: list[dict],
) -> str:
    """Return a doctype CUI.

    Priority: explicit specialty->doctype mapping (the 8 MTSamples report
    categories) beats heuristic rules over section headings. Rules fire in
    declared order; first rule whose `any_of` overlaps `section_types` by
    at least `min_hits` wins.
    """
    cui = explicit.get(specialty)
    if cui:
        return cui
    for rule in rules:
        if len(rule["any_of"] & section_types) >= rule["min_hits"]:
            return rule["cui"]
    return ""


def load_allowed_headings() -> set[str]:
    allowed: set[str] = set()
    with HEADINGS_FILE.open(encoding="utf-8") as f:
        for line in f:
            line = line.rstrip("\n")
            if not line or line.startswith("#"):
                continue
            _, _, h = line.partition("\t")
            h = h.strip()
            if h:
                allowed.add(h)
    return allowed


def slugify(h: str) -> str:
    return re.sub(r"[^A-Za-z0-9]+", "_", h).strip("_").lower()


def parse_sections(
    text: str,
    allowed: set[str],
    *,
    doc_id: int,
    specialty: str,
    specialty_cui: str,
    keywords: str,
) -> list[Section]:
    """Split a transcription on allowlisted ALL-CAPS headings.

    Finds every "HEADING:" match, keeps those in the allowlist, then carves
    the body between successive matches. Duplicate headings in one doc get
    their bodies joined with a newline. Returns one Section per heading.
    """
    matches: list[tuple[int, int, str]] = []
    for m in HEADING_RE.finditer(text):
        h = m.group(1).strip().rstrip(",").strip()
        if len(h.split()) > MAX_WORDS or h not in allowed:
            continue
        matches.append((m.start(), m.end(), h))

    bodies: dict[str, str] = {}
    order: list[str] = []
    for i, (_, end, h) in enumerate(matches):
        body_end = matches[i + 1][0] if i + 1 < len(matches) else len(text)
        body = text[end:body_end].lstrip(" ,\t\n").strip().rstrip(",").strip()
        if not body:
            continue
        if h in bodies:
            bodies[h] += "\n" + body
        else:
            bodies[h] = body
            order.append(h)

    return [
        Section(
            chunk_id=f"{doc_id}:{slugify(h)}",
            doc_id=doc_id,
            section_type=h,
            section_cui="",
            specialty=specialty,
            specialty_cui=specialty_cui,
            text=bodies[h],
            keywords=keywords,
        )
        for h in order
    ]


def _process_one(
    task: tuple[int, dict],
    allowed: set[str],
    specialty_cui_map: dict[str, str],
    doctype_explicit: dict[str, str],
    doctype_rules: list[dict],
) -> tuple[int, str, str, bool]:
    """Parse one CSV row and write its MTSampleDoc JSON file.

    Returns (doc_id, specialty, doctype_cui, specialty_unmapped) so the
    driver can aggregate stats without re-reading the written files.
    """
    doc_id, row = task
    specialty = str(row.get("medical_specialty") or "").strip()
    keywords = str(row.get("keywords") or "").strip()
    spec_cui = specialty_cui_map.get(specialty, "")
    alt_specialty_names = row.get("alt_specialties") or []
    sections = parse_sections(
        row["transcription"],
        allowed,
        doc_id=doc_id,
        specialty=specialty,
        specialty_cui=spec_cui,
        keywords=keywords,
    )
    section_types = {s.section_type for s in sections}
    doctype_cui = infer_doctype_cui(
        specialty, section_types, doctype_explicit, doctype_rules
    )
    # Carry full CUI coverage for every cross-filing: categories with an
    # empty specialty_cui (doc-type buckets like "Discharge Summary",
    # "Consult - History and Phy.") would otherwise have no CUI attached
    # to the alt record at all.
    alt_specialties = [
        {
            "specialty": alt,
            "specialty_cui": specialty_cui_map.get(alt, ""),
            "doctype_cui": infer_doctype_cui(
                alt, section_types, doctype_explicit, doctype_rules
            ),
        }
        for alt in alt_specialty_names
    ]
    doc = MTSampleDoc(
        doc_id=doc_id,
        description=str(row.get("description") or "").strip(),
        specialty=specialty,
        specialty_cui=spec_cui,
        doctype_cui=doctype_cui,
        sample_name=str(row.get("sample_name") or "").strip(),
        keywords=keywords,
        alt_specialties=alt_specialties,
        sections=sections,
    )
    path = OUT_DIR / f"{doc_id:04d}.json"
    path.write_text(json.dumps(asdict(doc), ensure_ascii=False), encoding="utf-8")
    return doc_id, specialty, doctype_cui, specialty not in specialty_cui_map


def _process_partition(partition: Iterable[tuple[int, dict]]) -> list[tuple]:
    """Dask worker entry point: load config once, process every row in this partition."""
    allowed = load_allowed_headings()
    specialty_cui_map = load_specialty_cui()
    doctype_explicit, doctype_rules = load_doctype_config()
    return [
        _process_one(task, allowed, specialty_cui_map, doctype_explicit, doctype_rules)
        for task in partition
    ]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--workers", type=int, default=16, help="dask worker processes")
    args = ap.parse_args()

    if not SRC_JSONL.exists():
        raise SystemExit(
            f"missing {SRC_JSONL} -- run python/ingestion/mtsamples/clean_mtsamples.py first"
        )

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    # doc_ids are assigned fresh each run from the cleaned-file row index, so
    # stale files from prior runs would leave a mix of old and new indices.
    stale = list(OUT_DIR.glob("*.json"))
    if stale:
        for p in stale:
            p.unlink()
        print(f"cleared {len(stale)} stale JSON files from {OUT_DIR}/")

    with SRC_JSONL.open(encoding="utf-8") as f:
        rows = [json.loads(line) for line in f if line.strip()]

    tasks = [(i, row) for i, row in enumerate(rows)]
    print(f"dispatching {len(tasks)} records across {args.workers} dask partitions -> {OUT_DIR}/")

    bag = db.from_sequence(tasks, npartitions=args.workers)
    results = bag.map_partitions(_process_partition).compute(
        scheduler="processes", num_workers=args.workers
    )

    n_doctype = sum(1 for _, _, dt, _ in results if dt)
    unmapped = sorted({s for _, s, _, u in results if u and s})
    print(f"wrote {len(results)} per-doc JSON files  (doctype_cui set on {n_doctype})")
    if unmapped:
        print(f"warning: {len(unmapped)} specialties missing from "
              f"{SPECIALTY_CUI_FILE.name}: {unmapped}")


if __name__ == "__main__":
    main()
