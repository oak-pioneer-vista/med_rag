"""Parse MTSamples transcriptions into section-level records.

MTSamples encodes notes inline ("SUBJECTIVE:, ... MEDICATIONS: , ..."),
so we carve each document on ALL-CAPS "HEADING:" tokens, keeping only
headings that appear in the cleaned allowlist at data/mt_section_headings.txt
(produced by extract_mt_headings.py).

Two stages:
  1) Per-document JSON with a `sections` list of Section objects.
  2) Section JSON (one per section, flattened), ready for embedding + NER.

`doc_id` matches the row position after dropna(transcription), so it lines
up with the point ids in `ingest_mtsamples.py`.

Outputs:
  - data/mtsamples_parsed.jsonl
  - data/mtsamples_chunks.jsonl

Usage:
  python python/ingestion/parse_mtsamples.py
"""

import json
import re
from dataclasses import asdict, dataclass, field
from pathlib import Path

import pandas as pd

REPO = Path(__file__).resolve().parent.parent.parent
SRC_CSV = REPO / "data" / "kaggle" / "mtsamples" / "mtsamples.csv"
HEADINGS_FILE = REPO / "data" / "mt_section_headings.txt"
OUT_PARSED = REPO / "data" / "mtsamples_parsed.jsonl"
OUT_CHUNKS = REPO / "data" / "mtsamples_chunks.jsonl"

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
    sample_name: str
    keywords: str
    sections: list[Section] = field(default_factory=list)


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


def main() -> None:
    allowed = load_allowed_headings()
    print(f"loaded {len(allowed)} allowed headings")

    df = pd.read_csv(SRC_CSV)
    df = df.dropna(subset=["transcription"]).reset_index(drop=True)
    print(f"processing {len(df)} records")

    OUT_PARSED.parent.mkdir(parents=True, exist_ok=True)
    n_chunks = 0
    with OUT_PARSED.open("w", encoding="utf-8") as out_parsed, \
         OUT_CHUNKS.open("w", encoding="utf-8") as out_chunks:
        for doc_id, row in df.iterrows():
            specialty = str(row.get("medical_specialty") or "").strip()
            keywords = str(row.get("keywords") or "").strip()
            doc = MTSampleDoc(
                doc_id=int(doc_id),
                description=str(row.get("description") or "").strip(),
                specialty=specialty,
                specialty_cui="",
                sample_name=str(row.get("sample_name") or "").strip(),
                keywords=keywords,
                sections=parse_sections(
                    row["transcription"],
                    allowed,
                    doc_id=int(doc_id),
                    specialty=specialty,
                    specialty_cui="",
                    keywords=keywords,
                ),
            )
            out_parsed.write(json.dumps(asdict(doc), ensure_ascii=False) + "\n")

            for chunk in doc.sections:
                out_chunks.write(json.dumps(asdict(chunk), ensure_ascii=False) + "\n")
                n_chunks += 1

    print(f"wrote {len(df)} docs to {OUT_PARSED}")
    print(f"wrote {n_chunks} section chunks to {OUT_CHUNKS}")


if __name__ == "__main__":
    main()
