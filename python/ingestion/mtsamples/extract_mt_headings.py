"""Extract section headings from MTSamples transcription text.

MTSamples encodes notes inline ("PROCEDURE:, ... ANESTHESIA:, ..."),
so headings are ALL-CAPS tokens ending in `:`. We capture them, dedupe
within each document, and emit a count-ranked list.

Cleaning rules:
  - drop headings observed in fewer than MIN_DOCS documents (default 2),
    which removes one-off typos and sentence fragments
  - drop headings with more than MAX_WORDS tokens, which removes
    accidental sentence captures

Usage:
  python python/ingestion/mtsamples/extract_mt_headings.py
"""

import csv
import re
import sys
from collections import Counter
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent.parent
SRC = REPO / "data" / "kaggle" / "mtsamples" / "mtsamples.csv"
OUT = REPO / "data" / "mt_section_headings.txt"

MIN_DOCS = 2
MAX_WORDS = 8
HEADING = re.compile(r"\b([A-Z][A-Z0-9 ,/&'\-]{2,78}):")


def main() -> None:
    csv.field_size_limit(sys.maxsize)
    counts: Counter[str] = Counter()
    docs = 0
    with SRC.open(newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            t = row.get("transcription") or ""
            if not t:
                continue
            docs += 1
            seen = set()
            for m in HEADING.finditer(t):
                h = m.group(1).strip().rstrip(",").strip()
                if len(h.split()) > MAX_WORDS:
                    continue
                seen.add(h)
            for h in seen:
                counts[h] += 1

    cleaned = [(h, n) for h, n in counts.most_common() if n >= MIN_DOCS]
    OUT.parent.mkdir(parents=True, exist_ok=True)
    with OUT.open("w", encoding="utf-8") as f:
        f.write(f"# MTSamples section headings\n")
        f.write(f"# source: {SRC.relative_to(REPO)}\n")
        f.write(f"# docs scanned: {docs}\n")
        f.write(f"# rules: regex={HEADING.pattern!r} "
                f"min_docs={MIN_DOCS} max_words={MAX_WORDS}\n")
        f.write(f"# kept: {len(cleaned)} of {len(counts)} unique headings\n")
        f.write(f"#\n# count\theading\n")
        for h, n in cleaned:
            f.write(f"{n}\t{h}\n")
    print(f"wrote {len(cleaned)} headings to {OUT}")


if __name__ == "__main__":
    main()
