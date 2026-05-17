"""Load and normalise all data sources into a unified document format"""

import json, csv
from pathlib import Path
from dataclasses import dataclass, field

RAW = Path(__file__).parent.parent / "data" / "raw"


@dataclass
class Document:
    text: str
    source: str          # "kjv" | "bsb" | "mhc"
    book: str
    chapter: int
    verse: int | None    # None for commentary chunks (chapter-level)
    metadata: dict = field(default_factory=dict)

    @property
    def reference(self) -> str:
        if self.verse:
            return f"{self.book} {self.chapter}:{self.verse}"
        return f"{self.book} {self.chapter}"


def load_kjv() -> list[Document]:
    data = json.loads((RAW / "kjv.json").read_text())
    docs = []
    for book in data["books"]:
        for chapter in book["chapters"]:
            for v in chapter["verses"]:
                docs.append(Document(
                    text=v["text"].strip(),
                    source="kjv",
                    book=book["name"],
                    chapter=chapter["chapter"],
                    verse=v["verse"],
                ))
    return docs


def load_bsb() -> list[Document]:
    data = json.loads((RAW / "web.json").read_text())
    docs = []
    for book in data["books"]:
        for chapter in book["chapters"]:
            for v in chapter["verses"]:
                docs.append(Document(
                    text=v["text"].strip(),
                    source="bsb",
                    book=book["name"],
                    chapter=chapter["chapter"],
                    verse=v["verse"],
                ))
    return docs


def load_commentary() -> list[Document]:
    """Load Matthew Henry chapter-level commentary. Chunking happens in chunker.py"""
    data = json.loads((RAW / "mhc_commentary.json").read_text())
    docs = []
    for book in data:
        for ch in book["chapters"]:
            text = ch["commentary"].strip()
            if text:
                docs.append(Document(
                    text=text,
                    source="mhc",
                    book=book["book"],
                    chapter=ch["chapter"],
                    verse=None,
                ))
    return docs


def load_cross_references() -> dict[str, list[str]]:
    """
    Returns a dict mapping verse reference → list of related verse references
    Eg {"Gen.1.1": ["Ps.96.5", "Isa.40.28", ...], ...}
    """
    xrefs: dict[str, list[str]] = {}
    with open(RAW / "cross_references.txt") as f:
        reader = csv.reader(f, delimiter="\t")
        next(reader)  # skip header
        for row in reader:
            if len(row) < 2:
                continue
            src, tgt = row[0].strip(), row[1].strip()
            xrefs.setdefault(src, []).append(tgt)
    return xrefs


def load_all() -> tuple[list[Document], list[Document], list[Document], dict]:
    """Return (kjv_docs, bsb_docs, commentary_docs, cross_references)"""
    print("Loading KJV...")
    kjv = load_kjv()
    print(f"  {len(kjv):,} verses")

    print("Loading BSB...")
    bsb = load_bsb()
    print(f"  {len(bsb):,} verses")

    print("Loading Matthew Henry Commentary...")
    mhc = load_commentary()
    print(f"  {len(mhc):,} chapters (before chunking)")

    print("Loading cross-references...")
    xrefs = load_cross_references()
    print(f"  {len(xrefs):,} verse cross-reference entries")

    return kjv, bsb, mhc, xrefs
