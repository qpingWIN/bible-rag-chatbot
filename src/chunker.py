"""
Sentence-transformer models truncate input at ~256 tokens. Long commentary
chapters are split into overlapping sentence-window chunks so every passage
is fully embedded and retrievable.
"""

import re
from dataclasses import dataclass, field
from .ingest import Document


@dataclass
class Chunk:
    text: str
    source: str        # "kjv" | "bsb" | "mhc"
    book: str
    chapter: int
    verse: int | None
    chunk_index: int
    metadata: dict = field(default_factory=dict)

    @property
    def reference(self) -> str:
        if self.verse:
            return f"{self.book} {self.chapter}:{self.verse}"
        return f"{self.book} {self.chapter} (commentary, chunk {self.chunk_index})"

    def to_dict(self) -> dict:
        return {
            "text": self.text,
            "source": self.source,
            "book": self.book,
            "chapter": self.chapter,
            "verse": self.verse,
            "chunk_index": self.chunk_index,
            "reference": self.reference,
            **self.metadata,
        }


def _split_sentences(text: str) -> list[str]:
    parts = re.split(r"(?<=[.!?])\s+", text.strip())
    return [p.strip() for p in parts if p.strip()]


def chunk_commentary(
    doc: Document,
    max_words: int = 250,
    overlap_sentences: int = 1,
) -> list[Chunk]:
    """
    Split a commentary document into overlapping sentence-window chunks.
    Slides a window across sentences until max_words is reached then steps
    forward by (window_size - overlap_sentences) to create overlap at boundaries.
    """
    sentences = _split_sentences(doc.text)
    if not sentences:
        return []

    chunks: list[Chunk] = []
    i = 0
    chunk_idx = 0

    while i < len(sentences):
        window: list[str] = []
        word_count = 0
        j = i

        while j < len(sentences) and word_count < max_words:
            word_count += len(sentences[j].split())
            window.append(sentences[j])
            j += 1

        chunk_text = " ".join(window)
        if chunk_text.strip():
            chunks.append(Chunk(
                text=chunk_text,
                source=doc.source,
                book=doc.book,
                chapter=doc.chapter,
                verse=doc.verse,
                chunk_index=chunk_idx,
            ))
            chunk_idx += 1

        step = max(1, len(window) - overlap_sentences)
        i += step

    return chunks


def chunk_verse(doc: Document) -> Chunk:
    """Bible verses are atomic, wrap as a single chunk."""
    return Chunk(
        text=doc.text,
        source=doc.source,
        book=doc.book,
        chapter=doc.chapter,
        verse=doc.verse,
        chunk_index=0,
    )


def build_all_chunks(
    kjv_docs: list[Document],
    bsb_docs: list[Document],
    mhc_docs: list[Document],
) -> list[Chunk]:
    chunks: list[Chunk] = []

    print("Chunking KJV verses...")
    for doc in kjv_docs:
        chunks.append(chunk_verse(doc))
    print(f"  {len(chunks):,} chunks")

    bsb_start = len(chunks)
    print("Chunking BSB verses...")
    for doc in bsb_docs:
        chunks.append(chunk_verse(doc))
    print(f"  {len(chunks) - bsb_start:,} chunks")

    mhc_start = len(chunks)
    print("Chunking Matthew Henry commentary...")
    for doc in mhc_docs:
        chunks.extend(chunk_commentary(doc))
    print(f"  {len(chunks) - mhc_start:,} chunks")

    print(f"\nTotal chunks: {len(chunks):,}")
    return chunks
