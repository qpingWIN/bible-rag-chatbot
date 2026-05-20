"""Gold-label matching: does a retrieved chunk count as a hit for a given gold label?
Two kinds of match:
- verse_match: "Mark.5.6" matches any chunk with that book/chapter/verse regardless of source (KJV vs BSB).
- mhc_match: "MHC.Lev.25" matches any chunk where source='mhc' and the book/chapter agree regardless of chunk_index.

Gold labels are translation-agnostic for verses (we test whether the right verse was found, not which translation surfaced it).
MHC gold is chapter-scope by design (see eval/methodology.md)
"""

def verse_match(gold_label: str, chunk: dict) -> bool:
    """Does this chunk match the gold verse reference?
    Expected gold format: "Mark.5.7"
    """
    original = gold_label
    gold_label = gold_label.strip().split(".")
    if len(gold_label) != 3:
        raise ValueError(f"verse_match expects 'Book.Chapter.Verse', got {original!r}")
    book, chapter_str, verse_str = gold_label
    
    try:
        chapter = int(chapter_str)
        verse = int(verse_str)
    except ValueError as e:
        raise ValueError(f"Bad chapter/verse in {gold_label!r}: {e}")
    
    return (
        chunk.get("book") == book
        and chunk.get("chapter") == chapter
        and chunk.get("verse") == verse
    )

def mhc_match(gold_label: str, chunk: dict) -> bool:
    """Does this chunk match the gold MHC reference?
    Expected gold format: "MHC.Lev.25"
    """
    original = gold_label
    gold_label = gold_label.strip().split(".")
    if len(gold_label) != 3 or gold_label[0] != "MHC":
        raise ValueError(f"mhc_match expects 'MHC.Book.Chapter', got {original!r}")
    
    _, book, chapter_str = gold_label
    
    try:
        chapter = int(chapter_str)
    except ValueError as e:
        raise ValueError(f"Bad chapter in {gold_label!r}: {e}")
    
    return (
        chunk.get("source") == "mhc"
        and chunk.get("book") == book
        and chunk.get("chapter") == chapter
    )
