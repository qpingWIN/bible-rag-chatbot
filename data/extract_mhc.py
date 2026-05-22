"""Extract Matthew Henry's Complete Commentary from SWORD module via diatheke"""

import subprocess, re, json
from pathlib import Path

BOOKS = [
    ("Gen", "Genesis", 50), ("Exod", "Exodus", 40), ("Lev", "Leviticus", 27),
    ("Num", "Numbers", 36), ("Deut", "Deuteronomy", 34), ("Josh", "Joshua", 24),
    ("Judg", "Judges", 21), ("Ruth", "Ruth", 4), ("1Sam", "1 Samuel", 31),
    ("2Sam", "2 Samuel", 24), ("1Kgs", "1 Kings", 22), ("2Kgs", "2 Kings", 25),
    ("1Chr", "1 Chronicles", 29), ("2Chr", "2 Chronicles", 36), ("Ezra", "Ezra", 10),
    ("Neh", "Nehemiah", 13), ("Esth", "Esther", 10), ("Job", "Job", 42),
    ("Ps", "Psalms", 150), ("Prov", "Proverbs", 31), ("Eccl", "Ecclesiastes", 12),
    ("Song", "Song of Solomon", 8), ("Isa", "Isaiah", 66), ("Jer", "Jeremiah", 52),
    ("Lam", "Lamentations", 5), ("Ezek", "Ezekiel", 48), ("Dan", "Daniel", 12),
    ("Hos", "Hosea", 14), ("Joel", "Joel", 3), ("Amos", "Amos", 9),
    ("Obad", "Obadiah", 1), ("Jonah", "Jonah", 4), ("Mic", "Micah", 7),
    ("Nah", "Nahum", 3), ("Hab", "Habakkuk", 3), ("Zeph", "Zephaniah", 3),
    ("Hag", "Haggai", 2), ("Zech", "Zechariah", 14), ("Mal", "Malachi", 4),
    ("Matt", "Matthew", 28), ("Mark", "Mark", 16), ("Luke", "Luke", 24),
    ("John", "John", 21), ("Acts", "Acts", 28), ("Rom", "Romans", 16),
    ("1Cor", "1 Corinthians", 16), ("2Cor", "2 Corinthians", 13),
    ("Gal", "Galatians", 6), ("Eph", "Ephesians", 6), ("Phil", "Philippians", 4),
    ("Col", "Colossians", 4), ("1Thess", "1 Thessalonians", 5),
    ("2Thess", "2 Thessalonians", 3), ("1Tim", "1 Timothy", 6),
    ("2Tim", "2 Timothy", 4), ("Titus", "Titus", 3), ("Phlm", "Philemon", 1),
    ("Heb", "Hebrews", 13), ("Jas", "James", 5), ("1Pet", "1 Peter", 5),
    ("2Pet", "2 Peter", 3), ("1John", "1 John", 5), ("2John", "2 John", 1),
    ("3John", "3 John", 1), ("Jude", "Jude", 1), ("Rev", "Revelation", 22),
]

TAG_RE = re.compile(r"<[^>]+>")
VERSE_RE = re.compile(r"^(\w[\w\s]*\d+:\d+):\s*")


def clean_osis(text: str) -> str:
    text = TAG_RE.sub("", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def fetch_chapter(book_abbr: str, chapter: int) -> str:
    """Fetch MHC commentary for a chapter.

    Diatheke's MHC module returns commentary structured by sections within
    a chapter. Each section's commentary is repeated once per verse in the
    section's range, distinguished only by the verse prefix. We deduplicate
    by content: each unique commentary block contributes once.

    Example: 2 Kings 5 has 4 sections (vv.1-8, 9-14, 15-19, 20-27), so
    diatheke returns 27 lines (8+6+5+8) plus a (MHC) footer. After
    deduplication we keep 4 unique commentary blocks and concatenate them.
    """
    key = f"{book_abbr} {chapter}"
    result = subprocess.run(
        ["diatheke", "-b", "MHC", "-k", key],
        capture_output=True, text=True, timeout=30
    )
    raw = result.stdout

    seen: set[str] = set()
    blocks: list[str] = []
    for line in raw.strip().splitlines():
        line = VERSE_RE.sub("", line).strip()
        if not line or line == "(MHC)":
            continue
        cleaned = clean_osis(line)
        if cleaned and cleaned not in seen:
            seen.add(cleaned)
            blocks.append(cleaned)

    return " ".join(blocks)


def main():
    out = []
    total = sum(chapters for _, _, chapters in BOOKS)
    done = 0

    for abbr, name, num_chapters in BOOKS:
        book_data = {"book": abbr, "chapters": []}
        for ch in range(1, num_chapters + 1):
            text = fetch_chapter(abbr, ch)
            if text:
                book_data["chapters"].append({"chapter": ch, "commentary": text})
            done += 1
            if done % 50 == 0:
                print(f"  {done}/{total} chapters done ({name} {ch})")
        out.append(book_data)
        print(f"✓ {name} ({num_chapters} chapters)")

    out_path = Path(__file__).parent / "raw" / "mhc_commentary.json"
    out_path.write_text(json.dumps(out, ensure_ascii=False, indent=2))
    print(f"\nSaved {sum(len(b['chapters']) for b in out)} chapters → {out_path}")


if __name__ == "__main__":
    main()
