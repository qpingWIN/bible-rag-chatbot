"""Tests for matching.py, run with pytest"""

import pytest

from eval.scoring.matching import verse_match, mhc_match


# Minimal chunk dicts representing each chunk type (verse chunk from KJV/BSB, MHC chapter chunk). Tests can modify these as needed.

def kjv_chunk(book="Mark", chapter=5, verse=6):
    return {"source": "kjv", "book": book, "chapter": chapter, "verse": verse, "chunk_index": 0}

def bsb_chunk(book="Mark", chapter=5, verse=6):
    return {"source": "bsb", "book": book, "chapter": chapter, "verse": verse, "chunk_index": 0}

def mhc_chunk(book="Lev", chapter=25, chunk_index=0):
    return {"source": "mhc", "book": book, "chapter": chapter, "verse": None, "chunk_index": chunk_index}


# verse_match tests
class TestVerseMatch:
    """verse_match: translation-agnostic verse gold"""

    def test_kjv_chunk_matches(self):
        assert verse_match("Mark.5.6", kjv_chunk())

    def test_bsb_chunk_matches(self):
        # Same verse in a different translation must still match.
        assert verse_match("Mark.5.6", bsb_chunk())

    def test_wrong_verse_rejected(self):
        assert not verse_match("Mark.5.6", kjv_chunk(verse=7))

    def test_wrong_chapter_rejected(self):
        assert not verse_match("Mark.5.6", kjv_chunk(chapter=4))

    def test_wrong_book_rejected(self):
        assert not verse_match("Mark.5.6", kjv_chunk(book="Matt"))

    def test_mhc_chunk_never_matches_verse_gold(self):
        # An MHC commentary chunk should not satisfy a verse gold even if book and chapter happen to align
        assert not verse_match("Lev.25.23", mhc_chunk())

    def test_malformed_gold_raises(self):
        with pytest.raises(ValueError, match="expects 'Book.Chapter.Verse'"):
            verse_match("Mark.5", kjv_chunk())

    def test_non_integer_chapter_raises(self):
        with pytest.raises(ValueError, match="Bad chapter/verse"):
            verse_match("Mark.X.6", kjv_chunk())



# mhc_match tests
class TestMhcMatch:
    """mhc_match: chapter-scope MHC gold, chunk_index ignored"""

    def test_mhc_chunk_matches(self):
        assert mhc_match("MHC.Lev.25", mhc_chunk())

    def test_different_chunk_index_still_matches(self):
        # Any chunk_index in the right chapter counts.
        assert mhc_match("MHC.Lev.25", mhc_chunk(chunk_index=7))

    def test_wrong_chapter_rejected(self):
        assert not mhc_match("MHC.Lev.25", mhc_chunk(chapter=24))

    def test_wrong_book_rejected(self):
        assert not mhc_match("MHC.Lev.25", mhc_chunk(book="Num"))

    def test_verse_chunk_never_matches_mhc_gold(self):
        # A KJV/BSB verse chunk should not satisfy MHC gold even if book and chapter happen to align
        assert not mhc_match("MHC.Lev.25", kjv_chunk(book="Lev", chapter=25, verse=23))

    def test_missing_mhc_prefix_raises(self):
        with pytest.raises(ValueError, match="expects 'MHC.Book.Chapter'"):
            mhc_match("Lev.25", mhc_chunk())

    def test_wrong_prefix_raises(self):
        with pytest.raises(ValueError, match="expects 'MHC.Book.Chapter'"):
            mhc_match("KJV.Lev.25", mhc_chunk())