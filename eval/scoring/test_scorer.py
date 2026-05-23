"""Tests for scorer.py. Run with pytest"""

import pytest

from eval.scoring.scorer import score_question

#Chunk fixtures (from test_matching.py)
def kjv_chunk(book="Mark", chapter=5, verse=6):
    return {"source": "kjv", "book": book, "chapter": chapter, "verse": verse, "chunk_index": 0}

def bsb_chunk(book="Mark", chapter=5, verse=6):
    return {"source": "bsb", "book": book, "chapter": chapter, "verse": verse, "chunk_index": 0}

def mhc_chunk(book="Lev", chapter=25, chunk_index=0):
    return {"source": "mhc", "book": book, "chapter": chapter, "verse": None, "chunk_index": chunk_index}


def make_question(
    qid="q_test",
    category="factual",
    gold_verses=None,
    gold_commentary_chunks=None,
    min_recall=1,
):
    return {
        "id": qid,
        "category": category,
        "gold_verses": gold_verses or [],
        "gold_commentary_chunks": gold_commentary_chunks or [],
        "min_recall": min_recall,
    }

class TestSingleVerseGold:
    """One gold verse, min_recall =1, non-interpretive category"""

    def test_passes_when_kjv_chunk_retrieved(self):
        q = make_question(gold_verses = ["Gen.1.1"], min_recall=1)
        retrieved = [kjv_chunk(book="Gen", chapter=1, verse=1)]
        result = score_question(q, retrieved)
        assert result["passed"]
        assert result["hits"] == 1
        assert result["verse_hits"] == 1
        assert result["mhc_hits"] == 0  
        assert result["total_gold"] == 1
        assert result["missed_gold"] == []  

    def test_passes_when_bsb_chunk_retrieved(self):
        #Translation-agnostic: BSB satisfies the same gold.
        q = make_question(gold_verses=["Gen.1.1"], min_recall=1)
        retrieved = [bsb_chunk(book="Gen", chapter=1, verse=1)]
        result = score_question(q, retrieved)
        assert result["passed"]
        assert result["hits"] == 1
        assert result["verse_hits"] == 1
        assert result["mhc_hits"] == 0  
        assert result["total_gold"] == 1
        assert result["missed_gold"] == []  

    def test_fails_when_nothing_matches(self):
        q = make_question(gold_verses=["Gen.1.1"], min_recall=1)
        retrieved = [kjv_chunk(book="Mark", chapter=5, verse=6)]
        result = score_question(q, retrieved)
        assert not result["passed"]
        assert result["hits"] == 0 
        assert result["missed_gold"] == ["Gen.1.1"]

class TestTranslationDedup:
    """KJV and BSB chunks of the same verse must count as ONE gold hit"""

    def test_kjv_bsb_same_verse_counts_once(self):
        q = make_question(gold_verses=["Gen.1.1"], min_recall=1)
        retrieved = [
            kjv_chunk(book="Gen", chapter=1, verse=1),
            bsb_chunk(book="Gen", chapter=1, verse=1),
        ]
        result = score_question(q, retrieved)

        assert result["passed"]
        assert result["hits"] ==1
        assert result["verse_hits"] == 1

class TestMultiverseThreshold:
    """min_recall > 1: must find multiple gold items"""

    def test_passes_when_threshold_met(self):
        q = make_question(
            category = "narrative",
            gold_verses=["1Sam.17.38", "1Sam.17.39", "1Sam.17.40"],
            min_recall=2,
        )
        retrieved = [
            kjv_chunk(book="1Sam", chapter=17, verse=38),
            kjv_chunk(book="1Sam", chapter=17, verse=39),
        ]
        result = score_question(q, retrieved)
        assert result["passed"]
        assert result["hits"] == 2
        assert result["missed_gold"] == ["1Sam.17.40"]

    def test_fails_when_threshold_not_met(self):
        q = make_question(
            category="narrative",
            gold_verses=["1Sam.17.38", "1Sam.17.39", "1Sam.17.40"],
            min_recall=2,
        )
        retrieved = [kjv_chunk(book="1Sam", chapter=17, verse=38)]
        result = score_question(q, retrieved)
        assert not result["passed"]
        assert result["hits"] == 1

class TestChapterDiversity:
    """q39 Judas case: min_recall = 2 across different gold items"""

    def test_two_matt_verses_count_as_two_hits(self):
        q = make_question(
            category="named_entity",
            gold_verses=["Matt.26.14", "Matt.26.15", "Mark.14.10"],
            min_recall=2,
        )
        retrieved = [
            kjv_chunk(book="Matt", chapter=26, verse=14),
            kjv_chunk(book="Matt", chapter=26, verse=15),
        ]
        result = score_question(q, retrieved)

        assert result["passed"]
        assert result["hits"] == 2

class TestInterpretiveCategory:
    """For interpretive questions, only MHC gold counts toward the threshold"""

    def test_passes_when_mhc_chunk_retrieved(self):
        q = make_question(
            category="interpretive",
            gold_verses=["Lev.25.23", "Lev.25.25"],
            gold_commentary_chunks=["MHC.Lev.25"],
            min_recall=1,
        )
        retrieved = [mhc_chunk(book="Lev", chapter=25, chunk_index=3)]
        result = score_question(q, retrieved)
        assert result["passed"]
        assert result["mhc_hits"] == 1
    
    def test_fails_when_only_verses_retrieved(self):
        """Verse hits should not help if category is interpretive"""

        q = make_question(
            category="interpretive",
            gold_verses=["Lev.25.23", "Lev.25.25"],
            gold_commentary_chunks=["MHC.Lev.25"],
            min_recall=1,
        )
        retrieved = [
            kjv_chunk(book="Lev", chapter=25, verse=23),
            kjv_chunk(book="Lev", chapter=25, verse=25),
        ]
        result = score_question(q, retrieved)
        assert not result["passed"]
        assert result["verse_hits"] == 2
        assert result["mhc_hits"] == 0
        assert result["hits"] == 2  # diagnostic info still tracked

    def test_verse_hits_still_tracked_when_mhc_passes(self):
        """MHC found AND verses found, passes and both counts visible"""

        q = make_question(
            category="interpretive",
            gold_verses=["Lev.25.23"],
            gold_commentary_chunks=["MHC.Lev.25"],
            min_recall=1,
        )
        retrieved = [
            kjv_chunk(book="Lev", chapter=25, verse=23),
            mhc_chunk(book="Lev", chapter=25),
        ]
        result = score_question(q, retrieved)
        assert result["passed"]
        assert result["verse_hits"] == 1
        assert result["mhc_hits"] == 1
        assert result["hits"] == 2

    def test_interpretive_with_higher_min_recall(self):
        """Generalisation: if min_recall=2 on an interpretive question with two MHC gold chunks then both must be found to pass (irrelevant 
        for the constructed questions but essential for potential future extensions of the methodology)"""

        q = make_question(
            category="interpretive",
            gold_verses=[],
            gold_commentary_chunks=["MHC.Lev.25", "MHC.Gen.28"],
            min_recall=2,
        )
        retrieved = [
            mhc_chunk(book="Lev", chapter=25),
            mhc_chunk(book="Gen", chapter=28),
        ]
        result = score_question(q, retrieved)
        assert result["passed"]
        assert result["mhc_hits"] == 2

class TestNonInterpretiveWithMhcGold:
    """Non-interpretive categories with MHC gold should still count both (irrelevant 
       for the constructed questions but essential for potential future extensions of the methodology)
    """

    def test_non_interpretive_counts_mhc_toward_threshold(self):
        q = make_question(
            category="thematic",  # not interpretive
            gold_verses=["Lev.25.23"],
            gold_commentary_chunks=["MHC.Lev.25"],
            min_recall=2,
        )
        retrieved = [
            kjv_chunk(book="Lev", chapter=25, verse=23),
            mhc_chunk(book="Lev", chapter=25),
        ]
        result = score_question(q, retrieved)
        assert result["passed"]  
        assert result["hits"] == 2

class TestEmptyCases:
    """Edge cases around empty inputs"""

    def test_empty_retrieved_fails(self):
        q = make_question(gold_verses=["Gen.1.1"], min_recall=1)
        result = score_question(q, [])
        assert not result["passed"]
        assert result["hits"] == 0
        assert result["missed_gold"] == ["Gen.1.1"]

    def test_empty_retrieved_on_interpretive(self):
        # Empty retrieved on interpretive fails because mhc_hits=0.
        q = make_question(
            category="interpretive",
            gold_verses=["Lev.25.23"],
            gold_commentary_chunks=["MHC.Lev.25"],
            min_recall=1,
        )
        result = score_question(q, [])
        assert not result["passed"]
        assert result["hits"] == 0
        assert result["mhc_hits"] == 0

class TestCtsMetrics:
    """Recall@k and MRR computed alongside pass/fail"""

    def test_recall_full_when_all_gold_found(self):
        q = make_question(
            gold_verses=["Gen.1.1", "Gen.1.2"],
            min_recall=1,
        )
        retrieved = [
            kjv_chunk(book="Gen", chapter=1, verse=1),
            kjv_chunk(book="Gen", chapter=1, verse=2),
        ]
        result = score_question(q, retrieved)
        assert result["recall_at_k"] == 1.0

    def test_recall_partial(self):
        q = make_question(
            gold_verses=["Gen.1.1", "Gen.1.2", "Gen.1.3"],
            min_recall=1,
        )
        retrieved = [kjv_chunk(book="Gen", chapter=1, verse=1)]
        result = score_question(q, retrieved)
        assert result["recall_at_k"] == 1/3

    def test_recall_zero_when_nothing_matches(self):
        q = make_question(gold_verses=["Gen.1.1"], min_recall=1)
        retrieved = [kjv_chunk(book="Mark", chapter=5, verse=6)]
        result = score_question(q, retrieved)
        assert result["recall_at_k"] == 0.0

    def test_recall_translation_dedup(self):
        # Two translations of the same gold verse should give recall=1.0, not 2.0
        q = make_question(gold_verses=["Gen.1.1"], min_recall=1)
        retrieved = [
            kjv_chunk(book="Gen", chapter=1, verse=1),
            bsb_chunk(book="Gen", chapter=1, verse=1),
        ]
        result = score_question(q, retrieved)
        assert result["recall_at_k"] == 1.0

    def test_mrr_gold_at_rank_one(self):
        q = make_question(gold_verses=["Gen.1.1"], min_recall=1)
        retrieved = [kjv_chunk(book="Gen", chapter=1, verse=1)]
        result = score_question(q, retrieved)
        assert result["first_hit_rank"] == 1
        assert result["reciprocal_rank"] == 1.0

    def test_mrr_gold_at_rank_three(self):
        q = make_question(gold_verses=["Gen.1.5"], min_recall=1)
        retrieved = [
            kjv_chunk(book="Mark", chapter=5, verse=6),
            kjv_chunk(book="John", chapter=1, verse=1),
            kjv_chunk(book="Gen", chapter=1, verse=5),
        ]
        result = score_question(q, retrieved)
        assert result["first_hit_rank"] == 3
        assert result["reciprocal_rank"] == 1/3

    def test_mrr_zero_when_no_gold_retrieved(self):
        q = make_question(gold_verses=["Gen.1.1"], min_recall=1)
        retrieved = [kjv_chunk(book="Mark", chapter=5, verse=6)]
        result = score_question(q, retrieved)
        assert result["first_hit_rank"] is None
        assert result["reciprocal_rank"] == 0.0

    def test_mrr_takes_earliest_of_multiple_gold(self):
        # Gold A and B in the set. A appears at rank 2, B at rank 4
        # MRR should use rank 2 (the earliest gold hit)
        q = make_question(
            gold_verses=["Gen.1.1", "Gen.1.2"],
            min_recall=1,
        )
        retrieved = [
            kjv_chunk(book="Mark", chapter=5, verse=6),
            kjv_chunk(book="Gen", chapter=1, verse=2),    #rank 2
            kjv_chunk(book="John", chapter=1, verse=1),
            kjv_chunk(book="Gen", chapter=1, verse=1),
        ]
        result = score_question(q, retrieved)
        assert result["first_hit_rank"] == 2
        assert result["reciprocal_rank"] == 0.5

    def test_mrr_uses_mhc_for_interpretive_too(self):
        # Interpretive scoring: 'passed' uses mhc_hits only, but MRR uses any gold (verse or MHC).
        # An MHC chunk at rank 1 should give reciprocal_rank=1.0
        q = make_question(
            category="interpretive",
            gold_verses=["Lev.25.23"],
            gold_commentary_chunks=["MHC.Lev.25"],
            min_recall=1,
        )
        retrieved = [
            mhc_chunk(book="Lev", chapter=25),
            kjv_chunk(book="Lev", chapter=25, verse=23),
        ]
        result = score_question(q, retrieved)
        assert result["first_hit_rank"] == 1
        assert result["reciprocal_rank"] == 1.0