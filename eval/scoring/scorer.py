"""Question-level scoring: pass-rate, Recall@k, MRR"""

from eval.scoring.matching import verse_match, mhc_match

def score_question(question:dict, retrieved: list[dict]) -> dict:
    """Scores one question against a list of retrieved chunks

    Returns a dict with:
      - passed: whether the question's threshold was met
      - hits: total gold items found (deduped: each gold counted at most once)
      - verse_hits: gold verses found
      - mhc_hits: gold MHC chunks found
      - total_gold: len(gold_verses) + len(gold_commentary_chunks)
      - missed_gold: gold labels not found in retrieved

    Translation dedup is implicit since we iterate over gold (not the retrieved), 
    thus the same gold verse can only be counted once.

    For "interpretive" category, 'passed' is calculated based on mhc_hits alone.
    Verse hits are recorded in hits and contribute to recall_at_k (broader coverage signal preserved)
    """

    gold_verses = question.get("gold_verses",[])
    gold_commentary = question.get("gold_commentary_chunks",[])
    category = question.get("category","")
    min_recall = question["min_recall"]
    verse_hits = 0
    mhc_hits =0
    missed = []

    for gold in gold_verses:
        if any(verse_match(gold, chunk) for chunk in retrieved):
            verse_hits += 1
        else:
            missed.append(gold)

    for gold in gold_commentary:
        if any(mhc_match(gold, chunk) for chunk in retrieved):
            mhc_hits += 1
        else:
            missed.append(gold)

    hits = verse_hits + mhc_hits
    total_gold = len(gold_verses) + len(gold_commentary)

    #find the 1st retrieved chunk that mathces any gold (for MRR)
    #Iterating in retrival order (rank 1, 2,...)
    first_hit_rank = None
    for rank, chunk in enumerate(retrieved, start=1):
        matched = (
            any(verse_match(g, chunk) for g in gold_verses)
            or any(mhc_match(g, chunk) for g in gold_commentary)
        )
        if matched:
            first_hit_rank = rank
            break
    reciprocal_rank = 1.0 / first_hit_rank if first_hit_rank else 0.0
    recall_at_k = hits / total_gold if total_gold else 0.0
        
    if category == "interpretive":
        passed = mhc_hits >=min_recall
    else:
        passed = hits >= min_recall

    return {
        "passed": passed,
        "hits": hits,
        "verse_hits": verse_hits,
        "mhc_hits": mhc_hits,
        "total_gold": total_gold,
        "missed_gold": missed,
        "recall_at_k": recall_at_k,
        "reciprocal_rank": reciprocal_rank,
        "first_hit_rank": first_hit_rank,
    }