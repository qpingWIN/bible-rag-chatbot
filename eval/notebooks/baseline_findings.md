# Baseline findings

What the first runnable eval revealed before any ablations beyond
the baseline trio.

## Setup

- 42 hand-labelled questions across five categories
- Three configs run: `k=10 + MHC on` (baseline), `k=20 + MHC on`,
  `k=20 + MHC off`

## Headline numbers

| Category     | k=10, MHC | k=20, MHC | k=20, no MHC |
|--------------|-----------|-----------|--------------|
| Factual      | 5/10      | 7/10      | 7/10         |
| Named entity | 5/9       | 5/9       | 5/9          |
| Narrative    | 0/8       | 1/8       | 1/8          |
| Thematic     | 0/10      | 2/10      | 2/10         |
| Interpretive | 4/5       | 4/5       | 0/5          |
| **Overall**  | **14/42** | **19/42** | **15/42**    |

## What the comparisons rule in and out

**Narrative and thematic are bottlenecked, but not by what I first
expected.**

Initial hypothesis: MHC commentary chunks were eating top-k slots that
verse-fragment gold needed. Disabling MHC (`use_commentary=False`)
should have lifted narrative/thematic pass-rate. It didn't.
Identical 1/8 and 2/10 with and without MHC. The slots that opened
when MHC was disabled filled with other irrelevant content, not with
previously-missing gold verses.

The actual bottleneck: gold verses for multi-verse questions are not
being ranked into top-k at all, regardless of how the budget is
allocated. Wider retrieval (k=10 → k=20) lifted narrative from 0/8 to
1/8 and thematic from 0/10 to 2/10 (small improvement). The remaining
gold lives outside top-20.

**Interpretive scoring works as designed**

The Option B′ rule (interpretive scored against MHC hits only) is
working: with MHC enabled, 4/5 passes. Strip MHC, drops to 0/5.
Confirms the category does what the methodology says it does.

## Diagnosis

Dense-only retrieval handles short verse fragments badly against
thematic-style queries. The embedding similarity between a
modern-English question ("What does the Bible say about anxiety?")
and a single KJV verse is modest. Longer commentary chunks covering
the same topic score higher because they share more semantic surface
area. For multi-verse gold, this means most of the gold never makes
top-k.

This is a known property of dense retrieval, not a system bug. The
fix is hybrid retrieval (combine dense + BM25 lexical scoring) so
short verses with strong keyword overlap get weighted up.

## What changes next

Hybrid retrieval is added in the next chunk. Hypothesis: narrative
and thematic recall will improve materially with no regression on
factual/named-entity/interpretive. If they don't, the bottleneck is
deeper than retrieval algorithm choice (e.g. embedding model
capacity) and the next move would be testing a stronger embedding
model.


## Update: continuous metrics

Re-ran the three baseline configs with Recall@k and MRR alongside
pass-rate. The continuous metrics reveal information that pass-rate
hides.

### Mean Recall@k and MRR by category

| Category     | k=10, MHC      | k=20, MHC      | k=20, no MHC   |
|--------------|----------------|----------------|----------------|
| Factual      | 0.350 / 0.278  | 0.408 / 0.291  | 0.408 / 0.314  |
| Named entity | 0.317 / 0.456  | 0.394 / 0.456  | 0.394 / 0.458  |
| Interpretive | 0.354 / 0.607  | 0.394 / 0.607  | 0.204 / 0.467  |
| Narrative    | 0.076 / 0.304  | 0.132 / 0.304  | 0.132 / 0.310  |
| Thematic     | 0.071 / 0.210  | 0.154 / 0.228  | 0.154 / 0.282  |
| **Overall**  | 0.225 / 0.344  | 0.291 / 0.351  | 0.268 / 0.355  |

Cells show *mean Recall@k / mean MRR*.

### What the metrics add to the story

**Narrative MRR (0.30) is comparable to factual MRR (0.28-0.31)** The
first gold verse for a narrative question typically appears around
rank 3-4, meaning that the retriever isn't bad at finding some gold narrative,
it's bad at finding multiple gold narratives. The pass-rate of 0%/13%
hides this: a question scoring 1/5 gold (recall@k = 0.20) reads the
same as 0/5 in pass-rate terms but reflects a real retrieval
capability.

**Wider retrieval lifts narrative/thematic recall more than other
categories** Going k=10 → k=20 lifts narrative recall by 74% and
thematic by 117%, vs. 17-24% for the other categories. Multi-gold
questions benefit proportionally more from additional retrieval
depth because each extra slot is another chance to hit a
not-yet-found gold item. But the absolute level remains low: even at
k=20, only 13-15% of gold is retrieved on average.

**Disabling MHC has zero effect on narrative/thematic recall** 
Confirms with precision that MHC chunks and
verse-fragment gold don't compete for the same slots in these
categories. The slots freed by removing MHC fill with other
irrelevant content.

**MHC dominates interpretive ranking** Interpretive MRR is 0.61 with
MHC enabled, meaning the first gold hit is typically at rank ~1.6. Disable
MHC and MRR drops to 0.47 with recall collapsing from 0.394 to 0.204.
The retriever ranks commentary chunks above verse fragments for
interpretive-style queries. Without commentary, the gold-verse
fallback is significantly worse.

### Refined hypothesis for hybrid retrieval

The bottleneck is not first-hit rank (already reasonable across
categories). It's coverage, namely finding multiple gold items per
question. Hybrid retrieval should help most where:

1. The question has many gold items (narrative, thematic)
2. The gold items share keyword overlap with the question (BM25's
   advantage over dense)

Specifically: thematic questions like "What does the Bible say about
anxiety?" have gold scattered across the canon, each containing the
word "anxious", "worry", "afraid", or "fear". Dense retrieval treats
these as semantically equivalent but ranks them on overall similarity
to the question phrasing. BM25 would weight up specifically those
verses where the keyword appears verbatim, lifting more of them into
top-k. Same logic for narrative: q01 mentions Goliath specifically
and the gold verses contain "Goliath" verbatim. BM25 should surface
those reliably.

Predicted outcome of hybrid retrieval:
- Narrative and thematic recall@k should rise meaningfully (>=2x at
  k=10), narrowing the gap to factual/named_entity
- Factual and named_entity should hold steady or improve slightly
  (BM25 is well-suited to "find me the verse mentioning X")
- Interpretive may not benefit much as MHC chunks are already winning
  on dense similarity, BM25 wouldn't lift them further

If hybrid lifts narrative/thematic but not interpretive, that's a
sharper diagnostic than dense-only could give us.
