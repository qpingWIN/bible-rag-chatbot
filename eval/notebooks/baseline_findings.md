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

**Interpretive scoring works as designed.**

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
