# Evaluation Methodology

## Overview

This evaluation suite measures retrieval performance over a corpus comprising the King James Version (KJV) Bible **and** Matthew Henry's Commentary (MHC). It contains 42 hand-labelled questions stratified across five categories, each targeting a distinct retrieval failure mode:

| Category      | n  | Tests                                                                      |
|---------------|----|----------------------------------------------------------------------------|
| Factual       | 10 | Baseline retrieval on direct facts                                         |
| Named entity  | 9  | Lexical handling across the proper-noun rarity spectrum (people + places)  |
| Narrative     | 8  | Multi-verse span coverage, sensitivity to chunking                         |
| Thematic      | 10 | Retrieval diversity on open-ended topics                                   |
| Interpretive  | 5  | Whether MHC commentary is surfaced when verses alone are interpretively incomplete |

The interpretive category specifically evaluates retrieval over MHC chunks and the other four evaluate verse retrieval.

## Sample size

42 questions reflects a deliberate tradeoff. Each question carries hand-verified gold labels (canonical verse IDs or MHC chunk IDs) and a documented construction path, making per-question labelling cost non-trivial. Per-category n is 5–10, sufficient to surface category-level failure modes when paired with bootstrap confidence intervals. The eval is designed to diagnose *where* the system fails, not to produce a single headline number.

## Question construction

Candidates were sourced as follows and verified against an authoritative source before being added to the set:

- **Factual** - Bible Hub keyword/phrase search.
- **Named entity** - well-known names plus LLM-generated candidates spanning the rarity spectrum (single-verse names like Diotrephes through cluster-frequent names like Bezalel, person names plus narrative-tied place names like Patmos and Troas), all verified via Bible Hub's concordance.
- **Narrative** - well-known stories with verse spans confirmed on Bible Hub, two LLM-generated less-famous candidates (Eutychus, Ananias and Sapphira) verified the same way.
- **Thematic** - OpenBible.info topic pages, supplemented with cross-references from a representative anchor verse.
- **Interpretive** - LLM-generated candidates, each verified against MHC's commentary on Bible Hub to confirm MHC supplies a load-bearing interpretive move the verses alone do not.

Each question's `source` field records its specific construction path. LLM-assisted candidate generation is disclosed wherever used, LLM was never used to write or approve gold labels.

Thematic questions are deliberately designed as discriminators between *clustering* and *diversifying* retrievers: gold sets span multiple canonical locations (e.g. q16 anxiety spans Phil 4 / Matt 6 / 1 Pet 5 / wisdom / prophets), so a retriever clustering on one famous anchor passes only superficially.

Interpretive questions are constructed so verse-only retrieval returns a coherent but interpretively-incomplete answer. The MHC chunk supplies the theological or other valuable interpretation (e.g. Jacob's ladder as Christ-mediator in q20, bronze serpent as cross-typology in q23). These questions are designed to fail without the commentary index.

## Gold labelling

Gold verses are stored as translation-agnostic canonical references (e.g. `Mark.5.6`), so the eval generalises across translation swaps in ablations.

For interpretive questions, gold is one MHC chunk reference at chapter scope (e.g. `MHC.Lev.25`). MHC chunking targets ~250 words per chunk so a single chapter's commentary typically produces multiple chunks but the gold label means *any* chunk derived from MHC's commentary on that chapter counts as a hit. This is chunking-invariant, it is important because chunking is itself an ablation variable but it has a known weakness: a retriever can return a chunk covering the wrong portion of the chapter's commentary and still pass the retrieval check. This is partly mitigated by the faithfulness scoring step on the generated answer (see Metrics).

## Thresholds and aggregation

Each question has a `min_recall` value: the minimum count of gold items the retriever must surface in the top-k for the question to be marked a pass.

- **Factual, named entity, interpretive** — `min_recall = 1`. The retriever either finds a relevant item or doesn't.
- **Narrative** — `min_recall` scales with gold span size, targeting roughly 50–60% recall. Narrative passes require enough span coverage for a downstream LLM to reconstruct the story.
- **Thematic** — `min_recall = 2–4` regardless of gold size, targeting roughly 30–40% recall. The threshold encodes "found N distinct thematic anchors," not "achieved high recall of the full gold set." This matches the category's purpose which is testing diversity rather than completeness.

One exception within named entity: q39 (Judas) uses `min_recall = 2` to test whether the retriever returns diverse gospel accounts of the betrayal rather than near-duplicate verses from a single passage.

A configuration is reported as passing overall only if it clears the threshold on every category, a partial pass is reported with the failing category named, preventing averaging from hiding category-specific regressions.

For categories with multi-item gold (narrative, thematic), binary pass-rate is supplemented with continuous Recall@k so that "found one verse" is distinguished from "found the full span" The threshold tells you whether the retriever found anything usable, the continuous metric tells you how completely.

## Metrics

Three classes of metric, reported at category level:

1. **Binary pass-rate** against `min_recall` thresholds. Headline category-level signal.
2. **Continuous retrieval metrics** — Recall@k and MRR to capture coverage and rank quality below the pass/fail threshold.
3. **Faithfulness** (interpretive, possibly thematic) — LLM-as-judge scoring of the generated answer against retrieved context, to catch cases where retrieval surfaced the right chunk but the answer doesn't reflect MHC's specific reading.

All metrics are reported with 95% bootstrap confidence intervals using **1,000 percentile-method resamples** with replacement. Resampling is applied to questions (not chunks or retrieval results) and is performed **within each category** when computing category-level CIs, since per-category n is the operative sample size for those numbers. With this sample size, point-estimate gaps between configurations are frequently within CI overlap, CIs are reported precisely to prevent over-interpretation of small differences in the ablation table. Per-question disagreement tables between configurations supplement aggregate metrics, since error analysis is often more diagnostic than mean differences.

## Limitations

- **Sample size** 42 questions, per-category n of 5–10. CIs are correspondingly wide, this is acknowledged in reporting rather than hidden.
- **Interpretive category n=5** MHC-verifiable candidates with structural (rather than purely applicative) commentary were the binding constraint. Acts 16 (Philippian jailer) was an initial candidate discarded after MHC verification because the commentary there is pastoral application rather than interpretive reframing.
- **Single-commentator verification** Interpretive scoping reflects MHC's specific theological tradition (Reformed, 18th-century English). A more robust setup would triangulate across multiple commentators.
- **LLM-assisted candidate generation** Used for some named-entity, two narrative, and all interpretive candidates. All verified against authoritative sources, but the candidate pool inherits whatever biases the LLM has about what counts as a "good" question.
- **Single-translation drafting** Questions phrased in modern English with gold derived from KJV-anchored Bible Hub searches, cross-translation drift on edge cases is possible.
- **Interpretive gold granularity** Chunk-level gold at chapter scope means a retriever can return a chunk on the wrong portion of MHC's chapter commentary and still pass the retrieval check. Partly mitigated by faithfulness scoring on the generated answer.
- **Not measured** Latency, cost, adversarial robustness (hallucination on out-of-distribution questions). These are separate concerns and out of scope here.

## Open decisions

The following are deliberately not pinned in this document and will be decided when the scoring harness is built:

- Retrieval k for Recall@k and threshold scoring.
- Faithfulness judge model, prompt, scoring scale (absolute vs pairwise), and human spot-check protocol.
- Aggregation rule for per-question disagreement analysis between configs.
- The ablation axis set. Candidates: embedding model, chunking strategy, dedup of synoptic and OT parallel passages, cross-reference expansion, hybrid lexical+dense retrieval, reranker.
