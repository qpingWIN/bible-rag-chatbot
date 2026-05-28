# Evaluation Methodology

## Overview

This evaluation suite measures retrieval performance over a corpus comprising the King James Version (KJV) Bible, the Berean Standard Bible (BSB), and Matthew Henry's Commentary (MHC). It contains 42 hand-labelled questions stratified across five categories, each targeting a distinct retrieval failure mode:

| Category      | n  | Tests                                                                      |
|---------------|----|----------------------------------------------------------------------------|
| Factual       | 10 | Baseline retrieval on direct facts                                         |
| Named entity  | 9  | Lexical handling across the proper-noun rarity spectrum (people + places)  |
| Narrative     | 8  | Multi-verse span coverage, sensitivity to chunking                         |
| Thematic      | 10 | Retrieval diversity on open-ended topics                                   |
| Interpretive  | 5  | Whether MHC commentary is surfaced when verses alone are interpretively incomplete |

The interpretive category specifically evaluates retrieval over MHC chunks and the other four evaluate verse retrieval.

## Sample size

42 questions reflects a deliberate tradeoff. Each question carries hand-verified gold labels and a documented construction path, making per-question labelling cost non-trivial. Per-category n is 5–10, sufficient to surface category-level failure modes when paired with bootstrap confidence intervals. The eval is designed to diagnose *where* the system fails, not to produce a single headline number.

## Question construction

Candidates were sourced as follows and verified against an authoritative source before being added to the set:

- **Factual** - Bible Hub keyword/phrase search.
- **Named entity** - well-known names plus LLM-generated candidates spanning the rarity spectrum (single-verse names like Diotrephes through cluster-frequent names like Bezalel, person names plus narrative-tied place names like Patmos and Troas), all verified via Bible Hub's concordance.
- **Narrative** - well-known stories with verse spans confirmed on Bible Hub, two LLM-generated less-famous candidates (Eutychus, Ananias and Sapphira) verified the same way.
- **Thematic** - OpenBible.info topic pages, supplemented with cross-references from a representative anchor verse.
- **Interpretive** - LLM-generated candidates, each verified against MHC's commentary on Bible Hub to confirm MHC supplies a load-bearing interpretive move the verses alone do not.

Each question's `source` field records its specific construction path. LLM-assisted candidate generation is disclosed wherever used, LLM was never used to write or approve gold labels.

Thematic questions are deliberately designed as discriminators between clustering and diversifying retrievers: gold sets span multiple canonical locations (e.g. q16 anxiety spans Phil 4 / Matt 6 / 1 Pet 5 / wisdom / prophets), so a retriever clustering on one famous anchor passes only superficially.

Interpretive questions are constructed so verse-only retrieval returns a coherent but interpretively-incomplete answer. The MHC chunk supplies the theological or other valuable interpretation (e.g. Jacob's ladder as Christ-mediator in q20, bronze serpent as cross-typology in q23). These questions are designed to fail without the commentary index.

## Gold labelling

The corpus stores each chunk with a `(source, book, chapter, verse, chunk_index)` tuple, where `source ∈ {"kjv", "bsb", "mhc"}`. Verse chunks have `verse` set and `chunk_index = 0`. MHC chunks have `verse = None` and a positional `chunk_index` within the chapter.

**Verse gold** is stored as translation-agnostic canonical references (e.g. `Mark.5.6`). At scoring time, a gold label matches any retrieved chunk where `(book, chapter, verse)` agrees, regardless of `source`. Retrieving `KJV.Mark.5.6` and retrieving `BSB.Mark.5.6` are both treated as hits because the underlying information is the same, the eval is testing whether the right verse was found, not whether a particular translation pipeline returned it. This also generalises cleanly if further translations are added later.

**MHC gold** is recorded at chapter scope (e.g. `MHC.Lev.25`) and matches any retrieved chunk where `source = "mhc"` and `(book, chapter)` agrees. This is a deliberate design choice rather than a labelling shortcut. MHC chunks are produced by a sliding sentence-window chunker (target ~250 words, 1-sentence overlap) operating on a per-chapter commentary, chunks carry a positional `chunk_index` but no verse-range metadata, since verse anchors are stripped during MHC extraction. Chunk-index gold would therefore mean "the Nth chunk of MHC's Lev 25 commentary under the current chunker settings", which would become invalid as soon as chunking parameters change, and chunking is itself a planned ablation axis. Chapter-scope gold is the only definition that survives changes to the chunker.

For interpretive questions, a faithfulness_rubric field records the specific interpretive content MHC supplies. This is the gold against which generated answers are scored by LLM-as-judge (see Metrics). Rubrics were drafted with LLM assistance and verified against MHC's commentary on Bible Hub.

The known weakness of this approach is that a retriever can return a chunk on the wrong portion of MHC's chapter commentary and still pass the retrieval check. It is addressed at a different layer rather than at the gold-label layer. See Metrics.

## Thresholds and aggregation

Each question has a `min_recall` value: the minimum count of gold items the retriever must surface in the top-k for the question to be marked a pass.

- **Factual, named entity** — `min_recall = 1`. The retriever either finds a relevant item or doesn't.
- **Interpretive** - `min_recall = 1` against MHC chunks specifically. Verse hits are not counted toward the threshold for this category since the category's purpose is testing whether the commentary index is surfaced. Finding only the verses should not count as a pass. Verse hits are still tracked in diagnostic output.
- **Narrative** — `min_recall` scales with gold span size, targeting roughly 50–60% recall. Narrative passes require enough span coverage for a downstream LLM to reconstruct the story.
- **Thematic** — `min_recall = 2–4` regardless of gold size, targeting roughly 30–40% recall. The threshold encodes "found N distinct thematic anchors" not "achieved high recall of the full gold set". This matches the category's purpose which is testing diversity rather than completeness.

One exception within named entity: q39 (Judas) uses `min_recall = 2` to test whether the retriever returns diverse gospel accounts of the betrayal rather than near-duplicate verses from a single passage.

A configuration is reported as passing overall only if it clears the threshold on every category, a partial pass is reported with the failing category named, preventing averaging from hiding category-specific regressions.

For categories with multi-item gold (narrative, thematic), binary pass-rate is supplemented with continuous Recall@k so that "found one verse" is distinguished from "found the full span". The threshold tells you whether the retriever found anything usable, the continuous metric tells you how completely.

For verse categories, deduplication is applied before scoring: if a retriever returns both `KJV.Mark.5.6` and `BSB.Mark.5.6`, this counts as one gold-verse hit, not two. Without dedup, a retriever could trivially inflate recall by returning every verse in both translations.

## Retrieval pipeline parameters

Several pipeline parameters are exposed to the eval rather than fixed at production defaults, because their optimal values are empirical questions the eval is designed to answer. These are not assumptions baked into the methodology but variables swept during the ablation:

- **Chunker** - `max_words`, `overlap_sentences` on the MHC sliding sentence-window chunker.
- **Cross-reference expansion** — `max_extra` (global cap on xref additions per query) and `max_per_seed` (per-seed contribution cap, so a single high-ranked seed cannot exhaust the xref budget on its own).
- **Retrieval `top_k`** - number of chunks retrieved before xref expansion.

Production defaults exist (`max_words=250`, `overlap_sentences=1`, `max_extra=3`, `max_per_seed=1`) but are starting points for the sweep, not fixed assumptions. The methodology commits to *how* these are evaluated (per-category metrics, bootstrap CIs, paired comparisons) and is agnostic to the specific values chosen.

## Metrics

Three classes of metric, reported at category level:

1. **Binary pass-rate** against `min_recall` thresholds. Headline category-level signal.
2. **Continuous retrieval metrics** — Recall@k and MRR to capture coverage and rank quality below the pass/fail threshold.
3. **Faithfulness** (interpretive, possibly thematic) — LLM-as-judge scoring of the generated answer against retrieved context, to catch cases where retrieval surfaced the right chunk but the answer doesn't reflect MHC's specific reading.

For the interpretive category, retrieval pass-rate is computed against MHC hits only (per the threshold rule above). A passing result means at least one MHC chunk from the correct chapter was returned, not that the specific load-bearing chunk was returned, since chapter-scope gold accepts any chunk in the chapter's commentary. Faithfulness then asks whether the answer reflects MHC's specific reading. A config that passes retrieval but fails faithfulness has surfaced a chunk from the right chapter but evidently not the right one within it.

All metrics are reported with 95% bootstrap confidence intervals using **1,000 percentile-method resamples** with replacement. Resampling is applied to questions (not chunks or retrieval results) and is performed **within each category** when computing category-level CIs, since per-category n is the operative sample size for those numbers. With this sample size, point-estimate gaps between configurations are frequently within CI overlap, CIs are reported precisely to prevent over-interpretation of small differences in the ablation table. Per-question disagreement tables between configurations supplement aggregate metrics, since error analysis is often more diagnostic than mean differences.

## Limitations

- **Sample size** 42 questions, per-category n of 5–10. CIs are correspondingly wide, this is acknowledged in reporting rather than hidden.
- **Interpretive category n=5** MHC-verifiable candidates with structural (rather than purely applicative) commentary were the binding constraint. Acts 16 (Philippian jailer) was an initial candidate discarded after MHC verification because the commentary there is pastoral application rather than interpretive reframing.
- **Single-commentator verification** Interpretive scoping reflects MHC's specific theological tradition (Reformed, 18th-century English). A more robust setup would triangulate across multiple commentators.
- **LLM-assisted candidate generation** Used for some named-entity, two narrative, and all interpretive candidates. All verified against authoritative sources, but the candidate pool inherits whatever biases the LLM has about what counts as a "good" question.
- **Question drafting against KJV phrasing** Questions were drafted in modern English with gold derived from KJV-anchored Bible Hub searches. Where a verse's wording differs materially between KJV and BSB, question phrasing may favour one over the other on edge cases. The scoring rule (translation-agnostic verse matching) is unaffected, but a retriever's surface-form sensitivity to translation may be.
- **MHC gold at chapter scope** Retrieval pass on an interpretive question confirms the right chapter surfaced, not that the specific load-bearing chunk within it was returned. Sub-chapter precision is handled at the faithfulness-scoring layer rather than at the gold-label layer, since chunking is a planned ablation axis and chunk-index gold would not survive chunker changes.
- **Thematic gold and cross-reference data are partly correlated** Thematic gold sets were constructed using OpenBible.info topic pages and cross-references from anchor verses. The cross-reference expansion step in the retrieval pipeline draws on the same family of cross-reference data. Configurations that include xref expansion will therefore show a structural advantage on thematic recall that is partly an artefact of this construction overlap rather than a generalisable retrieval improvement. The ablation should report the xref vs no-xref comparison alongside this caveat, narrative, factual, named-entity, and interpretive categories are unaffected since their gold was not derived from cross-reference data.
- **Not measured** Latency, cost, adversarial robustness (hallucination on out-of-distribution questions). These are separate concerns and out of scope here.

## Open decisions

The following are deliberately not pinned in this document and will be decided when the scoring harness is built and the ablation is run:

- Retrieval `top_k` for Recall@k and threshold scoring.
- Faithfulness judge model, prompt, scoring scale (absolute vs pairwise), and human spot-check protocol.
- Aggregation rule for per-question disagreement analysis between configs.
- Values swept for the pipeline parameters listed in *Retrieval pipeline parameters* (chunker, xref expansion, top_k).
- The full ablation axis set beyond those parameters. Candidates: embedding model, dedup of synoptic and OT parallel passages, hybrid lexical+dense retrieval, reranker.
