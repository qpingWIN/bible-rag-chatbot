# Stories from this project

Five debugging and engineering stories from building the eval. Each
is a self-contained narrative useful for interview prep.

---

## Story 1: The 5x MHC duplication that took the pass-rate from 4/42 to 8/42

The first time I ran the eval against my hand-labelled gold set, it
scored 4/42 (10%). I had expected somewhere around 30-40%. Something
was clearly wrong, but the system was returning plausible-looking
retrieved chunks for most queries. The mismatch was somewhere
upstream.

I started by hand-inspecting the MHC commentary chunks. The corpus
is Matthew Henry's Commentary extracted via `diatheke`, the
command-line tool for the SWORD project. When I opened a sample
chapter's chunks to check them, I found something strange. Genesis
chapter 1's commentary was about 50,000 words. Matthew Henry was
thorough but not that thorough. I checked a few other chapters and
found the same pattern: every MHC chunk was 4-5x longer than it
should be.

The root cause was in how `diatheke` returns chapter-scope
commentary. For each chapter, it iterates verse-by-verse and at
each verse returns the *entire* chapter's commentary, with the
verse number as a marker. My extraction code was concatenating
these returns. So Genesis 1's commentary was being included once
per verse in chapter 1, giving roughly 31 copies of the same text.

The fix was to deduplicate at the section level: within a chapter,
collect only unique commentary blocks rather than concatenating
every verse's return. After the fix and an index rebuild, the eval
scored 8/42 instead of 4/42 with no other changes. Doubling the
pass-rate by fixing one extraction bug.

What I took from this: when system output looks roughly right but
the eval score is wildly low, the bug is probably in data prep,
not in retrieval. Plausible-looking chunks can be silently wrong.

---

## Story 2: The book-name mismatch that silently broke every verse match

After fixing the MHC duplication, pass-rate was 8/42 (19%). Still
too low for what I was seeing in retrieved content. When I looked
at individual failed questions, the retriever was clearly finding
the right chapters and verses. But the scorer was returning 0 hits.

The scoring matcher does this:
```python
def verse_match(gold: str, chunk: dict) -> bool:
    # gold is "Gen.1.1", chunk has book="Genesis", chapter=1, verse=1
    return canonicalize(gold.book) == chunk["book"] and ...
```

I added a print statement to dump the chunk's `book` field for the
first failing match. Output: `book="Genesis"`. Gold reference:
`Gen.1.1`. The matcher was checking `"Gen" == "Genesis"`. False.
Every verse-level gold reference was silently failing.

Tracing back: my source JSON files for KJV and BSB used different
naming conventions. KJV's source used `"I Samuel"`,
`"Revelation of John"`, full Roman-numeral names. BSB used
`"1 Samuel"`, `"Revelation"`. My gold labels used OSIS abbreviations
like `1Sam` and `Rev`. The matcher only worked when chunk.book
happened to match the gold's canonical form, which it almost never
did.

Fixed by adding a `BOOK_NAME_TO_ABBR` lookup in `ingest.py` and
calling `_normalise_book()` at chunk creation time. I made it raise
`ValueError` on unrecognised names rather than silently passing
through. The point being: future data ingestion failures should be
loud, not silent.

After the fix and another index rebuild, pass-rate was 14/42 (33%).
The data layer was finally producing correct chunks.

What I took from this: silent string mismatches between data layers
are the worst kind of bug. The retriever was doing its job, the
scorer was doing its job, but they were speaking different dialects.
Add asserts at layer boundaries. Make canonical naming an enforced
invariant, not a hope.

---

## Story 3: Continuous metrics changed the diagnosis

After the data fixes, the eval showed dense k=10 at 14/42 pass-rate.
Factual, named entity, and interpretive were reasonable (50-80%
pass). Narrative and thematic were at 0%. My first reading was "the
retriever is bad at multi-verse questions." That framing pointed at
algorithm changes (hybrid retrieval, better embeddings).

Then I added Recall@k and MRR to the scorer alongside pass-rate.
Pass-rate is binary. A question scoring 1 of 5 gold items reads the
same as a question scoring 0 of 5. Continuous metrics distinguish
them.

The numbers came back differently than I expected. Narrative MRR
was 0.30. Factual MRR was 0.28. **They were comparable.** The
first gold verse for a narrative question typically appeared around
rank 3-4, the same depth as the first gold for a factual question.

The diagnosis shifted. The retriever wasn't bad at multi-verse
questions in the way I thought. It was finding *one* of the
multiple gold verses at reasonable depth. The failure mode was
breadth, not depth. After finding the first gold verse, subsequent
ones didn't make top-k because they were lexically distinct from
each other and from the question.

This changed what experiments were worth running. The original
plan included testing better embedding models. But if first-hit
rank was already good, a stronger embedding probably wouldn't help
much. What might help was something that could surface additional
related verses given that the first one was already in the result
set. That points at xref expansion, not at embedding upgrades.

What I took from this: pass-rate compresses information that
matters for diagnosis. Adding continuous metrics took 30 minutes
to implement and changed which interventions I prioritised. If
I'd jumped straight to "swap the embedding model" based on
pass-rate alone, I'd have spent a day on the wrong fix.

---

## Story 4: The hybrid retrieval negative result

This one didn't work. Worth recording precisely because it didn't.

The argument for hybrid retrieval was solid going in. Dense
embeddings are weak at exact name matching, short documents, and
direct keyword presence. BM25 is good at exactly those things. The
two should complement each other. Combining them with Reciprocal
Rank Fusion (RRF) is the standard production approach.

I implemented BM25 with `rank_bm25`, wrote a fusion layer, smoke-
tested it. The smoke tests looked encouraging. For "Who fell asleep
during Paul's sermon?", BM25 nailed `Acts 20:9` (Eutychus) at rank
1. Dense had it outside its top 30. RRF lifted it into the fused
top-2. That was the kind of win I expected hybrid to deliver on
the eval.

Then I ran the eval and hybrid scored 14/42 with MRR 0.262. Dense
alone was 14/42 with MRR 0.344. Pass-rate matched but ranking
quality was substantially worse. Something was wrong.

Inspecting factual fails under hybrid showed a clean pattern. For
q32 ("What did the Israelites eat in the wilderness?"), gold is
`Ex.16.31` (manna). The top hybrid result was `bsb Num 9:5`.
Num 9:5 is about Passover. But it contains "Israelites" and
"wilderness" verbatim. BM25 ranked it #1 confidently. Dense had it
at rank 24, which is the right place for it. The fusion gave
BM25's confident wrong answer a strong contribution, displacing
dense's correct ranking.

The pattern repeated. BM25 was producing high-confidence wrong
answers, not just moderately confident weak matches. The Bible
corpus has ~31k verses per translation with heavy repetition of
common nouns (Israel, Lord, wilderness, people, king). For most
queries, several non-gold verses share strong keyword overlap with
the question. BM25 ranks them confidently. RRF amplifies that
confidence into the fused top-k.

I tried two tuning moves. Widening the BM25 fetch window from
top-30 to top-100 made it worse, not better. More candidates means
more chances for noise to climb. Downweighting BM25 to 25% of the
fusion weight partially recovered the regression but still didn't
match dense baseline. The pattern across α settings was monotonic:
more BM25 influence, worse results.

The honest conclusion: BM25 doesn't fit this corpus at these
settings. The corpus property that makes it hard is high
repetition of common nouns across short verses. The retrievers
compete for ranking slots, and BM25's confident wrong picks
displace dense's correct picks.

I kept the infrastructure. Both `src/bm25_retriever.py` and
`src/hybrid.py` are in the repo, configurable from the runner. The
experiment is reproducible. The production config uses dense-only
retrieval.

What I took from this: negative results need to be honest, but
they shouldn't be reported as "I tried X and X failed, the end."
The useful version is "I tried X and X failed. Investigation
showed the mechanism was Y. Y is a property of this specific
corpus, not of X in general. X would likely work on a corpus
without property Y." That framing tells the reader what I learned
about the system, not just what I tried.

---

## Story 5: I almost stopped before the project's best result

After the hybrid retrieval detour didn't pay off, I went back to
the dense baseline and tested cross-reference expansion as a
separate ablation. Three configs at k=20: no xref, conservative
(max_per_seed=1, max_extra=3), aggressive (3 / 10).

All three sat at 19/42 pass-rate. MRR was identical to three
decimal places. Recall lifted very slightly at the aggressive
setting (+0.022) but mostly in factual where retrieval was already
strong. The reasonable conclusion was that xref expansion didn't
meaningfully help.

I was ready to stop here and write up. The project had a clean
ablation table, a negative result on hybrid, the production config
was dense k=20. 19/42 felt like a defensible ceiling given the
constraints.

Then I tried one more configuration out of curiosity. Top_k=30
with much more aggressive xref (max_per_seed=10, max_extra=15).
No specific theoretical motivation, just "I haven't tried going
further on both at once."

Result: 21/42 (50%). Recall jumped from 0.291 to 0.381 (+31%).
The gains weren't concentrated in one category. Recall lifted
substantially across all five categories: factual +33%, thematic
+46%, interpretive +31%. Two pass-rate flips (factual q29 and
named entity q41).

The mechanism became clear once I looked at it. At k=20 with
conservative xref, the xrefs were largely duplicating what dense
already found in its top 20. First-degree xref neighbours of
already-retrieved chunks tend to overlap with the retrieved set
itself. The xref expansion was effectively a no-op because
"chunks similar to chunks already returned" mostly already in the
result.

At k=30 with aggressive xref, the seed pool is larger, each seed
contributes up to 10 xrefs (vs 1), and the total budget is 15
(vs 3). The xrefs reaching the final result are no longer
duplicates of dense's picks but genuinely different verses that
the embedding-based retrieval didn't surface.

What I took from this: ablation dimensions interact. "Intervention
X doesn't help at default settings" is not the same as
"intervention X doesn't help." Xref expansion at k=20 looked like
a dead end across three tests. It became the project's largest
quality lift one config later.

This is a small-sample finding (one extra run beats three previous
ones), and I'd want to be careful framing it as a general result.
But for this specific corpus and these specific questions, the
interaction is real. The lesson generalises better than the
specific numbers: when a parameter has multiple tuning knobs, the
interaction matrix matters more than any single axis.

## Story 6: A "stronger" embedding model was worse for the use case

After the production config landed at 21/42 with MiniLM, the
obvious next experiment was swapping in a stronger embedding
model. I picked `BAAI/bge-base-en-v1.5`: 110M params (vs MiniLM's
23M), 768-dim (vs 384), trained specifically for retrieval rather
than general semantic similarity. Applied with BGE's recommended
instruction prefix for the queries.

Same config (k=30, aggressive xref), only the model changed. The
rebuild took 10 minutes. Then the eval.

Headline number: 22/42 (52%). Up by one question. A small win.

I almost stopped there and called it the new production config.
Then I looked at the per-category breakdown.

| Category | MiniLM | BGE-base | Recall change |
|---|---|---|---|
| Factual | 8/10 | 8/10 | -12% |
| Named entity | 6/9 | **8/9** | **+24%** |
| Narrative | 1/8 | 0/8 | -35% |
| Thematic | 2/10 | 2/10 | **-57%** |
| Interpretive | 4/5 | 4/5 | -24% |

Pass-rate up by one. Mean recall down 12%. And the per-category
pattern was striking: named entity jumped 24% in recall and gained
2 pass-rate, but thematic recall collapsed by 57% and narrative
dropped its only passing question.

This shouldn't happen with a normal model upgrade. Pass-rate and
recall should move together. They diverged because BGE was
concentrating its quality differently than MiniLM.

The mechanism became clear once I thought about how BGE was trained.
Contrastive retrieval loss: maximise similarity between queries and
their best documents, minimise similarity between queries and
irrelevant documents. The "minimise" side pushes the long tail of
"kind of relevant" content further from the query than models generally do.

For named entity questions ("Who was the disciple Jesus loved?"),
this sharpening is exactly right. The best-match document is the
one that mentions the named entity directly, and BGE finds it more
reliably than MiniLM.

For thematic questions ("What does the Bible say about
forgiveness?"), the gold is scattered across the canon (Matthew,
Ephesians, Colossians, 1 John, Mark, etc). Each gold verse is
"kind of relevant" to the question rather than "exactly the
answer." BGE pushes these tangentially related verses away from
the query in embedding space, so the 2nd, 3rd, 4th gold verses
that previously made top-30 with MiniLM no longer make it.

MRR being essentially flat (0.353 -> 0.357) confirmed this. First-
hit rank wasn't the lever. The difference was entirely in what
came after the first hit.

I made the call to keep MiniLM as production. One pass-rate
question gained on named entity didn't compensate for the recall
regression on thematic and narrative. For a Bible QA system where
both precision and breadth matter, MiniLM gave better balanced
retrieval.

What I took from this: "stronger model" is corpus dependent and
task dependent. The training objective matters as much as parameter
count. A retrieval-trained model trades breadth for precision, which
is the right tradeoff for some use cases (commercial product
search, where you want the single best match) and the wrong tradeoff
for others (thematic Bible search, where you want broad coverage).

Headline numbers can mislead. If I'd looked only at pass-rate and
not at recall, I would have shipped the worse production config.
The continuous metrics were what made the tradeoff visible.

