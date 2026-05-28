# Baseline findings

What the eval revealed across eleven retrieval configurations.

## Setup

42 hand-labelled questions across five categories. Each question
carries gold verse references and a min_recall threshold appropriate
to the category (factual=1, narrative=2-5, thematic=2-4, etc).

## Ablation dimensions

Four independent dimensions tested:

1. **Retrieval depth**: top_k=10 vs top_k=20 vs top_k=30
2. **Cross-reference expansion**: none vs conservative vs aggressive
3. **Hybrid retrieval**: pure dense vs dense + BM25 fused via RRF
4. **Embedding model**: MiniLM (general semantic) vs BGE (retrieval-trained)

Plus one ablation on commentary inclusion (use_commentary on/off)
to verify the Option B' interpretive scoring rule.

## Headline result

The production configuration is dense retrieval at top_k=30 with
aggressive cross-reference expansion (max_per_seed=10, max_extra=15)
using the `all-MiniLM-L6-v2` embedding model. **21/42 questions
pass (50%), mean Recall@k=0.381, mean MRR=0.353.**

A stronger embedding model (BGE-base) was tested and produced 22/42
pass-rate but with materially worse recall on multi-gold categories.
The MiniLM config was chosen as production for better balanced
retrieval. See Finding 7 below.

## Full results

See `ablation_summary.md` for the side-by-side comparison across
configurations. Below is the narrative interpretation.

---

## Finding 1: Retrieval depth matters, especially for multi-gold

Going from top_k=10 to top_k=20 lifted pass-rate from 14/42 to 19/42.
The lift wasn't even across categories.

| Category | k=10 pass | k=20 pass | Recall lift |
|---|---|---|---|
| Factual | 5/10 | 7/10 | +17% |
| Named entity | 5/9 | 5/9 | +24% |
| Interpretive | 4/5 | 4/5 | +11% |
| Narrative | 0/8 | 1/8 | +74% |
| Thematic | 0/10 | 2/10 | +117% |

Multi-gold categories (narrative, thematic) benefit proportionally
more from extra retrieval depth. Each additional slot is another
chance to hit a not-yet-found gold item. The absolute level at
k=20 still stayed low: only 13-15% of multi-gold items retrieved
on average.

## Finding 2: Continuous metrics reveal hidden capability

Pass-rate is binary. A question scoring 1/5 gold reads the same as
0/5 in pass-rate terms. Recall@k and MRR distinguish these cases.

The interesting observation: **narrative MRR (0.30) is comparable
to factual MRR (0.28-0.31)**. The first gold verse for a narrative
question typically appears around rank 3-4. The retriever isn't bad
at finding *some* narrative gold, it's bad at finding *multiple*
narrative gold. Pass-rate of 0%/12% hides this.

Same pattern for thematic. The system has capability that pass-rate
doesn't credit because the threshold requires breadth, not just
finding one item.

This finding directly shaped the next experiments. If first-hit
rank was already reasonable, embedding-level upgrades probably
wouldn't help much. What might help was something that surfaced
*additional* related verses given that the first one was in the
result set. That points at xref expansion.

## Finding 3: Interpretive scoring works as designed

The Option B' rule (interpretive scored against MHC chunks only)
is working. With MHC enabled: 4/5 passes. Strip MHC: 0/5. Confirms
the category requires commentary retrieval to pass, and that the
dense retriever is finding the right MHC chapter consistently when
it's available.

## Finding 4: MHC commentary doesn't crowd out verse gold

Initial hypothesis: long MHC commentary chunks were eating top-k
slots that verse-fragment gold needed. Disabling MHC should have
lifted narrative/thematic recall.

It didn't. Narrative and thematic recall are *identical to three
decimal places* with MHC on (0.132 / 0.154) and MHC off (0.132 /
0.154). The slots freed by removing MHC fill with other irrelevant
content, not with previously-missing gold verses.

The bottleneck for multi-gold categories is that gold verses don't
make top-k in the first place, not that MHC is displacing them.

## Finding 5: Cross-reference expansion interacts with retrieval depth

This is the most useful finding in the project.

Initial test at k=20 across three xref settings:

| Config | Recall@k | MRR | Pass |
|---|---|---|---|
| No xref (max_per_seed=0) | 0.291 | 0.351 | 19/42 |
| Conservative (1 / 3) | 0.291 | 0.351 | 19/42 |
| Aggressive (3 / 10) | 0.313 | 0.351 | 19/42 |

Pass-rate identical. MRR identical. Recall lifted +0.022 at the
aggressive setting, mostly in factual. The reasonable conclusion
was that xref expansion didn't help meaningfully on this eval set.

Then I tried one more configuration: top_k=30 with very aggressive
xref (max_per_seed=10, max_extra=15). The result:

| Config | Recall@k | MRR | Pass |
|---|---|---|---|
| Dense k=20 baseline | 0.291 | 0.351 | 19/42 |
| Dense k=30 + xref 10/15 | **0.381** | 0.353 | **21/42** |

Recall jumped +0.090 (+31% relative). Pass-rate went up by 2. The
gains aren't concentrated in one category, they spread across all
five.

| Category | k=20 recall | k=30 + aggressive xref recall | Change |
|---|---|---|---|
| Factual | 0.408 | 0.542 | +33% |
| Named entity | 0.394 | 0.506 | +28% |
| Narrative | 0.132 | 0.150 | +14% |
| Thematic | 0.154 | 0.225 | +46% |
| Interpretive | 0.394 | 0.517 | +31% |

### Why the interaction

At k=20 with conservative xref, the xrefs were largely duplicating
what dense already found. Each xref's "seed" was a top-20 chunk,
and the first-degree neighbours of top-20 chunks tend to overlap
with the top-20 set itself.

At k=30 with aggressive xref, the seed pool is larger (30 chunks
instead of 20), each seed can contribute up to 10 xrefs (vs 1),
and the total xref budget is 15 (vs 3). The xrefs reaching the
final result set are no longer duplicates of dense's top picks but
genuinely different verses that the original embedding-based
retrieval didn't surface.

### The lesson

Xref expansion at k=20 looked like a dead end after the first three
tests. The honest version of this finding is: I almost stopped
there. Trying one more, more aggressive configuration produced the
project's best result. The lesson is that ablation dimensions
interact, and "intervention X doesn't help" is a category mistake
when the intervention's effect depends on other parameters.

## Finding 6: Hybrid retrieval is a negative result

Implemented BM25 lexical retrieval combined with dense via
Reciprocal Rank Fusion (RRF). Tested four configurations across
two values of top_k and three combinations of fetch multiplier
and dense weight.

| Config | Pass | Recall@k | MRR |
|---|---|---|---|
| Dense k=10 (baseline) | 14/42 | 0.225 | 0.344 |
| Hybrid k=10, α=0.5, mult=3 | 14/42 | 0.209 | 0.262 |
| Hybrid k=10, α=0.5, mult=10 | 12/42 | 0.158 | 0.230 |
| Hybrid k=10, α=0.75, mult=3 | 13/42 | 0.205 | 0.250 |
| Dense k=20 | 19/42 | 0.291 | 0.351 |
| Hybrid k=20, α=0.5, mult=3 | 16/42 | 0.286 | 0.253 |

α is dense weight in weighted RRF. mult is the fetch multiplier
(candidates pulled from each retriever before fusion).

Every hybrid configuration produced lower MRR than the matched
dense baseline. The drop in MRR is large: 0.08 to 0.11 points,
substantial for this metric. Pass-rate matched or got worse.

### What went wrong

Inspection of factual fails under hybrid showed a consistent pattern:
BM25 was top-ranking high-keyword-overlap verses that weren't the
right answer.

q32 asks "What did the Israelites eat in the wilderness?" Gold is
`Ex.16.31` (manna). Under hybrid k=10, the top result is `bsb Num 9:5`
(BM25 rank 1, dense rank 24). Num 9:5 is about Passover, not manna.
But it contains "Israelites" and "wilderness" verbatim, two strong
BM25 signals. Dense ranked it 24th and was suspicious. BM25
elevated it to fused rank 1.

Same pattern for q26 (Moses receiving Ten Commandments). Gold is
`Ex.19.20`, `Ex.20.1`, `Deut.5.2`. Top hybrid result is `bsb Exod 34:28`
(BM25 rank 1, dense rank 6). Exod 34:28 mentions "ten commandments"
verbatim but describes Moses re-receiving the law after the golden
calf, not the original giving.

The mechanism: BM25 on Bible text produces high-confidence wrong
answers more often than expected. The corpus has ~31k verses per
translation with heavy repetition of common nouns (Israel, Lord,
wilderness, people, king). For most queries, several non-gold
verses contain strong keyword overlap with the question phrasing.
BM25 ranks them confidently. RRF gives those wrong picks a strong
contribution, displacing dense's correct rankings.

### Tuning attempts

Widening the fetch window (mult=10) made things worse, not better.
More candidates means more chances for chunks with two weak rank
contributions to beat chunks with one strong contribution.

Downweighting BM25 to α=0.75 (dense weight) partially recovered the
worst regression but still didn't reach dense baseline. The pattern
across α settings is monotonic: more BM25 influence, worse results.

One narrow positive: hybrid k=20 lifted named entity from 5/9 to
6/9 (q41 flipped). BM25 helped on the one category where exact
proper-noun matching is the main signal. But this gain was wiped
out by losses elsewhere.

### Honest conclusion

BM25 doesn't fit this corpus at default settings. The corpus
property that makes it hard is high repetition of common nouns
across short verses. The retrievers compete for ranking slots,
and BM25's confident wrong picks displace dense's correct picks
even at reduced weight.

The detour is documented as a negative result. The infrastructure
is kept (`src/bm25_retriever.py`, `src/hybrid.py`, configurable
from runner) so the experiment is reproducible. The production
system uses dense-only retrieval.

## Finding 7: A stronger embedding model traded categories rather than lifting all of them

After settling on the dense k=30 + aggressive xref config, I tested
one more intervention: swap the embedding model from
`sentence-transformers/all-MiniLM-L6-v2` (23M params, 384-dim,
general semantic similarity) to `BAAI/bge-base-en-v1.5` (110M params,
768-dim, retrieval-trained). BGE was applied with its recommended
instruction prefix
(`"Represent this sentence for searching relevant passages: "`)
prepended to query text but not to documents.

### Results

Same config (k=30, max_per_seed=10, max_extra=15), only the
embedding model changed:

| Category | MiniLM (prev best) | BGE-base | Pass Δ | Recall Δ |
|---|---|---|---|---|
| Factual | 8/10 (0.542 / 0.294) | 8/10 (0.475 / 0.308) | flat | -12% |
| Named entity | 6/9 (0.506 / 0.458) | **8/9 (0.626 / 0.529)** | **+2** | **+24%** |
| Narrative | 1/8 (0.150 / 0.307) | 0/8 (0.097 / 0.312) | -1 | -35% |
| Thematic | 2/10 (0.225 / 0.228) | 2/10 (0.098 / 0.129) | flat | -57% |
| Interpretive | 4/5 (0.517 / 0.607) | 4/5 (0.394 / 0.670) | flat | -24% |
| **Overall** | **21/42 (0.381 / 0.353)** | 22/42 (0.336 / 0.357) | **+1** | **-12%** |

Pass-rate up by one question. Mean recall down 12%. MRR essentially
flat. The headline number improved, but it masked a real tradeoff.

### Why pass-rate and recall moved in opposite directions

This is the unusual signal worth investigating. Normally pass-rate
and recall move together because they're correlated metrics. They
diverged here because BGE concentrated its quality differently than
MiniLM did.

Named entity recall lifted dramatically (+24%). Single-gold questions
became more likely to pass because BGE's retrieval-specific training
sharpened the embedding distance between queries and their best
matching documents.

Thematic recall collapsed (-57%). BGE's sharper embeddings push
tangentially-related content further from the query in embedding
space. For thematic questions where you want multiple related but
distinct verses (e.g. "What does the Bible say about forgiveness?"
has gold scattered across Matthew, Ephesians, Colossians, 1 John),
the additional 2nd, 3rd, 4th gold verses are getting pushed out of
top-k that they previously made.

MRR being flat (0.353 -> 0.357) confirms first-hit rank didn't
materially change. The system finds approximately the same first
gold verse at approximately the same depth. The difference is what
happens after the first hit.

### Why BGE behaves this way

BGE was trained with a contrastive retrieval objective: maximise
similarity between queries and their best documents, minimise
similarity between queries and irrelevant documents. The "minimise"
side of the loss pushes the long tail of "kind of relevant" content
further from the query than general-purpose models like MiniLM do.

For factual / named entity tasks where you want the single best
match, this is exactly the right behaviour. For thematic retrieval
where breadth matters more than precision, it's the wrong
behaviour.

This is a known property of retrieval-trained embeddings, not a
defect specific to BGE.

### Decision: keep MiniLM as production

22/42 vs 21/42 is a one-question improvement on pass-rate. The
recall regression on thematic (-57%) and narrative (-35%) is
substantially larger in magnitude. For a Bible QA system where
both single-best-answer and broad-topic-coverage matter, MiniLM
gives better balanced retrieval.

The BGE configuration is preserved (`src/embedder.py` has both
models, swap via the `MODEL_NAME` constant) so the result is
reproducible.

### What this rules out

Embedding model capacity is not the bottleneck on multi-gold
categories. The +5x parameter count and the retrieval-specific
training in BGE produced worse coverage on those categories.
Future work targeting multi-gold improvement should look at:

- Chunk-level changes (longer or overlapping chunks so each gold
  has more lexical surface area)
- Query rewriting (expand short questions before retrieval)
- A cross-encoder reranker on top of dense top-50, which could
  exploit BGE's precision without losing MiniLM's breadth

## Production configuration

```python
config = {
    "top_k": 30,
    "use_xrefs": True,
    "max_extra": 15,
    "max_per_seed": 10,
    "embedding_model": "sentence-transformers/all-MiniLM-L6-v2",
    "use_commentary": True,
    "use_hybrid": False,
}
```

Pass-rate 21/42 (50%), mean Recall@k 0.381, mean MRR 0.353.

## What's not tested

Things that might lift the multi-gold categories further but were
out of scope:

- Chunk-level changes (longer chunks, sliding-window overlap)
- Query rewriting (expand short questions into verse-style phrasing
  before retrieval)
- A cross-encoder reranker on top of dense top-50 that could combine
  BGE's precision with MiniLM-like breadth

These would be the next experiments with more time.
