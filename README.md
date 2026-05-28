# Bible RAG Chatbot

A locally-hosted retrieval-augmented question-answering system over a
~95k-chunk heterogeneous corpus: two Bible translations (KJV, BSB),
Matthew Henry's 18th-century commentary (1,189 chapters), and a
344k-edge cross-reference graph from OpenBible.info. Runs fully
locally with sentence-transformers + FAISS for retrieval, llama3.2
via Ollama for generation, and Gradio for the UI.

A custom evaluation framework was built around the system to measure
retrieval quality across a hand-labelled gold set. See the
[Evaluation](#evaluation) section for the methodology and results.

## Screenshots

|                                                                                                                                                                                               |                                                                                                                                                                                        |
| --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| [![Landing](https://github.com/qpingWIN/bible-rag-chatbot/raw/main/docs/screenshots/00_landing.png)](/qpingWIN/bible-rag-chatbot/blob/main/docs/screenshots/00_landing.png)                                     | [![Faith](https://github.com/qpingWIN/bible-rag-chatbot/raw/main/docs/screenshots/01_faith_answer.png)](/qpingWIN/bible-rag-chatbot/blob/main/docs/screenshots/01_faith_answer.png)                      |
| *Landing page*                                                                                                                                                                                | *"What does the Bible say about faith?"*                                                                                                                                               |
| [![David and Goliath](https://github.com/qpingWIN/bible-rag-chatbot/raw/main/docs/screenshots/02_david_goliath_answer.png)](/qpingWIN/bible-rag-chatbot/blob/main/docs/screenshots/02_david_goliath_answer.png) | [![Out of scope](https://github.com/qpingWIN/bible-rag-chatbot/raw/main/docs/screenshots/03_out_of_scope_answer.png)](/qpingWIN/bible-rag-chatbot/blob/main/docs/screenshots/03_out_of_scope_answer.png) |
| *"How did David defeat Goliath?"*                                                                                                                                                             | *"Will Jesus return to Earth in 2026?"*                                                                                                                                                |


## Stack

| Component     | Library                                                         |
| ------------- | --------------------------------------------------------------- |
| Embeddings    | `sentence-transformers/all-MiniLM-L6-v2` (384-dim, ~22M params) |
| Vector search | FAISS `IndexFlatIP` (exact cosine similarity)                   |
| LLM           | llama3.2 via Ollama                                             |
| UI            | Gradio                                                          |
| Hardware      | Apple Silicon MPS / CPU                                         |


## Features

- Semantic search across ~96k indexed chunks
- Two translations (KJV + BSB) with reference-key deduplication
- Matthew Henry's Complete Commentary chunked with a sliding 250-word window
- Cross-reference graph expansion over 344k verse-pair links
- Fully local, no data leaves the machine
- Every generated claim is citation-traceable to a source passage

---


## Evaluation

A custom evaluation framework was built around the chatbot to measure
retrieval quality rigorously. The eval framework, not the chatbot
itself, is the load-bearing portfolio claim.

### Headline result

The production configuration passes **21/42 questions (50%)** with
mean Recall@k 0.381 and mean MRR 0.353. Dense retrieval at top_k=30
with aggressive cross-reference expansion (max_per_seed=10,
max_extra=15).

Ten total configurations were tested across three intervention
dimensions to characterise the system's ceiling and rule out
interventions that didn't work.

### The gold set

42 hand-labelled questions across five categories. Each question
carries gold verse references and a min_recall threshold appropriate
to the category.

| Category | N | min_recall | What it tests |
|---|---|---|---|
| Factual | 10 | 1 | Direct facts with a single canonical answer |
| Named entity | 9 | 1 (or 2) | Identification of people and places |
| Narrative | 8 | 2-5 | Multi-verse story reconstruction |
| Thematic | 10 | 2-4 | Topic coverage across the canon |
| Interpretive | 5 | 1 | Typological readings requiring commentary |

Three scoring axes:
- **Pass-rate**: did the threshold get cleared (binary)
- **Recall@k**: fraction of gold items found (continuous)
- **MRR**: reciprocal rank of the first gold hit (continuous)

`eval/methodology.md` covers the design decisions in detail:
translation-agnostic verse matching, chapter-scope MHC matching, the
Option B' rule scoring interpretive questions against MHC commentary
chunks only, and the rationale for category-specific min_recall
thresholds.

### Key findings

| Lever | Effect |
|---|---|
| top_k 10 -> 20 | +5 pass, +29% recall. Most benefit in multi-gold categories. |
| top_k 20 -> 30 + aggressive xref | +2 pass, +31% recall. The production config. |
| MHC commentary on/off | Interpretive collapses without MHC. Other categories unchanged. |
| Cross-reference expansion at k=20 | Small recall lift only. Looked like a dead end. |
| Cross-reference expansion at k=30 | Substantial recall lift across all categories. |
| Hybrid retrieval (dense + BM25 via RRF) | Negative result. BM25 produced confident wrong answers. |
| Stronger embedding (BGE-base) | +1 pass, -12% recall. Sharper retrieval traded breadth for precision. MiniLM kept as production. |


The xref-expansion finding is the most interesting one. At k=20,
increasing xref aggressiveness from default to max_per_seed=3,
max_extra=10 lifted recall by only +0.022 with no pass-rate movement.
The reasonable conclusion was that xref expansion didn't help. One
more experiment at k=30 with even more aggressive expansion (10/15)
produced a +0.090 recall lift and 2 extra pass-rate. The two
parameters interact: xref expansion needs enough seed chunks to
surface genuinely new verses rather than duplicating what dense
already found.

The embedding swap is a tradeoff finding. BGE-base (110M params,
retrieval-trained) produced +1 pass-rate over MiniLM but lifted
single-gold categories (named entity recall +24%) at the cost of
multi-gold coverage (thematic recall -57%). The retrieval-specific
training sharpens the embedding distance between queries and their
best matches but pushes tangentially-related verses further away.
For balanced retrieval, MiniLM was kept as production.

The hybrid retrieval finding is documented in detail in
`eval/notebooks/baseline_findings.md`. Short version: Bible text has
heavy repetition of common nouns across short verses, so BM25
surfaces high-keyword-overlap non-gold verses at confident ranks,
which RRF amplifies into the fused top-k. The infrastructure is
kept (`src/bm25_retriever.py`, `src/hybrid.py`) so the negative
result is reproducible.

### Documents

- `eval/methodology.md` — eval design decisions and rationale
- `eval/notebooks/baseline_findings.md` — full ablation narrative
- `eval/notebooks/ablation_summary.md` — comparison across all configs
- `eval/notebooks/stories.md` — engineering stories from the build

### Reproducing the eval

```bash
# After running the chatbot Quick Start below
python -m eval.scoring.runner

# Run the test suite
pytest -v
```

The config dict at the bottom of `eval/scoring/runner.py` controls
all retrieval parameters. Each eval run saves a timestamped JSON
file to `eval/results/` containing the full config plus per-question
traces, so any row in the ablation table can be reproduced.

The production configuration:

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

### What I'd do next with more time

- Try query rewriting (expand short questions into verse-style
  phrasing before retrieval)
- Add a cross-encoder reranker on top of dense top-50
- Bootstrap confidence intervals on the per-category pass-rates
- Chunk-level changes (longer chunks, sliding-window overlap)

---


## Design Decisions

### Embedding Model

Using `all-MiniLM-L6-v2` from sentence-transformers: 384-dimensional
vectors, optimised for semantic similarity, lightweight inference.
Fast, low memory usage, good embedding quality. The 384-dim size is
a deliberate compromise between semantic representation quality and
retrieval efficiency.

Religious texts specifically benefit from semantic embeddings
because users tend to ask conceptually framed questions ("what does
the Bible say about forgiveness?") rather than exact keyword queries.
Sentence-transformer embeddings capture that semantic proximity far
better than traditional keyword search.

**Why not TF-IDF or BM25?** Poor semantic understanding. They excel
at exact matches but fail on conceptual queries. The eval also
confirmed this empirically: a hybrid dense + BM25 retrieval was
tested and produced lower MRR than dense alone, because BM25's
keyword matches on a heavily-repetitive corpus surfaced
confident wrong answers. See the [Evaluation](#evaluation) section.

**Why not a stronger retrieval-trained model (BGE, MPNet)?**
Tested empirically. BGE-base produced a slightly higher headline
pass-rate (22/42 vs 21/42) but worse balanced retrieval: named
entity recall lifted 24% while thematic recall collapsed 57%. The
retrieval-trained model sharpens precision at the cost of breadth,
which trades off poorly for thematic Bible search where multiple
related verses matter as much as the single best match.

**Alternatives considered:**

- OpenAI embeddings (`text-embedding-3-small/large`): state-of-the-art
  retrieval quality and excellent semantic clustering. If launching
  commercially, these would be worth the trade-offs (API cost,
  external dependency, network latency)

---


### Vector Index

Using FAISS `IndexFlatIP`: every query is compared against all
vectors in the database (exact nearest-neighbour search) to get
exact cosine similarity after L2 normalisation. With ~95k vectors at
384 dimensions, exact search is still computationally feasible
without compromising on approximation errors.

The main limitation is scalability. Since this project is
Bible-only, the corpus never needs to grow significantly, so this
is not a concern in practice.

**Alternatives considered:**

- `IndexIVFFlat`: approximate search, requires training, introduces
  retrieval errors that are unnecessary at this scale
- External vector databases (Pinecone, Weaviate, Chroma): better fit
  if managing vector retrieval as a scalable production system

---


### Chunking Strategy

Bible verses are short and self-contained, so each verse is indexed
as a single chunk (1 verse = 1 chunk). Embedding an entire verse
preserves its semantic meaning without any loss.

Commentary chapters are much longer. Embedding entire chapters would
dilute the meaning and make it harder to retrieve specific arguments.
Instead, a sliding 250-word window with 1-sentence overlap is
applied: the overlap prevents information loss at chunk boundaries
where an argument might span two windows.

**Alternatives considered:**

- Recursive character chunking (LangChain): flexible but less
  human-interpretable
- Semantic chunking: splits on meaning shifts rather than word
  count, more expensive to compute
- Token-based chunking: aligns with model limits but less intuitive
  to reason about

---


### Generation

The LLM is prompted with a strict grounding template:

```
You are a biblical scholar who answers questions using ONLY the provided source passages.
Rules:
- Base your answer strictly on the passages below. Do not add knowledge from outside these passages.
- After each claim, cite the source in parentheses: (KJV Genesis 1:1) or (MHC Matthew 5 commentary).
- If the passages don't contain enough information to answer, say: "The provided passages don't address this directly."
- Be concise but complete. No padding.
```

Temperature is set to `0.0`: minimal creativity, reduced
hallucinations, factual consistency, strict adherence to retrieved
passages.

---


### Translation Deduplication

**The problem:** KJV and BSB both contain every verse. A search for
"John 3:16" returns both `kjv:John 3:16` and `bsb:John 3:16` near
the top, scoring nearly identically. That wastes two of the k slots
on the same content.

**The solution:** as results come back from FAISS, a key is built
from `book + chapter + verse`. The second time the same key appears
(regardless of whether it is KJV or BSB), it is skipped. First one
in wins. The retriever fetches `top_k * 3` candidates upfront to
have enough raw results to fill `top_k` slots after dedup removes
some.

An earlier version of this had a second dedup layer based on exact
float32 score equality. It was removed during the eval work because
it could silently drop valid chunks when different vectors happened
to produce identical cosine scores.

---


### Cross-Reference Graph Expansion

**The problem:** dense retrieval finds passages that are semantically
similar to the query. But some questions require passages that are
theologically linked rather than semantically similar. "How does
Isaiah 53 relate to Jesus?" will not naturally retrieve Isaiah 53
from a query about Jesus, because they are from different testaments
with different vocabulary and the embedding space does not bridge
that gap.

The cross-reference graph from OpenBible.info is a 344k-edge directed
graph: `verse A -> verse B` means A cross-references B. It encodes
human-curated theological connections the embedding model cannot
capture.

After dense retrieval returns the top-k results, each retrieved
verse is looked up in the graph and its linked verses are appended
to the result set, capped by `max_extra` and `max_per_seed`. The
eval revealed that these parameters need to be set aggressively
(max_per_seed=10, max_extra=15) at higher retrieval depths (k=30)
to surface genuinely new verses rather than duplicates.

---


## Quick Start

**Prerequisites:** Python 3.11+, [Ollama](https://ollama.com)

```bash
git clone https://github.com/qpingWIN/bible-rag-chatbot
cd bible-rag-chatbot
pip install -r requirements.txt

ollama pull llama3.2

# Add data files to data/raw/ (see Data Setup below)

python build_index.py
ollama serve &
python app.py
# open http://localhost:7860
```

## Data Setup

Download and place in `data/raw/`:

| File                   | Source                                                                                                             |
| ---------------------- | ------------------------------------------------------------------------------------------------------------------ |
| `kjv.json`             | [scrollmapper/bible_databases](https://github.com/scrollmapper/bible_databases/blob/master/formats/json/KJV.json) |
| `web.json`             | [Berean Standard Bible](https://berean.bible)                                                                      |
| `mhc_commentary.json`  | Extract via `data/extract_mhc.py` from the SWORD MHC module                                                        |
| `cross_references.txt` | [OpenBible.info](https://www.openbible.info/labs/cross-references/)                                                |


## Project Structure

```
bible-rag-chatbot/
├── app.py                  # Gradio UI
├── build_index.py          # One-time index builder (FAISS + BM25)
├── requirements.txt
├── data/
│   ├── raw/                # Source data (not in repo)
│   ├── index/              # FAISS index, BM25 pickle (generated, not in repo)
│   └── extract_mhc.py
├── src/
│   ├── ingest.py           # Load raw data with book-name normalisation
│   ├── chunker.py          # Verse-level and sliding-window chunking
│   ├── embedder.py         # sentence-transformers wrapper
│   ├── retriever.py        # FAISS dense retrieval with translation dedup
│   ├── bm25_retriever.py   # BM25 lexical retrieval (for hybrid experiment)
│   ├── hybrid.py           # RRF fusion of dense + BM25
│   └── generator.py        # Ollama prompt construction and generation
├── eval/
│   ├── questions.jsonl     # 42 hand-labelled questions with gold refs
│   ├── methodology.md      # eval design decisions
│   ├── scoring/
│   │   ├── matching.py     # translation-agnostic verse matching
│   │   ├── scorer.py       # pass-rate, Recall@k, MRR computation
│   │   ├── runner.py       # eval orchestrator
│   │   └── test_*.py       # pytest test suite
│   ├── results/            # per-run JSON with full config and traces
│   └── notebooks/
│       ├── baseline_findings.md
│       ├── ablation_summary.md
│       └── stories.md
└── docs/screenshots/
```

## Data Sources

- **KJV**: [scrollmapper/bible_databases](https://github.com/scrollmapper/bible_databases) - Public Domain
- **BSB**: [Berean Bible](https://berean.bible) - Creative Commons
- **MHC**: Matthew Henry's Complete Commentary (1708-1714) - Public Domain, via [SWORD Project](https://www.crosswire.org/sword/)
- **Cross-references**: [OpenBible.info](https://www.openbible.info/labs/cross-references/) - CC BY 4.0
