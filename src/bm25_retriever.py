"""BM25 lexical retrieval over the same chunks as the FAISS index

BM25 is a keyword-based scoring function that complements dense embedding
retrieval. It's good at exact name matches, short documents(verses are really short in our case)
and exact keyword overlap (exactly where dense embeddings tend to be weakest)

The index is built once over all chunk texts and saved alongside
the FAISS index. Load it at runner startup just like the FAISS index.
"""

import pickle
import re
from pathlib import Path
from rank_bm25 import BM25Okapi
INDEX_DIR = Path(__file__).parent.parent / "data" / "index"
BM25_PATH = INDEX_DIR / "bm25.pkl"
def _tokenise(text: str) -> list[str]:
    """Lowercase, strip punctuation, split on whitespace
    Stemming+stopword removal not implemented since they would hurt the 
    rare nouns matching

    """
    text = text.lower()
    text = re.sub(r"[^\w\s]", " ", text)
    return text.split()

def build_bm25_index(chunks_meta: list[dict]) -> BM25Okapi:
    """Tokenise every chunk's text and build a BM25 index"""
    print(f"Tokenising {len(chunks_meta):,} chunks for BM25...")
    tokenised_corpus = [_tokenise(c["text"]) for c in chunks_meta]
    print("Building BM25 index...")
    bm25 = BM25Okapi(tokenised_corpus)
    print(f"BM25 index built: {len(tokenised_corpus):} documents")
    return bm25

def save_bm25_index(bm25: BM25Okapi) -> None:
    INDEX_DIR.mkdir(parents=True, exist_ok=True)
    with open(BM25_PATH, "wb") as f:
        pickle.dump(bm25, f)
    print(f"Saved BM25 index → {BM25_PATH}")

def load_bm25_index() -> BM25Okapi:
    with open(BM25_PATH, "rb") as f:
        bm25 = pickle.load(f)
    print(f"Loaded BM25 index from {BM25_PATH}")
    return bm25

def bm25_index_exists() -> bool:
    return BM25_PATH.exists()


def bm25_search(
    bm25: BM25Okapi,
    chunks_meta: list[dict],
    query: str,
    top_k: int = 10,
) -> list[dict]:
    """Return the top_k chunks by BM25 score deduplicated by verse reference.

    Mirrors the dedup pattern from retriever.search(): for verse chunks,
    collapse KJV/BSB versions of the same verse to a single result
    (the higher-scoring translation wins because BM25 results are
    already sorted descending). MHC chunks are kept distinct by
    their (book, chapter, chunk_index) tuple via their reference.
    """
    tokens = _tokenise(query)
    scores = bm25.get_scores(tokens)

    # argsort descending as we want indices of the highest-scoring chunks first
    ranked_indices = sorted(range(len(scores)), key=lambda i: -scores[i])

    seen_keys: set[str] = set()
    results: list[dict] = []

    for idx in ranked_indices:
        if scores[idx] <= 0:
            # BM25 score 0 means no query term matched this chunk.
            # No point including these as they pollute the ranking.
            break

        chunk = dict(chunks_meta[idx])
        chunk["score"] = float(scores[idx])

        if chunk.get("verse") is not None:
            key = f"{chunk['book']}_{chunk['chapter']}_{chunk['verse']}"
        else:
            key = chunk["reference"]

        if key in seen_keys:
            continue
        seen_keys.add(key)
        results.append(chunk)

        if len(results) == top_k:
            break

    return results
