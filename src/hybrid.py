"""Hybrid retrieval: fuse dense (FAISS) and lexical (BM25) rankings.

Uses Reciprocal Rank Fusion (RRF) to combine the two retrievers'
output rankings into a single ranked list. (avoids normalisation step between
cosine similariy scores [0-1] and BM 25 scores[0-30+] as it uses retrieval ranks directly)
"""

import numpy as np
from src.retriever import search
from src.bm25_retriever import bm25_search

def reciprocal_rank_fusion(
    dense_results: list[dict],
    bm25_results: list[dict],
    top_k: int = 10,
    rrf_k: int = 60,
    dense_weight: float = 0.75,
) -> list[dict]:
    """Fuse two ranked result lists with reciprocal rank fusion (RRF).

    Each chunk's RRF score is sum over retrievers of 1/(rrf_k + rank).

    rrf_k=60 is the standard from Cormack et al. 2009. It controls
    how steeply rank-1 dominates: smaller rrf_k means top ranks
    matter more, larger means flatter weighting.

    Returns top_k chunks sorted by fused RRF score descending.
    Each chunk dict gets a new 'rrf_score' field and 'dense_rank' / 'bm25_rank' fields for diagnostics.
    """
    # Build a {chunk_id: {dense_rank, bm25_rank, chunk_dict}} mapping.
    #Implicit dedup
    # (source, book, chapter, verse, chunk_index) which uniquely identifies a chunk in our index.
    def chunk_id(chunk: dict) -> tuple:
        return (
            chunk["source"],
            chunk["book"],
            chunk["chapter"],
            chunk.get("verse"),
            chunk.get("chunk_index", 0),
        )

    entries ={}

    for rank, chunk in enumerate(dense_results, start=1):
        cid = chunk_id(chunk)
        entries[cid] = {
            "chunk": chunk,
            "dense_rank": rank,
            "bm25_rank": None,
        }

    for rank, chunk in enumerate(bm25_results, start=1):
        cid = chunk_id(chunk)
        if cid in entries:
            entries[cid]["bm25_rank"] = rank
        else:
            entries[cid] = {
                "chunk": chunk,
                "dense_rank": None,
                "bm25_rank": rank,
            }

    # Compute RRF score for each entry
    fused = []
    for entry in entries.values():
        score = 0.0
        if entry["dense_rank"] is not None:
            score += dense_weight / (rrf_k + entry["dense_rank"])
        if entry["bm25_rank"] is not None:
            score += (1.0 - dense_weight) / (rrf_k + entry["bm25_rank"])

        chunk = dict(entry["chunk"]) 
        chunk["rrf_score"] = score
        chunk["dense_rank"] = entry["dense_rank"]
        chunk["bm25_rank"] = entry["bm25_rank"]
        fused.append(chunk)

    # Sort by RRF score descending and return top_k
    fused.sort(key=lambda c: c["rrf_score"], reverse=True)
    return fused[:top_k]

def hybrid_search(
    faiss_index,
    chunks_meta: list[dict],
    bm25_index,
    query: str,
    query_embedding: np.ndarray,
    top_k: int = 10,
    rrf_k: int = 60,
    fetch_multiplier: int = 3,
    dense_weight: float = 0.75,
) -> list[dict]:
    """Runs dense and BM25 retrieval, fuses with RRF, returns top_k

    Fetches top_k * fetch_multiplier from each retriever before fusion
    Returns chunks with rrf_score, dense_rank, bm25_rank fields added for diagnostics.
    """

    fetch_k = top_k * fetch_multiplier

    dense_results = search(faiss_index, chunks_meta, query_embedding, top_k=fetch_k)
    bm25_results = bm25_search(bm25_index, chunks_meta, query, top_k=fetch_k)

    return reciprocal_rank_fusion(
        dense_results=dense_results,
        bm25_results=bm25_results,
        top_k=top_k,
        rrf_k=rrf_k,
        dense_weight=dense_weight
    )