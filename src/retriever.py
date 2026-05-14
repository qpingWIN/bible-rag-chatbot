"""FAISS vector store: build, persist, and query the similarity index"""

import json
import numpy as np
import faiss
from pathlib import Path
from .chunker import Chunk

INDEX_DIR = Path(__file__).parent.parent / "data" / "index"


def build_index(chunks: list[Chunk], embeddings: np.ndarray) -> faiss.IndexFlatIP:
    """Build a FAISS flat inner-product index from pre-computed embeddings"""
    dim = embeddings.shape[1]
    index = faiss.IndexFlatIP(dim)
    index.add(embeddings)
    print(f"FAISS index built: {index.ntotal:,} vectors, dim={dim}")
    return index


def save_index(index: faiss.IndexFlatIP, chunks: list[Chunk]) -> None:
    INDEX_DIR.mkdir(parents=True, exist_ok=True)
    faiss.write_index(index, str(INDEX_DIR / "chunks.index"))
    meta = [c.to_dict() for c in chunks]
    (INDEX_DIR / "chunks_meta.json").write_text(
        json.dumps(meta, ensure_ascii=False)
    )
    print(f"Saved index + {len(meta):,} chunk records → {INDEX_DIR}")


def load_index() -> tuple[faiss.IndexFlatIP, list[dict]]:
    index = faiss.read_index(str(INDEX_DIR / "chunks.index"))
    meta = json.loads((INDEX_DIR / "chunks_meta.json").read_text())
    print(f"Loaded index: {index.ntotal:,} vectors, {len(meta):,} chunks")
    return index, meta


def index_exists() -> bool:
    return (INDEX_DIR / "chunks.index").exists()


def search(
    index: faiss.IndexFlatIP,
    chunks_meta: list[dict],
    query_embedding: np.ndarray,
    top_k: int = 3,
) -> list[dict]:
    """
    Return top_k most similar chunks, deduplicated by verse reference.

    Fetches top_k * 3 candidates first to absorb duplicates from KJV/BSB
    covering the same verse (near-identical embeddings, same cosine score).
    Commentary chunks have unique references and are never collapsed.
    """
    fetch_k = top_k * 3
    scores, indices = index.search(query_embedding, fetch_k)

    seen_keys: set[str] = set()
    seen_scores: set[float] = set()
    results: list[dict] = []

    for score, idx in zip(scores[0], indices[0]):
        if idx == -1:
            continue
        chunk = dict(chunks_meta[idx])
        chunk["score"] = float(score)

        if chunk.get("verse") is not None:
            key = f"{chunk['book']}_{chunk['chapter']}_{chunk['verse']}"
        else:
            key = chunk["reference"]

        if key in seen_keys or chunk["score"] in seen_scores:
            continue
        seen_keys.add(key)
        seen_scores.add(chunk["score"])
        results.append(chunk)

        if len(results) == top_k:
            break

    return results


def expand_with_cross_references(
    results: list[dict],
    index: faiss.IndexFlatIP,
    chunks_meta: list[dict],
    xrefs: dict[str, list[str]],
    ref_to_chunk: dict[str, int],
    max_extra: int = 3,
) -> list[dict]:
    """Augment results with cross-referenced verses"""
    seen_refs = {r["reference"] for r in results}
    extra: list[dict] = []

    for result in results:
        if result.get("verse") is None:
            continue
        ref_key = "{}.{}.{}".format(
            result["book"], result["chapter"], result["verse"]
        )
        for xref in xrefs.get(ref_key, [])[:5]:
            chunk_idx = ref_to_chunk.get(xref)
            if chunk_idx is None:
                continue
            chunk = dict(chunks_meta[chunk_idx])
            if chunk["reference"] in seen_refs:
                continue
            chunk["score"] = 0.0
            chunk["xref_from"] = result["reference"]
            extra.append(chunk)
            seen_refs.add(chunk["reference"])
            if len(extra) >= max_extra:
                break
        if len(extra) >= max_extra:
            break

    return results + extra


def build_ref_to_chunk_index(chunks_meta: list[dict]) -> dict[str, int]:
    """Map 'Book.Chapter.Verse' → index in chunks_meta for fast xref lookup"""
    ref_map: dict[str, int] = {}
    for i, chunk in enumerate(chunks_meta):
        if chunk.get("verse") is not None:
            key = "{}.{}.{}".format(chunk["book"], chunk["chapter"], chunk["verse"])
            ref_map[key] = i
    return ref_map
