"""Shared RAG pipeline: one loader used by both the Gradio UI and the API.

get_pipeline() loads the index, embedding model, and xref graph, and returns
them bundled as four callables (embed / search / expand / generate) plus
index_size. The lru_cache guarantees the heavy load happens exactly once per
process, no matter how many entrypoints (app.py, api.py, or both mounted
together) ask for it.

PRODUCTION_CONFIG is the single source of truth for the retrieval parameters
found by the eval ablation (see README, Evaluation section).
"""

from functools import lru_cache
from types import SimpleNamespace

PRODUCTION_CONFIG = {
    "top_k": 30,
    "use_xrefs": True,
    "max_extra": 15,
    "max_per_seed": 10,
    "use_commentary": True,
}


@lru_cache(maxsize=1)
def get_pipeline() -> SimpleNamespace:
    # Heavy imports deferred so this module stays importable without torch/faiss.
    from src.embedder import embed_texts, load_model
    from src.generator import generate
    from src.ingest import load_cross_references
    from src.retriever import (
        build_ref_to_chunk_index,
        expand_with_cross_references,
        index_exists,
        load_index,
        search,
    )

    if not index_exists():
        raise RuntimeError("Index not found. Run `python build_index.py` first.")

    index, chunks_meta = load_index()
    model = load_model()
    xrefs = load_cross_references()
    ref_to_chunk = build_ref_to_chunk_index(chunks_meta)

    return SimpleNamespace(
        index_size=index.ntotal,
        embed=lambda texts: embed_texts(model, texts, show_progress=False),
        search=lambda emb, k: search(index, chunks_meta, emb, top_k=k),
        expand=lambda results: expand_with_cross_references(
            results,
            index,
            chunks_meta,
            xrefs,
            ref_to_chunk,
            max_extra=PRODUCTION_CONFIG["max_extra"],
            max_per_seed=PRODUCTION_CONFIG["max_per_seed"],
        ),
        generate=lambda query, results: generate(query, results, stream=False),
    )
