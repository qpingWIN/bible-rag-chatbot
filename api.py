"""
FastAPI service exposing the Bible RAG pipeline over HTTP.

Run:
    uvicorn api:app --port 8000
Interactive docs (auto-generated):
    http://localhost:8000/docs

Endpoints:
    GET  /health   liveness + index stats
    POST /search   retrieval only (fast, no LLM) - returns ranked passages
    POST /ask      full RAG loop - retrieval + grounded generation via Ollama

Defaults are the production config found by the eval ablation study
(see README, Evaluation section): top_k=30, aggressive xref expansion.
"""

import os
import sys
from contextlib import asynccontextmanager
from pathlib import Path
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).parent))

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

# Production retrieval config - winner of the 11-config ablation study.
PRODUCTION_CONFIG = {
    "top_k": 30,
    "use_xrefs": True,
    "max_extra": 15,
    "max_per_seed": 10,
    "use_commentary": True,
}


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load the index, embedding model, and xref graph once at startup.

    Heavy imports (torch, faiss) happen here rather than at module import
    time, so the API module stays importable in test/CI environments
    (API_TEST_MODE=1) where those dependencies or the index are absent.
    """
    if os.environ.get("API_TEST_MODE") != "1":
        from src.embedder import load_model, embed_texts
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
            raise RuntimeError(
                "Index not found. Run `python build_index.py` first."
            )

        index, chunks_meta = load_index()
        model = load_model()
        xrefs = load_cross_references()
        ref_to_chunk = build_ref_to_chunk_index(chunks_meta)

        # The endpoints only ever touch these callables, which makes the
        # pipeline trivially fake-able in tests.
        app.state.rag = SimpleNamespace(
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
    yield


app = FastAPI(
    title="Bible RAG API",
    description=(
        "Retrieval-augmented question answering over KJV, BSB, and Matthew "
        "Henry's Commentary. Fully local: sentence-transformers + FAISS for "
        "retrieval, llama3.2 via Ollama for generation."
    ),
    version="1.0.0",
    lifespan=lifespan,
)


# ---------------------------------------------------------------------------
# Schemas
# ---------------------------------------------------------------------------

class AskRequest(BaseModel):
    question: str = Field(
        ...,
        min_length=3,
        max_length=500,
        examples=["What does the Bible say about faith?"],
    )
    top_k: int = Field(
        PRODUCTION_CONFIG["top_k"], ge=1, le=50,
        description="Number of passages to retrieve before xref expansion.",
    )
    use_commentary: bool = Field(
        PRODUCTION_CONFIG["use_commentary"],
        description="Include Matthew Henry commentary chunks.",
    )
    use_xrefs: bool = Field(
        PRODUCTION_CONFIG["use_xrefs"],
        description="Expand results via the OpenBible cross-reference graph.",
    )


class Passage(BaseModel):
    source: str = Field(description="kjv | bsb | mhc")
    reference: str
    score: float = Field(description="Cosine similarity; 0.0 for xref-added passages.")
    text: str
    xref_from: str | None = Field(
        None, description="Set when the passage was added via cross-reference expansion."
    )


class SearchResponse(BaseModel):
    question: str
    passages: list[Passage]


class AskResponse(SearchResponse):
    answer: str


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _get_rag(app: FastAPI) -> SimpleNamespace:
    rag = getattr(app.state, "rag", None)
    if rag is None:
        raise HTTPException(status_code=503, detail="RAG pipeline not initialised.")
    return rag


def _retrieve(rag: SimpleNamespace, req: AskRequest) -> list[dict]:
    emb = rag.embed([req.question])
    results = rag.search(emb, req.top_k)
    if not req.use_commentary:
        results = [r for r in results if r["source"] != "mhc"]
    if req.use_xrefs:
        results = rag.expand(results)
    return results


def _to_passages(results: list[dict]) -> list[Passage]:
    return [
        Passage(
            source=r["source"],
            reference=r["reference"],
            score=r["score"],
            text=r["text"],
            xref_from=r.get("xref_from"),
        )
        for r in results
    ]


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------

@app.get("/health")
def health():
    rag = getattr(app.state, "rag", None)
    return {
        "status": "ok" if rag else "starting",
        "index_size": rag.index_size if rag else None,
    }


@app.post("/search", response_model=SearchResponse)
def search_passages(req: AskRequest):
    """Retrieval only - no LLM call. Useful for inspecting what the
    generator would be grounded on as well as for latency-sensitive callers."""
    rag = _get_rag(app)
    results = _retrieve(rag, req)
    return SearchResponse(question=req.question, passages=_to_passages(results))


@app.post("/ask", response_model=AskResponse)
def ask(req: AskRequest):
    """Full RAG loop: retrieve, expand, generate a grounded, cited answer."""
    rag = _get_rag(app)
    results = _retrieve(rag, req)
    if not results:
        raise HTTPException(status_code=404, detail="No passages matched the query.")
    try:
        answer = rag.generate(req.question, results)
    except Exception as e:  # Ollama down / model missing
        raise HTTPException(
            status_code=503,
            detail=f"Generation backend unavailable: {e}. Is Ollama running?",
        ) from e
    return AskResponse(
        question=req.question,
        passages=_to_passages(results),
        answer=answer,
    )


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("api:app", host="127.0.0.1", port=8000)
