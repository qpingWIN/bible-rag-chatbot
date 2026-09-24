"""
FastAPI service exposing the Bible RAG pipeline over HTTP.

Run:
    uvicorn api:app --port 8000
Interactive docs (auto-generated):
    http://localhost:8000/docs

Endpoints:
    GET  /health   liveness + index stats
    POST /search   retrieval only (fast, no LLM) - returns ranked passages
    POST /ask      full RAG loop - retrieval + grounded generation via Ollama/Groq

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
from src.backend import backend_label

# Production retrieval config - winner of the 11-config ablation study.
# Defined once in src/pipeline.py, shared with the Gradio app.
from src.pipeline import PRODUCTION_CONFIG, get_pipeline


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load the index, embedding model, and xref graph once at startup.

    Heavy imports (torch, faiss) happen here rather than at module import
    time, so the API module stays importable in test/CI environments
    (API_TEST_MODE=1) where those dependencies or the index are absent.
    """
    if os.environ.get("API_TEST_MODE") != "1":
        # get_pipeline() is lru_cached: if the Gradio app (mounted below)
        # already loaded it, this returns the same objects instantly.
        # The endpoints only ever touch the pipeline's four callables,
        # which makes it trivially fake-able in tests.
        app.state.rag = get_pipeline()
    yield


app = FastAPI(
    title="Bible RAG API",
    description=(
        "Retrieval-augmented question answering over KJV, BSB, and Matthew "
        "Henry's Commentary. sentence-transformers + FAISS for retrieval, "
        f"{backend_label()} for generation."
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
    except Exception as e:  # generation backend down (Ollama or Groq)
        raise HTTPException(
            status_code=503,
            detail=f"Generation backend unavailable: {e}",
        ) from e
    return AskResponse(
        question=req.question,
        passages=_to_passages(results),
        answer=answer,
    )


# ---------------------------------------------------------------------------
# Gradio UI mounted at "/" — one process serves both the demo and the API.
# Skipped in tests: importing app would load the full pipeline.
# ---------------------------------------------------------------------------

if os.environ.get("API_TEST_MODE") != "1":
    import gradio as gr

    from app import demo

    # Same theme as the standalone app.py launch — mount bypasses launch(),
    # so theming must be applied here too
    app = gr.mount_gradio_app(app, demo, path="/", theme=gr.themes.Soft())


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("api:app", host="127.0.0.1", port=8000)
