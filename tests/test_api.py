"""API contract tests. Run with pytest.

These run without the FAISS index, torch, or Ollama: API_TEST_MODE=1 skips
the real pipeline load, and a fake pipeline is injected into app.state.
That keeps the suite runnable in CI in seconds.
"""

import os
from types import SimpleNamespace

os.environ["API_TEST_MODE"] = "1"

import pytest
from fastapi.testclient import TestClient

from api import app

FAKE_CHUNK = {
    "source": "kjv",
    "reference": "John 3:16",
    "score": 0.87,
    "text": "For God so loved the world...",
    "book": "John",
    "chapter": 3,
    "verse": 16,
}

MHC_CHUNK = {
    "source": "mhc",
    "reference": "John 3 commentary",
    "score": 0.55,
    "text": "Herein God commended his love...",
    "book": "John",
    "chapter": 3,
    "verse": None,
}


def make_fake_rag(results=None, generate_error=None):
    results = results if results is not None else [FAKE_CHUNK, MHC_CHUNK]

    def fake_generate(query, res):
        if generate_error:
            raise generate_error
        return "A grounded answer (KJV John 3:16)."

    return SimpleNamespace(
        index_size=95_000,
        embed=lambda texts: "fake-embedding",
        search=lambda emb, k: [dict(r) for r in results][:k],
        expand=lambda res: res + [{**FAKE_CHUNK, "reference": "Romans 5:8",
                                   "score": 0.0, "xref_from": "John 3:16"}],
        generate=fake_generate,
    )


@pytest.fixture
def client():
    app.state.rag = make_fake_rag()
    with TestClient(app) as c:
        yield c


def test_health(client):
    r = client.get("/health")
    assert r.status_code == 200
    assert r.json() == {"status": "ok", "index_size": 95_000}


def test_search_returns_passages(client):
    r = client.post("/search", json={"question": "What is love?"})
    assert r.status_code == 200
    body = r.json()
    assert body["question"] == "What is love?"
    refs = [p["reference"] for p in body["passages"]]
    assert "John 3:16" in refs
    assert "Romans 5:8" in refs  # added by xref expansion


def test_search_xrefs_off(client):
    r = client.post("/search", json={"question": "What is love?", "use_xrefs": False})
    refs = [p["reference"] for p in r.json()["passages"]]
    assert "Romans 5:8" not in refs


def test_search_commentary_off(client):
    r = client.post(
        "/search",
        json={"question": "What is love?", "use_commentary": False, "use_xrefs": False},
    )
    sources = {p["source"] for p in r.json()["passages"]}
    assert "mhc" not in sources


def test_ask_returns_grounded_answer(client):
    r = client.post("/ask", json={"question": "What is love?"})
    assert r.status_code == 200
    body = r.json()
    assert body["answer"].startswith("A grounded answer")
    assert len(body["passages"]) > 0


def test_ask_validation_rejects_empty_question(client):
    r = client.post("/ask", json={"question": ""})
    assert r.status_code == 422


def test_ask_validation_rejects_bad_top_k(client):
    r = client.post("/ask", json={"question": "What is love?", "top_k": 999})
    assert r.status_code == 422


def test_ask_503_when_generation_backend_down():
    app.state.rag = make_fake_rag(generate_error=ConnectionError("refused"))
    with TestClient(app) as c:
        r = c.post("/ask", json={"question": "What is love?"})
    assert r.status_code == 503
    assert "unavailable" in r.json()["detail"]


def test_503_when_pipeline_not_loaded():
    app.state.rag = None
    with TestClient(app) as c:
        r = c.post("/search", json={"question": "What is love?"})
    assert r.status_code == 503
