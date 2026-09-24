from src.backend import GROQ_MODEL, backend_label


def test_label_local(monkeypatch):
    monkeypatch.delenv("GROQ_API_KEY", raising=False)
    assert backend_label() == "llama3.2 via Ollama (fully local)"


def test_label_groq(monkeypatch):
    monkeypatch.setenv("GROQ_API_KEY", "test")
    assert backend_label() == f"{GROQ_MODEL} via the Groq API"