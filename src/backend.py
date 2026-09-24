"""Model names and the backend label, shared by generator.py, app.py and api.py.

Kept free of heavy imports (no ollama) so api.py stays importable in CI.
"""

import os

OLLAMA_MODEL = "llama3.2"
OLLAMA_HOST = "http://localhost:11434"
GROQ_MODEL = os.environ.get("GROQ_MODEL", "openai/gpt-oss-120b")


def backend_label() -> str:
    if os.environ.get("GROQ_API_KEY"):
        return f"{GROQ_MODEL} via the Groq API"
    return f"{OLLAMA_MODEL} via Ollama (fully local)"