"""Response generation: local Ollama by default, Groq API when GROQ_API_KEY is set.

The prompt, grounding rules, and temperature are identical across backends -
only the transport differs. The Groq path exists for the hosted demo (HF
Spaces has no GPU for local inference); local runs stay fully offline.
"""

import os

import httpx
from ollama import Client

OLLAMA_MODEL = "llama3.2"
OLLAMA_HOST = "http://localhost:11434"

GROQ_MODEL = os.environ.get("GROQ_MODEL", "openai/gpt-oss-120b")
GROQ_URL = "https://api.groq.com/openai/v1/chat/completions"

SYSTEM_TEMPLATE = """\
You are a biblical scholar who answers questions using ONLY the provided source passages.
Rules:
- Base your answer strictly on the passages below. Do not add knowledge from outside these passages.
- After each claim, cite the source in parentheses: (KJV Genesis 1:1) or (MHC Matthew 5 commentary).
- If the passages don't contain enough information to answer, say: "The provided passages don't address this directly".
- Be concise but complete. No padding.

SOURCE PASSAGES:
{context}
"""


def _format_context(results: list[dict]) -> str:
    lines = []
    for i, r in enumerate(results, 1):
        source_label = r["source"].upper()
        ref = r["reference"]
        text = r["text"].strip()
        lines.append(f"[{i}] ({source_label} {ref})\n{text}")
    return "\n\n".join(lines)


def build_messages(query: str, results: list[dict]) -> list[dict]:
    context = _format_context(results)
    system_prompt = SYSTEM_TEMPLATE.format(context=context)
    return [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": query},
    ]


def _generate_groq(messages: list[dict], temperature: float) -> str:
    """OpenAI-compatible chat completion against Groq's hosted llama."""
    response = httpx.post(
        GROQ_URL,
        headers={"Authorization": f"Bearer {os.environ['GROQ_API_KEY']}"},
        json={
            "model": GROQ_MODEL,
            "messages": messages,
            "temperature": temperature,
        },
        timeout=60.0,
    )
    response.raise_for_status()
    return response.json()["choices"][0]["message"]["content"]


def generate(
    query: str,
    results: list[dict],
    model: str = OLLAMA_MODEL,
    temperature: float = 0.0,
    stream: bool = True,
) -> str:
    messages = build_messages(query, results)

    if os.environ.get("GROQ_API_KEY"):
        return _generate_groq(messages, temperature)

    client = Client(host=OLLAMA_HOST)

    if stream:
        full_response = []
        for chunk in client.chat(
            model=model,
            messages=messages,
            options={"temperature": temperature},
            stream=True,
        ):
            token = chunk["message"]["content"]
            print(token, end="", flush=True)
            full_response.append(token)
        print()
        return "".join(full_response)
    else:
        response = client.chat(
            model=model,
            messages=messages,
            options={"temperature": temperature},
        )
        return response["message"]["content"]


def list_available_models() -> list[str]:
    client = Client(host=OLLAMA_HOST)
    return [m["name"] for m in client.list()["models"]]
