"""Response generation via Ollama (local LLM)"""

from ollama import Client

OLLAMA_MODEL = "llama3.2"
OLLAMA_HOST = "http://localhost:11434"

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


def generate(
    query: str,
    results: list[dict],
    model: str = OLLAMA_MODEL,
    temperature: float = 0.0,
    stream: bool = True,
) -> str:
    client = Client(host=OLLAMA_HOST)
    messages = build_messages(query, results)

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
