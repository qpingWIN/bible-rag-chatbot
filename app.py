"""
Bible RAG chatbot — Gradio interface.

Run:
  python app.py
Then open http://localhost:7860 in your browser.

THE FULL RAG LOOP (what happens when you submit a question)
-----------------------------------------------------------
1. Your question is embedded into a 384-dim vector encoding its meaning.
2. FAISS performs nearest-neighbour search: finds the top-K chunks whose
   embedding vectors are closest (highest cosine similarity) to yours.
3. [Optional]Cross-reference expansion: if matched verses have theological links, we add those too.
4. The retrieved chunks are injected into the system prompt as context.
5. Ollama (local llama3.2) reads the context + your question and generates a grounded answer that cites sources.

Nothing here calls the internet. Everything runs on your machine.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

import gradio as gr
from src.ingest import load_cross_references
from src.embedder import load_model, embed_texts
from src.retriever import (
    load_index,
    index_exists,
    search,
    expand_with_cross_references,
    build_ref_to_chunk_index,
)
from src.generator import generate, list_available_models

import numpy as np

# Starting checks
if not index_exists():
    print("ERROR: Index not found. Run `python build_index.py` first.")
    sys.exit(1)

print("Loading index and embedding model...")
index, chunks_meta = load_index()
embed_model = load_model()

print("Loading cross-references...")
xrefs = load_cross_references()
ref_to_chunk = build_ref_to_chunk_index(chunks_meta)

print("Checking Ollama models...")
try:
    available = list_available_models()
    print(f"Available models: {available}")
except Exception as e:
    print(f"WARNING: Could not connect to Ollama: {e}")
    print("Make sure Ollama is running: ollama serve")

# Production retrieval config — winner of the 10-config eval ablation
# (see README, Evaluation section). top_k and xref aggressiveness interact:
# expansion only surfaces genuinely new verses with enough seed chunks.
PROD_TOP_K = 30
PROD_MAX_EXTRA = 15
PROD_MAX_PER_SEED = 10

# Core query function

def answer_question(
    query: str,
    top_k: int,
    use_commentary: bool,
    use_xrefs: bool,
    source_filter: str,
) -> tuple[str, str]:
    """
    Process a user question and return (answer, sources_panel).

    Parameters are wired directly to the Gradio UI controls.
    """
    if not query.strip():
        return "Please enter a question.", ""

    # 1. Embed the query
    q_emb = embed_texts(embed_model, [query], show_progress=False)   # shape (1, 384)

    # 2. Retrieve
    results = search(index, chunks_meta, q_emb, top_k=top_k)

    # 3. Filter by source if requested
    if source_filter != "All":
        src_map = {"KJV only": "kjv", "BSB only": "bsb", "Commentary only": "mhc"}
        target = src_map[source_filter]
        results = [r for r in results if r["source"] == target]
        if not results:
            return f"No {source_filter} chunks matched. Try a different source filter.", ""

    # 4. Remove commentary chunks if user toggled off
    if not use_commentary:
        results = [r for r in results if r["source"] != "mhc"]
        if not results:
            return "No verse chunks matched. Try enabling commentary.", ""

    # 5. Cross-reference expansion
    if use_xrefs:
        results = expand_with_cross_references(
            results, index, chunks_meta, xrefs, ref_to_chunk,
            max_extra=PROD_MAX_EXTRA, max_per_seed=PROD_MAX_PER_SEED,
        )

    # 6. Generate answer
    answer = generate(query, results, stream=False)

    # 7. Build sources panel
    source_lines = []
    for r in results:
        score_str = f"  [score: {r['score']:.3f}]" if r["score"] > 0 else "  [xref]"
        xref_note = f"  ← linked from {r.get('xref_from', '')}" if r.get("xref_from") else ""
        source_lines.append(
            f"**{r['source'].upper()} · {r['reference']}**{score_str}{xref_note}\n> {r['text'][:300]}..."
        )
    sources_panel = "\n\n---\n\n".join(source_lines)

    return answer, sources_panel


# Gradio UI

with gr.Blocks(title="Bible RAG") as demo:
    gr.Markdown("""
    # Bible RAG Chatbot
    Ask any question about the Bible. Answers are grounded in KJV, BSB, and
    Matthew Henry's Complete Commentary - running 100% locally via Ollama.
    """)

    with gr.Row():
        with gr.Column(scale=3):
            question = gr.Textbox(
                label="Your question",
                placeholder="e.g. What does the Bible say about forgiveness?",
                lines=2,
            )
            submit_btn = gr.Button("Ask", variant="primary")
            answer_box = gr.Markdown(label="Answer")

        with gr.Column(scale=2):
            gr.Markdown("### Retrieved passages")
            sources_box = gr.Markdown(label="Sources")

    with gr.Accordion("Search settings", open=False):
        with gr.Row():
            top_k = gr.Slider(
                minimum=3, maximum=PROD_TOP_K, value=PROD_TOP_K, step=1,
                label="Passages to retrieve (K)",
                info="More passages = more context but longer prompts",
            )
            source_filter = gr.Dropdown(
                choices=["All", "KJV only", "BSB only", "Commentary only"],
                value="All",
                label="Source filter",
            )
        with gr.Row():
            use_commentary = gr.Checkbox(
                value=True, label="Include Matthew Henry commentary"
            )
            use_xrefs = gr.Checkbox(
                value=True, label="Expand with cross-references"
            )

    examples = gr.Examples(
        examples=[
            ["What does the Bible say about faith?"],
            ["Who was Abraham and why is he important?"],
            ["What happened at the Last Supper?"],
            ["Explain the parable of the prodigal son."],
            ["What is the meaning of John 3:16?"],
            ["How did David defeat Goliath?"],
        ],
        inputs=question,
    )

    submit_btn.click(
        fn=answer_question,
        inputs=[question, top_k, use_commentary, use_xrefs, source_filter],
        outputs=[answer_box, sources_box],
    )
    question.submit(
        fn=answer_question,
        inputs=[question, top_k, use_commentary, use_xrefs, source_filter],
        outputs=[answer_box, sources_box],
    )


if __name__ == "__main__":
    # Gradio 6 API: theme is a launch() parameter (moved out of gr.Blocks in 6.0)
    demo.launch(server_port=7860, share=False, theme=gr.themes.Soft())
