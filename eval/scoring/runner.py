"""Run the eval end-to-end: load questions, query retriever, score, save results

Loads the FAISS index and embedding model once, then for each question in
eval/questions.jsonl embeds the question, retrieves chunks, scores against
gold labels and records the result

Output: eval/results/run_YYYYMMDD_HHMMSS.json
Each run file contains the config used and per-question results so any run
can be reproduced or compared without re-running.

config = {
    "top_k": 10,
    "use_xrefs": True,
    "max_extra": 3,
    "max_per_seed": 1,
    "embedding_model": "sentence-transformers/all-MiniLM-L6-v2",
    "use_commentary": True,
}

"""

import sys 
import json
from pathlib import Path
from datetime import datetime

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.ingest import load_cross_references
from src.embedder import load_model, embed_texts
from src.retriever import (
    load_index,
    search,
    expand_with_cross_references,
    build_ref_to_chunk_index,
)
from eval.scoring.scorer import score_question

QUESTIONS_PATH = Path(__file__).parent.parent / "questions.jsonl"
RESULTS_DIR = Path(__file__).parent.parent / "results"

def load_questions() -> list[dict]:
    questions = []
    with open(QUESTIONS_PATH) as f:
        for line in f:
            line = line.strip()
            if line:
                questions.append(json.loads(line))
    return [q for q in questions if not q.get("excluded", False)]

def run_eval(config:dict) -> list[dict]:
    """Run the full eval for a given config. Returns result dicts for every question"""

    print("Loading index...")
    index, chunks_meta = load_index()
    print("Loading embedding model...")
    embed_model = load_model()
    print("Loading cross-references...")
    xrefs = load_cross_references()
    ref_to_chunk = build_ref_to_chunk_index(chunks_meta)
    print("Loading questions...")
    questions = load_questions()
    print(f"  {len(questions)} questions loaded (excluded filtered out)")

    results = []
    for i,q in enumerate(questions,1):
        q_emb = embed_texts(embed_model, [q["question"]], show_progress=False)
        retrieved = search(index, chunks_meta, q_emb, top_k = config["top_k"])
        if config["use_xrefs"]:
            retrieved = expand_with_cross_references(
                retrieved, index, chunks_meta, xrefs, ref_to_chunk,
                max_extra=config["max_extra"],
                max_per_seed=config["max_per_seed"],
            )
        if not config["use_commentary"]:
            retrieved = [r for r in retrieved if r["source"] != "mhc"]

        score = score_question(q, retrieved)
        result = {
            "id": q["id"],
            "category": q["category"],
            "question": q["question"],
            **score,
            "retrieved": retrieved,
        }
        results.append(result)

        status = "PASS" if score["passed"] else "FAIL"
        print(f"  [{i:02d}/{len(questions)}] {q['id']} ({q['category']}) → {status}  hits={score['hits']}/{score['total_gold']}")
    return results

def summarise (results: list[dict]) -> dict:
    """Summarise results by category and overall"""

    from collections import defaultdict
    cats: dict[str, dict] = defaultdict(lambda: {"passed": 0, "total": 0})

    for r in results:
        cat = r["category"]
        cats[cat]["total"] += 1
        if r["passed"]:
            cats[cat]["passed"] += 1
    summary = {}

    for cat, counts in sorted(cats.items()):
        rate = counts["passed"]/counts["total"]
        summary[cat] = {
            "passed": counts["passed"],
            "total": counts["total"],
            "pass_rate": round(rate,3),
        }
    overall_passed = sum(r["passed"] for r in results)
    summary["overall"] = {
        "passed": overall_passed,
        "total": len(results),
        "pass_rate": round(overall_passed / len(results), 3),
    }
    return summary

def save_results(config: dict, results: list[dict], summary: dict) -> Path:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_path = RESULTS_DIR / f"run_{timestamp}.json"
    payload = {
        "config": config,
        "summary": summary,
        "results": results,
    }
    out_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2))
    return out_path

        
def print_summary(summary: dict) -> None:
    print("\n" + "=" * 50)
    print("EVAL SUMMARY")
    print("=" * 50)
    for cat, counts in summary.items():
        bar = "█" * counts["passed"] + "░" * (counts["total"] - counts["passed"])
        print(f"  {cat:<16} {bar}  {counts['passed']}/{counts['total']}  ({counts['pass_rate']:.0%})")
    print("=" * 50)


if __name__ == "__main__":
    config = {
        "top_k": 20,
        "use_xrefs": True,
        "max_extra": 3,
        "max_per_seed": 1,
        "embedding_model": "sentence-transformers/all-MiniLM-L6-v2",
        "use_commentary": False,
    }

    results = run_eval(config)
    summary = summarise(results)
    print_summary(summary)
    out_path = save_results(config, results, summary)
    print(f"\nResults saved → {out_path}")