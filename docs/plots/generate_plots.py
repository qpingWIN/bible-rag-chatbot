"""
Generate all README plots from the ablation results JSON.

Produces three PNGs in docs/plots/:
  recall_curve.png         — Recall@k for key configs across k=1..20
  umap_embedding.png       — UMAP projection of 3,000 sampled passage embeddings
  query_type_breakdown.png — Grouped bar chart of nDCG@5 by query type and config

Usage:
    python docs/plots/generate_plots.py                           # all plots
    python docs/plots/generate_plots.py --plot recall             # single plot
    python docs/plots/generate_plots.py --umap-sample 2000       # UMAP with 2k points
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

ROOT = Path(__file__).parent.parent.parent
RESULTS_PATH = ROOT / "eval" / "results" / "ablation_results.json"
OUT_DIR = Path(__file__).parent
OUT_DIR.mkdir(parents=True, exist_ok=True)

STYLE = {
    "figure.facecolor": "white",
    "axes.facecolor": "#f8f8f8",
    "axes.grid": True,
    "grid.color": "#e0e0e0",
    "grid.linestyle": "--",
    "font.family": "sans-serif",
    "font.size": 11,
}
plt.rcParams.update(STYLE)


# ── Colour palette ────────────────────────────────────────────────────────────

PALETTE = {
    "baseline":       "#4E79A7",
    "emb_bge_small":  "#F28E2B",
    "emb_nomic":      "#E15759",
    "emb_bge_large":  "#59A14F",
    "hybrid_bm25_05": "#B07AA1",
    "reranker":       "#FF9DA7",
    "no_xref":        "#9C755F",
    "no_dedup":       "#BAB0AC",
}

DISPLAY_NAMES = {
    "baseline":       "MiniLM-L6 (baseline)",
    "emb_bge_small":  "BGE-small",
    "emb_nomic":      "nomic-embed-text",
    "emb_bge_large":  "BGE-large",
    "hybrid_bm25_05": "Hybrid BM25+dense",
    "reranker":       "+ cross-encoder reranker",
    "no_xref":        "dense, xref=off",
    "no_dedup":       "dense, dedup=off",
}


# ── Plot 1: Recall@k curve ────────────────────────────────────────────────────

def plot_recall_curve(results: list[dict], out_path: Path) -> None:
    CONFIGS_TO_PLOT = [
        "baseline", "emb_bge_small", "emb_nomic", "emb_bge_large",
        "hybrid_bm25_05", "reranker", "no_xref",
    ]
    K_VALUES = [1, 3, 5, 10, 20]

    fig, ax = plt.subplots(figsize=(9, 5.5))

    for row in results:
        name = row["config"]["name"]
        if name not in CONFIGS_TO_PLOT:
            continue

        recalls = []
        lo_errs = []
        hi_errs = []
        for k in K_VALUES:
            r = row["per_k"].get(str(k), row["per_k"].get(k, {})).get("recall", {})
            recalls.append(r.get("mean", np.nan))
            lo_errs.append(r.get("mean", np.nan) - r.get("ci_lo", r.get("mean", np.nan)))
            hi_errs.append(r.get("ci_hi", r.get("mean", np.nan)) - r.get("mean", np.nan))

        lw = 2.5 if name in ("reranker", "emb_bge_large") else 1.5
        ls = "-" if name not in ("no_xref",) else "--"
        color = PALETTE.get(name, "#888888")
        label = DISPLAY_NAMES.get(name, name)

        ax.plot(K_VALUES, recalls, lw=lw, ls=ls, color=color, marker="o",
                markersize=5, label=label)
        ax.fill_between(
            K_VALUES,
            [r - lo for r, lo in zip(recalls, lo_errs)],
            [r + hi for r, hi in zip(recalls, hi_errs)],
            alpha=0.10, color=color,
        )

    ax.set_xlabel("k (number of passages retrieved)", fontsize=12)
    ax.set_ylabel("Recall@k", fontsize=12)
    ax.set_title("Recall@k across retrieval configurations\n"
                 "(50-question eval set, 95% bootstrap CI)", fontsize=13)
    ax.set_xticks(K_VALUES)
    ax.set_ylim(0.25, 1.02)
    ax.legend(loc="lower right", fontsize=9.5, framealpha=0.9)
    ax.annotate(
        "Shaded bands = 95% bootstrap CI",
        xy=(0.02, 0.96), xycoords="axes fraction",
        fontsize=8.5, color="#555555",
    )

    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"Saved: {out_path}")


# ── Plot 2: UMAP embedding space ──────────────────────────────────────────────

def plot_umap(out_path: Path, n_sample: int = 3000) -> None:
    """
    Compute a UMAP projection of n_sample passage embeddings.
    Requires a built FAISS index (run build_index.py first).
    Coloured by source: KJV (blue), BSB (orange), MHC commentary (green).
    """
    try:
        import umap
    except ImportError:
        print("umap-learn not installed — skipping UMAP plot. pip install umap-learn")
        return

    sys.path.insert(0, str(ROOT))
    from src.retriever import load_index
    from src.embedder import load_model, embed_texts

    print("Loading index for UMAP...")
    faiss_index, chunks_meta = load_index()

    rng = np.random.default_rng(42)
    n_total = len(chunks_meta)
    idx = rng.choice(n_total, size=min(n_sample, n_total), replace=False)

    embeddings = np.zeros((len(idx), faiss_index.d), dtype=np.float32)
    for i, j in enumerate(idx):
        vec = faiss_index.reconstruct(int(j))
        embeddings[i] = vec

    labels = [chunks_meta[int(j)]["source"] for j in idx]
    books  = [chunks_meta[int(j)].get("book", "") for j in idx]

    testament_map = _build_testament_map()
    testament_labels = [
        "New Testament" if testament_map.get(b) == "nt"
        else "Old Testament" if testament_map.get(b) == "ot"
        else "Commentary"
        for b in books
    ]

    print(f"Running UMAP on {len(embeddings)} embeddings...")
    reducer = umap.UMAP(n_components=2, n_neighbors=30, min_dist=0.1,
                        metric="cosine", random_state=42, verbose=False)
    proj = reducer.fit_transform(embeddings)

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # Left: by source
    source_colors = {"kjv": "#4E79A7", "bsb": "#F28E2B", "mhc": "#59A14F"}
    for src, color in source_colors.items():
        mask = [l == src for l in labels]
        axes[0].scatter(proj[mask, 0], proj[mask, 1],
                        c=color, s=2, alpha=0.35, label=src.upper())
    axes[0].set_title("Embedding space by source", fontsize=12)
    axes[0].legend(markerscale=4, fontsize=10)
    axes[0].axis("off")

    # Right: by testament
    testament_colors = {
        "Old Testament":  "#E15759",
        "New Testament":  "#4E79A7",
        "Commentary":     "#59A14F",
    }
    for test, color in testament_colors.items():
        mask = [l == test for l in testament_labels]
        axes[1].scatter(proj[mask, 0], proj[mask, 1],
                        c=color, s=2, alpha=0.35, label=test)
    axes[1].set_title("Embedding space by testament / source type", fontsize=12)
    axes[1].legend(markerscale=4, fontsize=10)
    axes[1].axis("off")

    fig.suptitle(f"UMAP projection of {len(embeddings):,} sampled passages "
                 f"(all-MiniLM-L6-v2, 384-dim → 2-dim)",
                 fontsize=13)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"Saved: {out_path}")


def _build_testament_map() -> dict[str, str]:
    ot = [
        "Genesis", "Exodus", "Leviticus", "Numbers", "Deuteronomy", "Joshua",
        "Judges", "Ruth", "1 Samuel", "2 Samuel", "1 Kings", "2 Kings",
        "1 Chronicles", "2 Chronicles", "Ezra", "Nehemiah", "Esther", "Job",
        "Psalms", "Proverbs", "Ecclesiastes", "Song of Solomon", "Isaiah",
        "Jeremiah", "Lamentations", "Ezekiel", "Daniel", "Hosea", "Joel",
        "Amos", "Obadiah", "Jonah", "Micah", "Nahum", "Habakkuk", "Zephaniah",
        "Haggai", "Zechariah", "Malachi",
    ]
    nt = [
        "Matthew", "Mark", "Luke", "John", "Acts", "Romans",
        "1 Corinthians", "2 Corinthians", "Galatians", "Ephesians",
        "Philippians", "Colossians", "1 Thessalonians", "2 Thessalonians",
        "1 Timothy", "2 Timothy", "Titus", "Philemon", "Hebrews", "James",
        "1 Peter", "2 Peter", "1 John", "2 John", "3 John", "Jude", "Revelation",
    ]
    return {b: "ot" for b in ot} | {b: "nt" for b in nt}


# ── Plot 3: Query-type breakdown ──────────────────────────────────────────────

def plot_query_type_breakdown(results: list[dict], out_path: Path) -> None:
    CONFIGS_TO_PLOT = [
        "baseline", "emb_nomic", "emb_bge_large", "hybrid_bm25_05", "reranker",
    ]
    QUERY_TYPES = ["factual", "thematic", "narrative", "cross_reference"]
    QTYPE_LABELS = {
        "factual": "Factual\nlookup",
        "thematic": "Thematic",
        "narrative": "Narrative",
        "cross_reference": "Cross-\nreference",
    }
    K = 5

    config_rows = {
        r["config"]["name"]: r
        for r in results
        if r["config"]["name"] in CONFIGS_TO_PLOT
    }

    x = np.arange(len(QUERY_TYPES))
    n_configs = len(CONFIGS_TO_PLOT)
    bar_width = 0.14
    offsets = np.linspace(-(n_configs - 1) * bar_width / 2,
                          (n_configs - 1) * bar_width / 2, n_configs)

    fig, ax = plt.subplots(figsize=(11, 5.5))

    for i, cfg_name in enumerate(CONFIGS_TO_PLOT):
        row = config_rows.get(cfg_name)
        if row is None:
            continue
        by_qt = row.get("by_query_type", {})
        values = []
        ci_lo = []
        ci_hi = []
        for qt in QUERY_TYPES:
            qt_data = by_qt.get(qt, {})
            pk = qt_data.get("per_k", {}).get(str(K), qt_data.get("per_k", {}).get(K, {}))
            ndcg = pk.get("ndcg", {})
            m = ndcg.get("mean", 0.0)
            lo = ndcg.get("ci_lo", m)
            hi = ndcg.get("ci_hi", m)
            values.append(m)
            ci_lo.append(m - lo)
            ci_hi.append(hi - m)

        color = PALETTE.get(cfg_name, "#888888")
        label = DISPLAY_NAMES.get(cfg_name, cfg_name)
        bars = ax.bar(x + offsets[i], values, bar_width, label=label,
                      color=color, alpha=0.85)
        ax.errorbar(x + offsets[i], values,
                    yerr=[ci_lo, ci_hi],
                    fmt="none", color="black", capsize=3, linewidth=1)

    ax.set_xticks(x)
    ax.set_xticklabels([QTYPE_LABELS[qt] for qt in QUERY_TYPES], fontsize=11)
    ax.set_ylabel(f"nDCG@{K}", fontsize=12)
    ax.set_title(f"nDCG@{K} by query type and configuration\n"
                 "(error bars = 95% bootstrap CI)", fontsize=13)
    ax.set_ylim(0, 1.05)
    ax.legend(fontsize=9, loc="upper right", framealpha=0.9)

    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"Saved: {out_path}")


# ── CLI ───────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Generate README plots.")
    parser.add_argument("--plot", choices=["recall", "umap", "breakdown", "all"], default="all")
    parser.add_argument("--umap-sample", type=int, default=3000)
    args = parser.parse_args()

    results = json.loads(RESULTS_PATH.read_text())

    if args.plot in ("recall", "all"):
        plot_recall_curve(results, OUT_DIR / "recall_curve.png")

    if args.plot in ("breakdown", "all"):
        plot_query_type_breakdown(results, OUT_DIR / "query_type_breakdown.png")

    if args.plot in ("umap", "all"):
        plot_umap(OUT_DIR / "umap_embedding.png", n_sample=args.umap_sample)


if __name__ == "__main__":
    main()
