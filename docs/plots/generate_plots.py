"""Generate the README ablation plot from the documented results.

Produces docs/plots/ablation_overview.png: overall pass-rate and mean
Recall@k for all 11 evaluated configurations, production highlighted.

Numbers mirror eval/notebooks/ablation_summary.md (overall column),
which in turn is backed by the per-run JSONs in eval/results/.

Usage:
    python docs/plots/generate_plots.py
"""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

OUT_DIR = Path(__file__).parent

# (label, passed out of 42, mean recall@k, is_production)
# Rows 1-11 of the ablation matrix in eval/notebooks/ablation_summary.md.
CONFIGS = [
    ("Dense k=10 (baseline)",            14, 0.225, False),
    ("Dense k=20",                       19, 0.291, False),
    ("Dense k=20, no MHC",               15, 0.268, False),
    ("Dense k=20, no xref",              19, 0.291, False),
    ("Dense k=20, xref 3/10",            19, 0.313, False),
    ("Dense k=30, xref 10/15 (prod)",    21, 0.381, True),
    ("Hybrid k=10, a=.5, mult=3",        14, 0.209, False),
    ("Hybrid k=10, a=.5, mult=10",       12, 0.158, False),
    ("Hybrid k=10, a=.75, mult=3",       13, 0.209, False),
    ("Hybrid k=20, a=.5, mult=3",        16, 0.286, False),
    ("Dense k=30, xref 10/15, BGE-base", 22, 0.336, False),
]

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

PROD_COLOR = "#4E79A7"
OTHER_COLOR = "#BAB0AC"


def main() -> None:
    labels = [c[0] for c in CONFIGS][::-1]
    pass_rates = [c[1] / 42 for c in CONFIGS][::-1]
    recalls = [c[2] for c in CONFIGS][::-1]
    colors = [PROD_COLOR if c[3] else OTHER_COLOR for c in CONFIGS][::-1]

    fig, (ax1, ax2) = plt.subplots(
        1, 2, figsize=(11, 4.5), sharey=True, layout="constrained"
    )

    y = range(len(labels))
    ax1.barh(y, pass_rates, color=colors)
    ax1.set_yticks(y, labels)
    ax1.set_xlabel("Pass-rate (of 42 questions)")
    ax1.set_xlim(0, 0.65)
    for i, v in enumerate(pass_rates):
        ax1.text(v + 0.01, i, f"{round(v * 42)}/42", va="center", fontsize=9)

    ax2.barh(y, recalls, color=colors)
    ax2.set_xlabel("Mean Recall@k")
    ax2.set_xlim(0, 0.45)
    for i, v in enumerate(recalls):
        ax2.text(v + 0.005, i, f"{v:.3f}", va="center", fontsize=9)

    fig.suptitle(
        "Retrieval ablation: 11 configurations on the 42-question gold set",
        fontsize=12,
    )

    out = OUT_DIR / "ablation_overview.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    print(f"Saved {out}")


if __name__ == "__main__":
    main()
