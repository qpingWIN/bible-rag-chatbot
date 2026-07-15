"""Bootstrap confidence intervals for eval summary metrics.

The headline numbers (pass-rate 21/42, mean recall 0.381) are point
estimates over just 42 questions. This module quantifies how much they
could move under resampling: a percentile bootstrap draws B resamples of
the question set with replacement, recomputes the mean metric for each,
and reports the 2.5th/97.5th percentiles as a 95% CI.

Usage:
    python -m eval.scoring.bootstrap                          # latest run
    python -m eval.scoring.bootstrap eval/results/run_X.json  # specific run

Per-category CIs are printed too, but with 5-10 questions per category
they are wide by construction, thus we should treat them as honesty about uncertainty,
not as precise estimates.
"""

import json
import random
import sys
from pathlib import Path

RESULTS_DIR = Path(__file__).parent.parent / "results"

METRICS = {
    "pass_rate": "passed",
    "recall_at_k": "recall_at_k",
    "mrr": "reciprocal_rank",
}


def bootstrap_ci(
    values: list[float],
    n_resamples: int = 10_000,
    alpha: float = 0.05,
    seed: int = 42,
) -> tuple[float, float, float]:
    """Percentile-bootstrap CI for the mean of `values`.

    Returns (point_estimate, ci_low, ci_high). Seeded for reproducibility.
    """
    if not values:
        raise ValueError("bootstrap_ci needs at least one value")
    rng = random.Random(seed)
    n = len(values)
    means = sorted(
        sum(rng.choices(values, k=n)) / n for _ in range(n_resamples)
    )
    lo = means[int((alpha / 2) * n_resamples)]
    hi = means[min(int((1 - alpha / 2) * n_resamples), n_resamples - 1)]
    point = sum(values) / n
    return point, lo, hi


def summarise_with_ci(results: list[dict], **kwargs) -> dict:
    """Compute mean + 95% CI for each metric, overall and per category."""
    categories = sorted({r["category"] for r in results})
    groups = {"overall": results}
    groups.update(
        {cat: [r for r in results if r["category"] == cat] for cat in categories}
    )

    summary: dict = {}
    for name, group in groups.items():
        summary[name] = {"n": len(group)}
        for label, key in METRICS.items():
            values = [float(r[key]) for r in group]
            point, lo, hi = bootstrap_ci(values, **kwargs)
            summary[name][label] = {
                "mean": round(point, 3),
                "ci95": [round(lo, 3), round(hi, 3)],
            }
    return summary


def latest_run() -> Path:
    runs = sorted(RESULTS_DIR.glob("run_*.json"))
    if not runs:
        raise FileNotFoundError(f"No run files in {RESULTS_DIR}")
    return runs[-1]


def print_summary(summary: dict, run_path: Path) -> None:
    print(f"\nBootstrap 95% CIs (10,000 resamples) for {run_path.name}")
    print("=" * 100)
    header = f"  {'group':<14} {'n':>3}"
    for label in METRICS:
        header += f"  {label + ' [95% CI]':>26}"
    print(header)
    print("  " + "-" * 98)
    for name, stats in summary.items():
        row = f"  {name:<14} {stats['n']:>3}"
        for label in METRICS:
            m = stats[label]
            row += f"    {m['mean']:>6.3f} [{m['ci95'][0]:.3f}, {m['ci95'][1]:.3f}]"
        print(row)
    print("=" * 100)


if __name__ == "__main__":
    run_path = Path(sys.argv[1]) if len(sys.argv) > 1 else latest_run()
    run = json.loads(run_path.read_text())
    summary = summarise_with_ci(run["results"])
    print_summary(summary, run_path)
