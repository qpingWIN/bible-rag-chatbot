"""Tests for bootstrap.py, run with pytest"""

import pytest

from eval.scoring.bootstrap import bootstrap_ci, summarise_with_ci


def make_result(category="factual", passed=True, recall=1.0, rr=1.0):
    return {
        "category": category,
        "passed": passed,
        "recall_at_k": recall,
        "reciprocal_rank": rr,
    }


class TestBootstrapCI:
    def test_constant_values_give_degenerate_ci(self):
        point, lo, hi = bootstrap_ci([0.5] * 20)
        assert point == lo == hi == 0.5

    def test_point_estimate_is_sample_mean(self):
        values = [0.0, 0.0, 1.0, 1.0]
        point, _, _ = bootstrap_ci(values)
        assert point == 0.5

    def test_ci_brackets_the_mean(self):
        values = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0] * 7  # n=42
        point, lo, hi = bootstrap_ci(values)
        assert lo <= point <= hi
        assert lo < hi  # non-degenerate for varied data

    def test_seed_reproducibility(self):
        values = [0.1, 0.5, 0.9, 0.3, 0.7] * 8
        assert bootstrap_ci(values, seed=42) == bootstrap_ci(values, seed=42)
        assert bootstrap_ci(values, seed=1) != bootstrap_ci(values, seed=2)

    def test_more_data_narrows_ci(self):
        small = [0.0, 1.0] * 5
        large = [0.0, 1.0] * 100
        _, lo_s, hi_s = bootstrap_ci(small)
        _, lo_l, hi_l = bootstrap_ci(large)
        assert (hi_l - lo_l) < (hi_s - lo_s)

    def test_empty_values_raise(self):
        with pytest.raises(ValueError):
            bootstrap_ci([])


class TestSummariseWithCI:
    def test_groups_overall_and_per_category(self):
        results = [
            make_result("factual", passed=True),
            make_result("factual", passed=False),
            make_result("thematic", passed=True),
        ]
        summary = summarise_with_ci(results, n_resamples=100)
        assert set(summary) == {"overall", "factual", "thematic"}
        assert summary["overall"]["n"] == 3
        assert summary["factual"]["n"] == 2

    def test_pass_rate_mean_matches_fraction_passed(self):
        results = [make_result(passed=True)] * 3 + [make_result(passed=False)]
        summary = summarise_with_ci(results, n_resamples=100)
        assert summary["overall"]["pass_rate"]["mean"] == 0.75

    def test_all_metrics_present_with_ci_bounds(self):
        summary = summarise_with_ci([make_result()] * 5, n_resamples=100)
        for metric in ("pass_rate", "recall_at_k", "mrr"):
            m = summary["overall"][metric]
            assert m["ci95"][0] <= m["mean"] <= m["ci95"][1]
