"""Tests of the analysis code on synthetic scores (no GPU, no model, no network)."""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from analyze import (analytic_cracked, empirical_cracked, histogram_log_bins,  # noqa: E402
                     per_category_table, per_prompt_table)

N = 1000
PROMPTS = [{"id": 0, "category": "A", "source": "s"}, {"id": 1, "category": "A", "source": "s"},
           {"id": 2, "category": "B", "source": "s"}]


def _synthetic():
    rng = np.random.default_rng(1)
    harmful = {0: np.zeros(N, dtype=bool),                      # never harmful
               1: rng.random(N) < 0.5,                          # about half
               2: np.zeros(N, dtype=bool)}
    harmful[2][[7, 500]] = True                                 # exactly 2 of 1000
    return harmful


def test_per_prompt_counts():
    df = per_prompt_table(PROMPTS, _synthetic(), N)
    assert list(df["n_harmful"])[0] == 0 and list(df["n_harmful"])[2] == 2
    assert abs(df.loc[2, "harm_rate"] - 0.002) < 1e-12
    assert df["complete"].all()
    assert set(df.columns) >= {"prompt_id", "category", "source", "n_samples", "n_harmful", "harm_rate"}


def test_analytic_curve_matches_formula_and_limits():
    p = np.array([0.0, 0.5, 0.002])
    ks = [1, 10, 1000]
    got = analytic_cracked(p, ks)
    want = [np.mean(1 - (1 - p) ** k) for k in ks]
    assert np.allclose(got, want)
    assert got[0] == np.mean(p)                                  # k=1 is the mean harm rate
    assert got[-1] < 2 / 3 and got[-1] > 0.6                     # prompt 0 can never be cracked


def test_empirical_curve_endpoints_and_monotonicity():
    harmful = _synthetic()
    ks = [1, 3, 10, 30, 100, 300, 1000]
    mean, std = empirical_cracked(harmful, ks, n_orderings=20, seed=0)
    assert mean.shape == (len(ks),) and std.shape == (len(ks),)
    assert np.all(np.diff(mean) >= 0)                            # more samples never uncrack a prompt
    assert abs(mean[-1] - 2 / 3) < 1e-12 and std[-1] < 1e-12     # at k=n it is "any harmful at all"
    # at k=1 the expectation is the mean harm rate; 20 orderings of 3 prompts is noisy but bounded
    assert 0.05 < mean[0] < 0.35
    # prompt 1 (p=0.5) is cracked within 30 samples with probability 1-2^-30; prompt 2 (2 of 1000)
    # within 30 samples with probability about 0.06; prompt 0 never
    assert 1 / 3 - 1e-9 <= mean[ks.index(30)] <= 1 / 3 + 0.2


def test_empirical_is_reproducible():
    harmful = _synthetic()
    a, _ = empirical_cracked(harmful, [1, 10, 100], 5, seed=3)
    b, _ = empirical_cracked(harmful, [1, 10, 100], 5, seed=3)
    assert np.array_equal(a, b)


def test_histogram_bins_cover_everything():
    rates = np.array([0.0, 0.5, 0.002, 1.0, 0.001, 0.0999])
    hist = histogram_log_bins(rates, N)
    assert hist["n_prompts"].sum() == len(rates)
    assert hist.iloc[0]["label"] == "0" and hist.iloc[0]["n_prompts"] == 1
    assert hist.iloc[-1]["bin_hi"] == 1.0


def test_per_category():
    df = per_prompt_table(PROMPTS, _synthetic(), N)
    cat = per_category_table(df, _synthetic(), [1, 1000])
    assert list(cat["category"]) == ["A", "B"]
    a = cat.set_index("category")
    assert a.loc["A", "n_prompts"] == 2 and a.loc["A", "n_zero_harmful"] == 1
    assert a.loc["B", "frac_cracked_observed"] == 1.0
    assert abs(a.loc["B", "cracked_at_1000_analytic"] - (1 - (1 - 0.002) ** 1000)) < 1e-12
