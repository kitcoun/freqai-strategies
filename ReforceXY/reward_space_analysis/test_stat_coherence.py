#!/usr/bin/env python3
"""Quick statistical coherence checks for reward_space_analysis.

- Validates distribution shift metrics (KL, JS distance, Wasserstein, KS)
- Validates hypothesis tests: Spearman (idle), Kruskal ε², Mann–Whitney rank-biserial
- Validates QQ R^2 from distribution diagnostics
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from reward_space_analysis import (
    compute_distribution_shift_metrics,
    distribution_diagnostics,
    statistical_hypothesis_tests,
)


def make_df_for_spearman(n: int = 300) -> pd.DataFrame:
    rng = np.random.default_rng(42)
    idle = np.linspace(0, 100, n)
    # Strong negative relation: reward_idle decreases with idle_duration
    reward_idle = -0.1 * idle + rng.normal(0, 0.5, size=n)
    df = pd.DataFrame(
        {
            "idle_duration": idle,
            "reward_idle": reward_idle,
            # required fields (unused here)
            "reward_total": reward_idle,  # some signal
            "position": np.zeros(n),
            "is_force_exit": np.zeros(n),
            "reward_exit": np.zeros(n),
            "pnl": np.zeros(n),
            "trade_duration": np.zeros(n),
        }
    )
    return df


def make_df_for_kruskal(n_group: int = 100) -> pd.DataFrame:
    rng = np.random.default_rng(123)
    # three groups with different central tendencies
    g1 = rng.normal(0.0, 1.0, size=n_group)
    g2 = rng.normal(0.5, 1.0, size=n_group)
    g3 = rng.normal(1.0, 1.0, size=n_group)
    reward_total = np.concatenate([g1, g2, g3])
    position = np.concatenate(
        [np.full(n_group, 0.0), np.full(n_group, 0.5), np.full(n_group, 1.0)]
    )
    df = pd.DataFrame(
        {
            "reward_total": reward_total,
            "position": position,
            # fillers
            "reward_idle": np.zeros(n_group * 3),
            "is_force_exit": np.zeros(n_group * 3),
            "reward_exit": np.zeros(n_group * 3),
            "pnl": np.zeros(n_group * 3),
            "trade_duration": np.zeros(n_group * 3),
            "idle_duration": np.zeros(n_group * 3),
        }
    )
    return df


def make_df_for_mannwhitney(n: int = 200) -> pd.DataFrame:
    rng = np.random.default_rng(77)
    # force exits have larger rewards than regular exits
    force_exit_rewards = rng.normal(1.0, 0.5, size=n)
    regular_exit_rewards = rng.normal(0.2, 0.5, size=n)

    df_force = pd.DataFrame(
        {
            "is_force_exit": np.ones(n),
            "reward_exit": force_exit_rewards,
        }
    )
    df_regular = pd.DataFrame(
        {
            "is_force_exit": np.zeros(n),
            "reward_exit": regular_exit_rewards,
        }
    )
    df = pd.concat([df_force, df_regular], ignore_index=True)
    # fillers
    df["reward_total"] = df["reward_exit"]
    df["reward_idle"] = 0.0
    df["position"] = 0.0
    df["pnl"] = 0.0
    df["trade_duration"] = 0.0
    df["idle_duration"] = 0.0
    return df


def test_distribution_shift_metrics():
    # Identical distributions should yield near-zero KL/JS and high KS p-value
    rng = np.random.default_rng(2025)
    X = rng.normal(0, 1, size=5000)
    Y = rng.normal(0, 1, size=5000)

    s_df = pd.DataFrame({"pnl": X, "trade_duration": X, "idle_duration": X})
    r_df = pd.DataFrame({"pnl": Y, "trade_duration": Y, "idle_duration": Y})

    metrics = compute_distribution_shift_metrics(s_df, r_df)

    for feat in ["pnl", "trade_duration", "idle_duration"]:
        assert metrics[f"{feat}_kl_divergence"] >= 0
        assert metrics[f"{feat}_js_distance"] >= 0
        # should be small for identical distributions (allowing small numerical noise)
        assert metrics[f"{feat}_js_distance"] < 0.1
        # KS should not reject (p > 0.05)
        assert metrics[f"{feat}_ks_pvalue"] > 0.05


def test_hypothesis_tests():
    # Spearman negative correlation should be detected
    df_s = make_df_for_spearman()
    res_s = statistical_hypothesis_tests(df_s)
    assert "idle_correlation" in res_s
    assert res_s["idle_correlation"]["rho"] < 0
    assert res_s["idle_correlation"]["p_value"] < 0.05

    # Kruskal with 3 groups -> significant + epsilon^2 in [0, 1)
    df_k = make_df_for_kruskal()
    res_k = statistical_hypothesis_tests(df_k)
    assert "position_reward_difference" in res_k
    eps2 = res_k["position_reward_difference"]["effect_size_epsilon_sq"]
    assert 0 <= eps2 < 1
    assert res_k["position_reward_difference"]["p_value"] < 0.05

    # Mann–Whitney rank-biserial should be positive since force > regular
    df_m = make_df_for_mannwhitney()
    res_m = statistical_hypothesis_tests(df_m)
    assert "force_vs_regular_exits" in res_m
    assert res_m["force_vs_regular_exits"]["effect_size_rank_biserial"] > 0
    assert res_m["force_vs_regular_exits"]["p_value"] < 0.05


def test_distribution_diagnostics():
    rng = np.random.default_rng(99)
    data = rng.normal(0, 1, size=2000)
    df = pd.DataFrame(
        {
            "reward_total": data,
            "pnl": data,
            "trade_duration": data,
            "idle_duration": data,
        }
    )
    diag = distribution_diagnostics(df)
    # QQ R^2 should be high for normal data
    assert diag["reward_total_qq_r_squared"] > 0.97


if __name__ == "__main__":
    # Run tests sequentially
    test_distribution_shift_metrics()
    test_hypothesis_tests()
    test_distribution_diagnostics()
    print("All statistical coherence checks passed.")
