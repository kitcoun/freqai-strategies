#!/usr/bin/env python3
"""Utility tests for data loading, formatting, and parameter propagation."""

import json
import pickle
import re
import subprocess
import sys
import unittest
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

from reward_space_analysis import (
    DEFAULT_MODEL_REWARD_PARAMETERS,
    PBRS_INVARIANCE_TOL,
    _get_float_param,
    apply_potential_shaping,
    bootstrap_confidence_intervals,
    load_real_episodes,
    simulate_samples,
)

from .test_base import RewardSpaceTestBase


class TestLoadRealEpisodes(RewardSpaceTestBase):
    """Unit tests for load_real_episodes."""

    def write_pickle(self, obj, path: Path):
        with path.open("wb") as f:
            pickle.dump(obj, f)

    def test_top_level_dict_transitions(self):
        df = pd.DataFrame(
            {
                "pnl": [0.01],
                "trade_duration": [10],
                "idle_duration": [5],
                "position": [1.0],
                "action": [2.0],
                "reward": [1.0],
            }
        )
        p = Path(self.temp_dir) / "top.pkl"
        self.write_pickle({"transitions": df}, p)
        loaded = load_real_episodes(p)
        self.assertIsInstance(loaded, pd.DataFrame)
        self.assertEqual(list(loaded.columns).count("pnl"), 1)
        self.assertEqual(len(loaded), 1)

    def test_mixed_episode_list_warns_and_flattens(self):
        ep1 = {"episode_id": 1}
        ep2 = {
            "episode_id": 2,
            "transitions": [
                {
                    "pnl": 0.02,
                    "trade_duration": 5,
                    "idle_duration": 0,
                    "position": 1.0,
                    "action": 2.0,
                    "reward": 2.0,
                }
            ],
        }
        p = Path(self.temp_dir) / "mixed.pkl"
        self.write_pickle([ep1, ep2], p)
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            loaded = load_real_episodes(p)
            _ = w
        self.assertEqual(len(loaded), 1)
        self.assertPlacesEqual(float(loaded.iloc[0]["pnl"]), 0.02, places=7)

    def test_non_iterable_transitions_raises(self):
        bad = {"transitions": 123}
        p = Path(self.temp_dir) / "bad.pkl"
        self.write_pickle(bad, p)
        with self.assertRaises(ValueError):
            load_real_episodes(p)

    def test_enforce_columns_false_fills_na(self):
        trans = [
            {"pnl": 0.03, "trade_duration": 10, "idle_duration": 0, "position": 1.0, "action": 2.0}
        ]
        p = Path(self.temp_dir) / "fill.pkl"
        self.write_pickle(trans, p)
        loaded = load_real_episodes(p, enforce_columns=False)
        self.assertIn("reward", loaded.columns)
        self.assertTrue(loaded["reward"].isna().all())

    def test_casting_numeric_strings(self):
        trans = [
            {
                "pnl": "0.04",
                "trade_duration": "20",
                "idle_duration": "0",
                "position": "1.0",
                "action": "2.0",
                "reward": "3.0",
            }
        ]
        p = Path(self.temp_dir) / "strs.pkl"
        self.write_pickle(trans, p)
        loaded = load_real_episodes(p)
        self.assertIn("pnl", loaded.columns)
        self.assertIn(loaded["pnl"].dtype.kind, ("f", "i"))
        self.assertPlacesEqual(float(loaded.iloc[0]["pnl"]), 0.04, places=7)

    def test_pickled_dataframe_loads(self):
        """Ensure a directly pickled DataFrame loads correctly."""
        test_episodes = pd.DataFrame(
            {
                "pnl": [0.01, -0.02, 0.03],
                "trade_duration": [10, 20, 15],
                "idle_duration": [5, 0, 8],
                "position": [1.0, 0.0, 1.0],
                "action": [2.0, 0.0, 2.0],
                "reward": [10.5, -5.2, 15.8],
            }
        )
        p = Path(self.temp_dir) / "test_episodes.pkl"
        self.write_pickle(test_episodes, p)
        loaded_data = load_real_episodes(p)
        self.assertIsInstance(loaded_data, pd.DataFrame)
        self.assertEqual(len(loaded_data), 3)
        self.assertIn("pnl", loaded_data.columns)


class TestBootstrapStatistics(RewardSpaceTestBase):
    """Grouped tests for bootstrap confidence interval behavior."""

    def test_constant_distribution_bootstrap_and_diagnostics(self):
        """Degenerate columns produce (mean≈lo≈hi) zero-width intervals."""
        df = self._const_df(80)
        res = bootstrap_confidence_intervals(
            df, ["reward", "pnl"], n_bootstrap=200, confidence_level=0.95
        )
        for k, (mean, lo, hi) in res.items():
            self.assertAlmostEqualFloat(mean, lo, tolerance=2e-09)
            self.assertAlmostEqualFloat(mean, hi, tolerance=2e-09)
            self.assertLessEqual(hi - lo, 2e-09)

    def test_bootstrap_shrinkage_with_sample_size(self):
        """Half-width decreases with larger sample (~1/sqrt(n) heuristic)."""
        small = self._shift_scale_df(80)
        large = self._shift_scale_df(800)
        res_small = bootstrap_confidence_intervals(small, ["reward"], n_bootstrap=400)
        res_large = bootstrap_confidence_intervals(large, ["reward"], n_bootstrap=400)
        _, lo_s, hi_s = list(res_small.values())[0]
        _, lo_l, hi_l = list(res_large.values())[0]
        hw_small = (hi_s - lo_s) / 2.0
        hw_large = (hi_l - lo_l) / 2.0
        self.assertFinite(hw_small, name="hw_small")
        self.assertFinite(hw_large, name="hw_large")
        self.assertLess(hw_large, hw_small * 0.55)

    def test_bootstrap_confidence_intervals_basic(self):
        """Basic CI computation returns ordered finite bounds."""
        test_data = self.make_stats_df(n=100, seed=self.SEED)
        results = bootstrap_confidence_intervals(test_data, ["reward", "pnl"], n_bootstrap=100)
        for metric, (mean, ci_low, ci_high) in results.items():
            self.assertFinite(mean, name=f"mean[{metric}]")
            self.assertFinite(ci_low, name=f"ci_low[{metric}]")
            self.assertFinite(ci_high, name=f"ci_high[{metric}]")
            self.assertLess(ci_low, ci_high)

    def test_canonical_invariance_flag_and_sum(self):
        """Canonical mode + no additives -> pbrs_invariant True and Σ shaping ≈ 0."""
        params = self.base_params(
            exit_potential_mode="canonical",
            entry_additive_enabled=False,
            exit_additive_enabled=False,
            hold_potential_enabled=True,
        )
        df = simulate_samples(
            params={**params, "max_trade_duration_candles": 100},
            num_samples=400,
            seed=self.SEED,
            base_factor=self.TEST_BASE_FACTOR,
            profit_target=self.TEST_PROFIT_TARGET,
            risk_reward_ratio=self.TEST_RR,
            max_duration_ratio=2.0,
            trading_mode="margin",
            pnl_base_std=self.TEST_PNL_STD,
            pnl_duration_vol_scale=self.TEST_PNL_DUR_VOL_SCALE,
        )
        unique_flags = set(df["pbrs_invariant"].unique().tolist())
        self.assertEqual(unique_flags, {True}, f"Unexpected invariant flags: {unique_flags}")
        total_shaping = float(df["reward_shaping"].sum())
        self.assertLess(
            abs(total_shaping),
            PBRS_INVARIANCE_TOL,
            f"Canonical invariance violated: Σ shaping = {total_shaping}",
        )

    def test_non_canonical_flag_false_and_sum_nonzero(self):
        """Non-canonical exit potential (progressive_release) -> pbrs_invariant False and Σ shaping != 0."""
        params = self.base_params(
            exit_potential_mode="progressive_release",
            exit_potential_decay=0.25,
            entry_additive_enabled=False,
            exit_additive_enabled=False,
            hold_potential_enabled=True,
        )
        df = simulate_samples(
            params={**params, "max_trade_duration_candles": 100},
            num_samples=400,
            seed=self.SEED,
            base_factor=self.TEST_BASE_FACTOR,
            profit_target=self.TEST_PROFIT_TARGET,
            risk_reward_ratio=self.TEST_RR,
            max_duration_ratio=2.0,
            trading_mode="margin",
            pnl_base_std=self.TEST_PNL_STD,
            pnl_duration_vol_scale=self.TEST_PNL_DUR_VOL_SCALE,
        )
        unique_flags = set(df["pbrs_invariant"].unique().tolist())
        self.assertEqual(unique_flags, {False}, f"Unexpected invariant flags: {unique_flags}")
        total_shaping = float(df["reward_shaping"].sum())
        self.assertGreater(
            abs(total_shaping),
            PBRS_INVARIANCE_TOL * 10,
            f"Expected non-zero Σ shaping in non-canonical mode (got {total_shaping})",
        )


class TestReportFormatting(RewardSpaceTestBase):
    """Tests for report formatting elements not covered elsewhere."""

    def test_abs_shaping_line_present_and_constant(self):
        """Abs Σ Shaping Reward line present, formatted, uses constant not literal."""
        df = pd.DataFrame(
            {
                "reward_shaping": [self.TOL_IDENTITY_STRICT, -self.TOL_IDENTITY_STRICT],
                "reward_entry_additive": [0.0, 0.0],
                "reward_exit_additive": [0.0, 0.0],
            }
        )
        total_shaping = df["reward_shaping"].sum()
        self.assertTrue(abs(total_shaping) < PBRS_INVARIANCE_TOL)
        lines = [f"| Abs Σ Shaping Reward | {abs(total_shaping):.6e} |"]
        content = "\n".join(lines)
        m = re.search("\\| Abs Σ Shaping Reward \\| ([0-9]+\\.[0-9]{6}e[+-][0-9]{2}) \\|", content)
        self.assertIsNotNone(m, "Abs Σ Shaping Reward line missing or misformatted")
        val = float(m.group(1)) if m else None
        if val is not None:
            self.assertLess(val, self.TOL_NEGLIGIBLE + self.TOL_IDENTITY_STRICT)
        self.assertNotIn(
            str(self.TOL_GENERIC_EQ),
            content,
            "Tolerance constant value should appear, not raw literal",
        )

    def test_pbrs_non_canonical_report_generation(self):
        """Generate synthetic invariance section with non-zero shaping to assert Non-canonical classification."""
        df = pd.DataFrame(
            {
                "reward_shaping": [0.01, -0.002],
                "reward_entry_additive": [0.0, 0.0],
                "reward_exit_additive": [0.001, 0.0],
            }
        )
        total_shaping = df["reward_shaping"].sum()
        self.assertGreater(abs(total_shaping), PBRS_INVARIANCE_TOL)
        invariance_status = "❌ Non-canonical"
        section = []
        section.append("**PBRS Invariance Summary:**\n")
        section.append("| Field | Value |\n")
        section.append("|-------|-------|\n")
        section.append(f"| Invariance | {invariance_status} |\n")
        section.append(f"| Note | Total shaping = {total_shaping:.6f} (non-zero) |\n")
        section.append(f"| Σ Shaping Reward | {total_shaping:.6f} |\n")
        section.append(f"| Abs Σ Shaping Reward | {abs(total_shaping):.6e} |\n")
        section.append(f"| Σ Entry Additive | {df['reward_entry_additive'].sum():.6f} |\n")
        section.append(f"| Σ Exit Additive | {df['reward_exit_additive'].sum():.6f} |\n")
        content = "".join(section)
        self.assertIn("❌ Non-canonical", content)
        self.assertRegex(content, "Σ Shaping Reward \\| 0\\.008000 \\|")
        m_abs = re.search("Abs Σ Shaping Reward \\| ([0-9.]+e[+-][0-9]{2}) \\|", content)
        self.assertIsNotNone(m_abs)
        if m_abs:
            self.assertAlmostEqual(abs(total_shaping), float(m_abs.group(1)), places=12)

    def test_additive_activation_deterministic_contribution(self):
        """Additives enabled increase total reward; shaping impact limited."""
        base = self.base_params(
            hold_potential_enabled=True,
            entry_additive_enabled=False,
            exit_additive_enabled=False,
            exit_potential_mode="non_canonical",
        )
        with_add = base.copy()
        with_add.update(
            {
                "entry_additive_enabled": True,
                "exit_additive_enabled": True,
                "entry_additive_scale": 0.4,
                "exit_additive_scale": 0.4,
                "entry_additive_gain": 1.0,
                "exit_additive_gain": 1.0,
            }
        )
        ctx = {
            "base_reward": 0.05,
            "current_pnl": 0.01,
            "current_duration_ratio": 0.2,
            "next_pnl": 0.012,
            "next_duration_ratio": 0.25,
            "is_entry": True,
            "is_exit": False,
        }
        _t0, s0, _n0 = apply_potential_shaping(last_potential=0.0, params=base, **ctx)
        t1, s1, _n1 = apply_potential_shaping(last_potential=0.0, params=with_add, **ctx)
        self.assertFinite(t1)
        self.assertFinite(s1)
        self.assertLess(abs(s1 - s0), 0.2)
        self.assertGreater(t1 - _t0, 0.0, "Total reward should increase with additives present")

    def test_report_cumulative_invariance_aggregation(self):
        """Canonical telescoping term: small per-step mean drift, bounded increments."""
        params = self.base_params(
            hold_potential_enabled=True,
            entry_additive_enabled=False,
            exit_additive_enabled=False,
            exit_potential_mode="canonical",
        )
        gamma = _get_float_param(
            params, "potential_gamma", DEFAULT_MODEL_REWARD_PARAMETERS.get("potential_gamma", 0.95)
        )
        rng = np.random.default_rng(321)
        last_potential = 0.0
        telescoping_sum = 0.0
        max_abs_step = 0.0
        steps = 0
        for _ in range(500):
            is_exit = rng.uniform() < 0.1
            current_pnl = float(rng.normal(0, 0.05))
            current_dur = float(rng.uniform(0, 1))
            next_pnl = 0.0 if is_exit else float(rng.normal(0, 0.05))
            next_dur = 0.0 if is_exit else float(rng.uniform(0, 1))
            _tot, _shap, next_potential = apply_potential_shaping(
                base_reward=0.0,
                current_pnl=current_pnl,
                current_duration_ratio=current_dur,
                next_pnl=next_pnl,
                next_duration_ratio=next_dur,
                is_exit=is_exit,
                last_potential=last_potential,
                params=params,
            )
            inc = gamma * next_potential - last_potential
            telescoping_sum += inc
            if abs(inc) > max_abs_step:
                max_abs_step = abs(inc)
            steps += 1
            if is_exit:
                last_potential = 0.0
            else:
                last_potential = next_potential
        mean_drift = telescoping_sum / max(1, steps)
        self.assertLess(
            abs(mean_drift),
            0.02,
            f"Per-step telescoping drift too large (mean={mean_drift}, steps={steps})",
        )
        self.assertLessEqual(
            max_abs_step,
            self.PBRS_MAX_ABS_SHAPING,
            f"Unexpected large telescoping increment (max={max_abs_step})",
        )

    def test_report_explicit_non_invariance_progressive_release(self):
        """progressive_release should generally yield non-zero cumulative shaping (release leak)."""
        params = self.base_params(
            hold_potential_enabled=True,
            entry_additive_enabled=False,
            exit_additive_enabled=False,
            exit_potential_mode="progressive_release",
            exit_potential_decay=0.25,
        )
        rng = np.random.default_rng(321)
        last_potential = 0.0
        shaping_sum = 0.0
        for _ in range(160):
            is_exit = rng.uniform() < 0.15
            next_pnl = 0.0 if is_exit else float(rng.normal(0, 0.07))
            next_dur = 0.0 if is_exit else float(rng.uniform(0, 1))
            _tot, shap, next_pot = apply_potential_shaping(
                base_reward=0.0,
                current_pnl=float(rng.normal(0, 0.07)),
                current_duration_ratio=float(rng.uniform(0, 1)),
                next_pnl=next_pnl,
                next_duration_ratio=next_dur,
                is_exit=is_exit,
                last_potential=last_potential,
                params=params,
            )
            shaping_sum += shap
            last_potential = 0.0 if is_exit else next_pot
        self.assertGreater(
            abs(shaping_sum),
            PBRS_INVARIANCE_TOL * 50,
            f"Expected non-zero Σ shaping (got {shaping_sum})",
        )

    def test_gamma_extremes(self):
        """Gamma=0 and gamma≈1 boundary behaviours produce bounded shaping and finite potentials."""
        for gamma in [0.0, 0.999999]:
            params = self.base_params(
                hold_potential_enabled=True,
                entry_additive_enabled=False,
                exit_additive_enabled=False,
                exit_potential_mode="canonical",
                potential_gamma=gamma,
            )
            _tot, shap, next_pot = apply_potential_shaping(
                base_reward=0.0,
                current_pnl=0.02,
                current_duration_ratio=0.3,
                next_pnl=0.025,
                next_duration_ratio=0.35,
                is_exit=False,
                last_potential=0.0,
                params=params,
            )
            self.assertTrue(np.isfinite(shap))
            self.assertTrue(np.isfinite(next_pot))
            self.assertLessEqual(abs(shap), self.PBRS_MAX_ABS_SHAPING)


class TestCsvAndSimulationOptions(RewardSpaceTestBase):
    """CLI-level tests: CSV encoding and simulate_unrealized_pnl option effects."""

    def test_action_column_integer_in_csv(self):
        """Ensure 'action' column in reward_samples.csv is encoded as integers."""
        out_dir = self.output_path / "csv_int_check"
        cmd = [
            sys.executable,
            "reward_space_analysis.py",
            "--num_samples",
            "200",
            "--seed",
            str(self.SEED),
            "--out_dir",
            str(out_dir),
        ]
        result = subprocess.run(
            cmd, capture_output=True, text=True, cwd=Path(__file__).parent.parent
        )
        self.assertEqual(result.returncode, 0, f"CLI failed: {result.stderr}")
        csv_path = out_dir / "reward_samples.csv"
        self.assertTrue(csv_path.exists(), "Missing reward_samples.csv")
        df = pd.read_csv(csv_path)
        self.assertIn("action", df.columns)
        values = df["action"].tolist()
        self.assertTrue(
            all((float(v).is_integer() for v in values)),
            "Non-integer values detected in 'action' column",
        )
        allowed = {0, 1, 2, 3, 4}
        self.assertTrue(set((int(v) for v in values)).issubset(allowed))

    def test_unrealized_pnl_affects_hold_potential(self):
        """--unrealized_pnl should alter hold next_potential distribution vs default."""
        out_default = self.output_path / "sim_default"
        out_sim = self.output_path / "sim_unrealized"
        base_args = ["--num_samples", "800", "--seed", str(self.SEED), "--out_dir"]
        cmd_default = [sys.executable, "reward_space_analysis.py", *base_args, str(out_default)]
        res_def = subprocess.run(
            cmd_default, capture_output=True, text=True, cwd=Path(__file__).parent.parent
        )
        self.assertEqual(res_def.returncode, 0, f"CLI default run failed: {res_def.stderr}")
        cmd_sim = [
            sys.executable,
            "reward_space_analysis.py",
            *base_args,
            str(out_sim),
            "--unrealized_pnl",
        ]
        res_sim = subprocess.run(
            cmd_sim, capture_output=True, text=True, cwd=Path(__file__).parent.parent
        )
        self.assertEqual(res_sim.returncode, 0, f"CLI simulated run failed: {res_sim.stderr}")
        df_def = pd.read_csv(out_default / "reward_samples.csv")
        df_sim = pd.read_csv(out_sim / "reward_samples.csv")
        mask_hold_def = (df_def["action"] == 0) & df_def["position"].isin([0.0, 1.0])
        mask_hold_sim = (df_sim["action"] == 0) & df_sim["position"].isin([0.0, 1.0])
        self.assertGreater(int(mask_hold_def.sum()), 0, "No hold samples in default run")
        self.assertGreater(int(mask_hold_sim.sum()), 0, "No hold samples in simulate run")
        mean_next_def = float(df_def.loc[mask_hold_def, "next_potential"].mean())
        mean_next_sim = float(df_sim.loc[mask_hold_sim, "next_potential"].mean())
        self.assertFinite(mean_next_def, name="mean_next_def")
        self.assertFinite(mean_next_sim, name="mean_next_sim")
        self.assertGreater(
            abs(mean_next_sim - mean_next_def),
            self.TOL_GENERIC_EQ,
            f"No detectable effect of --unrealized_pnl on Φ(s): def={mean_next_def:.6f}, sim={mean_next_sim:.6f}",
        )


class TestParamsPropagation(RewardSpaceTestBase):
    """Integration tests to validate max_trade_duration_candles propagation via CLI params and dynamic flag."""

    def test_max_trade_duration_candles_propagation_params(self):
        """--params max_trade_duration_candles=X propagates to manifest and simulation params."""
        out_dir = self.output_path / "mtd_params"
        cmd = [
            sys.executable,
            "reward_space_analysis.py",
            "--num_samples",
            "120",
            "--seed",
            str(self.SEED),
            "--out_dir",
            str(out_dir),
            "--params",
            "max_trade_duration_candles=96",
        ]
        result = subprocess.run(
            cmd, capture_output=True, text=True, cwd=Path(__file__).parent.parent
        )
        self.assertEqual(result.returncode, 0, f"CLI failed: {result.stderr}")
        manifest_path = out_dir / "manifest.json"
        self.assertTrue(manifest_path.exists(), "Missing manifest.json")
        with open(manifest_path, "r") as f:
            manifest = json.load(f)
        self.assertIn("reward_params", manifest)
        self.assertIn("simulation_params", manifest)
        rp = manifest["reward_params"]
        self.assertIn("max_trade_duration_candles", rp)
        self.assertEqual(int(rp["max_trade_duration_candles"]), 96)

    def test_max_trade_duration_candles_propagation_flag(self):
        """Dynamic flag --max_trade_duration_candles X propagates identically."""
        out_dir = self.output_path / "mtd_flag"
        cmd = [
            sys.executable,
            "reward_space_analysis.py",
            "--num_samples",
            "120",
            "--seed",
            str(self.SEED),
            "--out_dir",
            str(out_dir),
            "--max_trade_duration_candles",
            "64",
        ]
        result = subprocess.run(
            cmd, capture_output=True, text=True, cwd=Path(__file__).parent.parent
        )
        self.assertEqual(result.returncode, 0, f"CLI failed: {result.stderr}")
        manifest_path = out_dir / "manifest.json"
        self.assertTrue(manifest_path.exists(), "Missing manifest.json")
        with open(manifest_path, "r") as f:
            manifest = json.load(f)
        self.assertIn("reward_params", manifest)
        self.assertIn("simulation_params", manifest)
        rp = manifest["reward_params"]
        self.assertIn("max_trade_duration_candles", rp)
        self.assertEqual(int(rp["max_trade_duration_candles"]), 64)


if __name__ == "__main__":
    unittest.main()
