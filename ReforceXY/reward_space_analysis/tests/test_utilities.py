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

import pandas as pd

from reward_space_analysis import (
    PBRS_INVARIANCE_TOL,
    apply_potential_shaping,
    load_real_episodes,
)

from .test_base import RewardSpaceTestBase


class TestLoadRealEpisodes(RewardSpaceTestBase):
    """Unit tests for load_real_episodes."""

    def write_pickle(self, obj, path: Path):
        with path.open("wb") as f:
            pickle.dump(obj, f)

    def test_top_level_dict_transitions(self):
        """Test top level dict transitions."""
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
        """Test mixed episode list warns and flattens."""
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
        """Test non iterable transitions raises."""
        bad = {"transitions": 123}
        p = Path(self.temp_dir) / "bad.pkl"
        self.write_pickle(bad, p)
        with self.assertRaises(ValueError):
            load_real_episodes(p)

    def test_enforce_columns_false_fills_na(self):
        """Test enforce columns false fills na."""
        trans = [
            {"pnl": 0.03, "trade_duration": 10, "idle_duration": 0, "position": 1.0, "action": 2.0}
        ]
        p = Path(self.temp_dir) / "fill.pkl"
        self.write_pickle(trans, p)
        loaded = load_real_episodes(p, enforce_columns=False)
        self.assertIn("reward", loaded.columns)
        self.assertTrue(loaded["reward"].isna().all())

    def test_casting_numeric_strings(self):
        """Test casting numeric strings."""
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
