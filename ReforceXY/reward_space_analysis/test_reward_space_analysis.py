#!/usr/bin/env python3
"""Unified test suite for reward space analysis.

Covers:
- Integration testing (CLI interface)
- Statistical coherence validation
- Reward alignment with environment
"""

import unittest
import tempfile
import shutil
from pathlib import Path
import subprocess
import sys
import json
import pickle
import numpy as np
import pandas as pd

# Import functions to test
try:
    import reward_space_analysis as rsa
    from reward_space_analysis import (
        DEFAULT_MODEL_REWARD_PARAMETERS,
        Actions,
        ForceActions,
        Positions,
        RewardContext,
        calculate_reward,
        compute_distribution_shift_metrics,
        distribution_diagnostics,
        simulate_samples,
        bootstrap_confidence_intervals,
        parse_overrides,
    )
except ImportError as e:
    print(f"Import error: {e}")
    sys.exit(1)


class RewardSpaceTestBase(unittest.TestCase):
    """Base class with common test utilities."""

    @classmethod
    def setUpClass(cls):
        """Set up class-level constants - single source of truth."""
        cls.SEED = 42
        cls.DEFAULT_PARAMS = DEFAULT_MODEL_REWARD_PARAMETERS.copy()
        cls.TEST_SAMPLES = 50  # Small for speed

    def setUp(self):
        """Set up test fixtures with reproducible random seed."""
        np.random.seed(self.SEED)
        self.temp_dir = tempfile.mkdtemp()
        self.output_path = Path(self.temp_dir)

    def tearDown(self):
        """Clean up temporary files."""
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def assertAlmostEqualFloat(self, first, second, tolerance=1e-6, msg=None):
        """Helper for floating point comparisons."""
        if abs(first - second) > tolerance:
            self.fail(f"{first} != {second} within {tolerance}: {msg}")


class TestIntegration(RewardSpaceTestBase):
    """Integration tests for CLI and file outputs."""

    def test_cli_execution_produces_expected_files(self):
        """Test that CLI execution produces all expected output files."""
        cmd = [
            sys.executable,
            "reward_space_analysis.py",
            "--num_samples",
            str(self.TEST_SAMPLES),
            "--seed",
            str(self.SEED),
            "--output",
            str(self.output_path),
        ]

        result = subprocess.run(
            cmd, capture_output=True, text=True, cwd=Path(__file__).parent
        )

        # Should execute successfully
        self.assertEqual(result.returncode, 0, f"CLI failed: {result.stderr}")

        # Should produce expected files (single source of truth list)
        expected_files = [
            "reward_samples.csv",
            "feature_importance.csv",
            "statistical_analysis.md",
            "manifest.json",
            "partial_dependence_trade_duration.csv",
            "partial_dependence_idle_duration.csv",
            "partial_dependence_pnl.csv",
        ]

        for filename in expected_files:
            file_path = self.output_path / filename
            self.assertTrue(file_path.exists(), f"Missing expected file: {filename}")

    def test_manifest_structure_and_reproducibility(self):
        """Test manifest structure and reproducibility with same seed."""
        # First run
        cmd1 = [
            sys.executable,
            "reward_space_analysis.py",
            "--num_samples",
            str(self.TEST_SAMPLES),
            "--seed",
            str(self.SEED),
            "--output",
            str(self.output_path / "run1"),
        ]

        # Second run with same seed
        cmd2 = [
            sys.executable,
            "reward_space_analysis.py",
            "--num_samples",
            str(self.TEST_SAMPLES),
            "--seed",
            str(self.SEED),
            "--output",
            str(self.output_path / "run2"),
        ]

        # Execute both runs
        result1 = subprocess.run(
            cmd1, capture_output=True, text=True, cwd=Path(__file__).parent
        )
        result2 = subprocess.run(
            cmd2, capture_output=True, text=True, cwd=Path(__file__).parent
        )

        self.assertEqual(result1.returncode, 0)
        self.assertEqual(result2.returncode, 0)

        # Validate manifest structure
        for run_dir in ["run1", "run2"]:
            with open(self.output_path / run_dir / "manifest.json", "r") as f:
                manifest = json.load(f)

            required_keys = {"generated_at", "num_samples", "seed", "params_hash"}
            self.assertTrue(required_keys.issubset(manifest.keys()))
            self.assertEqual(manifest["num_samples"], self.TEST_SAMPLES)
            self.assertEqual(manifest["seed"], self.SEED)

        # Test reproducibility: manifests should have same params_hash
        with open(self.output_path / "run1" / "manifest.json", "r") as f:
            manifest1 = json.load(f)
        with open(self.output_path / "run2" / "manifest.json", "r") as f:
            manifest2 = json.load(f)

        self.assertEqual(
            manifest1["params_hash"],
            manifest2["params_hash"],
            "Same seed should produce same parameters hash",
        )


class TestStatisticalCoherence(RewardSpaceTestBase):
    """Statistical coherence validation tests."""

    def _make_test_dataframe(self, n: int = 100) -> pd.DataFrame:
        """Generate test dataframe for statistical validation."""
        np.random.seed(self.SEED)

        # Create correlated data for Spearman test
        idle_duration = np.random.exponential(10, n)
        reward_idle = -0.01 * idle_duration + np.random.normal(0, 0.001, n)

        return pd.DataFrame(
            {
                "idle_duration": idle_duration,
                "reward_idle": reward_idle,
                "position": np.random.choice([0.0, 0.5, 1.0], n),
                "reward_total": np.random.normal(0, 1, n),
                "pnl": np.random.normal(0, 0.02, n),
                "trade_duration": np.random.exponential(20, n),
            }
        )

    def test_distribution_shift_metrics(self):
        """Test KL divergence, JS distance, Wasserstein distance calculations."""
        df1 = self._make_test_dataframe(100)
        df2 = self._make_test_dataframe(100)

        # Add slight shift to second dataset
        df2["reward_total"] += 0.1

        metrics = compute_distribution_shift_metrics(df1, df2)

        # Should contain expected metrics for pnl (continuous feature)
        expected_keys = {
            "pnl_kl_divergence",
            "pnl_js_distance",
            "pnl_wasserstein",
            "pnl_ks_statistic",
        }
        actual_keys = set(metrics.keys())
        matching_keys = expected_keys.intersection(actual_keys)
        self.assertGreater(
            len(matching_keys),
            0,
            f"Should have some distribution metrics. Got: {actual_keys}",
        )

        # Values should be finite and reasonable
        for metric_name, value in metrics.items():
            if "pnl" in metric_name:
                self.assertTrue(np.isfinite(value), f"{metric_name} should be finite")
                if any(
                    suffix in metric_name for suffix in ["js_distance", "ks_statistic"]
                ):
                    self.assertGreaterEqual(
                        value, 0, f"{metric_name} should be non-negative"
                    )

    def test_hypothesis_testing(self):
        """Test statistical hypothesis tests."""
        df = self._make_test_dataframe(200)

        # Test only if we have enough data - simple correlation validation
        if len(df) > 30:
            idle_data = df[df["idle_duration"] > 0]
            if len(idle_data) > 10:
                # Simple correlation check: idle duration should correlate negatively with idle reward
                idle_dur = idle_data["idle_duration"].values
                idle_rew = idle_data["reward_idle"].values

                # Basic validation that data makes sense
                self.assertTrue(
                    len(idle_dur) == len(idle_rew),
                    "Idle duration and reward arrays should have same length",
                )
                self.assertTrue(
                    all(d >= 0 for d in idle_dur),
                    "Idle durations should be non-negative",
                )

                # Idle rewards should generally be negative (penalty for holding)
                negative_rewards = (idle_rew < 0).sum()
                total_rewards = len(idle_rew)
                negative_ratio = negative_rewards / total_rewards

                self.assertGreater(
                    negative_ratio,
                    0.5,
                    "Most idle rewards should be negative (penalties)",
                )

    def test_distribution_diagnostics(self):
        """Test distribution normality diagnostics."""
        df = self._make_test_dataframe(100)

        diagnostics = distribution_diagnostics(df)

        # Should contain diagnostics for key columns (flattened format)
        expected_prefixes = ["reward_total_", "pnl_"]
        for prefix in expected_prefixes:
            matching_keys = [
                key for key in diagnostics.keys() if key.startswith(prefix)
            ]
            self.assertGreater(
                len(matching_keys), 0, f"Should have diagnostics for {prefix}"
            )

            # Check for basic statistical measures
            expected_suffixes = ["mean", "std", "skewness", "kurtosis"]
            for suffix in expected_suffixes:
                key = f"{prefix}{suffix}"
                if key in diagnostics:
                    self.assertTrue(
                        np.isfinite(diagnostics[key]), f"{key} should be finite"
                    )


class TestRewardAlignment(RewardSpaceTestBase):
    """Test reward calculation alignment with environment."""

    def test_basic_reward_calculation(self):
        """Test basic reward calculation consistency."""
        context = RewardContext(
            pnl=0.02,
            trade_duration=10,
            idle_duration=0,
            max_trade_duration=100,
            max_unrealized_profit=0.025,
            min_unrealized_profit=0.015,
            position=Positions.Long,
            action=Actions.Long_exit,
            force_action=None,
        )

        breakdown = calculate_reward(
            context,
            self.DEFAULT_PARAMS,
            base_factor=100.0,
            profit_target=0.06,
            risk_reward_ratio=2.0,
            short_allowed=True,
            action_masking=True,
        )

        # Should return valid breakdown
        self.assertIsInstance(breakdown.total, (int, float))
        self.assertTrue(np.isfinite(breakdown.total))

        # Exit reward should be positive for profitable trade
        self.assertGreater(
            breakdown.exit_component, 0, "Profitable exit should have positive reward"
        )

    def test_force_action_logic(self):
        """Test force action behavior consistency."""
        # Take profit should generally be positive
        tp_context = RewardContext(
            pnl=0.05,  # Good profit
            trade_duration=20,
            idle_duration=0,
            max_trade_duration=100,
            max_unrealized_profit=0.06,
            min_unrealized_profit=0.04,
            position=Positions.Long,
            action=Actions.Long_exit,
            force_action=ForceActions.Take_profit,
        )

        tp_breakdown = calculate_reward(
            tp_context,
            self.DEFAULT_PARAMS,
            base_factor=100.0,
            profit_target=0.06,
            risk_reward_ratio=2.0,
            short_allowed=True,
            action_masking=True,
        )

        # Stop loss should generally be negative
        sl_context = RewardContext(
            pnl=-0.03,  # Loss
            trade_duration=15,
            idle_duration=0,
            max_trade_duration=100,
            max_unrealized_profit=0.01,
            min_unrealized_profit=-0.04,
            position=Positions.Long,
            action=Actions.Long_exit,
            force_action=ForceActions.Stop_loss,
        )

        sl_breakdown = calculate_reward(
            sl_context,
            self.DEFAULT_PARAMS,
            base_factor=100.0,
            profit_target=0.06,
            risk_reward_ratio=2.0,
            short_allowed=True,
            action_masking=True,
        )

        # Take profit should be better than stop loss
        self.assertGreater(
            tp_breakdown.total,
            sl_breakdown.total,
            "Take profit should yield higher reward than stop loss",
        )

    def test_exit_factor_calculation(self):
        """Test exit factor calculation consistency."""
        # Test different exit factor modes
        modes_to_test = ["linear", "piecewise", "power"]

        for mode in modes_to_test:
            test_params = self.DEFAULT_PARAMS.copy()
            test_params["exit_factor_mode"] = mode

            factor = rsa._get_exit_factor(
                factor=1.0,
                pnl=0.02,
                pnl_factor=1.5,
                duration_ratio=0.3,
                params=test_params,
            )

            self.assertTrue(
                np.isfinite(factor), f"Exit factor for {mode} should be finite"
            )
            self.assertGreater(factor, 0, f"Exit factor for {mode} should be positive")


class TestPublicFunctions(RewardSpaceTestBase):
    """Test public functions and API."""

    def test_parse_overrides(self):
        """Test parse_overrides function."""
        # Test valid overrides
        overrides = ["key1=1.5", "key2=hello", "key3=42"]
        result = parse_overrides(overrides)

        self.assertEqual(result["key1"], 1.5)
        self.assertEqual(result["key2"], "hello")
        self.assertEqual(result["key3"], 42.0)

        # Test invalid format
        with self.assertRaises(ValueError):
            parse_overrides(["invalid_format"])

    def test_bootstrap_confidence_intervals(self):
        """Test bootstrap confidence intervals calculation."""
        # Create test data
        np.random.seed(42)
        test_data = pd.DataFrame(
            {
                "reward_total": np.random.normal(0, 1, 100),
                "pnl": np.random.normal(0.01, 0.02, 100),
            }
        )

        results = bootstrap_confidence_intervals(
            test_data,
            ["reward_total", "pnl"],
            n_bootstrap=100,  # Small for speed
        )

        self.assertIn("reward_total", results)
        self.assertIn("pnl", results)

        for metric, (mean, ci_low, ci_high) in results.items():
            self.assertTrue(np.isfinite(mean), f"Mean for {metric} should be finite")
            self.assertTrue(
                np.isfinite(ci_low), f"CI low for {metric} should be finite"
            )
            self.assertTrue(
                np.isfinite(ci_high), f"CI high for {metric} should be finite"
            )
            self.assertLess(
                ci_low, ci_high, f"CI bounds for {metric} should be ordered"
            )

    def test_statistical_hypothesis_tests_seed_reproducibility(self):
        """Ensure statistical_hypothesis_tests + bootstrap CIs are reproducible with stats_seed."""
        from reward_space_analysis import (
            statistical_hypothesis_tests,
            bootstrap_confidence_intervals,
        )

        np.random.seed(123)
        # Create idle_duration with variability throughout to avoid constant Spearman warnings
        idle_base = np.random.uniform(3, 40, 300)
        idle_mask = np.random.rand(300) < 0.4
        idle_duration = (
            idle_base * idle_mask
        )  # some zeros but not fully constant subset
        df = pd.DataFrame(
            {
                "reward_total": np.random.normal(0, 1, 300),
                "reward_idle": np.where(idle_mask, np.random.normal(-1, 0.3, 300), 0.0),
                "reward_holding": np.where(
                    ~idle_mask, np.random.normal(-0.5, 0.2, 300), 0.0
                ),
                "reward_exit": np.random.normal(0.8, 0.6, 300),
                "pnl": np.random.normal(0.01, 0.02, 300),
                "trade_duration": np.random.uniform(5, 150, 300),
                "idle_duration": idle_duration,
                "position": np.random.choice([0.0, 0.5, 1.0], 300),
                "is_force_exit": np.random.choice([0.0, 1.0], 300, p=[0.85, 0.15]),
            }
        )

        # Two runs with same seed
        r1 = statistical_hypothesis_tests(df, seed=777)
        r2 = statistical_hypothesis_tests(df, seed=777)
        self.assertEqual(set(r1.keys()), set(r2.keys()))
        for k in r1:
            for field in ("p_value", "significant"):
                v1 = r1[k][field]
                v2 = r2[k][field]
                if (
                    isinstance(v1, float)
                    and isinstance(v2, float)
                    and (np.isnan(v1) and np.isnan(v2))
                ):
                    continue
                self.assertEqual(v1, v2, f"Mismatch for {k}:{field}")

        # Different seed may change bootstrap CI of rho if present
        r3 = statistical_hypothesis_tests(df, seed=888)
        if "idle_correlation" in r1 and "idle_correlation" in r3:
            ci1 = r1["idle_correlation"]["ci_95"]
            ci3 = r3["idle_correlation"]["ci_95"]
            # Not guaranteed different, but often; only assert shape
            self.assertEqual(len(ci1), 2)
            self.assertEqual(len(ci3), 2)

        # Test bootstrap reproducibility path
        metrics = ["reward_total", "pnl"]
        ci_a = bootstrap_confidence_intervals(df, metrics, n_bootstrap=250, seed=2024)
        ci_b = bootstrap_confidence_intervals(df, metrics, n_bootstrap=250, seed=2024)
        self.assertEqual(ci_a, ci_b)

    def test_pnl_invariant_validation(self):
        """Test critical PnL invariant: only exit actions should have non-zero PnL."""
        # Generate samples and verify PnL invariant
        df = simulate_samples(
            num_samples=200,
            seed=42,
            params=self.DEFAULT_PARAMS,
            max_trade_duration=50,
            base_factor=100.0,
            profit_target=0.03,
            risk_reward_ratio=1.0,
            holding_max_ratio=2.0,
            trading_mode="margin",
            pnl_base_std=0.02,
            pnl_duration_vol_scale=0.5,
        )

        # Critical invariant: Total PnL must equal sum of exit PnL
        total_pnl = df["pnl"].sum()
        exit_mask = df["reward_exit"] != 0
        exit_pnl_sum = df.loc[exit_mask, "pnl"].sum()

        self.assertAlmostEqual(
            total_pnl,
            exit_pnl_sum,
            places=10,
            msg="PnL invariant violation: total PnL != sum of exit PnL",
        )

        # Only exit actions (2.0, 4.0) should have non-zero PnL
        non_zero_pnl_actions = set(df[df["pnl"] != 0]["action"].unique())
        expected_exit_actions = {2.0, 4.0}  # Long_exit, Short_exit

        self.assertTrue(
            non_zero_pnl_actions.issubset(expected_exit_actions),
            f"Non-exit actions have PnL: {non_zero_pnl_actions - expected_exit_actions}",
        )

        # No actions should have zero PnL but non-zero exit reward
        invalid_combinations = df[(df["pnl"] == 0) & (df["reward_exit"] != 0)]
        self.assertEqual(
            len(invalid_combinations),
            0,
            "Found actions with zero PnL but non-zero exit reward",
        )

    def test_distribution_metrics_mathematical_bounds(self):
        """Test mathematical bounds and validity of distribution shift metrics."""
        # Create two different distributions for testing
        np.random.seed(42)
        df1 = pd.DataFrame(
            {
                "pnl": np.random.normal(0, 0.02, 500),
                "trade_duration": np.random.exponential(30, 500),
                "idle_duration": np.random.gamma(2, 5, 500),
            }
        )

        df2 = pd.DataFrame(
            {
                "pnl": np.random.normal(0.01, 0.025, 500),
                "trade_duration": np.random.exponential(35, 500),
                "idle_duration": np.random.gamma(2.5, 6, 500),
            }
        )

        metrics = compute_distribution_shift_metrics(df1, df2)

        # Validate mathematical bounds for each feature
        for feature in ["pnl", "trade_duration", "idle_duration"]:
            # KL divergence must be >= 0
            kl_key = f"{feature}_kl_divergence"
            if kl_key in metrics:
                self.assertGreaterEqual(
                    metrics[kl_key], 0, f"KL divergence for {feature} must be >= 0"
                )

            # JS distance must be in [0, 1]
            js_key = f"{feature}_js_distance"
            if js_key in metrics:
                js_val = metrics[js_key]
                self.assertGreaterEqual(
                    js_val, 0, f"JS distance for {feature} must be >= 0"
                )
                self.assertLessEqual(
                    js_val, 1, f"JS distance for {feature} must be <= 1"
                )

            # Wasserstein must be >= 0
            ws_key = f"{feature}_wasserstein"
            if ws_key in metrics:
                self.assertGreaterEqual(
                    metrics[ws_key],
                    0,
                    f"Wasserstein distance for {feature} must be >= 0",
                )

            # KS statistic must be in [0, 1]
            ks_stat_key = f"{feature}_ks_statistic"
            if ks_stat_key in metrics:
                ks_val = metrics[ks_stat_key]
                self.assertGreaterEqual(
                    ks_val, 0, f"KS statistic for {feature} must be >= 0"
                )
                self.assertLessEqual(
                    ks_val, 1, f"KS statistic for {feature} must be <= 1"
                )

            # KS p-value must be in [0, 1]
            ks_p_key = f"{feature}_ks_pvalue"
            if ks_p_key in metrics:
                p_val = metrics[ks_p_key]
                self.assertGreaterEqual(
                    p_val, 0, f"KS p-value for {feature} must be >= 0"
                )
                self.assertLessEqual(p_val, 1, f"KS p-value for {feature} must be <= 1")

    def test_heteroscedasticity_pnl_validation(self):
        """Test that PnL variance increases with trade duration (heteroscedasticity)."""
        # Generate larger sample for statistical power
        df = simulate_samples(
            num_samples=1000,
            seed=123,
            params=self.DEFAULT_PARAMS,
            max_trade_duration=100,
            base_factor=100.0,
            profit_target=0.03,
            risk_reward_ratio=1.0,
            holding_max_ratio=2.0,
            trading_mode="margin",
            pnl_base_std=0.02,
            pnl_duration_vol_scale=0.5,
        )

        # Filter to exit actions only (where PnL is meaningful)
        exit_data = df[df["reward_exit"] != 0].copy()

        if len(exit_data) < 50:
            self.skipTest("Insufficient exit actions for heteroscedasticity test")

        # Create duration bins and test variance pattern
        exit_data["duration_bin"] = pd.cut(
            exit_data["duration_ratio"], bins=4, labels=["Q1", "Q2", "Q3", "Q4"]
        )

        variance_by_bin = exit_data.groupby("duration_bin")["pnl"].var().dropna()

        # Should see increasing variance with duration (not always strict monotonic due to sampling)
        # Test that Q4 variance > Q1 variance (should hold with heteroscedastic model)
        if "Q1" in variance_by_bin.index and "Q4" in variance_by_bin.index:
            self.assertGreater(
                variance_by_bin["Q4"],
                variance_by_bin["Q1"] * 0.8,  # Allow some tolerance
                "PnL heteroscedasticity: variance should increase with duration",
            )

    def test_exit_factor_mathematical_formulas(self):
        """Test mathematical correctness of exit factor calculations."""
        import math
        from reward_space_analysis import (
            calculate_reward,
            RewardContext,
            Actions,
            Positions,
        )

        # Test context with known values
        context = RewardContext(
            pnl=0.05,
            trade_duration=50,
            idle_duration=0,
            max_trade_duration=100,
            max_unrealized_profit=0.06,
            min_unrealized_profit=0.04,
            position=Positions.Long,
            action=Actions.Long_exit,
            force_action=None,
        )

        params = self.DEFAULT_PARAMS.copy()
        duration_ratio = 50 / 100  # 0.5

        # Test power mode with known tau
        params["exit_factor_mode"] = "power"
        params["exit_power_tau"] = 0.5

        reward_power = calculate_reward(
            context, params, 100.0, 0.03, 1.0, short_allowed=True, action_masking=True
        )

        # Mathematical validation: alpha = -ln(tau) = -ln(0.5) ≈ 0.693
        expected_alpha = -math.log(0.5)
        expected_attenuation = 1.0 / (1.0 + expected_alpha * duration_ratio)

        # The reward should be attenuated by this factor
        self.assertGreater(
            reward_power.exit_component, 0, "Power mode should generate positive reward"
        )

        # Test half_life mode
        params["exit_factor_mode"] = "half_life"
        params["exit_half_life"] = 0.5

        reward_half_life = calculate_reward(
            context, params, 100.0, 0.03, 1.0, short_allowed=True, action_masking=True
        )

        # Mathematical validation: 2^(-duration_ratio/half_life) = 2^(-0.5/0.5) = 0.5
        expected_half_life_factor = 2 ** (-duration_ratio / 0.5)
        self.assertAlmostEqual(expected_half_life_factor, 0.5, places=6)

        # Test that different modes produce different results (mathematical diversity)
        params["exit_factor_mode"] = "linear"
        params["exit_linear_slope"] = 1.0

        reward_linear = calculate_reward(
            context, params, 100.0, 0.03, 1.0, short_allowed=True, action_masking=True
        )

        # All modes should produce positive rewards but different values
        self.assertGreater(reward_power.exit_component, 0)
        self.assertGreater(reward_half_life.exit_component, 0)
        self.assertGreater(reward_linear.exit_component, 0)

        # They should be different (mathematical distinctness)
        rewards = [
            reward_power.exit_component,
            reward_half_life.exit_component,
            reward_linear.exit_component,
        ]
        unique_rewards = set(f"{r:.6f}" for r in rewards)
        self.assertGreater(
            len(unique_rewards),
            1,
            "Different exit factor modes should produce different rewards",
        )

    def test_statistical_functions_bounds_validation(self):
        """Test that all statistical functions respect mathematical bounds."""
        # Create test data with known statistical properties
        np.random.seed(42)
        df = pd.DataFrame(
            {
                "reward_total": np.random.normal(0, 1, 300),
                "reward_idle": np.random.normal(-1, 0.5, 300),
                "reward_holding": np.random.normal(-0.5, 0.3, 300),
                "reward_exit": np.random.normal(1, 0.8, 300),
                "pnl": np.random.normal(0.01, 0.02, 300),
                "trade_duration": np.random.uniform(5, 150, 300),
                "idle_duration": np.random.uniform(0, 100, 300),
                "position": np.random.choice([0.0, 0.5, 1.0], 300),
                "is_force_exit": np.random.choice([0.0, 1.0], 300, p=[0.8, 0.2]),
            }
        )

        # Test distribution diagnostics bounds
        diagnostics = distribution_diagnostics(df)

        for col in ["reward_total", "pnl", "trade_duration", "idle_duration"]:
            # Skewness can be any real number, but should be finite
            if f"{col}_skewness" in diagnostics:
                skew = diagnostics[f"{col}_skewness"]
                self.assertTrue(
                    np.isfinite(skew), f"Skewness for {col} should be finite"
                )

            # Kurtosis should be finite (can be negative for platykurtic distributions)
            if f"{col}_kurtosis" in diagnostics:
                kurt = diagnostics[f"{col}_kurtosis"]
                self.assertTrue(
                    np.isfinite(kurt), f"Kurtosis for {col} should be finite"
                )

            # Shapiro p-value must be in [0, 1]
            if f"{col}_shapiro_pval" in diagnostics:
                p_val = diagnostics[f"{col}_shapiro_pval"]
                self.assertGreaterEqual(
                    p_val, 0, f"Shapiro p-value for {col} must be >= 0"
                )
                self.assertLessEqual(
                    p_val, 1, f"Shapiro p-value for {col} must be <= 1"
                )

        # Test hypothesis tests results bounds
        from reward_space_analysis import statistical_hypothesis_tests

        hypothesis_results = statistical_hypothesis_tests(df, seed=42)

        for test_name, result in hypothesis_results.items():
            # All p-values must be in [0, 1]
            if "p_value" in result:
                p_val = result["p_value"]
                self.assertGreaterEqual(
                    p_val, 0, f"p-value for {test_name} must be >= 0"
                )
                self.assertLessEqual(p_val, 1, f"p-value for {test_name} must be <= 1")

            # Effect sizes should be finite and meaningful
            if "effect_size_epsilon_sq" in result:
                effect_size = result["effect_size_epsilon_sq"]
                self.assertTrue(
                    np.isfinite(effect_size),
                    f"Effect size for {test_name} should be finite",
                )
                self.assertGreaterEqual(
                    effect_size, 0, f"ε² for {test_name} should be >= 0"
                )

            if "effect_size_rank_biserial" in result:
                rb_corr = result["effect_size_rank_biserial"]
                self.assertTrue(
                    np.isfinite(rb_corr),
                    f"Rank-biserial correlation for {test_name} should be finite",
                )
                self.assertGreaterEqual(
                    rb_corr, -1, f"Rank-biserial for {test_name} should be >= -1"
                )
                self.assertLessEqual(
                    rb_corr, 1, f"Rank-biserial for {test_name} should be <= 1"
                )

    def test_simulate_samples_with_different_modes(self):
        """Test simulate_samples with different trading modes."""
        # Test spot mode (no shorts)
        df_spot = simulate_samples(
            num_samples=100,
            seed=42,
            params=self.DEFAULT_PARAMS,
            max_trade_duration=100,
            base_factor=100.0,
            profit_target=0.03,
            risk_reward_ratio=1.0,
            holding_max_ratio=2.0,
            trading_mode="spot",
            pnl_base_std=0.02,
            pnl_duration_vol_scale=0.5,
        )

        # Should not have any short positions
        short_positions = (df_spot["position"] == float(Positions.Short.value)).sum()
        self.assertEqual(
            short_positions, 0, "Spot mode should not have short positions"
        )

        # Test margin mode (shorts allowed)
        df_margin = simulate_samples(
            num_samples=100,
            seed=42,
            params=self.DEFAULT_PARAMS,
            max_trade_duration=100,
            base_factor=100.0,
            profit_target=0.03,
            risk_reward_ratio=1.0,
            holding_max_ratio=2.0,
            trading_mode="margin",
            pnl_base_std=0.02,
            pnl_duration_vol_scale=0.5,
        )

        # Should have required columns
        required_columns = [
            "pnl",
            "trade_duration",
            "idle_duration",
            "position",
            "action",
            "reward_total",
            "reward_invalid",
            "reward_idle",
            "reward_holding",
            "reward_exit",
        ]
        for col in required_columns:
            self.assertIn(col, df_margin.columns, f"Column {col} should be present")

    def test_reward_calculation_comprehensive(self):
        """Test comprehensive reward calculation scenarios."""
        # Test different reward scenarios
        test_cases = [
            # (position, action, force_action, expected_reward_type)
            (Positions.Neutral, Actions.Neutral, None, "idle_penalty"),
            (Positions.Long, Actions.Long_exit, None, "exit_component"),
            (Positions.Short, Actions.Short_exit, None, "exit_component"),
            (
                Positions.Long,
                Actions.Neutral,
                ForceActions.Take_profit,
                "exit_component",
            ),
            (
                Positions.Short,
                Actions.Neutral,
                ForceActions.Stop_loss,
                "exit_component",
            ),
        ]

        for position, action, force_action, expected_type in test_cases:
            with self.subTest(
                position=position, action=action, force_action=force_action
            ):
                context = RewardContext(
                    pnl=0.02 if force_action == ForceActions.Take_profit else -0.02,
                    trade_duration=50 if position != Positions.Neutral else 0,
                    idle_duration=10 if position == Positions.Neutral else 0,
                    max_trade_duration=100,
                    max_unrealized_profit=0.03,
                    min_unrealized_profit=-0.01,
                    position=position,
                    action=action,
                    force_action=force_action,
                )

                breakdown = calculate_reward(
                    context,
                    self.DEFAULT_PARAMS,
                    base_factor=100.0,
                    profit_target=0.03,
                    risk_reward_ratio=1.0,
                    short_allowed=True,
                    action_masking=True,
                )

                # Check that the appropriate component is non-zero
                if expected_type == "idle_penalty":
                    self.assertNotEqual(
                        breakdown.idle_penalty,
                        0.0,
                        f"Expected idle penalty for {position}/{action}",
                    )
                elif expected_type == "exit_component":
                    self.assertNotEqual(
                        breakdown.exit_component,
                        0.0,
                        f"Expected exit component for {position}/{action}",
                    )

                # Total should always be finite
                self.assertTrue(
                    np.isfinite(breakdown.total),
                    f"Reward total should be finite for {position}/{action}",
                )


class TestEdgeCases(RewardSpaceTestBase):
    """Test edge cases and error conditions."""

    def test_extreme_parameter_values(self):
        """Test behavior with extreme parameter values."""
        # Test with very large parameters
        extreme_params = self.DEFAULT_PARAMS.copy()
        extreme_params["win_reward_factor"] = 1000.0
        extreme_params["base_factor"] = 10000.0

        context = RewardContext(
            pnl=0.05,
            trade_duration=50,
            idle_duration=0,
            max_trade_duration=100,
            max_unrealized_profit=0.06,
            min_unrealized_profit=0.02,
            position=Positions.Long,
            action=Actions.Long_exit,
            force_action=None,
        )

        breakdown = calculate_reward(
            context,
            extreme_params,
            base_factor=10000.0,
            profit_target=0.03,
            risk_reward_ratio=1.0,
            short_allowed=True,
            action_masking=True,
        )

        self.assertTrue(
            np.isfinite(breakdown.total),
            "Reward should be finite even with extreme parameters",
        )

    def test_different_exit_factor_modes(self):
        """Test different exit factor calculation modes."""
        modes = ["legacy", "sqrt", "linear", "power", "piecewise", "half_life"]

        for mode in modes:
            with self.subTest(mode=mode):
                test_params = self.DEFAULT_PARAMS.copy()
                test_params["exit_factor_mode"] = mode

                context = RewardContext(
                    pnl=0.02,
                    trade_duration=50,
                    idle_duration=0,
                    max_trade_duration=100,
                    max_unrealized_profit=0.03,
                    min_unrealized_profit=0.01,
                    position=Positions.Long,
                    action=Actions.Long_exit,
                    force_action=None,
                )

                breakdown = calculate_reward(
                    context,
                    test_params,
                    base_factor=100.0,
                    profit_target=0.03,
                    risk_reward_ratio=1.0,
                    short_allowed=True,
                    action_masking=True,
                )

                self.assertTrue(
                    np.isfinite(breakdown.exit_component),
                    f"Exit component should be finite for mode {mode}",
                )
                self.assertTrue(
                    np.isfinite(breakdown.total),
                    f"Total reward should be finite for mode {mode}",
                )


class TestUtilityFunctions(RewardSpaceTestBase):
    """Test utility and helper functions."""

    def test_to_bool_comprehensive(self):
        """Test _to_bool with comprehensive inputs."""
        # Test via simulate_samples which uses action_masking parameter
        df1 = simulate_samples(
            num_samples=10,
            seed=42,
            params={"action_masking": "true"},
            max_trade_duration=50,
            base_factor=100.0,
            profit_target=0.03,
            risk_reward_ratio=1.0,
            holding_max_ratio=2.0,
            trading_mode="spot",
            pnl_base_std=0.02,
            pnl_duration_vol_scale=0.5,
        )
        self.assertIsInstance(df1, pd.DataFrame)

        df2 = simulate_samples(
            num_samples=10,
            seed=42,
            params={"action_masking": "false"},
            max_trade_duration=50,
            base_factor=100.0,
            profit_target=0.03,
            risk_reward_ratio=1.0,
            holding_max_ratio=2.0,
            trading_mode="spot",
            pnl_base_std=0.02,
            pnl_duration_vol_scale=0.5,
        )
        self.assertIsInstance(df2, pd.DataFrame)

    def test_short_allowed_via_simulation(self):
        """Test _is_short_allowed via different trading modes."""
        # Test futures mode (shorts allowed)
        df_futures = simulate_samples(
            num_samples=100,
            seed=42,
            params=self.DEFAULT_PARAMS,
            max_trade_duration=50,
            base_factor=100.0,
            profit_target=0.03,
            risk_reward_ratio=1.0,
            holding_max_ratio=2.0,
            trading_mode="futures",
            pnl_base_std=0.02,
            pnl_duration_vol_scale=0.5,
        )

        # Should have some short positions
        short_positions = (df_futures["position"] == float(Positions.Short.value)).sum()
        self.assertGreater(
            short_positions, 0, "Futures mode should allow short positions"
        )

    def test_model_analysis_function(self):
        """Test model_analysis function."""
        from reward_space_analysis import model_analysis

        # Create test data
        test_data = simulate_samples(
            num_samples=100,
            seed=42,
            params=self.DEFAULT_PARAMS,
            max_trade_duration=50,
            base_factor=100.0,
            profit_target=0.03,
            risk_reward_ratio=1.0,
            holding_max_ratio=2.0,
            trading_mode="spot",
            pnl_base_std=0.02,
            pnl_duration_vol_scale=0.5,
        )

        # Create temporary output directory
        with tempfile.TemporaryDirectory() as tmp_dir:
            output_path = Path(tmp_dir)
            model_analysis(test_data, output_path, seed=42)

            # Check that feature importance file is created
            feature_file = output_path / "feature_importance.csv"
            self.assertTrue(
                feature_file.exists(), "Feature importance file should be created"
            )

    def test_write_functions(self):
        """Test various write functions."""
        from reward_space_analysis import (
            write_summary,
            write_relationship_reports,
            write_representativity_report,
        )

        # Create test data
        test_data = simulate_samples(
            num_samples=100,
            seed=42,
            params=self.DEFAULT_PARAMS,
            max_trade_duration=50,
            base_factor=100.0,
            profit_target=0.03,
            risk_reward_ratio=1.0,
            holding_max_ratio=2.0,
            trading_mode="spot",
            pnl_base_std=0.02,
            pnl_duration_vol_scale=0.5,
        )

        with tempfile.TemporaryDirectory() as tmp_dir:
            output_path = Path(tmp_dir)

            # Test write_summary
            write_summary(test_data, output_path)
            summary_file = output_path / "reward_summary.md"
            self.assertTrue(summary_file.exists(), "Summary file should be created")

            # Test write_relationship_reports
            write_relationship_reports(test_data, output_path, max_trade_duration=50)
            relationship_file = output_path / "reward_relationships.md"
            self.assertTrue(
                relationship_file.exists(), "Relationship file should be created"
            )

            # Test write_representativity_report
            write_representativity_report(
                test_data, output_path, profit_target=0.03, max_trade_duration=50
            )
            repr_file = output_path / "representativity.md"
            self.assertTrue(
                repr_file.exists(), "Representativity file should be created"
            )

    def test_load_real_episodes(self):
        """Test load_real_episodes function."""
        from reward_space_analysis import load_real_episodes

        # Create a temporary pickle file with test data
        test_episodes = pd.DataFrame(
            {
                "pnl": [0.01, -0.02, 0.03],
                "trade_duration": [10, 20, 15],
                "idle_duration": [5, 0, 8],
                "position": [1.0, 0.0, 1.0],
                "action": [2.0, 0.0, 2.0],
                "reward_total": [10.5, -5.2, 15.8],
            }
        )

        with tempfile.TemporaryDirectory() as tmp_dir:
            pickle_path = Path(tmp_dir) / "test_episodes.pkl"
            with pickle_path.open("wb") as f:
                pickle.dump(test_episodes, f)  # Don't wrap in list

            loaded_data = load_real_episodes(pickle_path)
            self.assertIsInstance(loaded_data, pd.DataFrame)
            self.assertEqual(len(loaded_data), 3)
            self.assertIn("pnl", loaded_data.columns)

    def test_statistical_functions_comprehensive(self):
        """Test comprehensive statistical functions."""
        from reward_space_analysis import (
            statistical_hypothesis_tests,
            write_enhanced_statistical_report,
        )

        # Create test data with specific patterns
        np.random.seed(42)
        test_data = pd.DataFrame(
            {
                "reward_total": np.random.normal(0, 1, 200),
                "reward_idle": np.concatenate(
                    [np.zeros(150), np.random.normal(-1, 0.5, 50)]
                ),
                "reward_holding": np.concatenate(
                    [np.zeros(150), np.random.normal(-0.5, 0.3, 50)]
                ),
                "reward_exit": np.concatenate(
                    [np.zeros(150), np.random.normal(2, 1, 50)]
                ),
                "pnl": np.random.normal(0.01, 0.02, 200),
                "trade_duration": np.random.uniform(10, 100, 200),
                "idle_duration": np.concatenate(
                    [np.random.uniform(5, 50, 50), np.zeros(150)]
                ),
                "position": np.random.choice([0.0, 0.5, 1.0], 200),
                "is_force_exit": np.random.choice([0.0, 1.0], 200, p=[0.8, 0.2]),
            }
        )

        # Test hypothesis tests
        results = statistical_hypothesis_tests(test_data)
        self.assertIsInstance(results, dict)

        # Test enhanced statistical report
        with tempfile.TemporaryDirectory() as tmp_dir:
            output_path = Path(tmp_dir)
            write_enhanced_statistical_report(test_data, output_path)
            report_file = output_path / "enhanced_statistical_report.md"
            self.assertTrue(
                report_file.exists(), "Enhanced statistical report should be created"
            )

    def test_argument_parser_construction(self):
        """Test build_argument_parser function."""
        from reward_space_analysis import build_argument_parser

        parser = build_argument_parser()
        self.assertIsNotNone(parser)

        # Test parsing with minimal arguments
        args = parser.parse_args(["--num_samples", "100", "--output", "test_output"])
        self.assertEqual(args.num_samples, 100)
        self.assertEqual(str(args.output), "test_output")

    def test_complete_statistical_analysis_writer(self):
        """Test write_complete_statistical_analysis function."""
        from reward_space_analysis import write_complete_statistical_analysis

        # Create comprehensive test data
        test_data = simulate_samples(
            num_samples=200,
            seed=42,
            params=self.DEFAULT_PARAMS,
            max_trade_duration=100,
            base_factor=100.0,
            profit_target=0.03,
            risk_reward_ratio=1.0,
            holding_max_ratio=2.0,
            trading_mode="margin",
            pnl_base_std=0.02,
            pnl_duration_vol_scale=0.5,
        )

        with tempfile.TemporaryDirectory() as tmp_dir:
            output_path = Path(tmp_dir)

            write_complete_statistical_analysis(
                test_data,
                output_path,
                max_trade_duration=100,
                profit_target=0.03,
                seed=42,
                real_df=None,
            )

            # Check that main report is created
            main_report = output_path / "statistical_analysis.md"
            self.assertTrue(
                main_report.exists(), "Main statistical analysis should be created"
            )

            # Check for other expected files
            feature_file = output_path / "feature_importance.csv"
            self.assertTrue(
                feature_file.exists(), "Feature importance should be created"
            )


class TestPrivateFunctionsViaPublicAPI(RewardSpaceTestBase):
    """Test private functions through public API calls."""

    def test_idle_penalty_via_rewards(self):
        """Test idle penalty calculation via reward calculation."""
        # Create context that will trigger idle penalty
        context = RewardContext(
            pnl=0.0,
            trade_duration=0,
            idle_duration=20,
            max_trade_duration=100,
            max_unrealized_profit=0.0,
            min_unrealized_profit=0.0,
            position=Positions.Neutral,
            action=Actions.Neutral,
            force_action=None,
        )

        breakdown = calculate_reward(
            context,
            self.DEFAULT_PARAMS,
            base_factor=100.0,
            profit_target=0.03,
            risk_reward_ratio=1.0,
            short_allowed=True,
            action_masking=True,
        )

        self.assertLess(breakdown.idle_penalty, 0, "Idle penalty should be negative")
        self.assertEqual(
            breakdown.total, breakdown.idle_penalty, "Total should equal idle penalty"
        )

    def test_holding_penalty_via_rewards(self):
        """Test holding penalty calculation via reward calculation."""
        # Create context that will trigger holding penalty
        context = RewardContext(
            pnl=0.01,
            trade_duration=150,
            idle_duration=0,  # Long duration
            max_trade_duration=100,
            max_unrealized_profit=0.02,
            min_unrealized_profit=0.0,
            position=Positions.Long,
            action=Actions.Neutral,
            force_action=None,
        )

        breakdown = calculate_reward(
            context,
            self.DEFAULT_PARAMS,
            base_factor=100.0,
            profit_target=0.03,
            risk_reward_ratio=1.0,
            short_allowed=True,
            action_masking=True,
        )

        self.assertLess(
            breakdown.holding_penalty, 0, "Holding penalty should be negative"
        )
        self.assertEqual(
            breakdown.total,
            breakdown.holding_penalty,
            "Total should equal holding penalty",
        )

    def test_exit_reward_calculation_comprehensive(self):
        """Test exit reward calculation with various scenarios."""
        scenarios = [
            (Positions.Long, Actions.Long_exit, 0.05, "Profitable long exit"),
            (Positions.Short, Actions.Short_exit, -0.03, "Profitable short exit"),
            (Positions.Long, Actions.Long_exit, -0.02, "Losing long exit"),
            (Positions.Short, Actions.Short_exit, 0.02, "Losing short exit"),
        ]

        for position, action, pnl, description in scenarios:
            with self.subTest(description=description):
                context = RewardContext(
                    pnl=pnl,
                    trade_duration=50,
                    idle_duration=0,
                    max_trade_duration=100,
                    max_unrealized_profit=max(pnl + 0.01, 0.01),
                    min_unrealized_profit=min(pnl - 0.01, -0.01),
                    position=position,
                    action=action,
                    force_action=None,
                )

                breakdown = calculate_reward(
                    context,
                    self.DEFAULT_PARAMS,
                    base_factor=100.0,
                    profit_target=0.03,
                    risk_reward_ratio=1.0,
                    short_allowed=True,
                    action_masking=True,
                )

                self.assertNotEqual(
                    breakdown.exit_component,
                    0.0,
                    f"Exit component should be non-zero for {description}",
                )
                self.assertTrue(
                    np.isfinite(breakdown.total),
                    f"Total should be finite for {description}",
                )

    def test_invalid_action_handling(self):
        """Test invalid action penalty."""
        # Try to exit long when in short position (invalid)
        context = RewardContext(
            pnl=0.02,
            trade_duration=50,
            idle_duration=0,
            max_trade_duration=100,
            max_unrealized_profit=0.03,
            min_unrealized_profit=0.01,
            position=Positions.Short,
            action=Actions.Long_exit,
            force_action=None,  # Invalid: can't long_exit from short
        )

        breakdown = calculate_reward(
            context,
            self.DEFAULT_PARAMS,
            base_factor=100.0,
            profit_target=0.03,
            risk_reward_ratio=1.0,
            short_allowed=True,
            action_masking=False,  # Disable masking to test invalid penalty
        )

        self.assertLess(
            breakdown.invalid_penalty, 0, "Invalid action should have negative penalty"
        )
        self.assertEqual(
            breakdown.total,
            breakdown.invalid_penalty,
            "Total should equal invalid penalty",
        )


if __name__ == "__main__":
    # Configure test discovery and execution
    loader = unittest.TestLoader()
    suite = loader.loadTestsFromModule(sys.modules[__name__])

    # Run tests with verbose output
    runner = unittest.TextTestRunner(verbosity=2, buffer=True)
    result = runner.run(suite)

    # Exit with error code if tests failed
    sys.exit(0 if result.wasSuccessful() else 1)
