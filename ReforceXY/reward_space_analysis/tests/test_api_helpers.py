#!/usr/bin/env python3
"""Tests for public API and helper functions."""

import math
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

from reward_space_analysis import (
    Actions,
    Positions,
    _get_bool_param,
    _get_float_param,
    _get_int_param,
    _get_str_param,
    build_argument_parser,
    calculate_reward,
    parse_overrides,
    simulate_samples,
    write_complete_statistical_analysis,
)

from .test_base import RewardSpaceTestBase


class TestAPIAndHelpers(RewardSpaceTestBase):
    """Public API + helper utility tests."""

    def test_parse_overrides(self):
        overrides = ["alpha=1.5", "mode=linear", "limit=42"]
        result = parse_overrides(overrides)
        self.assertEqual(result["alpha"], 1.5)
        self.assertEqual(result["mode"], "linear")
        self.assertEqual(result["limit"], 42.0)
        with self.assertRaises(ValueError):
            parse_overrides(["badpair"])

    def test_api_simulation_and_reward_smoke(self):
        df = simulate_samples(
            params=self.base_params(max_trade_duration_candles=40),
            num_samples=20,
            seed=self.SEED_SMOKE_TEST,
            base_factor=self.TEST_BASE_FACTOR,
            profit_target=self.TEST_PROFIT_TARGET,
            risk_reward_ratio=self.TEST_RR,
            max_duration_ratio=1.5,
            trading_mode="margin",
            pnl_base_std=self.TEST_PNL_STD,
            pnl_duration_vol_scale=self.TEST_PNL_DUR_VOL_SCALE,
        )
        self.assertGreater(len(df), 0)
        any_exit = df[df["reward_exit"] != 0].head(1)
        if not any_exit.empty:
            row = any_exit.iloc[0]
            ctx = self.make_ctx(
                pnl=float(row["pnl"]),
                trade_duration=int(row["trade_duration"]),
                idle_duration=int(row["idle_duration"]),
                max_unrealized_profit=float(row["pnl"]) + 0.01,
                min_unrealized_profit=float(row["pnl"]) - 0.01,
                position=Positions.Long,
                action=Actions.Long_exit,
            )
            breakdown = calculate_reward(
                ctx,
                self.DEFAULT_PARAMS,
                base_factor=self.TEST_BASE_FACTOR,
                profit_target=self.TEST_PROFIT_TARGET,
                risk_reward_ratio=self.TEST_RR,
                short_allowed=True,
                action_masking=True,
            )
            self.assertFinite(breakdown.total)

    def test_simulate_samples_trading_modes_spot_vs_margin(self):
        """simulate_samples coverage: spot should forbid shorts, margin should allow them."""
        df_spot = simulate_samples(
            params=self.base_params(max_trade_duration_candles=100),
            num_samples=80,
            seed=self.SEED,
            base_factor=self.TEST_BASE_FACTOR,
            profit_target=self.TEST_PROFIT_TARGET,
            risk_reward_ratio=self.TEST_RR,
            max_duration_ratio=2.0,
            trading_mode="spot",
            pnl_base_std=self.TEST_PNL_STD,
            pnl_duration_vol_scale=self.TEST_PNL_DUR_VOL_SCALE,
        )
        short_positions_spot = (df_spot["position"] == float(Positions.Short.value)).sum()
        self.assertEqual(short_positions_spot, 0, "Spot mode must not contain short positions")
        df_margin = simulate_samples(
            params=self.base_params(max_trade_duration_candles=100),
            num_samples=80,
            seed=self.SEED,
            base_factor=self.TEST_BASE_FACTOR,
            profit_target=self.TEST_PROFIT_TARGET,
            risk_reward_ratio=self.TEST_RR,
            max_duration_ratio=2.0,
            trading_mode="margin",
            pnl_base_std=self.TEST_PNL_STD,
            pnl_duration_vol_scale=self.TEST_PNL_DUR_VOL_SCALE,
        )
        for col in [
            "pnl",
            "trade_duration",
            "idle_duration",
            "position",
            "action",
            "reward",
            "reward_invalid",
            "reward_idle",
            "reward_hold",
            "reward_exit",
        ]:
            self.assertIn(col, df_margin.columns)

    def test_to_bool(self):
        """Test _to_bool with various inputs."""
        df1 = simulate_samples(
            params=self.base_params(action_masking="true", max_trade_duration_candles=50),
            num_samples=10,
            seed=self.SEED,
            base_factor=self.TEST_BASE_FACTOR,
            profit_target=self.TEST_PROFIT_TARGET,
            risk_reward_ratio=self.TEST_RR,
            max_duration_ratio=2.0,
            trading_mode="spot",
            pnl_base_std=self.TEST_PNL_STD,
            pnl_duration_vol_scale=self.TEST_PNL_DUR_VOL_SCALE,
        )
        self.assertIsInstance(df1, pd.DataFrame)
        df2 = simulate_samples(
            params=self.base_params(action_masking="false", max_trade_duration_candles=50),
            num_samples=10,
            seed=self.SEED,
            base_factor=self.TEST_BASE_FACTOR,
            profit_target=self.TEST_PROFIT_TARGET,
            risk_reward_ratio=self.TEST_RR,
            max_duration_ratio=2.0,
            trading_mode="spot",
            pnl_base_std=self.TEST_PNL_STD,
            pnl_duration_vol_scale=self.TEST_PNL_DUR_VOL_SCALE,
        )
        self.assertIsInstance(df2, pd.DataFrame)

    def test_short_allowed_via_simulation(self):
        """Test _is_short_allowed via different trading modes."""
        df_futures = simulate_samples(
            params=self.base_params(max_trade_duration_candles=50),
            num_samples=100,
            seed=self.SEED,
            base_factor=self.TEST_BASE_FACTOR,
            profit_target=self.TEST_PROFIT_TARGET,
            risk_reward_ratio=self.TEST_RR,
            max_duration_ratio=2.0,
            trading_mode="futures",
            pnl_base_std=self.TEST_PNL_STD,
            pnl_duration_vol_scale=self.TEST_PNL_DUR_VOL_SCALE,
        )
        short_positions = (df_futures["position"] == float(Positions.Short.value)).sum()
        self.assertGreater(short_positions, 0, "Futures mode should allow short positions")

    def test_get_float_param(self):
        """Test float parameter extraction."""
        params = {"test_float": 1.5, "test_int": 2, "test_str": "hello"}
        self.assertEqual(_get_float_param(params, "test_float", 0.0), 1.5)
        self.assertEqual(_get_float_param(params, "test_int", 0.0), 2.0)
        val_str = _get_float_param(params, "test_str", 0.0)
        self.assertTrue(isinstance(val_str, float))
        self.assertTrue(math.isnan(val_str))
        self.assertEqual(_get_float_param(params, "missing", 3.14), 3.14)

    def test_get_str_param(self):
        """Test string parameter extraction."""
        params = {"test_str": "hello", "test_int": 2}
        self.assertEqual(_get_str_param(params, "test_str", "default"), "hello")
        self.assertEqual(_get_str_param(params, "test_int", "default"), "default")
        self.assertEqual(_get_str_param(params, "missing", "default"), "default")

    def test_get_bool_param(self):
        """Test boolean parameter extraction."""
        params = {"test_true": True, "test_false": False, "test_int": 1, "test_str": "yes"}
        self.assertTrue(_get_bool_param(params, "test_true", False))
        self.assertFalse(_get_bool_param(params, "test_false", True))
        self.assertTrue(_get_bool_param(params, "test_int", False))
        self.assertTrue(_get_bool_param(params, "test_str", False))
        self.assertFalse(_get_bool_param(params, "missing", False))

    def test_get_int_param_coercions(self):
        """Robust coercion paths of _get_int_param (bool/int/float/str/None/unsupported)."""
        self.assertEqual(_get_int_param({"k": None}, "k", 5), 5)
        self.assertEqual(_get_int_param({"k": None}, "k", "x"), 0)
        self.assertEqual(_get_int_param({"k": True}, "k", 0), 1)
        self.assertEqual(_get_int_param({"k": False}, "k", 7), 0)
        self.assertEqual(_get_int_param({"k": -12}, "k", 0), -12)
        self.assertEqual(_get_int_param({"k": 9.99}, "k", 0), 9)
        self.assertEqual(_get_int_param({"k": -3.7}, "k", 0), -3)
        self.assertEqual(_get_int_param({"k": np.nan}, "k", 4), 4)
        self.assertEqual(_get_int_param({"k": float("inf")}, "k", 4), 4)
        self.assertEqual(_get_int_param({"k": "42"}, "k", 0), 42)
        self.assertEqual(_get_int_param({"k": " 17 "}, "k", 0), 17)
        self.assertEqual(_get_int_param({"k": "3.9"}, "k", 0), 3)
        self.assertEqual(_get_int_param({"k": "1e2"}, "k", 0), 100)
        self.assertEqual(_get_int_param({"k": ""}, "k", 5), 5)
        self.assertEqual(_get_int_param({"k": "abc"}, "k", 5), 5)
        self.assertEqual(_get_int_param({"k": "NaN"}, "k", 5), 5)
        self.assertEqual(_get_int_param({"k": [1, 2, 3]}, "k", 3), 3)
        self.assertEqual(_get_int_param({}, "missing", "zzz"), 0)

    def test_argument_parser_construction(self):
        """Test build_argument_parser function."""
        parser = build_argument_parser()
        self.assertIsNotNone(parser)
        args = parser.parse_args(["--num_samples", "100", "--out_dir", "test_output"])
        self.assertEqual(args.num_samples, 100)
        self.assertEqual(str(args.out_dir), "test_output")

    def test_complete_statistical_analysis_writer(self):
        """Test write_complete_statistical_analysis function."""
        test_data = simulate_samples(
            params=self.base_params(max_trade_duration_candles=100),
            num_samples=200,
            seed=self.SEED,
            base_factor=self.TEST_BASE_FACTOR,
            profit_target=self.TEST_PROFIT_TARGET,
            risk_reward_ratio=self.TEST_RR,
            max_duration_ratio=2.0,
            trading_mode="margin",
            pnl_base_std=self.TEST_PNL_STD,
            pnl_duration_vol_scale=self.TEST_PNL_DUR_VOL_SCALE,
        )
        with tempfile.TemporaryDirectory() as tmp_dir:
            output_path = Path(tmp_dir)
            write_complete_statistical_analysis(
                test_data,
                output_path,
                profit_target=self.TEST_PROFIT_TARGET,
                seed=self.SEED,
                real_df=None,
            )
            main_report = output_path / "statistical_analysis.md"
            self.assertTrue(main_report.exists(), "Main statistical analysis should be created")
            feature_file = output_path / "feature_importance.csv"
            self.assertTrue(feature_file.exists(), "Feature importance should be created")


class TestPrivateFunctions(RewardSpaceTestBase):
    """Test private functions through public API calls."""

    def test_idle_penalty_via_rewards(self):
        """Test idle penalty calculation via reward calculation."""
        context = self.make_ctx(
            pnl=0.0,
            trade_duration=0,
            idle_duration=20,
            max_unrealized_profit=0.0,
            min_unrealized_profit=0.0,
            position=Positions.Neutral,
            action=Actions.Neutral,
        )
        breakdown = calculate_reward(
            context,
            self.DEFAULT_PARAMS,
            base_factor=self.TEST_BASE_FACTOR,
            profit_target=self.TEST_PROFIT_TARGET,
            risk_reward_ratio=1.0,
            short_allowed=True,
            action_masking=True,
        )
        self.assertLess(breakdown.idle_penalty, 0, "Idle penalty should be negative")
        self.assertAlmostEqualFloat(
            breakdown.total,
            breakdown.idle_penalty
            + breakdown.reward_shaping
            + breakdown.entry_additive
            + breakdown.exit_additive,
            tolerance=self.TOL_IDENTITY_RELAXED,
            msg="Total should equal sum of components (idle + shaping/additives)",
        )

    def test_hold_penalty_via_rewards(self):
        """Test hold penalty calculation via reward calculation."""
        context = self.make_ctx(
            pnl=0.01,
            trade_duration=150,
            idle_duration=0,
            max_unrealized_profit=0.02,
            min_unrealized_profit=0.0,
            position=Positions.Long,
            action=Actions.Neutral,
        )
        breakdown = calculate_reward(
            context,
            self.DEFAULT_PARAMS,
            base_factor=self.TEST_BASE_FACTOR,
            profit_target=self.TEST_PROFIT_TARGET,
            risk_reward_ratio=self.TEST_RR,
            short_allowed=True,
            action_masking=True,
        )
        self.assertLess(breakdown.hold_penalty, 0, "Hold penalty should be negative")
        self.assertAlmostEqualFloat(
            breakdown.total,
            breakdown.hold_penalty
            + breakdown.reward_shaping
            + breakdown.entry_additive
            + breakdown.exit_additive,
            tolerance=self.TOL_IDENTITY_RELAXED,
            msg="Total should equal sum of components (hold + shaping/additives)",
        )

    def test_exit_reward_calculation(self):
        """Test exit reward calculation with various scenarios."""
        scenarios = [
            (Positions.Long, Actions.Long_exit, 0.05, "Profitable long exit"),
            (Positions.Short, Actions.Short_exit, -0.03, "Profitable short exit"),
            (Positions.Long, Actions.Long_exit, -0.02, "Losing long exit"),
            (Positions.Short, Actions.Short_exit, 0.02, "Losing short exit"),
        ]
        for position, action, pnl, description in scenarios:
            with self.subTest(description=description):
                context = self.make_ctx(
                    pnl=pnl,
                    trade_duration=50,
                    idle_duration=0,
                    max_unrealized_profit=max(pnl + 0.01, 0.01),
                    min_unrealized_profit=min(pnl - 0.01, -0.01),
                    position=position,
                    action=action,
                )
                breakdown = calculate_reward(
                    context,
                    self.DEFAULT_PARAMS,
                    base_factor=self.TEST_BASE_FACTOR,
                    profit_target=self.TEST_PROFIT_TARGET,
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
                    np.isfinite(breakdown.total), f"Total should be finite for {description}"
                )

    def test_invalid_action_handling(self):
        """Test invalid action penalty."""
        context = self.make_ctx(
            pnl=0.02,
            trade_duration=50,
            idle_duration=0,
            max_unrealized_profit=0.03,
            min_unrealized_profit=0.01,
            position=Positions.Short,
            action=Actions.Long_exit,
        )
        breakdown = calculate_reward(
            context,
            self.DEFAULT_PARAMS,
            base_factor=self.TEST_BASE_FACTOR,
            profit_target=self.TEST_PROFIT_TARGET,
            risk_reward_ratio=1.0,
            short_allowed=True,
            action_masking=False,
        )
        self.assertLess(breakdown.invalid_penalty, 0, "Invalid action should have negative penalty")
        self.assertAlmostEqualFloat(
            breakdown.total,
            breakdown.invalid_penalty
            + breakdown.reward_shaping
            + breakdown.entry_additive
            + breakdown.exit_additive,
            tolerance=self.TOL_IDENTITY_RELAXED,
            msg="Total should equal invalid penalty plus shaping/additives",
        )

    def test_hold_penalty_zero_before_max_duration(self):
        """Test hold penalty logic: zero penalty before max_trade_duration."""
        max_duration = 128
        test_cases = [
            (64, "before max_duration"),
            (127, "just before max_duration"),
            (128, "exactly at max_duration"),
            (129, "just after max_duration"),
            (192, "well after max_duration"),
        ]
        for trade_duration, description in test_cases:
            with self.subTest(duration=trade_duration, desc=description):
                context = self.make_ctx(
                    pnl=0.0,
                    trade_duration=trade_duration,
                    idle_duration=0,
                    max_unrealized_profit=0.0,
                    min_unrealized_profit=0.0,
                    position=Positions.Long,
                    action=Actions.Neutral,
                )
                breakdown = calculate_reward(
                    context,
                    self.DEFAULT_PARAMS,
                    base_factor=self.TEST_BASE_FACTOR,
                    profit_target=self.TEST_PROFIT_TARGET,
                    risk_reward_ratio=1.0,
                    short_allowed=True,
                    action_masking=True,
                )
                duration_ratio = trade_duration / max_duration
                if duration_ratio < 1.0:
                    self.assertEqual(
                        breakdown.hold_penalty,
                        0.0,
                        f"Hold penalty should be 0.0 {description} (ratio={duration_ratio:.2f})",
                    )
                elif duration_ratio == 1.0:
                    self.assertEqual(
                        breakdown.hold_penalty,
                        0.0,
                        f"Hold penalty should be 0.0 {description} (ratio={duration_ratio:.2f})",
                    )
                else:
                    self.assertLess(
                        breakdown.hold_penalty,
                        0.0,
                        f"Hold penalty should be negative {description} (ratio={duration_ratio:.2f})",
                    )
                self.assertAlmostEqualFloat(
                    breakdown.total,
                    breakdown.hold_penalty
                    + breakdown.reward_shaping
                    + breakdown.entry_additive
                    + breakdown.exit_additive,
                    tolerance=self.TOL_IDENTITY_RELAXED,
                    msg=f"Total mismatch including shaping {description}",
                )

    def test_hold_penalty_progressive_scaling(self):
        """Test that hold penalty scales progressively after max_duration."""
        params = self.base_params(max_trade_duration_candles=100)
        durations = [150, 200, 300]
        penalties: list[float] = []
        for duration in durations:
            context = self.make_ctx(
                pnl=0.0,
                trade_duration=duration,
                idle_duration=0,
                max_unrealized_profit=0.0,
                min_unrealized_profit=0.0,
                position=Positions.Long,
                action=Actions.Neutral,
            )
            breakdown = calculate_reward(
                context,
                params,
                base_factor=self.TEST_BASE_FACTOR,
                profit_target=self.TEST_PROFIT_TARGET,
                risk_reward_ratio=self.TEST_RR,
                short_allowed=True,
                action_masking=True,
            )
            penalties.append(breakdown.hold_penalty)
        for i in range(1, len(penalties)):
            self.assertLessEqual(
                penalties[i],
                penalties[i - 1],
                f"Penalty should increase with duration: {penalties[i]} > {penalties[i - 1]}",
            )

    def test_new_invariant_and_warn_parameters(self):
        """Ensure new tunables (check_invariants, exit_factor_threshold) exist and behave.

        Uses a very large base_factor to trigger potential warning condition without capping.
        """
        params = self.base_params()
        self.assertIn("check_invariants", params)
        self.assertIn("exit_factor_threshold", params)
        context = self.make_ctx(
            pnl=0.05,
            trade_duration=300,
            idle_duration=0,
            max_unrealized_profit=0.06,
            min_unrealized_profit=0.0,
            position=Positions.Long,
            action=Actions.Long_exit,
        )
        breakdown = calculate_reward(
            context,
            params,
            base_factor=10000000.0,
            profit_target=self.TEST_PROFIT_TARGET,
            risk_reward_ratio=self.TEST_RR,
            short_allowed=True,
            action_masking=True,
        )
        self.assertFinite(breakdown.exit_component, name="exit_component")


if __name__ == "__main__":
    unittest.main()
