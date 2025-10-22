#!/usr/bin/env python3
"""Tests for reward calculation components and algorithms."""

import dataclasses
import math
import unittest

from reward_space_analysis import (
    Actions,
    Positions,
    RewardContext,
    _compute_hold_potential,
    _get_exit_factor,
    _get_float_param,
    _get_pnl_factor,
    calculate_reward,
)

from .test_base import RewardSpaceTestBase


class TestRewardComponents(RewardSpaceTestBase):
    def test_hold_potential_computation_finite(self):
        """Test hold potential computation returns finite values."""
        params = {
            "hold_potential_enabled": True,
            "hold_potential_scale": 1.0,
            "hold_potential_gain": 1.0,
            "hold_potential_transform_pnl": "tanh",
            "hold_potential_transform_duration": "tanh",
        }
        val = _compute_hold_potential(0.5, 0.3, params)
        self.assertFinite(val, name="hold_potential")

    def test_hold_penalty_comprehensive(self):
        """Comprehensive hold penalty test: calculation, thresholds, and progressive scaling."""
        # Test 1: Basic hold penalty calculation via reward calculation (trade_duration > max_duration)
        context = self.make_ctx(
            pnl=0.01,
            trade_duration=150,  # > default max_duration (128)
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

        # Test 2: Zero penalty before max_duration threshold
        max_duration = 128
        test_cases = [
            (64, "before max_duration"),
            (127, "just before max_duration"),
            (128, "exactly at max_duration"),
            (129, "just after max_duration"),
        ]
        for trade_duration, description in test_cases:
            with self.subTest(duration=trade_duration, desc=description):
                context = self.make_ctx(
                    pnl=0.0,
                    trade_duration=trade_duration,
                    idle_duration=0,
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
                    # At exact max duration, penalty can be 0.0 or slightly negative (implementation dependent)
                    self.assertLessEqual(
                        breakdown.hold_penalty,
                        0.0,
                        f"Hold penalty should be <= 0.0 {description} (ratio={duration_ratio:.2f})",
                    )
                else:
                    # Beyond max duration, penalty should be strictly negative
                    self.assertLess(
                        breakdown.hold_penalty,
                        0.0,
                        f"Hold penalty should be negative {description} (ratio={duration_ratio:.2f})",
                    )

        # Test 3: Progressive scaling after max_duration
        params = self.base_params(max_trade_duration_candles=100)
        durations = [150, 200, 300]
        penalties: list[float] = []
        for duration in durations:
            context = self.make_ctx(
                pnl=0.0,
                trade_duration=duration,
                idle_duration=0,
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
                f"Penalty should increase (more negative) with duration: {penalties[i]} <= {penalties[i - 1]}",
            )

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

    """Core reward component tests."""

    def test_reward_calculation_component_activation(self):
        """Test reward component activation: idle_penalty and exit_component trigger correctly."""
        test_cases = [
            (Positions.Neutral, Actions.Neutral, "idle_penalty"),
            (Positions.Long, Actions.Long_exit, "exit_component"),
            (Positions.Short, Actions.Short_exit, "exit_component"),
        ]
        for position, action, expected_type in test_cases:
            with self.subTest(position=position, action=action):
                context = self.make_ctx(
                    pnl=0.02 if expected_type == "exit_component" else 0.0,
                    trade_duration=50 if position != Positions.Neutral else 0,
                    idle_duration=10 if position == Positions.Neutral else 0,
                    max_unrealized_profit=0.03,
                    min_unrealized_profit=-0.01,
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
                if expected_type == "idle_penalty":
                    self.assertNotEqual(breakdown.idle_penalty, 0.0)
                elif expected_type == "exit_component":
                    self.assertNotEqual(breakdown.exit_component, 0.0)
                self.assertFinite(breakdown.total, name="breakdown.total")

    def test_efficiency_zero_policy(self):
        """Test efficiency zero policy."""
        ctx = self.make_ctx(
            pnl=0.0,
            trade_duration=1,
            max_unrealized_profit=0.0,
            min_unrealized_profit=-0.02,
            position=Positions.Long,
            action=Actions.Long_exit,
        )
        params = self.base_params()
        profit_target = self.TEST_PROFIT_TARGET * self.TEST_RR
        pnl_factor = _get_pnl_factor(params, ctx, profit_target, self.TEST_RR)
        self.assertFinite(pnl_factor, name="pnl_factor")
        self.assertAlmostEqualFloat(pnl_factor, 1.0, tolerance=self.TOL_GENERIC_EQ)

    def test_max_idle_duration_candles_logic(self):
        """Test max idle duration candles logic."""
        params_small = self.base_params(max_idle_duration_candles=50)
        params_large = self.base_params(max_idle_duration_candles=200)
        base_factor = self.TEST_BASE_FACTOR
        context = self.make_ctx(
            pnl=0.0,
            trade_duration=0,
            idle_duration=40,
            position=Positions.Neutral,
            action=Actions.Neutral,
        )
        small = calculate_reward(
            context,
            params_small,
            base_factor,
            profit_target=self.TEST_PROFIT_TARGET,
            risk_reward_ratio=self.TEST_RR,
            short_allowed=True,
            action_masking=True,
        )
        large = calculate_reward(
            context,
            params_large,
            base_factor=self.TEST_BASE_FACTOR,
            profit_target=0.06,
            risk_reward_ratio=self.TEST_RR,
            short_allowed=True,
            action_masking=True,
        )
        self.assertLess(small.idle_penalty, 0.0)
        self.assertLess(large.idle_penalty, 0.0)
        self.assertGreater(large.idle_penalty, small.idle_penalty)

    def test_exit_factor_calculation(self):
        """Exit factor calculation across core modes + plateau variant (plateau via exit_plateau=True)."""
        modes_to_test = ["linear", "power"]
        for mode in modes_to_test:
            test_params = self.base_params(exit_attenuation_mode=mode)
            factor = _get_exit_factor(
                base_factor=1.0, pnl=0.02, pnl_factor=1.5, duration_ratio=0.3, params=test_params
            )
            self.assertFinite(factor, name=f"exit_factor[{mode}]")
            self.assertGreater(factor, 0, f"Exit factor for {mode} should be positive")
        plateau_params = self.base_params(
            exit_attenuation_mode="linear",
            exit_plateau=True,
            exit_plateau_grace=0.5,
            exit_linear_slope=1.0,
        )
        plateau_factor_pre = _get_exit_factor(
            base_factor=1.0, pnl=0.02, pnl_factor=1.5, duration_ratio=0.4, params=plateau_params
        )
        plateau_factor_post = _get_exit_factor(
            base_factor=1.0, pnl=0.02, pnl_factor=1.5, duration_ratio=0.8, params=plateau_params
        )
        self.assertGreater(plateau_factor_pre, 0)
        self.assertGreater(plateau_factor_post, 0)
        self.assertGreaterEqual(
            plateau_factor_pre,
            plateau_factor_post - self.TOL_IDENTITY_STRICT,
            "Plateau pre-grace factor should be >= post-grace factor",
        )

    def test_idle_penalty_zero_when_profit_target_zero(self):
        """If profit_target=0 → idle_factor=0 → idle penalty must be exactly 0 for neutral idle state."""
        context = self.make_ctx(
            pnl=0.0,
            trade_duration=0,
            idle_duration=30,
            position=Positions.Neutral,
            action=Actions.Neutral,
        )
        br = calculate_reward(
            context,
            self.DEFAULT_PARAMS,
            base_factor=self.TEST_BASE_FACTOR,
            profit_target=0.0,
            risk_reward_ratio=self.TEST_RR,
            short_allowed=True,
            action_masking=True,
        )
        self.assertEqual(br.idle_penalty, 0.0, "Idle penalty should be zero when profit_target=0")
        self.assertEqual(br.total, 0.0, "Total reward should be zero in this configuration")

    def test_win_reward_factor_saturation(self):
        """Saturation test: pnl amplification factor should monotonically approach (1 + win_reward_factor)."""
        win_reward_factor = 3.0
        beta = 0.5
        profit_target = self.TEST_PROFIT_TARGET
        params = self.base_params(
            win_reward_factor=win_reward_factor,
            pnl_factor_beta=beta,
            efficiency_weight=0.0,
            exit_attenuation_mode="linear",
            exit_plateau=False,
            exit_linear_slope=0.0,
        )
        params.pop("base_factor", None)
        pnl_values = [profit_target * m for m in (1.05, self.TEST_RR_HIGH, 5.0, 10.0)]
        ratios_observed: list[float] = []
        for pnl in pnl_values:
            context = self.make_ctx(
                pnl=pnl,
                trade_duration=0,
                idle_duration=0,
                max_unrealized_profit=pnl,
                min_unrealized_profit=0.0,
                position=Positions.Long,
                action=Actions.Long_exit,
            )
            br = calculate_reward(
                context,
                params,
                base_factor=1.0,
                profit_target=profit_target,
                risk_reward_ratio=1.0,
                short_allowed=True,
                action_masking=True,
            )
            ratio = br.exit_component / pnl if pnl != 0 else 0.0
            ratios_observed.append(float(ratio))
        self.assertMonotonic(
            ratios_observed,
            non_decreasing=True,
            tolerance=self.TOL_IDENTITY_STRICT,
            name="pnl_amplification_ratio",
        )
        asymptote = 1.0 + win_reward_factor
        final_ratio = ratios_observed[-1]
        self.assertFinite(final_ratio, name="final_ratio")
        self.assertLess(
            abs(final_ratio - asymptote),
            0.001,
            f"Final amplification {final_ratio:.6f} not close to asymptote {asymptote:.6f}",
        )
        expected_ratios: list[float] = []
        for pnl in pnl_values:
            pnl_ratio = pnl / profit_target
            expected = 1.0 + win_reward_factor * math.tanh(beta * (pnl_ratio - 1.0))
            expected_ratios.append(expected)
        for obs, exp in zip(ratios_observed, expected_ratios):
            self.assertFinite(obs, name="observed_ratio")
            self.assertFinite(exp, name="expected_ratio")
            self.assertLess(
                abs(obs - exp),
                5e-06,
                f"Observed amplification {obs:.8f} deviates from expected {exp:.8f}",
            )

    def test_scale_invariance_and_decomposition(self):
        """Components scale ~ linearly with base_factor; total equals sum(core + shaping + additives)."""
        params = self.base_params()
        params.pop("base_factor", None)
        base_factor = 80.0
        k = 7.5
        profit_target = self.TEST_PROFIT_TARGET
        rr = 1.5
        contexts: list[RewardContext] = [
            self.make_ctx(
                pnl=0.025,
                trade_duration=40,
                idle_duration=0,
                max_unrealized_profit=0.03,
                min_unrealized_profit=0.0,
                position=Positions.Long,
                action=Actions.Long_exit,
            ),
            self.make_ctx(
                pnl=-self.TEST_PNL_STD,
                trade_duration=60,
                idle_duration=0,
                max_unrealized_profit=0.01,
                min_unrealized_profit=-0.04,
                position=Positions.Long,
                action=Actions.Long_exit,
            ),
            self.make_ctx(
                pnl=0.0,
                trade_duration=0,
                idle_duration=35,
                max_unrealized_profit=0.0,
                min_unrealized_profit=0.0,
                position=Positions.Neutral,
                action=Actions.Neutral,
            ),
            self.make_ctx(
                pnl=0.0,
                trade_duration=80,
                idle_duration=0,
                max_unrealized_profit=0.04,
                min_unrealized_profit=-0.01,
                position=Positions.Long,
                action=Actions.Neutral,
            ),
        ]
        tol_scale = self.TOL_RELATIVE
        for ctx in contexts:
            br1 = calculate_reward(
                ctx,
                params,
                base_factor=base_factor,
                profit_target=profit_target,
                risk_reward_ratio=rr,
                short_allowed=True,
                action_masking=True,
            )
            br2 = calculate_reward(
                ctx,
                params,
                base_factor=base_factor * k,
                profit_target=profit_target,
                risk_reward_ratio=rr,
                short_allowed=True,
                action_masking=True,
            )
            for br in (br1, br2):
                comp_sum = (
                    br.exit_component
                    + br.idle_penalty
                    + br.hold_penalty
                    + br.invalid_penalty
                    + br.reward_shaping
                    + br.entry_additive
                    + br.exit_additive
                )
                self.assertAlmostEqual(
                    br.total,
                    comp_sum,
                    places=12,
                    msg=f"Decomposition mismatch (ctx={ctx}, total={br.total}, sum={comp_sum})",
                )
            components1 = {
                "exit_component": br1.exit_component,
                "idle_penalty": br1.idle_penalty,
                "hold_penalty": br1.hold_penalty,
                "invalid_penalty": br1.invalid_penalty,
                "total": br1.exit_component
                + br1.idle_penalty
                + br1.hold_penalty
                + br1.invalid_penalty,
            }
            components2 = {
                "exit_component": br2.exit_component,
                "idle_penalty": br2.idle_penalty,
                "hold_penalty": br2.hold_penalty,
                "invalid_penalty": br2.invalid_penalty,
                "total": br2.exit_component
                + br2.idle_penalty
                + br2.hold_penalty
                + br2.invalid_penalty,
            }
            for key, v1 in components1.items():
                v2 = components2[key]
                if abs(v1) < 1e-15 and abs(v2) < 1e-15:
                    continue
                self.assertLess(
                    abs(v2 - k * v1),
                    tol_scale * max(1.0, abs(k * v1)),
                    f"Scale invariance failed for {key}: v1={v1}, v2={v2}, k={k}",
                )

    def test_long_short_symmetry(self):
        """Long vs Short exit reward magnitudes should match in absolute value for identical PnL (no directional bias)."""
        params = self.base_params()
        params.pop("base_factor", None)
        base_factor = 120.0
        profit_target = 0.04
        rr = self.TEST_RR_HIGH
        pnls = [0.018, -0.022]
        for pnl in pnls:
            ctx_long = self.make_ctx(
                pnl=pnl,
                trade_duration=55,
                idle_duration=0,
                max_unrealized_profit=pnl if pnl > 0 else 0.01,
                min_unrealized_profit=pnl if pnl < 0 else -0.01,
                position=Positions.Long,
                action=Actions.Long_exit,
            )
            ctx_short = self.make_ctx(
                pnl=pnl,
                trade_duration=55,
                idle_duration=0,
                max_unrealized_profit=pnl if pnl > 0 else 0.01,
                min_unrealized_profit=pnl if pnl < 0 else -0.01,
                position=Positions.Short,
                action=Actions.Short_exit,
            )
            br_long = calculate_reward(
                ctx_long,
                params,
                base_factor=base_factor,
                profit_target=profit_target,
                risk_reward_ratio=rr,
                short_allowed=True,
                action_masking=True,
            )
            br_short = calculate_reward(
                ctx_short,
                params,
                base_factor=base_factor,
                profit_target=profit_target,
                risk_reward_ratio=rr,
                short_allowed=True,
                action_masking=True,
            )
            if pnl > 0:
                self.assertGreater(br_long.exit_component, 0)
                self.assertGreater(br_short.exit_component, 0)
            else:
                self.assertLess(br_long.exit_component, 0)
                self.assertLess(br_short.exit_component, 0)
            self.assertLess(
                abs(abs(br_long.exit_component) - abs(br_short.exit_component)),
                self.TOL_RELATIVE * max(1.0, abs(br_long.exit_component)),
                f"Long/Short asymmetry pnl={pnl}: long={br_long.exit_component}, short={br_short.exit_component}",
            )

    def test_idle_penalty_fallback_and_proportionality(self):
        """Idle penalty fallback denominator & proportional scaling."""
        params = self.base_params(max_idle_duration_candles=None, max_trade_duration_candles=100)
        base_factor = 90.0
        profit_target = self.TEST_PROFIT_TARGET
        risk_reward_ratio = 1.0
        ctx_a = self.make_ctx(
            pnl=0.0,
            trade_duration=0,
            idle_duration=20,
            position=Positions.Neutral,
            action=Actions.Neutral,
        )
        ctx_b = dataclasses.replace(ctx_a, idle_duration=40)
        br_a = calculate_reward(
            ctx_a,
            params,
            base_factor=base_factor,
            profit_target=profit_target,
            risk_reward_ratio=risk_reward_ratio,
            short_allowed=True,
            action_masking=True,
        )
        br_b = calculate_reward(
            ctx_b,
            params,
            base_factor=base_factor,
            profit_target=profit_target,
            risk_reward_ratio=risk_reward_ratio,
            short_allowed=True,
            action_masking=True,
        )
        self.assertLess(br_a.idle_penalty, 0.0)
        self.assertLess(br_b.idle_penalty, 0.0)
        ratio = br_b.idle_penalty / br_a.idle_penalty if br_a.idle_penalty != 0 else None
        self.assertIsNotNone(ratio)
        if ratio is not None:
            self.assertAlmostEqualFloat(abs(ratio), 2.0, tolerance=0.2)
        ctx_mid = dataclasses.replace(ctx_a, idle_duration=120)
        br_mid = calculate_reward(
            ctx_mid,
            params,
            base_factor=base_factor,
            profit_target=profit_target,
            risk_reward_ratio=risk_reward_ratio,
            short_allowed=True,
            action_masking=True,
        )
        self.assertLess(br_mid.idle_penalty, 0.0)
        idle_penalty_scale = _get_float_param(params, "idle_penalty_scale", 0.5)
        idle_penalty_power = _get_float_param(params, "idle_penalty_power", 1.025)
        factor = _get_float_param(params, "base_factor", float(base_factor))
        idle_factor = factor * (profit_target * risk_reward_ratio) / 4.0
        observed_ratio = abs(br_mid.idle_penalty) / (idle_factor * idle_penalty_scale)
        if observed_ratio > 0:
            implied_D = 120 / observed_ratio ** (1 / idle_penalty_power)
            self.assertAlmostEqualFloat(implied_D, 400.0, tolerance=20.0)


if __name__ == "__main__":
    unittest.main()
