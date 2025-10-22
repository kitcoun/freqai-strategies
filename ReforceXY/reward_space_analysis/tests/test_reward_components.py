#!/usr/bin/env python3
"""Tests for reward calculation components and algorithms."""

import math
import unittest

from reward_space_analysis import (
    Actions,
    Positions,
    RewardContext,
    _get_exit_factor,
    _get_pnl_factor,
    calculate_reward,
)

from .test_base import RewardSpaceTestBase


class TestRewardComponents(RewardSpaceTestBase):
    """Core reward component tests."""

    def test_reward_calculation_scenarios_basic(self):
        """Reward calculation scenarios: expected components become non-zero."""
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


if __name__ == "__main__":
    unittest.main()
