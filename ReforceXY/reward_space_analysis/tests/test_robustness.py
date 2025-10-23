#!/usr/bin/env python3
"""Robustness tests and boundary condition validation."""

import math
import unittest
import warnings

import numpy as np

from reward_space_analysis import (
    ATTENUATION_MODES,
    ATTENUATION_MODES_WITH_LEGACY,
    Actions,
    Positions,
    RewardContext,
    _get_exit_factor,
    _get_pnl_factor,
    calculate_reward,
    simulate_samples,
)

from .test_base import RewardSpaceTestBase


class TestRewardRobustnessAndBoundaries(RewardSpaceTestBase):
    """Robustness & boundary assertions: invariants, attenuation maths, parameter edges, scaling, warnings."""

    def test_decomposition_integrity(self):
        """reward must equal the single active core component under mutually exclusive scenarios (idle/hold/exit/invalid)."""
        scenarios = [
            dict(
                ctx=self.make_ctx(
                    pnl=0.0,
                    trade_duration=0,
                    idle_duration=25,
                    max_unrealized_profit=0.0,
                    min_unrealized_profit=0.0,
                    position=Positions.Neutral,
                    action=Actions.Neutral,
                ),
                active="idle_penalty",
            ),
            dict(
                ctx=self.make_ctx(
                    pnl=0.0,
                    trade_duration=150,
                    idle_duration=0,
                    max_unrealized_profit=0.0,
                    min_unrealized_profit=0.0,
                    position=Positions.Long,
                    action=Actions.Neutral,
                ),
                active="hold_penalty",
            ),
            dict(
                ctx=self.make_ctx(
                    pnl=self.TEST_PROFIT_TARGET,
                    trade_duration=60,
                    idle_duration=0,
                    max_unrealized_profit=0.05,
                    min_unrealized_profit=0.01,
                    position=Positions.Long,
                    action=Actions.Long_exit,
                ),
                active="exit_component",
            ),
            dict(
                ctx=self.make_ctx(
                    pnl=0.01,
                    trade_duration=10,
                    idle_duration=0,
                    max_unrealized_profit=0.02,
                    min_unrealized_profit=0.0,
                    position=Positions.Short,
                    action=Actions.Long_exit,
                ),
                active="invalid_penalty",
            ),
        ]
        for sc in scenarios:
            ctx_obj: RewardContext = sc["ctx"]
            active_label: str = sc["active"]
            with self.subTest(active=active_label):
                params = self.base_params(
                    entry_additive_enabled=False,
                    exit_additive_enabled=False,
                    hold_potential_enabled=False,
                    potential_gamma=0.0,
                    check_invariants=False,
                )
                br = calculate_reward(
                    ctx_obj,
                    params,
                    base_factor=self.TEST_BASE_FACTOR,
                    profit_target=self.TEST_PROFIT_TARGET,
                    risk_reward_ratio=self.TEST_RR,
                    short_allowed=True,
                    action_masking=True,
                )
                core_components = {
                    "exit_component": br.exit_component,
                    "idle_penalty": br.idle_penalty,
                    "hold_penalty": br.hold_penalty,
                    "invalid_penalty": br.invalid_penalty,
                }
                for name, value in core_components.items():
                    if name == active_label:
                        self.assertAlmostEqualFloat(
                            value,
                            br.total,
                            tolerance=self.TOL_IDENTITY_RELAXED,
                            msg=f"Active component {name} != total",
                        )
                    else:
                        self.assertNearZero(
                            value,
                            atol=self.TOL_IDENTITY_RELAXED,
                            msg=f"Inactive component {name} not near zero (val={value})",
                        )
                self.assertAlmostEqualFloat(
                    br.reward_shaping, 0.0, tolerance=self.TOL_IDENTITY_RELAXED
                )
                self.assertAlmostEqualFloat(
                    br.entry_additive, 0.0, tolerance=self.TOL_IDENTITY_RELAXED
                )
                self.assertAlmostEqualFloat(
                    br.exit_additive, 0.0, tolerance=self.TOL_IDENTITY_RELAXED
                )

    def test_pnl_invariant_exit_only(self):
        """Invariant: only exit actions have non-zero PnL (robustness category)."""
        df = simulate_samples(
            params=self.base_params(max_trade_duration_candles=50),
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
        total_pnl = df["pnl"].sum()
        exit_mask = df["reward_exit"] != 0
        exit_pnl_sum = df.loc[exit_mask, "pnl"].sum()
        self.assertAlmostEqual(
            total_pnl,
            exit_pnl_sum,
            places=10,
            msg="PnL invariant violation: total PnL != sum of exit PnL",
        )
        non_zero_pnl_actions = set(df[df["pnl"].abs() > self.EPS_BASE]["action"].unique())
        expected_exit_actions = {2.0, 4.0}
        self.assertTrue(
            non_zero_pnl_actions.issubset(expected_exit_actions),
            f"Non-exit actions have PnL: {non_zero_pnl_actions - expected_exit_actions}",
        )
        invalid_combinations = df[(df["pnl"].abs() <= self.EPS_BASE) & (df["reward_exit"] != 0)]
        self.assertEqual(len(invalid_combinations), 0)

    def test_exit_factor_comprehensive(self):
        """Comprehensive exit factor test: mathematical correctness and monotonic attenuation."""
        # Part 1: Mathematical formulas validation
        context = self.make_ctx(
            pnl=0.05,
            trade_duration=50,
            idle_duration=0,
            max_unrealized_profit=0.05,
            min_unrealized_profit=0.01,
            position=Positions.Long,
            action=Actions.Long_exit,
        )
        params = self.DEFAULT_PARAMS.copy()
        duration_ratio = 50 / 100

        # Test power mode
        params["exit_attenuation_mode"] = "power"
        params["exit_power_tau"] = 0.5
        params["exit_plateau"] = False
        reward_power = calculate_reward(
            context,
            params,
            self.TEST_BASE_FACTOR,
            self.TEST_PROFIT_TARGET,
            self.TEST_RR,
            short_allowed=True,
            action_masking=True,
        )
        self.assertGreater(reward_power.exit_component, 0)

        # Test half_life mode with mathematical validation
        params["exit_attenuation_mode"] = "half_life"
        params["exit_half_life"] = 0.5
        reward_half_life = calculate_reward(
            context,
            params,
            self.TEST_BASE_FACTOR,
            self.TEST_PROFIT_TARGET,
            self.TEST_RR,
            short_allowed=True,
            action_masking=True,
        )
        pnl_factor_hl = _get_pnl_factor(params, context, self.TEST_PROFIT_TARGET, self.TEST_RR)
        observed_exit_factor = _get_exit_factor(
            self.TEST_BASE_FACTOR, context.pnl, pnl_factor_hl, duration_ratio, params
        )
        observed_half_life_factor = observed_exit_factor / (
            self.TEST_BASE_FACTOR * max(pnl_factor_hl, self.EPS_BASE)
        )
        expected_half_life_factor = 2 ** (-duration_ratio / params["exit_half_life"])
        self.assertAlmostEqualFloat(
            observed_half_life_factor,
            expected_half_life_factor,
            tolerance=self.TOL_IDENTITY_RELAXED,
            msg="Half-life attenuation mismatch: observed vs expected",
        )

        # Test linear mode
        params["exit_attenuation_mode"] = "linear"
        params["exit_linear_slope"] = 1.0
        reward_linear = calculate_reward(
            context,
            params,
            self.TEST_BASE_FACTOR,
            self.TEST_PROFIT_TARGET,
            self.TEST_RR,
            short_allowed=True,
            action_masking=True,
        )
        rewards = [
            reward_power.exit_component,
            reward_half_life.exit_component,
            reward_linear.exit_component,
        ]
        self.assertTrue(all((r > 0 for r in rewards)))
        unique_rewards = set((f"{r:.6f}" for r in rewards))
        self.assertGreater(len(unique_rewards), 1)

        # Part 2: Monotonic attenuation validation
        modes = list(ATTENUATION_MODES) + ["plateau_linear"]
        base_factor = self.TEST_BASE_FACTOR
        pnl = 0.05
        pnl_factor = 1.0
        for mode in modes:
            with self.subTest(mode=mode):
                if mode == "plateau_linear":
                    mode_params = self.base_params(
                        exit_attenuation_mode="linear",
                        exit_plateau=True,
                        exit_plateau_grace=0.2,
                        exit_linear_slope=1.0,
                    )
                elif mode == "linear":
                    mode_params = self.base_params(
                        exit_attenuation_mode="linear", exit_linear_slope=1.2
                    )
                elif mode == "power":
                    mode_params = self.base_params(
                        exit_attenuation_mode="power", exit_power_tau=0.5
                    )
                elif mode == "half_life":
                    mode_params = self.base_params(
                        exit_attenuation_mode="half_life", exit_half_life=0.7
                    )
                else:
                    mode_params = self.base_params(exit_attenuation_mode="sqrt")

                ratios = np.linspace(0, 2, 15)
                values = [
                    _get_exit_factor(base_factor, pnl, pnl_factor, r, mode_params) for r in ratios
                ]

                if mode == "plateau_linear":
                    grace = float(mode_params["exit_plateau_grace"])
                    filtered = [
                        (r, v)
                        for r, v in zip(ratios, values)
                        if r >= grace - self.TOL_IDENTITY_RELAXED
                    ]
                    values_to_check = [v for _, v in filtered]
                else:
                    values_to_check = values

                for earlier, later in zip(values_to_check, values_to_check[1:]):
                    self.assertLessEqual(
                        later,
                        earlier + self.TOL_IDENTITY_RELAXED,
                        f"Non-monotonic attenuation in mode={mode}",
                    )

    def test_exit_factor_threshold_warning_and_non_capping(self):
        """Warning emission without capping when exit_factor_threshold exceeded."""
        params = self.base_params(exit_factor_threshold=10.0)
        params.pop("base_factor", None)
        context = self.make_ctx(
            pnl=0.08,
            trade_duration=10,
            idle_duration=0,
            max_unrealized_profit=0.09,
            min_unrealized_profit=0.0,
            position=Positions.Long,
            action=Actions.Long_exit,
        )
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            baseline = calculate_reward(
                context,
                params,
                base_factor=self.TEST_BASE_FACTOR,
                profit_target=self.TEST_PROFIT_TARGET,
                risk_reward_ratio=self.TEST_RR_HIGH,
                short_allowed=True,
                action_masking=True,
            )
            amplified_base_factor = self.TEST_BASE_FACTOR * 200.0
            amplified = calculate_reward(
                context,
                params,
                base_factor=amplified_base_factor,
                profit_target=self.TEST_PROFIT_TARGET,
                risk_reward_ratio=self.TEST_RR_HIGH,
                short_allowed=True,
                action_masking=True,
            )
        self.assertGreater(baseline.exit_component, 0.0)
        self.assertGreater(amplified.exit_component, baseline.exit_component)
        scale = amplified.exit_component / baseline.exit_component
        self.assertGreater(scale, 10.0)
        runtime_warnings = [w for w in caught if issubclass(w.category, RuntimeWarning)]
        self.assertTrue(runtime_warnings)
        self.assertTrue(
            any(
                (
                    "exceeded threshold" in str(w.message)
                    or "exceeds threshold" in str(w.message)
                    or "|factor|=" in str(w.message)
                    for w in runtime_warnings
                )
            )
        )

    def test_negative_slope_sanitization(self):
        """Negative exit_linear_slope is sanitized to 0.0; resulting exit factors must match slope=0.0 within tolerance."""
        base_factor = 100.0
        pnl = 0.03
        pnl_factor = 1.0
        duration_ratios = [0.0, 0.2, 0.5, 1.0, 1.5]
        params_bad = self.base_params(
            exit_attenuation_mode="linear", exit_linear_slope=-5.0, exit_plateau=False
        )
        params_ref = self.base_params(
            exit_attenuation_mode="linear", exit_linear_slope=0.0, exit_plateau=False
        )
        for dr in duration_ratios:
            f_bad = _get_exit_factor(base_factor, pnl, pnl_factor, dr, params_bad)
            f_ref = _get_exit_factor(base_factor, pnl, pnl_factor, dr, params_ref)
            self.assertAlmostEqualFloat(
                f_bad,
                f_ref,
                tolerance=self.TOL_IDENTITY_RELAXED,
                msg=f"Sanitized slope mismatch at dr={dr} f_bad={f_bad} f_ref={f_ref}",
            )

    def test_power_mode_alpha_formula(self):
        """Power mode attenuation: ratio f(dr=1)/f(dr=0) must equal 1/(1+1)^alpha with alpha=-log(tau)/log(2)."""
        base_factor = 200.0
        pnl = 0.04
        pnl_factor = 1.0
        duration_ratio = 1.0
        taus = [0.9, 0.5, 0.25, 1.0]
        for tau in taus:
            params = self.base_params(
                exit_attenuation_mode="power", exit_power_tau=tau, exit_plateau=False
            )
            f0 = _get_exit_factor(base_factor, pnl, pnl_factor, 0.0, params)
            f1 = _get_exit_factor(base_factor, pnl, pnl_factor, duration_ratio, params)
            if 0.0 < tau <= 1.0:
                alpha = -math.log(tau) / math.log(2.0)
            else:
                alpha = 1.0
            expected_ratio = 1.0 / (1.0 + duration_ratio) ** alpha
            observed_ratio = f1 / f0 if f0 != 0 else np.nan
            self.assertFinite(observed_ratio, name="observed_ratio")
            self.assertLess(
                abs(observed_ratio - expected_ratio),
                5e-12 if tau == 1.0 else 5e-09,
                f"Alpha attenuation mismatch tau={tau} alpha={alpha} obs_ratio={observed_ratio} exp_ratio={expected_ratio}",
            )

    def test_reward_calculation_extreme_parameters_stability(self):
        """Test reward calculation extreme parameters stability."""
        extreme_params = self.base_params(win_reward_factor=1000.0, base_factor=10000.0)
        context = RewardContext(
            pnl=0.05,
            trade_duration=50,
            idle_duration=0,
            max_unrealized_profit=0.06,
            min_unrealized_profit=0.02,
            position=Positions.Long,
            action=Actions.Long_exit,
        )
        br = calculate_reward(
            context,
            extreme_params,
            base_factor=10000.0,
            profit_target=self.TEST_PROFIT_TARGET,
            risk_reward_ratio=self.TEST_RR,
            short_allowed=True,
            action_masking=True,
        )
        self.assertFinite(br.total, name="breakdown.total")

    def test_exit_attenuation_modes_enumeration(self):
        """Test exit attenuation modes enumeration."""
        modes = ATTENUATION_MODES_WITH_LEGACY
        for mode in modes:
            with self.subTest(mode=mode):
                test_params = self.base_params(exit_attenuation_mode=mode)
                ctx = self.make_ctx(
                    pnl=0.02,
                    trade_duration=50,
                    idle_duration=0,
                    max_unrealized_profit=0.03,
                    min_unrealized_profit=0.01,
                    position=Positions.Long,
                    action=Actions.Long_exit,
                )
                br = calculate_reward(
                    ctx,
                    test_params,
                    base_factor=self.TEST_BASE_FACTOR,
                    profit_target=self.TEST_PROFIT_TARGET,
                    risk_reward_ratio=self.TEST_RR,
                    short_allowed=True,
                    action_masking=True,
                )
                self.assertFinite(br.exit_component, name="breakdown.exit_component")
                self.assertFinite(br.total, name="breakdown.total")

    def test_exit_factor_boundary_parameters(self):
        """Test parameter edge cases: tau extremes, plateau grace edges, slope zero."""
        base_factor = 50.0
        pnl = 0.02
        pnl_factor = 1.0
        params_hi = self.base_params(exit_attenuation_mode="power", exit_power_tau=0.999999)
        params_lo = self.base_params(
            exit_attenuation_mode="power", exit_power_tau=self.MIN_EXIT_POWER_TAU
        )
        r = 1.5
        hi_val = _get_exit_factor(base_factor, pnl, pnl_factor, r, params_hi)
        lo_val = _get_exit_factor(base_factor, pnl, pnl_factor, r, params_lo)
        self.assertGreater(
            hi_val, lo_val, "Power mode: higher tau (≈1) should attenuate less than tiny tau"
        )
        params_g0 = self.base_params(
            exit_attenuation_mode="linear",
            exit_plateau=True,
            exit_plateau_grace=0.0,
            exit_linear_slope=1.0,
        )
        params_g1 = self.base_params(
            exit_attenuation_mode="linear",
            exit_plateau=True,
            exit_plateau_grace=1.0,
            exit_linear_slope=1.0,
        )
        val_g0 = _get_exit_factor(base_factor, pnl, pnl_factor, 0.5, params_g0)
        val_g1 = _get_exit_factor(base_factor, pnl, pnl_factor, 0.5, params_g1)
        self.assertGreater(
            val_g1, val_g0, "Plateau grace=1.0 should delay attenuation vs grace=0.0"
        )
        params_lin0 = self.base_params(
            exit_attenuation_mode="linear", exit_linear_slope=0.0, exit_plateau=False
        )
        params_lin1 = self.base_params(
            exit_attenuation_mode="linear", exit_linear_slope=2.0, exit_plateau=False
        )
        val_lin0 = _get_exit_factor(base_factor, pnl, pnl_factor, 1.0, params_lin0)
        val_lin1 = _get_exit_factor(base_factor, pnl, pnl_factor, 1.0, params_lin1)
        self.assertGreater(
            val_lin0, val_lin1, "Linear slope=0 should yield no attenuation vs slope>0"
        )

    def test_plateau_linear_slope_zero_constant_after_grace(self):
        """Plateau+linear slope=0 should yield flat factor after grace boundary (no attenuation)."""
        params = self.base_params(
            exit_attenuation_mode="linear",
            exit_plateau=True,
            exit_plateau_grace=0.3,
            exit_linear_slope=0.0,
        )
        base_factor = self.TEST_BASE_FACTOR
        pnl = 0.04
        pnl_factor = 1.2
        ratios = [0.3, 0.6, 1.0, 1.4]
        values = [_get_exit_factor(base_factor, pnl, pnl_factor, r, params) for r in ratios]
        first = values[0]
        for v in values[1:]:
            self.assertAlmostEqualFloat(
                v,
                first,
                tolerance=self.TOL_IDENTITY_RELAXED,
                msg=f"Plateau+linear slope=0 factor drift at ratio set {ratios} => {values}",
            )

    def test_plateau_grace_extends_beyond_one(self):
        """Plateau grace >1.0 should keep full strength (no attenuation) past duration_ratio=1."""
        params = self.DEFAULT_PARAMS.copy()
        params.update(
            {
                "exit_attenuation_mode": "linear",
                "exit_plateau": True,
                "exit_plateau_grace": 1.5,
                "exit_linear_slope": 2.0,
            }
        )
        base_factor = 80.0
        pnl = self.TEST_PROFIT_TARGET
        pnl_factor = 1.1
        ratios = [0.8, 1.0, 1.2, 1.4, 1.6]
        vals = [_get_exit_factor(base_factor, pnl, pnl_factor, r, params) for r in ratios]
        ref = vals[0]
        for i, r in enumerate(ratios[:-1]):
            self.assertAlmostEqualFloat(
                vals[i],
                ref,
                tolerance=self.TOL_IDENTITY_RELAXED,
                msg=f"Unexpected attenuation before grace end at ratio {r}",
            )
        self.assertLess(vals[-1], ref, "Attenuation should begin after grace boundary")

    def test_plateau_continuity_at_grace_boundary(self):
        """Test plateau continuity at grace boundary."""
        modes = ["sqrt", "linear", "power", "half_life"]
        grace = 0.8
        eps = self.CONTINUITY_EPS_SMALL
        base_factor = self.TEST_BASE_FACTOR
        pnl = 0.01
        pnl_factor = 1.0
        tau = 0.5
        half_life = 0.5
        slope = 1.3
        for mode in modes:
            with self.subTest(mode=mode):
                params = self.DEFAULT_PARAMS.copy()
                params.update(
                    {
                        "exit_attenuation_mode": mode,
                        "exit_plateau": True,
                        "exit_plateau_grace": grace,
                        "exit_linear_slope": slope,
                        "exit_power_tau": tau,
                        "exit_half_life": half_life,
                    }
                )
                left = _get_exit_factor(base_factor, pnl, pnl_factor, grace - eps, params)
                boundary = _get_exit_factor(base_factor, pnl, pnl_factor, grace, params)
                right = _get_exit_factor(base_factor, pnl, pnl_factor, grace + eps, params)
                self.assertAlmostEqualFloat(
                    left,
                    boundary,
                    tolerance=self.TOL_IDENTITY_RELAXED,
                    msg=f"Left/boundary mismatch for mode {mode}",
                )
                self.assertLess(
                    right, boundary, f"No attenuation detected just after grace for mode {mode}"
                )
                diff = boundary - right
                if mode == "linear":
                    bound = base_factor * slope * eps * 2.0
                elif mode == "sqrt":
                    bound = base_factor * 0.5 * eps * 2.0
                elif mode == "power":
                    alpha = -math.log(tau) / math.log(2.0)
                    bound = base_factor * alpha * eps * 2.0
                elif mode == "half_life":
                    bound = base_factor * (math.log(2.0) / half_life) * eps * 2.5
                else:
                    bound = base_factor * eps * 5.0
                self.assertLessEqual(
                    diff,
                    bound,
                    f"Attenuation jump too large at boundary for mode {mode} (diff={diff:.6e} > bound={bound:.6e})",
                )

    def test_plateau_continuity_multiple_eps_scaling(self):
        """Verify attenuation difference scales approximately linearly with epsilon (first-order continuity heuristic)."""
        mode = "linear"
        grace = 0.6
        eps1 = self.CONTINUITY_EPS_LARGE
        eps2 = self.CONTINUITY_EPS_SMALL
        base_factor = 80.0
        pnl = 0.02
        params = self.DEFAULT_PARAMS.copy()
        params.update(
            {
                "exit_attenuation_mode": mode,
                "exit_plateau": True,
                "exit_plateau_grace": grace,
                "exit_linear_slope": 1.1,
            }
        )
        f_boundary = _get_exit_factor(base_factor, pnl, 1.0, grace, params)
        f1 = _get_exit_factor(base_factor, pnl, 1.0, grace + eps1, params)
        f2 = _get_exit_factor(base_factor, pnl, 1.0, grace + eps2, params)
        diff1 = f_boundary - f1
        diff2 = f_boundary - f2
        ratio = diff1 / max(diff2, self.TOL_NUMERIC_GUARD)
        self.assertGreater(ratio, 5.0, f"Scaling ratio too small (ratio={ratio:.2f})")
        self.assertLess(ratio, 15.0, f"Scaling ratio too large (ratio={ratio:.2f})")


if __name__ == "__main__":
    unittest.main()
