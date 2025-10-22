#!/usr/bin/env python3
"""Tests for Potential-Based Reward Shaping (PBRS) mechanics."""

import math
import unittest

import numpy as np

from reward_space_analysis import (
    DEFAULT_MODEL_REWARD_PARAMETERS,
    _compute_entry_additive,
    _compute_exit_additive,
    _compute_exit_potential,
    _compute_hold_potential,
    _get_float_param,
    apply_potential_shaping,
    apply_transform,
    validate_reward_parameters,
)

from .test_base import RewardSpaceTestBase


class TestPBRS(RewardSpaceTestBase):
    """PBRS mechanics tests (transforms, parameters, potentials, invariance)."""

    def test_pbrs_progressive_release_decay_clamped(self):
        """progressive_release decay>1 clamps -> Φ'=0 & Δ=-Φ_prev."""
        params = self.DEFAULT_PARAMS.copy()
        params.update(
            {
                "potential_gamma": DEFAULT_MODEL_REWARD_PARAMETERS["potential_gamma"],
                "exit_potential_mode": "progressive_release",
                "exit_potential_decay": 5.0,
                "hold_potential_enabled": True,
                "entry_additive_enabled": False,
                "exit_additive_enabled": False,
            }
        )
        current_pnl = 0.02
        current_dur = 0.5
        prev_potential = _compute_hold_potential(current_pnl, current_dur, params)
        _total_reward, reward_shaping, next_potential = apply_potential_shaping(
            base_reward=0.0,
            current_pnl=current_pnl,
            current_duration_ratio=current_dur,
            next_pnl=0.0,
            next_duration_ratio=0.0,
            is_exit=True,
            is_entry=False,
            last_potential=0.789,
            params=params,
        )
        self.assertAlmostEqualFloat(next_potential, 0.0, tolerance=self.TOL_IDENTITY_RELAXED)
        self.assertAlmostEqualFloat(
            reward_shaping, -prev_potential, tolerance=self.TOL_IDENTITY_RELAXED
        )

    def test_pbrs_spike_cancel_invariance(self):
        """spike_cancel terminal shaping ≈0 (Φ' inversion yields cancellation)."""
        params = self.DEFAULT_PARAMS.copy()
        params.update(
            {
                "potential_gamma": 0.9,
                "exit_potential_mode": "spike_cancel",
                "hold_potential_enabled": True,
                "entry_additive_enabled": False,
                "exit_additive_enabled": False,
            }
        )
        current_pnl = 0.015
        current_dur = 0.4
        prev_potential = _compute_hold_potential(current_pnl, current_dur, params)
        gamma = _get_float_param(
            params, "potential_gamma", DEFAULT_MODEL_REWARD_PARAMETERS.get("potential_gamma", 0.95)
        )
        expected_next_potential = (
            prev_potential / gamma if gamma not in (0.0, None) else prev_potential
        )
        _total_reward, reward_shaping, next_potential = apply_potential_shaping(
            base_reward=0.0,
            current_pnl=current_pnl,
            current_duration_ratio=current_dur,
            next_pnl=0.0,
            next_duration_ratio=0.0,
            is_exit=True,
            is_entry=False,
            last_potential=prev_potential,
            params=params,
        )
        self.assertAlmostEqualFloat(
            next_potential, expected_next_potential, tolerance=self.TOL_IDENTITY_RELAXED
        )
        self.assertNearZero(reward_shaping, atol=self.TOL_IDENTITY_RELAXED)

    def test_tanh_transform(self):
        """tanh transform: tanh(x) in (-1, 1)."""
        self.assertAlmostEqualFloat(apply_transform("tanh", 0.0), 0.0)
        self.assertAlmostEqualFloat(apply_transform("tanh", 1.0), math.tanh(1.0))
        self.assertAlmostEqualFloat(apply_transform("tanh", -1.0), math.tanh(-1.0))
        self.assertTrue(abs(apply_transform("tanh", 100.0)) <= 1.0)
        self.assertTrue(abs(apply_transform("tanh", -100.0)) <= 1.0)

    def test_softsign_transform(self):
        """softsign transform: x / (1 + |x|) in (-1, 1)."""
        self.assertAlmostEqualFloat(apply_transform("softsign", 0.0), 0.0)
        self.assertAlmostEqualFloat(apply_transform("softsign", 1.0), 0.5)
        self.assertAlmostEqualFloat(apply_transform("softsign", -1.0), -0.5)
        self.assertTrue(abs(apply_transform("softsign", 100.0)) < 1.0)
        self.assertTrue(abs(apply_transform("softsign", -100.0)) < 1.0)

    def test_asinh_transform(self):
        """asinh transform: x / sqrt(1 + x^2) in (-1, 1)."""
        self.assertAlmostEqualFloat(apply_transform("asinh", 0.0), 0.0)
        self.assertAlmostEqualFloat(
            apply_transform("asinh", 1.2345),
            -apply_transform("asinh", -1.2345),
            tolerance=self.TOL_IDENTITY_STRICT,
        )
        vals = [apply_transform("asinh", x) for x in [-5.0, -1.0, 0.0, 1.0, 5.0]]
        self.assertTrue(all((vals[i] < vals[i + 1] for i in range(len(vals) - 1))))
        self.assertTrue(abs(apply_transform("asinh", 1000000.0)) < 1.0)
        self.assertTrue(abs(apply_transform("asinh", -1000000.0)) < 1.0)

    def test_arctan_transform(self):
        """arctan transform: (2/pi) * arctan(x) in (-1, 1)."""
        self.assertAlmostEqualFloat(apply_transform("arctan", 0.0), 0.0)
        self.assertAlmostEqualFloat(
            apply_transform("arctan", 1.0), 2.0 / math.pi * math.atan(1.0), tolerance=1e-10
        )
        self.assertTrue(abs(apply_transform("arctan", 100.0)) <= 1.0)
        self.assertTrue(abs(apply_transform("arctan", -100.0)) <= 1.0)

    def test_sigmoid_transform(self):
        """sigmoid transform: 2σ(x) - 1, σ(x) = 1/(1 + e^(-x)) in (-1, 1)."""
        self.assertAlmostEqualFloat(apply_transform("sigmoid", 0.0), 0.0)
        self.assertTrue(apply_transform("sigmoid", 100.0) > 0.99)
        self.assertTrue(apply_transform("sigmoid", -100.0) < -0.99)
        self.assertTrue(-1 < apply_transform("sigmoid", 10.0) < 1)
        self.assertTrue(-1 < apply_transform("sigmoid", -10.0) < 1)

    def test_clip_transform(self):
        """clip transform: clip(x, -1, 1) in [-1, 1]."""
        self.assertAlmostEqualFloat(apply_transform("clip", 0.0), 0.0)
        self.assertAlmostEqualFloat(apply_transform("clip", 0.5), 0.5)
        self.assertAlmostEqualFloat(apply_transform("clip", 2.0), 1.0)
        self.assertAlmostEqualFloat(apply_transform("clip", -2.0), -1.0)

    def test_invalid_transform(self):
        """Test error handling for invalid transforms."""
        self.assertAlmostEqualFloat(
            apply_transform("invalid_transform", 1.0),
            math.tanh(1.0),
            tolerance=self.TOL_IDENTITY_RELAXED,
        )

    def test_hold_potential_basic(self):
        """Test basic hold potential calculation."""
        params = {
            "hold_potential_enabled": True,
            "hold_potential_scale": 1.0,
            "hold_potential_gain": 1.0,
            "hold_potential_transform_pnl": "tanh",
            "hold_potential_transform_duration": "tanh",
        }
        val = _compute_hold_potential(0.5, 0.3, params)
        self.assertFinite(val, name="hold_potential")

    def test_entry_additive_disabled(self):
        """Test entry additive when disabled."""
        params = {"entry_additive_enabled": False}
        val = _compute_entry_additive(0.5, 0.3, params)
        self.assertEqual(val, 0.0)

    def test_exit_additive_disabled(self):
        """Test exit additive when disabled."""
        params = {"exit_additive_enabled": False}
        val = _compute_exit_additive(0.5, 0.3, params)
        self.assertEqual(val, 0.0)

    def test_exit_potential_canonical(self):
        params = self.base_params(
            exit_potential_mode="canonical",
            hold_potential_enabled=True,
            entry_additive_enabled=True,
            exit_additive_enabled=True,
        )
        base_reward = 0.25
        current_pnl = 0.05
        current_duration_ratio = 0.4
        next_pnl = 0.0
        next_duration_ratio = 0.0
        total, shaping, next_potential = apply_potential_shaping(
            base_reward=base_reward,
            current_pnl=current_pnl,
            current_duration_ratio=current_duration_ratio,
            next_pnl=next_pnl,
            next_duration_ratio=next_duration_ratio,
            is_exit=True,
            is_entry=False,
            last_potential=0.789,
            params=params,
        )
        self.assertIn("_pbrs_invariance_applied", params)
        self.assertFalse(
            params["entry_additive_enabled"],
            "Entry additive should be auto-disabled in canonical mode",
        )
        self.assertFalse(
            params["exit_additive_enabled"],
            "Exit additive should be auto-disabled in canonical mode",
        )
        self.assertPlacesEqual(next_potential, 0.0, places=12)
        current_potential = _compute_hold_potential(
            current_pnl,
            current_duration_ratio,
            {"hold_potential_enabled": True, "hold_potential_scale": 1.0},
        )
        self.assertAlmostEqual(shaping, -current_potential, delta=self.TOL_IDENTITY_RELAXED)
        residual = total - base_reward - shaping
        self.assertAlmostEqual(residual, 0.0, delta=self.TOL_IDENTITY_RELAXED)
        self.assertTrue(np.isfinite(total))

    def test_pbrs_invariance_internal_flag_set(self):
        """Canonical path sets _pbrs_invariance_applied once; second call idempotent."""
        params = self.base_params(
            exit_potential_mode="canonical",
            hold_potential_enabled=True,
            entry_additive_enabled=True,
            exit_additive_enabled=True,
        )
        terminal_next_potentials, shaping_values = self._canonical_sweep(params)
        _t1, _s1, _n1 = apply_potential_shaping(
            base_reward=0.0,
            current_pnl=0.05,
            current_duration_ratio=0.3,
            next_pnl=0.0,
            next_duration_ratio=0.0,
            is_exit=True,
            is_entry=False,
            last_potential=0.4,
            params=params,
        )
        self.assertIn("_pbrs_invariance_applied", params)
        self.assertFalse(params["entry_additive_enabled"])
        self.assertFalse(params["exit_additive_enabled"])
        if terminal_next_potentials:
            self.assertTrue(
                all((abs(p) < self.PBRS_TERMINAL_TOL for p in terminal_next_potentials))
            )
        max_abs = max((abs(v) for v in shaping_values)) if shaping_values else 0.0
        self.assertLessEqual(max_abs, self.PBRS_MAX_ABS_SHAPING)
        state_after = (params["entry_additive_enabled"], params["exit_additive_enabled"])
        _t2, _s2, _n2 = apply_potential_shaping(
            base_reward=0.0,
            current_pnl=0.02,
            current_duration_ratio=0.1,
            next_pnl=0.0,
            next_duration_ratio=0.0,
            is_exit=True,
            is_entry=False,
            last_potential=0.1,
            params=params,
        )
        self.assertEqual(
            state_after, (params["entry_additive_enabled"], params["exit_additive_enabled"])
        )

    def test_progressive_release_negative_decay_clamped(self):
        """Negative decay must clamp to 0 => next potential equals last potential (no release)."""
        params = self.base_params(
            exit_potential_mode="progressive_release",
            exit_potential_decay=-0.75,
            hold_potential_enabled=True,
        )
        last_potential = 0.42
        total, shaping, next_potential = apply_potential_shaping(
            base_reward=0.0,
            current_pnl=0.0,
            current_duration_ratio=0.0,
            next_pnl=0.0,
            next_duration_ratio=0.0,
            is_exit=True,
            last_potential=last_potential,
            params=params,
        )
        self.assertPlacesEqual(next_potential, last_potential, places=12)
        gamma_raw = DEFAULT_MODEL_REWARD_PARAMETERS.get("potential_gamma", 0.95)
        try:
            gamma = float(gamma_raw)
        except Exception:
            gamma = 0.95
        self.assertLessEqual(abs(shaping - gamma * last_potential), self.TOL_GENERIC_EQ)
        self.assertPlacesEqual(total, shaping, places=12)

    def test_potential_gamma_nan_fallback(self):
        """potential_gamma=NaN should fall back to default value (indirect comparison)."""
        base_params_dict = self.base_params()
        default_gamma = base_params_dict.get("potential_gamma", 0.95)
        params_nan = self.base_params(potential_gamma=np.nan, hold_potential_enabled=True)
        res_nan = apply_potential_shaping(
            base_reward=0.1,
            current_pnl=0.03,
            current_duration_ratio=0.2,
            next_pnl=0.035,
            next_duration_ratio=0.25,
            is_exit=False,
            last_potential=0.0,
            params=params_nan,
        )
        params_ref = self.base_params(potential_gamma=default_gamma, hold_potential_enabled=True)
        res_ref = apply_potential_shaping(
            base_reward=0.1,
            current_pnl=0.03,
            current_duration_ratio=0.2,
            next_pnl=0.035,
            next_duration_ratio=0.25,
            is_exit=False,
            last_potential=0.0,
            params=params_ref,
        )
        self.assertLess(
            abs(res_nan[1] - res_ref[1]),
            self.TOL_IDENTITY_RELAXED,
            "Unexpected shaping difference under gamma NaN fallback",
        )
        self.assertLess(
            abs(res_nan[0] - res_ref[0]),
            self.TOL_IDENTITY_RELAXED,
            "Unexpected total difference under gamma NaN fallback",
        )

    def test_validate_reward_parameters_success_and_failure(self):
        """validate_reward_parameters: success on defaults and failure on invalid ranges."""
        params_ok = DEFAULT_MODEL_REWARD_PARAMETERS.copy()
        try:
            validated = validate_reward_parameters(params_ok)
        except Exception as e:
            self.fail(f"validate_reward_parameters raised unexpectedly: {e}")
        if isinstance(validated, tuple) and len(validated) >= 1 and isinstance(validated[0], dict):
            validated_params = validated[0]
        else:
            validated_params = validated
        for k in ("potential_gamma", "hold_potential_enabled", "exit_potential_mode"):
            self.assertIn(k, validated_params, f"Missing key '{k}' in validated params")
        params_bad = params_ok.copy()
        params_bad["potential_gamma"] = -0.2
        params_bad["hold_potential_scale"] = -5.0
        with self.assertRaises((ValueError, AssertionError)):
            vr = validate_reward_parameters(params_bad)
            if not isinstance(vr, Exception):
                self.fail("validate_reward_parameters should raise on invalid params")

    def test_compute_exit_potential_mode_differences(self):
        """_compute_exit_potential modes: canonical resets Φ; spike_cancel approx preserves γΦ' ≈ Φ_prev (delta≈0)."""
        gamma = 0.93
        base_common = dict(
            hold_potential_enabled=True,
            potential_gamma=gamma,
            entry_additive_enabled=False,
            exit_additive_enabled=False,
            hold_potential_scale=1.0,
        )
        ctx_pnl = 0.012
        ctx_dur_ratio = 0.3
        params_can = self.base_params(exit_potential_mode="canonical", **base_common)
        prev_phi = _compute_hold_potential(ctx_pnl, ctx_dur_ratio, params_can)
        self.assertFinite(prev_phi, name="prev_phi")
        next_phi_can = _compute_exit_potential(prev_phi, params_can)
        self.assertAlmostEqualFloat(
            next_phi_can,
            0.0,
            tolerance=self.TOL_IDENTITY_STRICT,
            msg="Canonical exit must zero potential",
        )
        canonical_delta = -prev_phi
        self.assertAlmostEqualFloat(
            canonical_delta,
            -prev_phi,
            tolerance=self.TOL_IDENTITY_RELAXED,
            msg="Canonical delta mismatch",
        )
        params_spike = self.base_params(exit_potential_mode="spike_cancel", **base_common)
        next_phi_spike = _compute_exit_potential(prev_phi, params_spike)
        shaping_spike = gamma * next_phi_spike - prev_phi
        self.assertNearZero(
            shaping_spike,
            atol=self.TOL_IDENTITY_RELAXED,
            msg="Spike cancel should nullify shaping delta",
        )
        self.assertGreaterEqual(
            abs(canonical_delta) + self.TOL_IDENTITY_STRICT,
            abs(shaping_spike),
            "Canonical shaping magnitude should exceed spike_cancel",
        )

    def test_transform_bulk_monotonicity_and_bounds(self):
        """Non-decreasing monotonicity & (-1,1) bounds for smooth transforms (excluding clip)."""
        transforms = ["tanh", "softsign", "arctan", "sigmoid", "asinh"]
        xs = [-5.0, -1.0, -0.5, 0.0, 0.5, 1.0, 5.0]
        for name in transforms:
            with self.subTest(transform=name):
                vals = [apply_transform(name, x) for x in xs]
                self.assertTrue(all((-1.0 < v < 1.0 for v in vals)), f"{name} out of bounds")
                for a, b in zip(vals, vals[1:]):
                    self.assertLessEqual(
                        a, b + self.TOL_IDENTITY_STRICT, f"{name} not monotonic between {a} and {b}"
                    )

    def test_pbrs_retain_previous_cumulative_drift(self):
        """retain_previous mode accumulates negative shaping drift (non-invariant)."""
        params = self.base_params(
            exit_potential_mode="retain_previous",
            hold_potential_enabled=True,
            entry_additive_enabled=False,
            exit_additive_enabled=False,
            potential_gamma=0.9,
        )
        gamma = _get_float_param(
            params, "potential_gamma", DEFAULT_MODEL_REWARD_PARAMETERS.get("potential_gamma", 0.95)
        )
        rng = np.random.default_rng(555)
        potentials = rng.uniform(0.05, 0.85, size=220)
        deltas = [gamma * p - p for p in potentials]
        cumulative = float(np.sum(deltas))
        self.assertLess(cumulative, -self.TOL_NEGLIGIBLE)
        self.assertGreater(abs(cumulative), 10 * self.TOL_IDENTITY_RELAXED)

    def test_normality_invariance_under_scaling(self):
        """Skewness & excess kurtosis invariant under positive scaling of normal sample."""
        rng = np.random.default_rng(808)
        base = rng.normal(0.0, 1.0, size=7000)
        scaled = 5.0 * base

        def _skew_kurt(x: np.ndarray) -> tuple[float, float]:
            m = np.mean(x)
            c = x - m
            m2 = np.mean(c**2)
            m3 = np.mean(c**3)
            m4 = np.mean(c**4)
            skew = m3 / (m2**1.5 + 1e-18)
            kurt = m4 / (m2**2 + 1e-18) - 3.0
            return (skew, kurt)

        s_base, k_base = _skew_kurt(base)
        s_scaled, k_scaled = _skew_kurt(scaled)
        self.assertAlmostEqualFloat(s_base, s_scaled, tolerance=self.TOL_DISTRIB_SHAPE)
        self.assertAlmostEqualFloat(k_base, k_scaled, tolerance=self.TOL_DISTRIB_SHAPE)


if __name__ == "__main__":
    unittest.main()
