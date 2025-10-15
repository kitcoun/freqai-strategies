#!/usr/bin/env python3
"""Synthetic reward space analysis for the ReforceXY environment.

Capabilities:
- Hypothesis testing (Spearman, Kruskal-Wallis, Mann-Whitney).
- Percentile bootstrap confidence intervals (BCa not yet implemented).
- Distribution diagnostics (Shapiro, Anderson, skewness, kurtosis, Q-Q R²).
- Distribution shift metrics (KL divergence, JS distance, Wasserstein, KS test) with
    degenerate (constant) distribution safeguards.
- Unified RandomForest feature importance + partial dependence.
- Heteroscedastic PnL simulation (variance scales with duration).

Exit attenuation mode normalization:
- User supplied ``exit_attenuation_mode`` is taken as-is (case-sensitive) and validated
    against the allowed set. Any invalid value (including casing mismatch) results in a
    silent fallback to ``'linear'``.

Architecture principles:
- Single source of truth: ``DEFAULT_MODEL_REWARD_PARAMETERS`` (dynamic CLI generation).
- Determinism: explicit seeding, parameter hashing for manifest traceability.
- Extensibility: modular helpers (sampling, reward calculation, statistics, reporting).
"""

from __future__ import annotations

import argparse
import dataclasses
import math
import pickle
import random
import warnings
from enum import Enum, IntEnum
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
from scipy import stats
from scipy.spatial.distance import jensenshannon
from scipy.stats import entropy
from sklearn.ensemble import RandomForestRegressor
from sklearn.inspection import partial_dependence, permutation_importance
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split


class Actions(IntEnum):
    Neutral = 0
    Long_enter = 1
    Long_exit = 2
    Short_enter = 3
    Short_exit = 4


class Positions(Enum):
    Short = 0
    Long = 1
    Neutral = 0.5


# Tolerance for PBRS invariance classification (canonical if |Σ shaping| < PBRS_INVARIANCE_TOL)
PBRS_INVARIANCE_TOL: float = 1e-6
# Default discount factor γ for potential-based reward shaping
POTENTIAL_GAMMA_DEFAULT: float = 0.95

# Attenuation mode sets (centralized for tests and validation)
ATTENUATION_MODES: Tuple[str, ...] = ("sqrt", "linear", "power", "half_life")
ATTENUATION_MODES_WITH_LEGACY: Tuple[str, ...] = ATTENUATION_MODES + ("legacy",)

# Centralized internal numeric guards & behavior toggles (single source of truth for internal tunables)
INTERNAL_GUARDS: dict[str, float] = {
    "degenerate_ci_epsilon": 1e-9,
    "distribution_constant_fallback_moment": 0.0,
    "distribution_constant_fallback_qq_r2": 1.0,
    "moment_extreme_threshold": 1e4,
    "bootstrap_min_recommended": 200,
}


class RewardDiagnosticsWarning(RuntimeWarning):
    """Warning category for reward space diagnostic graceful fallbacks.

    Enables selective filtering in tests or CLI harness while leaving other
    runtime warnings unaffected.
    """

    pass


def _to_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return bool(int(value))
    if value is None:
        return False
    text = str(value).strip().lower()
    if text in {"true", "1", "yes", "y", "on"}:
        return True
    if text in {"false", "0", "no", "n", "off"}:
        return False
    return bool(text)


def _get_float_param(
    params: RewardParams, key: str, default: RewardParamValue
) -> float:
    """Extract float parameter with type safety and default fallback."""
    value = params.get(key, default)
    # None -> NaN
    if value is None:
        return np.nan
    # Bool: treat explicitly (avoid surprising True->1.0 unless intentional)
    if isinstance(value, bool):
        return float(int(value))
    # Numeric
    if isinstance(value, (int, float)):
        try:
            fval = float(value)
        except (ValueError, TypeError):
            return np.nan
        return fval if np.isfinite(fval) else np.nan
    # String parsing
    if isinstance(value, str):
        stripped = value.strip()
        if stripped == "":
            return np.nan
        try:
            fval = float(stripped)
        except ValueError:
            return np.nan
        return fval if np.isfinite(fval) else np.nan
    # Unsupported type
    return np.nan


def _compute_duration_ratio(trade_duration: int, max_trade_duration: int) -> float:
    """Compute duration ratio with safe division."""
    return trade_duration / max(1, max_trade_duration)


def _is_short_allowed(trading_mode: str) -> bool:
    mode = trading_mode.lower()
    if mode in {"margin", "futures"}:
        return True
    if mode == "spot":
        return False
    raise ValueError("Unsupported trading mode. Expected one of: spot, margin, futures")


# Mathematical constants pre-computed for performance
_LOG_2 = math.log(2.0)
DEFAULT_IDLE_DURATION_MULTIPLIER = 4

RewardParamValue = Union[float, str, bool, None]
RewardParams = Dict[str, RewardParamValue]


# Internal safe fallback helper for numeric failures (centralizes semantics)
def _fail_safely(reason: str) -> float:
    """Return 0.0 on recoverable numeric failure (reason available for future debug hooks)."""
    # NOTE: presently silent to preserve legacy behavior; hook logging here if needed.
    _ = reason
    return 0.0


# PBRS constants
ALLOWED_TRANSFORMS = {
    "tanh",
    "softsign",
    "softsign_sharp",
    "arctan",
    "logistic",
    "asinh_norm",
    "clip",
}
ALLOWED_EXIT_POTENTIAL_MODES = {
    "canonical",
    "non-canonical",
    "progressive_release",
    "spike_cancel",
    "retain_previous",
}

DEFAULT_MODEL_REWARD_PARAMETERS: RewardParams = {
    "invalid_action": -2.0,
    "base_factor": 100.0,
    # Idle penalty (env defaults)
    "idle_penalty_scale": 0.5,
    "idle_penalty_power": 1.025,
    # Fallback: 2 * max_trade_duration_candles
    "max_idle_duration_candles": None,
    # Hold keys (env defaults)
    "hold_penalty_scale": 0.25,
    "hold_penalty_power": 1.025,
    # Exit attenuation configuration (env default)
    "exit_attenuation_mode": "linear",
    "exit_plateau": True,
    "exit_plateau_grace": 1.0,
    "exit_linear_slope": 1.0,
    "exit_power_tau": 0.5,
    "exit_half_life": 0.5,
    # Efficiency keys (env defaults)
    "efficiency_weight": 1.0,
    "efficiency_center": 0.5,
    # Profit factor params (env defaults)
    "win_reward_factor": 2.0,
    "pnl_factor_beta": 0.5,
    # Invariant / safety controls (env defaults)
    "check_invariants": True,
    "exit_factor_threshold": 10000.0,
    # === PBRS PARAMETERS ===
    # Potential-based reward shaping core parameters
    # Discount factor γ for potential term (0 ≤ γ ≤ 1)
    "potential_gamma": POTENTIAL_GAMMA_DEFAULT,
    "potential_softsign_sharpness": 1.0,
    # Exit potential modes: canonical | non-canonical | progressive_release | spike_cancel | retain_previous
    "exit_potential_mode": "canonical",
    "exit_potential_decay": 0.5,
    # Hold potential (PBRS function Φ)
    "hold_potential_enabled": True,
    "hold_potential_scale": 1.0,
    "hold_potential_gain": 1.0,
    "hold_potential_transform_pnl": "tanh",
    "hold_potential_transform_duration": "tanh",
    # Entry additive (non-PBRS additive term)
    "entry_additive_enabled": False,
    "entry_additive_scale": 1.0,
    "entry_additive_gain": 1.0,
    "entry_additive_transform_pnl": "tanh",
    "entry_additive_transform_duration": "tanh",
    # Exit additive (non-PBRS additive term)
    "exit_additive_enabled": False,
    "exit_additive_scale": 1.0,
    "exit_additive_gain": 1.0,
    "exit_additive_transform_pnl": "tanh",
    "exit_additive_transform_duration": "tanh",
}

DEFAULT_MODEL_REWARD_PARAMETERS_HELP: Dict[str, str] = {
    "invalid_action": "Penalty for invalid actions.",
    "base_factor": "Base reward factor used inside the environment.",
    "idle_penalty_power": "Power applied to idle penalty scaling.",
    "idle_penalty_scale": "Scale of idle penalty.",
    "max_idle_duration_candles": "Maximum idle duration candles before full idle penalty scaling.",
    "hold_penalty_scale": "Scale of hold penalty.",
    "hold_penalty_power": "Power applied to hold penalty scaling.",
    "exit_attenuation_mode": "Attenuation kernel (legacy|sqrt|linear|power|half_life).",
    "exit_plateau": "Enable plateau. If true, full strength until grace boundary then apply attenuation.",
    "exit_plateau_grace": "Grace boundary duration ratio for plateau (full strength until this boundary).",
    "exit_linear_slope": "Slope for linear exit attenuation.",
    "exit_power_tau": "Tau in (0,1] to derive alpha for power mode.",
    "exit_half_life": "Half-life for exponential decay exit mode.",
    "efficiency_weight": "Weight for efficiency factor in exit reward.",
    "efficiency_center": "Pivot (in [0,1]) for linear efficiency factor; efficiency_ratio above this increases factor, below decreases.",
    "win_reward_factor": "Asymptotic bonus multiplier for pnl above target: approaches (1 + win_reward_factor); combined with efficiency_factor the final product can exceed this bound.",
    "pnl_factor_beta": "Sensitivity of amplification around target.",
    "check_invariants": "Boolean flag (true/false) to enable runtime invariant & safety checks.",
    "exit_factor_threshold": "If |exit factor| exceeds this threshold, emit warning.",
    # PBRS parameters
    "potential_gamma": "Discount factor γ for PBRS potential-based reward shaping (0 ≤ γ ≤ 1).",
    "potential_softsign_sharpness": "Sharpness parameter for softsign_sharp transform (smaller = sharper).",
    "exit_potential_mode": "Exit potential mode: 'canonical' (Φ=0 & additives disabled), 'non-canonical' (Φ=0 & additives allowed), 'progressive_release', 'spike_cancel', 'retain_previous'.",
    "exit_potential_decay": "Decay factor for progressive_release exit mode (0 ≤ decay ≤ 1).",
    "hold_potential_enabled": "Enable PBRS hold potential function Φ(s).",
    "hold_potential_scale": "Scale factor for hold potential function.",
    "hold_potential_gain": "Gain factor applied before transforms in hold potential.",
    "hold_potential_transform_pnl": "Transform function for PnL in hold potential: tanh, softsign, softsign_sharp, arctan, logistic, asinh_norm, clip.",
    "hold_potential_transform_duration": "Transform function for duration ratio in hold potential.",
    "entry_additive_enabled": "Enable entry additive reward (non-PBRS component).",
    "entry_additive_scale": "Scale factor for entry additive reward.",
    "entry_additive_gain": "Gain factor for entry additive reward.",
    "entry_additive_transform_pnl": "Transform function for PnL in entry additive.",
    "entry_additive_transform_duration": "Transform function for duration ratio in entry additive.",
    "exit_additive_enabled": "Enable exit additive reward (non-PBRS component).",
    "exit_additive_scale": "Scale factor for exit additive reward.",
    "exit_additive_gain": "Gain factor for exit additive reward.",
    "exit_additive_transform_pnl": "Transform function for PnL in exit additive.",
    "exit_additive_transform_duration": "Transform function for duration ratio in exit additive.",
}


# ---------------------------------------------------------------------------
# Parameter validation utilities
# ---------------------------------------------------------------------------

_PARAMETER_BOUNDS: Dict[str, Dict[str, float]] = {
    # key: {min: ..., max: ...}  (bounds are inclusive where it makes sense)
    "invalid_action": {"max": 0.0},  # penalty should be <= 0
    "base_factor": {"min": 0.0},
    "idle_penalty_power": {"min": 0.0},
    "idle_penalty_scale": {"min": 0.0},
    "max_idle_duration_candles": {"min": 0.0},
    "hold_penalty_scale": {"min": 0.0},
    "hold_penalty_power": {"min": 0.0},
    "exit_linear_slope": {"min": 0.0},
    "exit_plateau_grace": {"min": 0.0},
    "exit_power_tau": {"min": 1e-6, "max": 1.0},  # open (0,1] approximated
    "exit_half_life": {"min": 1e-6},
    "efficiency_weight": {"min": 0.0, "max": 2.0},
    "efficiency_center": {"min": 0.0, "max": 1.0},
    "win_reward_factor": {"min": 0.0},
    "pnl_factor_beta": {"min": 1e-6},
    # PBRS parameter bounds
    "potential_gamma": {"min": 0.0, "max": 1.0},
    # Softsign sharpness: only lower bound enforced (upper bound limited implicitly by transform stability)
    "potential_softsign_sharpness": {"min": 1e-6},
    "exit_potential_decay": {"min": 0.0, "max": 1.0},
    "hold_potential_scale": {"min": 0.0},
    "hold_potential_gain": {"min": 0.0},
    "entry_additive_scale": {"min": 0.0},
    "entry_additive_gain": {"min": 0.0},
    "exit_additive_scale": {"min": 0.0},
    "exit_additive_gain": {"min": 0.0},
}


def validate_reward_parameters(
    params: RewardParams,
) -> Tuple[RewardParams, Dict[str, Dict[str, Any]]]:
    """Validate and clamp reward parameter values.

    This function enforces numeric bounds declared in ``_PARAMETER_BOUNDS``. Values
    outside their allowed range are clamped and an entry is recorded in the
    ``adjustments`` mapping describing the original value, the adjusted value and the
    reason (which bound triggered the change). Non‑finite values are reset to the
    minimum bound (or 0.0 if no explicit minimum is defined).

    It does NOT perform schema validation of any DataFrame (legacy text removed).

    Parameters
    ----------
    params : dict
        Raw user supplied reward parameter overrides (already merged with defaults
        upstream). The dict is not mutated in‑place; a sanitized copy is returned.

    Returns
    -------
    sanitized_params : dict
        Possibly adjusted copy of the provided parameters.
    adjustments : dict[str, dict]
        Mapping: param -> {original, adjusted, reason} for every modified entry.
    """
    sanitized = dict(params)
    adjustments: Dict[str, Dict[str, Any]] = {}
    # Normalize boolean-like parameters explicitly to avoid inconsistent types
    _bool_keys = [
        "check_invariants",
        "hold_potential_enabled",
        "entry_additive_enabled",
        "exit_additive_enabled",
    ]
    for bkey in _bool_keys:
        if bkey in sanitized:
            original_val = sanitized[bkey]
            coerced = _to_bool(original_val)
            if coerced is not original_val:
                sanitized[bkey] = coerced
                adjustments.setdefault(
                    bkey,
                    {
                        "original": original_val,
                        "adjusted": coerced,
                        "reason": "bool_coerce",
                    },
                )
    for key, bounds in _PARAMETER_BOUNDS.items():
        if key not in sanitized:
            continue
        value = sanitized[key]
        if not isinstance(value, (int, float)):
            continue
        original = float(value)
        adjusted = original
        reason_parts: List[str] = []
        if "min" in bounds and adjusted < bounds["min"]:
            adjusted = bounds["min"]
            reason_parts.append(f"min={bounds['min']}")
        if "max" in bounds and adjusted > bounds["max"]:
            adjusted = bounds["max"]
            reason_parts.append(f"max={bounds['max']}")
        if not np.isfinite(adjusted):
            adjusted = bounds.get("min", 0.0)
            reason_parts.append("non_finite_reset")
        if not np.isclose(adjusted, original):
            sanitized[key] = adjusted
            adjustments[key] = {
                "original": original,
                "adjusted": adjusted,
                "reason": ",".join(reason_parts),  # textual reason directly
            }
    return sanitized, adjustments


def _normalize_and_validate_mode(params: RewardParams) -> None:
    """Align normalization of ``exit_attenuation_mode`` with ReforceXY environment.

    Behavior (mirrors in-env logic):
    - Do not force lowercase or strip user formatting; use the value as provided.
    - Supported modes (case-sensitive): {legacy, sqrt, linear, power, half_life}.
    - If the value is not among supported keys, silently fallback to 'linear'
      without emitting a warning (environment side performs a silent fallback).
    - If the key is absent or value is ``None``: leave untouched (upstream defaults
      will inject 'linear').
    """
    if "exit_attenuation_mode" not in params:
        return

    exit_attenuation_mode = _get_str_param(params, "exit_attenuation_mode", "linear")
    if exit_attenuation_mode not in ATTENUATION_MODES_WITH_LEGACY:
        params["exit_attenuation_mode"] = "linear"


def add_tunable_cli_args(parser: argparse.ArgumentParser) -> None:
    """Dynamically add CLI options for each tunable in DEFAULT_MODEL_REWARD_PARAMETERS.

    Rules:
    - Use the same underscored names as option flags (e.g., --idle_penalty_scale).
    - Defaults are None so only user-provided values override params.
    - For exit_attenuation_mode, enforce allowed choices (case-sensitive).
    - Skip keys already managed as top-level options (e.g., base_factor) to avoid duplicates.
    """
    skip_keys = {"base_factor"}  # already defined as top-level
    for key, default in DEFAULT_MODEL_REWARD_PARAMETERS.items():
        if key in skip_keys:
            continue
        help_text = DEFAULT_MODEL_REWARD_PARAMETERS_HELP.get(
            key, f"Override tunable '{key}'."
        )
        if key == "exit_attenuation_mode":
            parser.add_argument(
                f"--{key}",
                type=str,
                choices=sorted(ATTENUATION_MODES_WITH_LEGACY),
                default=None,
                help=help_text,
            )
        elif key == "exit_plateau":
            parser.add_argument(
                f"--{key}",
                type=int,
                choices=[0, 1],
                default=None,
                help=help_text,
            )
        elif key == "exit_potential_mode":
            parser.add_argument(
                f"--{key}",
                type=str,
                choices=sorted(ALLOWED_EXIT_POTENTIAL_MODES),
                default=None,
                help=help_text,
            )
        elif key in [
            "hold_potential_transform_pnl",
            "hold_potential_transform_duration",
            "entry_additive_transform_pnl",
            "entry_additive_transform_duration",
            "exit_additive_transform_pnl",
            "exit_additive_transform_duration",
        ]:
            parser.add_argument(
                f"--{key}",
                type=str,
                choices=sorted(ALLOWED_TRANSFORMS),
                default=None,
                help=help_text,
            )
        elif key in [
            "hold_potential_enabled",
            "entry_additive_enabled",
            "exit_additive_enabled",
        ]:
            parser.add_argument(
                f"--{key}",
                type=int,
                choices=[0, 1],
                default=None,
                help=help_text,
            )
        else:
            # Map numerics to float; leave strings as str
            if isinstance(default, (int, float)):
                parser.add_argument(
                    f"--{key}", type=float, default=None, help=help_text
                )
            else:
                parser.add_argument(f"--{key}", type=str, default=None, help=help_text)


@dataclasses.dataclass
class RewardContext:
    pnl: float
    trade_duration: int
    idle_duration: int
    max_trade_duration: int
    max_unrealized_profit: float
    min_unrealized_profit: float
    position: Positions
    action: Actions


@dataclasses.dataclass
class RewardBreakdown:
    total: float = 0.0
    invalid_penalty: float = 0.0
    idle_penalty: float = 0.0
    hold_penalty: float = 0.0
    exit_component: float = 0.0
    # PBRS components
    shaping_reward: float = 0.0
    entry_additive: float = 0.0
    exit_additive: float = 0.0
    current_potential: float = 0.0
    next_potential: float = 0.0


def _get_exit_factor(
    base_factor: float,
    pnl: float,
    pnl_factor: float,
    duration_ratio: float,
    params: RewardParams,
) -> float:
    """Compute exit factor controlling time attenuation of exit reward.

    Purpose
    -------
    Produces a multiplicative factor applied to raw PnL at exit:
        exit_reward = pnl * exit_factor
    where:
        exit_factor = time_kernel(base_factor, effective_duration_ratio) * pnl_factor

    Parity
    ------
    Mirrors environment method ``ReforceXY._get_exit_factor`` for offline / synthetic analysis.

    Algorithm
    ---------
    1. Validate finiteness & clamp negative duration to 0.
    2. Apply optional plateau: effective_dr = 0 while duration_ratio <= grace else (dr - grace).
    3. Select kernel: legacy | sqrt | linear | power | half_life (all monotonic except legacy discontinuity at dr=1).
    4. Multiply by externally supplied ``pnl_factor`` (profit & efficiency modulation).
    5. Invariants & safety: non-finite -> 0; prevent negative factor on non-negative pnl; warn on large magnitude.

    Parameters
    ----------
    base_factor : float
        Base scaling constant before temporal attenuation.
    pnl : float
        Realized (or unrealized at exit decision) profit/loss.
    pnl_factor : float
        PnL modulation factor (win amplification + efficiency) computed separately.
    duration_ratio : float
        trade_duration / max_trade_duration (pre-clamped upstream).
    params : dict
        Reward parameter mapping.

    Returns
    -------
    float
        Exit factor (>=0 unless pnl < 0 and invariants disabled).

    Notes
    -----
    - Legacy kernel is discontinuous and maintained for backward compatibility only.
    - Combine with ``_get_pnl_factor`` for full exit reward shaping (non-PBRS, path dependent).
    - Plateau introduces a derivative kink at the grace boundary.
    """
    # Basic finiteness checks
    if (
        not np.isfinite(base_factor)
        or not np.isfinite(pnl)
        or not np.isfinite(duration_ratio)
    ):
        return _fail_safely("non_finite_exit_factor_inputs")

    # Guard: duration ratio should never be negative
    if duration_ratio < 0.0:
        duration_ratio = 0.0

    exit_attenuation_mode = _get_str_param(params, "exit_attenuation_mode", "linear")
    exit_plateau = _get_bool_param(params, "exit_plateau", True)

    exit_plateau_grace = _get_float_param(params, "exit_plateau_grace", 1.0)
    if exit_plateau_grace < 0.0:
        exit_plateau_grace = 1.0
    exit_linear_slope = _get_float_param(params, "exit_linear_slope", 1.0)
    if exit_linear_slope < 0.0:
        exit_linear_slope = 1.0

    def _legacy_kernel(f: float, dr: float) -> float:
        return f * (1.5 if dr <= 1.0 else 0.5)

    def _sqrt_kernel(f: float, dr: float) -> float:
        return f / math.sqrt(1.0 + dr)

    def _linear_kernel(f: float, dr: float) -> float:
        return f / (1.0 + exit_linear_slope * dr)

    def _power_kernel(f: float, dr: float) -> float:
        tau = _get_float_param(
            params,
            "exit_power_tau",
            DEFAULT_MODEL_REWARD_PARAMETERS.get("exit_power_tau", 0.5),
        )
        if 0.0 < tau <= 1.0:
            alpha = -math.log(tau) / _LOG_2
        else:
            alpha = 1.0
        return f / math.pow(1.0 + dr, alpha)

    def _half_life_kernel(f: float, dr: float) -> float:
        hl = _get_float_param(params, "exit_half_life", 0.5)
        if hl <= 0.0:
            hl = 0.5
        return f * math.pow(2.0, -dr / hl)

    kernels = {
        "legacy": _legacy_kernel,
        "sqrt": _sqrt_kernel,
        "linear": _linear_kernel,
        "power": _power_kernel,
        "half_life": _half_life_kernel,
    }
    if exit_plateau:
        if duration_ratio <= exit_plateau_grace:
            effective_dr = 0.0
        else:
            effective_dr = duration_ratio - exit_plateau_grace
    else:
        effective_dr = duration_ratio

    kernel = kernels.get(exit_attenuation_mode, None)
    if kernel is None:
        warnings.warn(
            (
                f"Unknown exit_attenuation_mode '{exit_attenuation_mode}'; defaulting to 'linear' "
                f"(effective_dr={effective_dr:.5f})"
            ),
            RuntimeWarning,
            stacklevel=2,
        )
        kernel = _linear_kernel

    try:
        base_factor = kernel(base_factor, effective_dr)
    except Exception as e:
        warnings.warn(
            f"exit_attenuation_mode '{exit_attenuation_mode}' failed ({e!r}); fallback linear (effective_dr={effective_dr:.5f})",
            RuntimeWarning,
            stacklevel=2,
        )
        base_factor = _linear_kernel(base_factor, effective_dr)

    # Apply pnl_factor after time attenuation
    base_factor *= pnl_factor

    # Invariant & safety checks
    if _get_bool_param(params, "check_invariants", True):
        if not np.isfinite(base_factor):
            return _fail_safely("non_finite_exit_factor_after_kernel")
        if base_factor < 0.0 and pnl >= 0.0:
            # Clamp: avoid negative amplification on non-negative pnl
            base_factor = 0.0
        exit_factor_threshold = _get_float_param(
            params,
            "exit_factor_threshold",
            DEFAULT_MODEL_REWARD_PARAMETERS.get("exit_factor_threshold", 10000.0),
        )
        if exit_factor_threshold > 0 and np.isfinite(exit_factor_threshold):
            if abs(base_factor) > exit_factor_threshold:
                warnings.warn(
                    (
                        f"_get_exit_factor |factor|={abs(base_factor):.2f} exceeds threshold {exit_factor_threshold:.2f}"
                    ),
                    RuntimeWarning,
                    stacklevel=2,
                )

    return base_factor


def _get_pnl_factor(
    params: RewardParams,
    context: RewardContext,
    profit_target: float,
    risk_reward_ratio: float,
) -> float:
    """Compute PnL amplification factor (profit target over/under performance + efficiency).

    Purpose
    -------
    Encapsulates profit overshoot bonus and controlled loss penalty plus an efficiency tilt
    based on intra-trade utilization of observed profit range.

    Parity
    ------
    Mirrors environment method ``MyRLEnv._get_pnl_factor``.

    Algorithm
    ---------
    1. Compute pnl_ratio = pnl / profit_target (profit_target already includes RR upstream).
    2. If |pnl_ratio| <= 1: pnl_target_factor = 1.0.
    3. Else compute base = tanh(beta * (|pnl_ratio| - 1)).
       a. Gain branch (pnl_ratio > 1): 1 + win_reward_factor * base
       b. Loss branch (pnl_ratio < -1/rr): 1 + (win_reward_factor * rr) * base
    4. Efficiency: derive efficiency_ratio within [0,1] from intra-trade min/max; add linear tilt
       around efficiency_center scaled by efficiency_weight (sign-flipped for losses).
    5. Return max(0, pnl_target_factor * efficiency_factor).

    Parameters
    ----------
    params : dict
        Reward parameter mapping.
    context : RewardContext
        Current PnL and intra-trade extrema.
    profit_target : float
        Profit objective (already RR-scaled when called from calculate_reward).
    risk_reward_ratio : float
        RR used to set asymmetric loss trigger threshold (pnl_ratio < -1/RR).

    Returns
    -------
    float
        Non-negative factor (0 if invalid inputs or degenerate target).

    Notes
    -----
    - Symmetric tanh avoids unbounded amplification.
    - Loss penalty magnitude scaled by RR to keep incentive structure consistent across setups.
    - Efficiency tilt introduces path dependence (non-PBRS component).
    """
    pnl = context.pnl
    if (
        not np.isfinite(pnl)
        or not np.isfinite(profit_target)
        or not np.isfinite(risk_reward_ratio)
    ):
        return _fail_safely("non_finite_inputs_pnl_factor")
    if profit_target <= 0.0:
        return 0.0

    win_reward_factor = _get_float_param(
        params,
        "win_reward_factor",
        DEFAULT_MODEL_REWARD_PARAMETERS.get("win_reward_factor", 2.0),
    )
    pnl_factor_beta = _get_float_param(
        params,
        "pnl_factor_beta",
        DEFAULT_MODEL_REWARD_PARAMETERS.get("pnl_factor_beta", 0.5),
    )
    rr = risk_reward_ratio if risk_reward_ratio > 0 else 1.0

    pnl_ratio = pnl / profit_target
    pnl_target_factor = 1.0
    if abs(pnl_ratio) > 1.0:
        # Environment uses tanh(beta * (abs(ratio)-1)) for magnitude beyond target.
        base_pnl_target_factor = math.tanh(pnl_factor_beta * (abs(pnl_ratio) - 1.0))
        if pnl_ratio > 1.0:
            pnl_target_factor = 1.0 + win_reward_factor * base_pnl_target_factor
        elif pnl_ratio < -(1.0 / rr):
            loss_penalty_factor = win_reward_factor * rr
            pnl_target_factor = 1.0 + loss_penalty_factor * base_pnl_target_factor

    efficiency_factor = 1.0
    efficiency_weight = _get_float_param(
        params,
        "efficiency_weight",
        DEFAULT_MODEL_REWARD_PARAMETERS.get("efficiency_weight", 1.0),
    )
    efficiency_center = _get_float_param(
        params,
        "efficiency_center",
        DEFAULT_MODEL_REWARD_PARAMETERS.get("efficiency_center", 0.5),
    )
    if efficiency_weight != 0.0 and not np.isclose(pnl, 0.0):
        max_pnl = max(context.max_unrealized_profit, pnl)
        min_pnl = min(context.min_unrealized_profit, pnl)
        range_pnl = max_pnl - min_pnl
        if np.isfinite(range_pnl) and not np.isclose(range_pnl, 0.0):
            efficiency_ratio = (pnl - min_pnl) / range_pnl
            if pnl > 0.0:
                efficiency_factor = 1.0 + efficiency_weight * (
                    efficiency_ratio - efficiency_center
                )
            elif pnl < 0.0:
                efficiency_factor = 1.0 + efficiency_weight * (
                    efficiency_center - efficiency_ratio
                )

    return max(0.0, pnl_target_factor * efficiency_factor)


def _is_valid_action(
    position: Positions,
    action: Actions,
    *,
    short_allowed: bool,
) -> bool:
    if action == Actions.Neutral:
        return True
    if action == Actions.Long_enter:
        return position == Positions.Neutral
    if action == Actions.Short_enter:
        return short_allowed and position == Positions.Neutral
    if action == Actions.Long_exit:
        return position == Positions.Long
    if action == Actions.Short_exit:
        return position == Positions.Short
    return False


def _idle_penalty(
    context: RewardContext, idle_factor: float, params: RewardParams
) -> float:
    """Mirror the environment's idle penalty behavior."""
    idle_penalty_scale = _get_float_param(
        params,
        "idle_penalty_scale",
        DEFAULT_MODEL_REWARD_PARAMETERS.get("idle_penalty_scale", 0.5),
    )
    idle_penalty_power = _get_float_param(
        params,
        "idle_penalty_power",
        DEFAULT_MODEL_REWARD_PARAMETERS.get("idle_penalty_power", 1.025),
    )
    max_trade_duration_candles = params.get("max_trade_duration_candles")
    try:
        if max_trade_duration_candles is not None:
            max_trade_duration_candles = int(max_trade_duration_candles)
        else:
            max_trade_duration_candles = int(context.max_trade_duration)
    except (TypeError, ValueError):
        max_trade_duration_candles = int(context.max_trade_duration)

    max_idle_duration_candles = params.get("max_idle_duration_candles")
    if max_idle_duration_candles is None:
        max_idle_duration = (
            DEFAULT_IDLE_DURATION_MULTIPLIER * max_trade_duration_candles
        )
    else:
        try:
            max_idle_duration = int(max_idle_duration_candles)
        except (TypeError, ValueError):
            max_idle_duration = (
                DEFAULT_IDLE_DURATION_MULTIPLIER * max_trade_duration_candles
            )

    idle_duration_ratio = context.idle_duration / max(1, max_idle_duration)
    return -idle_factor * idle_penalty_scale * idle_duration_ratio**idle_penalty_power


def _hold_penalty(
    context: RewardContext, hold_factor: float, params: RewardParams
) -> float:
    """Mirror the environment's hold penalty behavior."""
    hold_penalty_scale = _get_float_param(
        params,
        "hold_penalty_scale",
        DEFAULT_MODEL_REWARD_PARAMETERS.get("hold_penalty_scale", 0.25),
    )
    hold_penalty_power = _get_float_param(
        params,
        "hold_penalty_power",
        DEFAULT_MODEL_REWARD_PARAMETERS.get("hold_penalty_power", 1.025),
    )
    duration_ratio = _compute_duration_ratio(
        context.trade_duration, context.max_trade_duration
    )

    if duration_ratio < 1.0:
        return _fail_safely("hold_penalty_duration_ratio_lt_1")

    return (
        -hold_factor * hold_penalty_scale * (duration_ratio - 1.0) ** hold_penalty_power
    )


def _compute_exit_reward(
    base_factor: float,
    pnl_factor: float,
    context: RewardContext,
    params: RewardParams,
) -> float:
    """Compose the exit reward: pnl * exit_factor."""
    duration_ratio = _compute_duration_ratio(
        context.trade_duration, context.max_trade_duration
    )
    exit_factor = _get_exit_factor(
        base_factor, context.pnl, pnl_factor, duration_ratio, params
    )
    return context.pnl * exit_factor


def calculate_reward(
    context: RewardContext,
    params: RewardParams,
    base_factor: float,
    profit_target: float,
    risk_reward_ratio: float,
    *,
    short_allowed: bool,
    action_masking: bool,
    previous_potential: float = 0.0,
) -> RewardBreakdown:
    breakdown = RewardBreakdown()

    is_valid = _is_valid_action(
        context.position,
        context.action,
        short_allowed=short_allowed,
    )
    if not is_valid and not action_masking:
        breakdown.invalid_penalty = _get_float_param(params, "invalid_action", -2.0)
        breakdown.total = breakdown.invalid_penalty
        return breakdown

    factor = _get_float_param(params, "base_factor", base_factor)

    if "profit_target" in params:
        profit_target = _get_float_param(params, "profit_target", float(profit_target))

    if "risk_reward_ratio" in params:
        risk_reward_ratio = _get_float_param(
            params, "risk_reward_ratio", float(risk_reward_ratio)
        )

    profit_target_final = profit_target * risk_reward_ratio
    idle_factor = factor * profit_target_final / 4.0
    pnl_factor = _get_pnl_factor(
        params,
        context,
        profit_target_final,
        risk_reward_ratio,
    )
    hold_factor = idle_factor

    # Base reward calculation (existing logic)
    base_reward = 0.0

    if context.action == Actions.Neutral and context.position == Positions.Neutral:
        base_reward = _idle_penalty(context, idle_factor, params)
        breakdown.idle_penalty = base_reward
    elif (
        context.position in (Positions.Long, Positions.Short)
        and context.action == Actions.Neutral
    ):
        base_reward = _hold_penalty(context, hold_factor, params)
        breakdown.hold_penalty = base_reward
    elif context.action == Actions.Long_exit and context.position == Positions.Long:
        base_reward = _compute_exit_reward(factor, pnl_factor, context, params)
        breakdown.exit_component = base_reward
    elif context.action == Actions.Short_exit and context.position == Positions.Short:
        base_reward = _compute_exit_reward(factor, pnl_factor, context, params)
        breakdown.exit_component = base_reward
    else:
        base_reward = 0.0

    # === PBRS INTEGRATION ===
    # Determine state transitions for PBRS
    current_pnl = context.pnl if context.position != Positions.Neutral else 0.0
    current_duration_ratio = (
        context.trade_duration / context.max_trade_duration
        if context.position != Positions.Neutral and context.max_trade_duration > 0
        else 0.0
    )

    # Simulate next state for PBRS calculation
    is_terminal = context.action in (Actions.Long_exit, Actions.Short_exit)

    # For terminal transitions, next state is neutral (PnL=0, duration=0)
    if is_terminal:
        next_pnl = 0.0
        next_duration_ratio = 0.0
    else:
        # For non-terminal, use current values (simplified simulation)
        next_pnl = current_pnl
        next_duration_ratio = current_duration_ratio

    # Apply PBRS if any PBRS parameters are enabled
    pbrs_enabled = (
        _get_bool_param(params, "hold_potential_enabled", True)
        or _get_bool_param(params, "entry_additive_enabled", False)
        or _get_bool_param(params, "exit_additive_enabled", False)
    )

    if pbrs_enabled:
        total_reward, shaping_reward, next_potential = apply_potential_shaping(
            base_reward=base_reward,
            current_pnl=current_pnl,
            current_duration_ratio=current_duration_ratio,
            next_pnl=next_pnl,
            next_duration_ratio=next_duration_ratio,
            is_terminal=is_terminal,
            last_potential=previous_potential,
            params=params,
        )

        # Update breakdown with PBRS components
        breakdown.shaping_reward = shaping_reward
        breakdown.current_potential = _compute_hold_potential(
            current_pnl, current_duration_ratio, params
        )
        breakdown.next_potential = next_potential
        breakdown.entry_additive = _compute_entry_additive(
            current_pnl, current_duration_ratio, params
        )
        breakdown.exit_additive = (
            _compute_exit_additive(next_pnl, next_duration_ratio, params)
            if is_terminal
            else 0.0
        )
        breakdown.total = total_reward
    else:
        breakdown.total = base_reward

    return breakdown


def _sample_action(
    position: Positions,
    rng: random.Random,
    *,
    short_allowed: bool,
) -> Actions:
    if position == Positions.Neutral:
        if short_allowed:
            choices = [Actions.Neutral, Actions.Long_enter, Actions.Short_enter]
            weights = [0.6, 0.2, 0.2]
        else:
            choices = [Actions.Neutral, Actions.Long_enter]
            weights = [0.7, 0.3]
    elif position == Positions.Long:
        choices = [Actions.Neutral, Actions.Long_exit]
        weights = [0.55, 0.45]
    else:  # Positions.Short
        choices = [Actions.Neutral, Actions.Short_exit]
        weights = [0.55, 0.45]
    return rng.choices(choices, weights=weights, k=1)[0]


def parse_overrides(overrides: Iterable[str]) -> RewardParams:
    parsed: RewardParams = {}
    for override in overrides:
        if "=" not in override:
            raise ValueError(f"Invalid override format: '{override}'")
        key, value = override.split("=", 1)
        try:
            parsed[key] = float(value)
        except ValueError:
            parsed[key] = value
    return parsed


def simulate_samples(
    num_samples: int,
    seed: int,
    params: RewardParams,
    max_trade_duration: int,
    base_factor: float,
    profit_target: float,
    risk_reward_ratio: float,
    max_duration_ratio: float,
    trading_mode: str,
    pnl_base_std: float,
    pnl_duration_vol_scale: float,
) -> pd.DataFrame:
    rng = random.Random(seed)
    short_allowed = _is_short_allowed(trading_mode)
    action_masking = _get_bool_param(params, "action_masking", True)
    samples: list[Dict[str, float]] = []
    for _ in range(num_samples):
        if short_allowed:
            position_choices = [Positions.Neutral, Positions.Long, Positions.Short]
            position_weights = [0.45, 0.3, 0.25]
        else:
            position_choices = [Positions.Neutral, Positions.Long]
            position_weights = [0.6, 0.4]

        position = rng.choices(position_choices, weights=position_weights, k=1)[0]
        action = _sample_action(position, rng, short_allowed=short_allowed)

        if position == Positions.Neutral:
            trade_duration = 0
            max_idle_duration_candles = params.get("max_idle_duration_candles")
            try:
                if max_idle_duration_candles is not None:
                    max_idle_duration_candles = int(max_idle_duration_candles)
                else:
                    max_idle_duration_candles = int(
                        max_trade_duration * max_duration_ratio
                    )
            except (TypeError, ValueError):
                max_idle_duration_candles = int(max_trade_duration * max_duration_ratio)

            idle_duration = int(rng.uniform(0, max_idle_duration_candles))
        else:
            trade_duration = int(
                rng.uniform(1, max_trade_duration * max_duration_ratio)
            )
            trade_duration = max(1, trade_duration)
            idle_duration = 0

        # Only exit actions should have non-zero PnL
        pnl = 0.0  # Initialize as zero for all actions

        # Generate PnL only for exit actions (Long_exit=2, Short_exit=4)
        if action in (Actions.Long_exit, Actions.Short_exit):
            # Apply directional bias for positions
            duration_factor = trade_duration / max(1, max_trade_duration)

            # PnL variance scales with duration for more realistic heteroscedasticity
            pnl_std = pnl_base_std * (1.0 + pnl_duration_vol_scale * duration_factor)
            pnl = rng.gauss(0.0, pnl_std)
            if position == Positions.Long:
                pnl += 0.005 * duration_factor
            elif position == Positions.Short:
                pnl -= 0.005 * duration_factor

            # Clip PnL to realistic range
            pnl = max(min(pnl, 0.15), -0.15)

        if position == Positions.Neutral:
            max_unrealized_profit = 0.0
            min_unrealized_profit = 0.0
        else:
            # Unrealized profits should bracket the final PnL
            # Max represents peak profit during trade, min represents lowest point
            span = abs(rng.gauss(0.0, 0.015))
            # Ensure max >= pnl >= min by construction
            max_unrealized_profit = pnl + abs(rng.gauss(0.0, span))
            min_unrealized_profit = pnl - abs(rng.gauss(0.0, span))

        context = RewardContext(
            pnl=pnl,
            trade_duration=trade_duration,
            idle_duration=idle_duration,
            max_trade_duration=max_trade_duration,
            max_unrealized_profit=max_unrealized_profit,
            min_unrealized_profit=min_unrealized_profit,
            position=position,
            action=action,
        )

        breakdown = calculate_reward(
            context,
            params,
            base_factor,
            profit_target,
            risk_reward_ratio,
            short_allowed=short_allowed,
            action_masking=action_masking,
        )

        samples.append(
            {
                "pnl": context.pnl,
                "trade_duration": context.trade_duration,
                "idle_duration": context.idle_duration,
                "duration_ratio": context.trade_duration / max(1, max_trade_duration),
                "idle_ratio": context.idle_duration / max(1, max_trade_duration),
                "position": float(context.position.value),
                "action": float(context.action.value),
                "reward_total": breakdown.total,
                "reward_invalid": breakdown.invalid_penalty,
                "reward_idle": breakdown.idle_penalty,
                "reward_hold": breakdown.hold_penalty,
                "reward_exit": breakdown.exit_component,
                # PBRS components
                "reward_shaping": breakdown.shaping_reward,
                "reward_entry_additive": breakdown.entry_additive,
                "reward_exit_additive": breakdown.exit_additive,
                "current_potential": breakdown.current_potential,
                "next_potential": breakdown.next_potential,
                "is_invalid": float(breakdown.invalid_penalty != 0.0),
            }
        )

    df = pd.DataFrame(samples)

    # Validate critical algorithmic invariants
    _validate_simulation_invariants(df)

    return df


def _validate_simulation_invariants(df: pd.DataFrame) -> None:
    """Validate critical algorithmic invariants in simulated data.

    This function ensures mathematical correctness and catches algorithmic bugs.
    Failures here indicate fundamental implementation errors that must be fixed.
    """
    # INVARIANT 1: PnL Conservation - Total PnL must equal sum of exit PnL
    total_pnl = df["pnl"].sum()
    exit_action_mask = df["action"].isin([2.0, 4.0])
    exit_pnl_sum = df.loc[exit_action_mask, "pnl"].sum()

    pnl_diff = abs(total_pnl - exit_pnl_sum)
    if pnl_diff > 1e-10:
        raise AssertionError(
            f"PnL INVARIANT VIOLATION: Total PnL ({total_pnl:.6f}) != "
            f"Exit PnL sum ({exit_pnl_sum:.6f}), difference = {pnl_diff:.2e}"
        )

    # INVARIANT 2: PnL Exclusivity - Only exit actions should have non-zero PnL
    non_zero_pnl_actions = set(df[df["pnl"] != 0]["action"].unique())
    valid_exit_actions = {2.0, 4.0}
    invalid_actions = non_zero_pnl_actions - valid_exit_actions
    if invalid_actions:
        raise AssertionError(
            f"PnL EXCLUSIVITY VIOLATION: Non-exit actions {invalid_actions} have non-zero PnL"
        )

    # INVARIANT 3: Exit Reward Consistency - Non-zero exit rewards require non-zero PnL
    inconsistent_exits = df[(df["pnl"] == 0) & (df["reward_exit"] != 0)]
    if len(inconsistent_exits) > 0:
        raise AssertionError(
            f"EXIT REWARD INCONSISTENCY: {len(inconsistent_exits)} actions have "
            f"zero PnL but non-zero exit reward"
        )

    # INVARIANT 4: Action-Position Compatibility
    # Validate that exit actions match positions
    long_exits = df[
        (df["action"] == 2.0) & (df["position"] != 1.0)
    ]  # Long_exit but not Long position
    short_exits = df[
        (df["action"] == 4.0) & (df["position"] != 0.0)
    ]  # Short_exit but not Short position

    if len(long_exits) > 0:
        raise AssertionError(
            f"ACTION-POSITION INCONSISTENCY: {len(long_exits)} Long_exit actions "
            f"without Long position"
        )

    if len(short_exits) > 0:
        raise AssertionError(
            f"ACTION-POSITION INCONSISTENCY: {len(short_exits)} Short_exit actions "
            f"without Short position"
        )

    # INVARIANT 5: Duration Logic - Neutral positions should have trade_duration = 0
    neutral_with_trade = df[(df["position"] == 0.5) & (df["trade_duration"] > 0)]
    if len(neutral_with_trade) > 0:
        raise AssertionError(
            f"DURATION LOGIC VIOLATION: {len(neutral_with_trade)} Neutral positions "
            f"with non-zero trade_duration"
        )

    # INVARIANT 6: Bounded Values - Check realistic bounds
    extreme_pnl = df[(df["pnl"].abs() > 0.2)]  # Beyond reasonable range
    if len(extreme_pnl) > 0:
        max_abs_pnl = df["pnl"].abs().max()
        raise AssertionError(
            f"BOUNDS VIOLATION: {len(extreme_pnl)} samples with extreme PnL, "
            f"max |PnL| = {max_abs_pnl:.6f}"
        )


def _compute_summary_stats(df: pd.DataFrame) -> Dict[str, Any]:
    """Compute summary statistics without writing to file."""
    action_summary = df.groupby("action")["reward_total"].agg(
        ["count", "mean", "std", "min", "max"]
    )
    component_share = df[
        [
            "reward_invalid",
            "reward_idle",
            "reward_hold",
            "reward_exit",
            "reward_shaping",
            "reward_entry_additive",
            "reward_exit_additive",
        ]
    ].apply(lambda col: (col != 0).mean())

    components = [
        "reward_invalid",
        "reward_idle",
        "reward_hold",
        "reward_exit",
        "reward_total",
    ]
    component_bounds = (
        df[components]
        .agg(["min", "mean", "max"])
        .T.rename(
            columns={
                "min": "component_min",
                "mean": "component_mean",
                "max": "component_max",
            }
        )
        .round(6)
    )

    global_stats = df["reward_total"].describe(
        percentiles=[0.01, 0.1, 0.25, 0.5, 0.75, 0.9, 0.99]
    )

    return {
        "global_stats": global_stats,
        "action_summary": action_summary,
        "component_share": component_share,
        "component_bounds": component_bounds,
    }


def _binned_stats(
    df: pd.DataFrame,
    column: str,
    target: str,
    bins: Iterable[float],
) -> pd.DataFrame:
    """Compute aggregated statistics of a target variable across value bins.

    Purpose
    -------
    Provide consistent binned descriptive statistics (count, mean, std, min, max)
    for exploratory diagnostics of reward component relationships.

    Parameters
    ----------
    df : pd.DataFrame
        Source dataframe containing at minimum the ``column`` and ``target`` fields.
    column : str
        Name of the column whose values are used to create bins.
    target : str
        Column whose statistics are aggregated per bin.
    bins : Iterable[float]
        Monotonic sequence of bin edges (len >= 2). Values outside the range are
        clipped to the boundary edges prior to bin assignment.

    Returns
    -------
    pd.DataFrame
        Dataframe indexed by stringified interval with columns:
        ``count``, ``mean``, ``std``, ``min``, ``max``.

    Notes
    -----
    - Duplicate bin edges are dropped via pandas ``cut(duplicates='drop')``.
    - Non-finite or missing bins after clipping are excluded prior to grouping.
    - This helper is deterministic and side-effect free.
    """
    bins_arr = np.asarray(list(bins), dtype=float)
    if bins_arr.ndim != 1 or bins_arr.size < 2:
        raise ValueError("bins must contain at least two edges")
    clipped = df[column].clip(lower=float(bins_arr[0]), upper=float(bins_arr[-1]))
    categories = pd.cut(
        clipped,
        bins=bins_arr,
        include_lowest=True,
        duplicates="drop",
    )
    aggregated = (
        pd.DataFrame({"bin": categories, target: df[target]})
        .dropna(subset=["bin"])
        .groupby("bin", observed=False)[target]
        .agg(["count", "mean", "std", "min", "max"])
    )
    aggregated.index = aggregated.index.astype(str)
    return aggregated


def _compute_relationship_stats(
    df: pd.DataFrame, max_trade_duration: int
) -> Dict[str, Any]:
    """Compute binned relationship statistics among core reward drivers.

    Purpose
    -------
    Generate uniformly binned summaries for idle duration, trade duration and
    realized PnL to facilitate downstream comparative or visual analyses.

    Parameters
    ----------
    df : pd.DataFrame
        Input dataframe containing ``idle_duration``, ``trade_duration``, ``pnl`` and
        corresponding reward component columns (``reward_idle``, ``reward_hold``,
        ``reward_exit``).
    max_trade_duration : int
        Maximum configured trade duration used to scale bin ranges.

    Returns
    -------
    Dict[str, Any]
        Dictionary with keys ``idle_stats``, ``hold_stats`` and ``exit_stats`` each
        containing a binned statistics dataframe.

    Notes
    -----
    - PnL bin upper bound is adjusted by a tiny epsilon when min ≈ max to avoid
      degenerate intervals.
    - All statistics are rounded to 6 decimal places for compactness.
    """
    idle_bins = np.linspace(0, max_trade_duration * 3.0, 13)
    trade_bins = np.linspace(0, max_trade_duration * 3.0, 13)
    pnl_min = float(df["pnl"].min())
    pnl_max = float(df["pnl"].max())
    if np.isclose(pnl_min, pnl_max):
        pnl_max = pnl_min + 1e-6
    pnl_bins = np.linspace(pnl_min, pnl_max, 13)

    idle_stats = _binned_stats(df, "idle_duration", "reward_idle", idle_bins)
    hold_stats = _binned_stats(df, "trade_duration", "reward_hold", trade_bins)
    exit_stats = _binned_stats(df, "pnl", "reward_exit", pnl_bins)

    idle_stats = idle_stats.round(6)
    hold_stats = hold_stats.round(6)
    exit_stats = exit_stats.round(6)

    correlation_fields = [
        "reward_total",
        "reward_invalid",
        "reward_idle",
        "reward_hold",
        "reward_exit",
        "pnl",
        "trade_duration",
        "idle_duration",
    ]
    # Drop columns that are constant (std == 0) to avoid all-NaN correlation rows
    numeric_subset = df[correlation_fields]
    constant_cols = [
        c for c in numeric_subset.columns if numeric_subset[c].nunique() <= 1
    ]
    if constant_cols:
        filtered = numeric_subset.drop(columns=constant_cols)
    else:
        filtered = numeric_subset
    correlation = filtered.corr().round(4)

    return {
        "idle_stats": idle_stats,
        "hold_stats": hold_stats,
        "exit_stats": exit_stats,
        "correlation": correlation,
        "correlation_dropped": constant_cols,
    }


def _compute_representativity_stats(
    df: pd.DataFrame, profit_target: float, max_trade_duration: int | None = None
) -> Dict[str, Any]:
    """Compute representativity statistics for the reward space.

    NOTE: The max_trade_duration parameter is reserved for future duration coverage metrics.
    """
    total = len(df)
    # Map numeric position codes to readable labels to avoid casting Neutral (0.5) to 0
    pos_label_map = {0.0: "Short", 0.5: "Neutral", 1.0: "Long"}
    pos_labeled = df["position"].map(pos_label_map)
    pos_counts = (
        pos_labeled.value_counts()
        .reindex(["Short", "Neutral", "Long"])
        .fillna(0)
        .astype(int)
    )
    # Actions are encoded as float enum values, casting to int is safe here
    act_counts = df["action"].astype(int).value_counts().sort_index()

    pnl_above_target = float((df["pnl"] > profit_target).mean())
    pnl_near_target = float(
        ((df["pnl"] >= 0.8 * profit_target) & (df["pnl"] <= 1.2 * profit_target)).mean()
    )
    pnl_extreme = float((df["pnl"].abs() >= 0.14).mean())

    duration_overage_share = float((df["duration_ratio"] > 1.0).mean())
    idle_activated = float((df["reward_idle"] != 0).mean())
    hold_activated = float((df["reward_hold"] != 0).mean())
    exit_activated = float((df["reward_exit"] != 0).mean())

    return {
        "total": total,
        "pos_counts": pos_counts,
        "act_counts": act_counts,
        "pnl_above_target": pnl_above_target,
        "pnl_near_target": pnl_near_target,
        "pnl_extreme": pnl_extreme,
        "duration_overage_share": duration_overage_share,
        "idle_activated": idle_activated,
        "hold_activated": hold_activated,
        "exit_activated": exit_activated,
    }


def _perform_feature_analysis(
    df: pd.DataFrame, seed: int, *, skip_partial_dependence: bool = False
) -> Tuple[
    pd.DataFrame, Dict[str, Any], Dict[str, pd.DataFrame], RandomForestRegressor
]:
    """Run RandomForest-based feature analysis.

    Returns
    -------
    importance_df : pd.DataFrame
        Permutation importance summary (mean/std per feature).
    analysis_stats : Dict[str, Any]
        Core diagnostics (R², sample counts, top feature & score).
    partial_deps : Dict[str, pd.DataFrame]
        Partial dependence data frames keyed by feature.
    model : RandomForestRegressor
        Fitted model instance (for optional downstream inspection).
    """
    feature_cols = [
        "pnl",
        "trade_duration",
        "idle_duration",
        "duration_ratio",
        "idle_ratio",
        "position",
        "action",
        "is_invalid",
    ]
    X = df[feature_cols]
    for col in ("trade_duration", "idle_duration"):
        if col in X.columns and pd.api.types.is_integer_dtype(X[col]):
            X.loc[:, col] = X[col].astype(float)
    y = df["reward_total"]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=seed
    )

    # Canonical RandomForest configuration - single source of truth
    model = RandomForestRegressor(
        n_estimators=400,
        max_depth=None,
        random_state=seed,
        n_jobs=1,
    )
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    r2 = r2_score(y_test, y_pred)

    perm = permutation_importance(
        model,
        X_test,
        y_test,
        n_repeats=25,
        random_state=seed,
        n_jobs=1,
    )

    importance_df = (
        pd.DataFrame(
            {
                "feature": feature_cols,
                "importance_mean": perm.importances_mean,
                "importance_std": perm.importances_std,
            }
        )
        .sort_values("importance_mean", ascending=False)
        .reset_index(drop=True)
    )

    # Compute partial dependence for key features unless skipped
    partial_deps = {}
    if not skip_partial_dependence:
        for feature in ["trade_duration", "idle_duration", "pnl"]:
            pd_result = partial_dependence(
                model,
                X_test,
                [feature],
                grid_resolution=50,
                kind="average",
            )
            value_key = "values" if "values" in pd_result else "grid_values"
            values = pd_result[value_key][0]
            averaged = pd_result["average"][0]
            partial_deps[feature] = pd.DataFrame(
                {feature: values, "partial_dependence": averaged}
            )

    analysis_stats = {
        "r2_score": r2,
        "n_features": len(feature_cols),
        "n_samples_train": len(X_train),
        "n_samples_test": len(X_test),
        "top_feature": importance_df.iloc[0]["feature"],
        "top_importance": importance_df.iloc[0]["importance_mean"],
    }

    return importance_df, analysis_stats, partial_deps, model


def load_real_episodes(path: Path, *, enforce_columns: bool = True) -> pd.DataFrame:
    """Load serialized episodes / transitions into a normalized DataFrame.

    Accepted payload topologies (pickled):
    - DataFrame directly.
    - Dict with key ``'transitions'`` whose value is:
        * a DataFrame OR
        * an iterable of transition dicts.
    - List of episode dicts where each dict with key ``'transitions'`` provides:
        * a DataFrame OR
        * an iterable of transition dicts.
    - Fallback: any object convertible to ``pd.DataFrame`` (best‑effort).

    Normalization steps:
    1. Flatten nested episode structures.
    2. Coerce numeric columns (expected + optional) with safe NaN coercion tracking.
    3. Ensure required columns exist (raise or fill with NaN depending on ``enforce_columns``).
    4. Add missing optional numeric columns (set to NaN) to obtain a stable schema.
    5. (Light) Deduplication: drop exact duplicate transition rows if any.

    Security NOTE:
        Pickle loading executes arbitrary code if the file is malicious. Only load
        trusted artifacts. This helper assumes the calling context enforces trust.

    Parameters
    ----------
    path : Path
        Pickle file path.
    enforce_columns : bool, default True
        If True, missing required columns cause a ValueError; else they are created
        and filled with NaN and a warning is emitted.

    Returns
    -------
    pd.DataFrame
        Normalized transition frame including (at minimum) required columns.

    Raises
    ------
    ValueError
        On unpickle failure, structure mismatch (when ``enforce_columns`` is True),
        or irrecoverable conversion errors.
    """

    try:
        with path.open("rb") as f:
            episodes_data = pickle.load(f)
    except Exception as e:
        raise ValueError(f"Failed to unpickle '{path}': {e!r}") from e

    # Top-level dict with 'transitions'
    if isinstance(episodes_data, dict) and "transitions" in episodes_data:
        candidate = episodes_data["transitions"]
        if isinstance(candidate, pd.DataFrame):
            df = candidate.copy()
        else:
            try:
                df = pd.DataFrame(list(candidate))
            except TypeError:
                raise ValueError(
                    f"Top-level 'transitions' in '{path}' is not iterable (type {type(candidate)!r})."
                )
            except Exception as e:
                raise ValueError(
                    f"Could not build DataFrame from top-level 'transitions' in '{path}': {e!r}"
                ) from e
    # List of episodes where some entries have 'transitions'
    elif isinstance(episodes_data, list) and any(
        isinstance(e, dict) and "transitions" in e for e in episodes_data
    ):
        all_transitions = []
        skipped = 0
        for episode in episodes_data:
            if isinstance(episode, dict) and "transitions" in episode:
                trans = episode["transitions"]
                if isinstance(trans, pd.DataFrame):
                    all_transitions.extend(trans.to_dict(orient="records"))
                else:
                    try:
                        all_transitions.extend(list(trans))
                    except TypeError:
                        raise ValueError(
                            f"Episode 'transitions' is not iterable in file '{path}'; found type {type(trans)!r}"
                        )
            else:
                skipped += 1
        if skipped:
            warnings.warn(
                f"Ignored {skipped} episode(s) without 'transitions' when loading '{path}'",
                RuntimeWarning,
                stacklevel=2,
            )
        try:
            df = pd.DataFrame(all_transitions)
        except Exception as e:
            raise ValueError(
                f"Could not build DataFrame from flattened transitions in '{path}': {e!r}"
            ) from e
    else:
        try:
            if isinstance(episodes_data, pd.DataFrame):
                df = episodes_data.copy()
            else:
                df = pd.DataFrame(episodes_data)
        except Exception as e:
            raise ValueError(
                f"Could not convert pickled object from '{path}' to DataFrame: {e!r}"
            ) from e

    # Coerce common numeric fields; warn when values are coerced to NaN
    numeric_expected = {
        "pnl",
        "trade_duration",
        "idle_duration",
        "position",
        "action",
        "reward_total",
    }

    # Keep optional list stable and explicit
    numeric_optional = {
        "reward_exit",
        "reward_idle",
        "reward_hold",
        "reward_invalid",
        "duration_ratio",
        "idle_ratio",
        "max_unrealized_profit",
        "min_unrealized_profit",
        # Additive / shaping components
        "reward_entry_additive",
        "reward_exit_additive",
        "current_potential",
        "next_potential",
        "is_invalid",
    }

    for col in list(numeric_expected | numeric_optional):
        if col in df.columns:
            before_na = df[col].isna().sum()
            df[col] = pd.to_numeric(df[col], errors="coerce")
            coerced = df[col].isna().sum() - before_na
            if coerced > 0:
                frac = coerced / len(df) if len(df) > 0 else 0.0
                warnings.warn(
                    (
                        f"Column '{col}' contained {coerced} non-numeric value(s) "
                        f"({frac:.1%}) that were coerced to NaN when loading '{path}'."
                    ),
                    RuntimeWarning,
                    stacklevel=2,
                )

    # Ensure required columns exist (or fill with NaN if allowed)
    required = {
        "pnl",
        "trade_duration",
        "idle_duration",
        "position",
        "action",
        "reward_total",
    }
    missing_required = required - set(df.columns)
    if missing_required:
        if enforce_columns:
            raise ValueError(
                f"Loaded episodes data is missing required columns: {sorted(missing_required)}. "
                f"Found columns: {sorted(list(df.columns))}."
            )
        warnings.warn(
            f"Loaded episodes data is missing columns {sorted(missing_required)}; filling with NaN (enforce_columns=False)",
            RuntimeWarning,
            stacklevel=2,
        )
        for col in missing_required:
            df[col] = np.nan

    # Ensure all optional numeric columns exist for schema stability
    for opt_col in numeric_optional:
        if opt_col not in df.columns:
            df[opt_col] = np.nan

    # Drop exact duplicates (rare but can appear after flattening)
    before_dupes = len(df)
    df = df.drop_duplicates()
    if len(df) != before_dupes:
        warnings.warn(
            f"Removed {before_dupes - len(df)} duplicate transition row(s) while loading '{path}'.",
            RuntimeWarning,
            stacklevel=2,
        )

    return df


def compute_distribution_shift_metrics(
    synthetic_df: pd.DataFrame,
    real_df: pd.DataFrame,
) -> Dict[str, float]:
    """Compute KL, JS, Wasserstein divergences between synthetic and real distributions."""
    metrics = {}
    continuous_features = ["pnl", "trade_duration", "idle_duration"]

    for feature in continuous_features:
        synth_values = synthetic_df[feature].dropna().values
        real_values = real_df[feature].dropna().values

        if len(synth_values) < 10 or len(real_values) < 10:
            continue

        min_val = min(synth_values.min(), real_values.min())
        max_val = max(synth_values.max(), real_values.max())
        # Guard against degenerate distributions (all values identical)
        if not np.isfinite(min_val) or not np.isfinite(max_val):
            continue
        if np.isclose(max_val, min_val, rtol=0, atol=1e-12):
            # All mass at a single point -> shift metrics are all zero by definition
            metrics[f"{feature}_kl_divergence"] = 0.0
            metrics[f"{feature}_js_distance"] = 0.0
            metrics[f"{feature}_wasserstein"] = 0.0
            metrics[f"{feature}_ks_statistic"] = 0.0
            metrics[f"{feature}_ks_pvalue"] = 1.0
            continue
        bins = np.linspace(min_val, max_val, 50)

        # Use density=False to get counts, then normalize to probabilities
        hist_synth, _ = np.histogram(synth_values, bins=bins, density=False)
        hist_real, _ = np.histogram(real_values, bins=bins, density=False)

        # Add small epsilon to avoid log(0) in KL divergence
        epsilon = 1e-10
        hist_synth = hist_synth + epsilon
        hist_real = hist_real + epsilon
        # Normalize to create probability distributions (sum to 1)
        hist_synth = hist_synth / hist_synth.sum()
        hist_real = hist_real / hist_real.sum()

        # KL(synthetic||real): measures how much synthetic diverges from real
        metrics[f"{feature}_kl_divergence"] = float(entropy(hist_synth, hist_real))
        # JS distance (square root of JS divergence) is symmetric
        metrics[f"{feature}_js_distance"] = float(jensenshannon(hist_synth, hist_real))
        metrics[f"{feature}_wasserstein"] = float(
            stats.wasserstein_distance(synth_values, real_values)
        )

        ks_stat, ks_pval = stats.ks_2samp(synth_values, real_values)
        metrics[f"{feature}_ks_statistic"] = float(ks_stat)
        metrics[f"{feature}_ks_pvalue"] = float(ks_pval)

    # Validate distribution shift metrics bounds
    _validate_distribution_metrics(metrics)

    return metrics


def _validate_distribution_metrics(metrics: Dict[str, float]) -> None:
    """Validate mathematical bounds of distribution shift metrics."""
    for key, value in metrics.items():
        if not np.isfinite(value):
            raise AssertionError(f"Distribution metric {key} is not finite: {value}")

        # KL divergence must be >= 0
        if "kl_divergence" in key and value < 0:
            raise AssertionError(f"KL divergence {key} must be >= 0, got {value:.6f}")

        # JS distance must be in [0, 1]
        if "js_distance" in key:
            if not (0 <= value <= 1):
                raise AssertionError(
                    f"JS distance {key} must be in [0,1], got {value:.6f}"
                )

        # Wasserstein distance must be >= 0
        if "wasserstein" in key and value < 0:
            raise AssertionError(
                f"Wasserstein distance {key} must be >= 0, got {value:.6f}"
            )

        # KS statistic must be in [0, 1]
        if "ks_statistic" in key:
            if not (0 <= value <= 1):
                raise AssertionError(
                    f"KS statistic {key} must be in [0,1], got {value:.6f}"
                )

        # p-values must be in [0, 1]
        if "pvalue" in key:
            if not (0 <= value <= 1):
                raise AssertionError(f"p-value {key} must be in [0,1], got {value:.6f}")


def statistical_hypothesis_tests(
    df: pd.DataFrame, *, adjust_method: str = "none", seed: int = 42
) -> Dict[str, Any]:
    """Statistical hypothesis tests (Spearman, Kruskal-Wallis, Mann-Whitney).

    Parameters
    ----------
    df : pd.DataFrame
        Synthetic samples.
    adjust_method : {'none','benjamini_hochberg'}
        Optional p-value multiple test correction.
    seed : int
        Random seed for bootstrap resampling.
    """
    results = {}
    alpha = 0.05

    # Test 1: Idle correlation
    idle_mask = df["reward_idle"] != 0
    if idle_mask.sum() >= 30:
        idle_dur = df.loc[idle_mask, "idle_duration"]
        idle_rew = df.loc[idle_mask, "reward_idle"]

        rho, p_val = stats.spearmanr(idle_dur, idle_rew)

        # Bootstrap CI: resample pairs to preserve bivariate structure
        bootstrap_rhos = []
        n_boot = 1000
        rng = np.random.default_rng(seed)
        for _ in range(n_boot):
            idx = rng.choice(len(idle_dur), size=len(idle_dur), replace=True)
            boot_rho, _ = stats.spearmanr(idle_dur.iloc[idx], idle_rew.iloc[idx])
            # Handle NaN case if all values identical in bootstrap sample
            if np.isfinite(boot_rho):
                bootstrap_rhos.append(boot_rho)

        if len(bootstrap_rhos) > 0:
            ci_low, ci_high = np.percentile(bootstrap_rhos, [2.5, 97.5])
        else:
            # Fallback if no valid bootstrap samples
            ci_low, ci_high = np.nan, np.nan

        results["idle_correlation"] = {
            "test": "Spearman rank correlation",
            "rho": float(rho),
            "p_value": float(p_val),
            "ci_95": (float(ci_low), float(ci_high)),
            "significant": bool(p_val < alpha),
            "interpretation": "Strong negative correlation expected"
            if rho < -0.7
            else "Check the logic",
            "n_samples": int(idle_mask.sum()),
        }

    # Test 2: Position reward differences
    position_groups = [
        df[df["position"] == pos]["reward_total"].dropna().values
        for pos in df["position"].unique()
    ]
    position_groups = [g for g in position_groups if len(g) >= 10]

    if len(position_groups) >= 2:
        h_stat, p_val = stats.kruskal(*position_groups)
        n_total = sum(len(g) for g in position_groups)
        epsilon_sq = h_stat / (n_total - 1) if n_total > 1 else 0.0

        results["position_reward_difference"] = {
            "test": "Kruskal-Wallis H",
            "statistic": float(h_stat),
            "p_value": float(p_val),
            "significant": bool(p_val < alpha),
            "effect_size_epsilon_sq": float(epsilon_sq),
            "interpretation": "Large effect"
            if epsilon_sq > 0.14
            else "Medium effect"
            if epsilon_sq > 0.06
            else "Small effect",
            "n_groups": len(position_groups),
        }

    # Test 3: PnL sign differences
    pnl_positive = df[df["pnl"] > 0]["reward_total"].dropna()
    pnl_negative = df[df["pnl"] < 0]["reward_total"].dropna()

    if len(pnl_positive) >= 30 and len(pnl_negative) >= 30:
        u_stat, p_val = stats.mannwhitneyu(pnl_positive, pnl_negative)

        results["pnl_sign_reward_difference"] = {
            "test": "Mann-Whitney U (pnl+ vs pnl-)",
            "statistic": float(u_stat),
            "p_value": float(p_val),
            "significant": bool(p_val < alpha),
            "median_pnl_positive": float(pnl_positive.median()),
            "median_pnl_negative": float(pnl_negative.median()),
        }

    # Optional multiple testing correction (Benjamini-Hochberg)
    if adjust_method not in {"none", "benjamini_hochberg"}:
        raise ValueError(
            "Unsupported adjust_method. Use 'none' or 'benjamini_hochberg'."
        )
    if adjust_method == "benjamini_hochberg" and results:
        # Collect p-values
        items = list(results.items())
        pvals = np.array([v[1]["p_value"] for v in items])
        m = len(pvals)
        order = np.argsort(pvals)
        ranks = np.empty_like(order)
        ranks[order] = np.arange(1, m + 1)
        adj = pvals * m / ranks
        # enforce monotonicity
        adj_sorted = np.minimum.accumulate(adj[order][::-1])[::-1]
        adj_final = np.empty_like(adj_sorted)
        adj_final[order] = np.clip(adj_sorted, 0, 1)
        # Attach adjusted p-values and recompute significance
        for (name, res), p_adj in zip(items, adj_final):
            res["p_value_adj"] = float(p_adj)
            res["significant_adj"] = bool(p_adj < alpha)
            results[name] = res

    # Validate hypothesis test results
    _validate_hypothesis_test_results(results)

    return results


def _validate_hypothesis_test_results(results: Dict[str, Any]) -> None:
    """Validate statistical properties of hypothesis test results."""
    for test_name, result in results.items():
        # All p-values must be in [0, 1] or NaN (for cases like constant input)
        if "p_value" in result:
            p_val = result["p_value"]
            if not (np.isnan(p_val) or (0 <= p_val <= 1)):
                raise AssertionError(
                    f"Invalid p-value for {test_name}: {p_val:.6f} not in [0,1] or NaN"
                )

        # Adjusted p-values must also be in [0, 1] or NaN
        if "p_value_adj" in result:
            p_adj = result["p_value_adj"]
            if not (np.isnan(p_adj) or (0 <= p_adj <= 1)):
                raise AssertionError(
                    f"Invalid adjusted p-value for {test_name}: {p_adj:.6f} not in [0,1] or NaN"
                )

        # Effect sizes must be finite and in valid ranges
        if "effect_size_epsilon_sq" in result:
            epsilon_sq = result["effect_size_epsilon_sq"]
            if not np.isfinite(epsilon_sq) or epsilon_sq < 0:
                raise AssertionError(
                    f"Invalid ε² for {test_name}: {epsilon_sq:.6f} (must be finite and >= 0)"
                )

        if "effect_size_rank_biserial" in result:
            rb_corr = result["effect_size_rank_biserial"]
            if not np.isfinite(rb_corr) or not (-1 <= rb_corr <= 1):
                raise AssertionError(
                    f"Invalid rank-biserial correlation for {test_name}: {rb_corr:.6f} "
                    f"(must be finite and in [-1,1])"
                )

        # Correlation coefficients must be in [-1, 1]
        if "rho" in result:
            rho = result["rho"]
            if np.isfinite(rho) and not (-1 <= rho <= 1):
                raise AssertionError(
                    f"Invalid correlation coefficient for {test_name}: {rho:.6f} "
                    f"not in [-1,1]"
                )

        # Confidence intervals must be properly ordered
        if (
            "ci_95" in result
            and isinstance(result["ci_95"], (tuple, list))
            and len(result["ci_95"]) == 2
        ):
            ci_low, ci_high = result["ci_95"]
            if np.isfinite(ci_low) and np.isfinite(ci_high) and ci_low > ci_high:
                raise AssertionError(
                    f"Invalid CI ordering for {test_name}: [{ci_low:.6f}, {ci_high:.6f}]"
                )


def bootstrap_confidence_intervals(
    df: pd.DataFrame,
    metrics: List[str],
    n_bootstrap: int = 10000,
    confidence_level: float = 0.95,
    seed: int = 42,
    *,
    strict_diagnostics: bool = False,
) -> Dict[str, Tuple[float, float, float]]:
    """Estimate confidence intervals for mean of selected metrics via bootstrap.

    Graceful mode policy (``strict_diagnostics=False``):
      - If the computed CI has zero or negative width (including inverted bounds),
        it is automatically widened symmetrically around the sample mean by an
        epsilon (``INTERNAL_GUARDS['degenerate_ci_epsilon']``) so the reporting
        pipeline is not interrupted. A ``RewardDiagnosticsWarning`` is emitted to
        make the adjustment transparent.
    Strict mode policy (``strict_diagnostics=True``):
      - The same anomalous condition triggers an immediate ``AssertionError``
        (fail-fast) to surface upstream causes such as a constant column or a
        prior data sanitization issue.
    Rationale: This dual mode avoids silently masking structural problems while
    still supporting uninterrupted exploratory / CLI smoke runs.

    Purpose
    -------
    Provide non-parametric uncertainty estimates for the mean of reward-related
    metrics, robust to unknown or asymmetric distributions.

    Parameters
    ----------
    df : pd.DataFrame
        Source dataframe containing metric columns.
    metrics : List[str]
        Names of numeric columns to evaluate; silently skipped if absent.
    n_bootstrap : int, default 10000
        Number of bootstrap resamples (with replacement).
    confidence_level : float, default 0.95
        Nominal coverage probability for the two-sided interval.
    seed : int, default 42
        Seed for local reproducible RNG (does not mutate global state).

    Returns
    -------
    Dict[str, Tuple[float, float, float]]
        Mapping metric -> (mean_estimate, ci_lower, ci_upper).

    Notes
    -----
    - Metrics with < 10 non-null observations are skipped to avoid unstable CIs.
    - Percentile method used; no bias correction or acceleration applied.
    - Validation enforces finite, ordered, positive-width intervals.
    """
    alpha = 1 - confidence_level
    lower_percentile = 100 * alpha / 2
    upper_percentile = 100 * (1 - alpha / 2)

    results = {}

    # Advisory: very low bootstrap counts produce unstable CI widths
    min_rec = int(INTERNAL_GUARDS.get("bootstrap_min_recommended", 200))
    if n_bootstrap < min_rec:
        warnings.warn(
            f"n_bootstrap={n_bootstrap} < recommended minimum {min_rec}; confidence intervals may be unstable",
            RewardDiagnosticsWarning,
        )

    # Local RNG to avoid mutating global NumPy RNG state
    rng = np.random.default_rng(seed)

    for metric in metrics:
        if metric not in df.columns:
            continue

        data = df[metric].dropna()
        if len(data) < 10:
            continue

        point_est = float(data.mean())

        bootstrap_means = []
        data_array = data.values  # speed
        n = len(data_array)
        for _ in range(n_bootstrap):
            indices = rng.integers(0, n, size=n)
            bootstrap_means.append(float(data_array[indices].mean()))

        ci_low = float(np.percentile(bootstrap_means, lower_percentile))
        ci_high = float(np.percentile(bootstrap_means, upper_percentile))

        results[metric] = (point_est, ci_low, ci_high)

    # Validate bootstrap confidence intervals
    _validate_bootstrap_results(results, strict_diagnostics=strict_diagnostics)

    return results


def _validate_bootstrap_results(
    results: Dict[str, Tuple[float, float, float]], *, strict_diagnostics: bool
) -> None:
    """Validate structural and numerical integrity of bootstrap CI outputs.

    Purpose
    -------
    Fail fast if any generated confidence interval violates expected invariants.

    Parameters
    ----------
    results : Dict[str, Tuple[float, float, float]]
        Mapping from metric name to (mean, ci_low, ci_high).

    Returns
    -------
    None

    Notes
    -----
    - Raises AssertionError on first violation (finite bounds, ordering, width > 0).
    - Intentionally internal; external callers should rely on exceptions for flow.
    """
    for metric, (mean, ci_low, ci_high) in results.items():
        # CI bounds must be finite
        if not (np.isfinite(mean) and np.isfinite(ci_low) and np.isfinite(ci_high)):
            raise AssertionError(
                f"Bootstrap CI for {metric}: non-finite values "
                f"(mean={mean}, ci_low={ci_low}, ci_high={ci_high})"
            )

        # CI must be properly ordered
        if not (ci_low <= mean <= ci_high):
            raise AssertionError(
                f"Bootstrap CI for {metric}: ordering violation "
                f"({ci_low:.6f} <= {mean:.6f} <= {ci_high:.6f})"
            )

        # CI width should be positive (non-degenerate)
        width = ci_high - ci_low
        if width <= 0:
            if strict_diagnostics:
                raise AssertionError(
                    f"Bootstrap CI for {metric}: non-positive width {width:.6f}"
                )
            # Graceful mode: expand interval symmetrically
            epsilon = (
                INTERNAL_GUARDS.get("degenerate_ci_epsilon", 1e-9)
                if width == 0
                else abs(width) * 1e-6
            )
            center = mean
            # Adjust only if current bounds are identical; otherwise enforce ordering minimally.
            if ci_low == ci_high:
                ci_low = center - epsilon
                ci_high = center + epsilon
            else:
                # Ensure proper ordering if inverted or collapsed negatively.
                lower = min(ci_low, ci_high) - epsilon
                upper = max(ci_low, ci_high) + epsilon
                ci_low, ci_high = lower, upper
            results[metric] = (mean, ci_low, ci_high)
            warnings.warn(
                f"Degenerate bootstrap CI for {metric} adjusted to maintain positive width;"
                f" original width={width:.6e}, epsilon={epsilon:.1e}",
                RewardDiagnosticsWarning,
            )


def distribution_diagnostics(
    df: pd.DataFrame, *, seed: int | None = None, strict_diagnostics: bool = False
) -> Dict[str, Any]:
    """Compute distributional diagnostics for selected numeric columns.

    Purpose
    -------
    Aggregate normality test statistics and moment-based shape descriptors to
    support assessment of modelling assumptions or transformation needs.

    Parameters
    ----------
    df : pd.DataFrame
        Dataframe containing relevant numeric columns.
    seed : int | None, optional
        Reserved for potential future stochastic extensions; unused presently.

    Returns
    -------
    Dict[str, Any]
        Mapping of column name -> diagnostic results (tests, moments, p-values).

    Notes
    -----
    - Skips columns absent from the dataframe.
    - Applies Shapiro-Wilk for n <= 5000 else D'Agostino's K2 due to cost.
    - All numeric outputs are floats; non-finite intermediate results are ignored.
    """
    diagnostics = {}
    _ = seed  # placeholder to keep signature for future reproducibility extensions

    for col in ["reward_total", "pnl", "trade_duration", "idle_duration"]:
        if col not in df.columns:
            continue

        data = df[col].dropna().values
        if len(data) < 10:
            continue

        diagnostics[f"{col}_mean"] = float(np.mean(data))
        diagnostics[f"{col}_std"] = float(np.std(data, ddof=1))
        skew_v = float(stats.skew(data))
        kurt_v = float(stats.kurtosis(data, fisher=True))
        diagnostics[f"{col}_skewness"] = skew_v
        diagnostics[f"{col}_kurtosis"] = kurt_v
        thr = INTERNAL_GUARDS.get("moment_extreme_threshold", 1e4)
        if abs(skew_v) > thr or abs(kurt_v) > thr:
            msg = f"Extreme moment(s) for {col}: skew={skew_v:.3e}, kurtosis={kurt_v:.3e} exceeds threshold {thr}."
            if strict_diagnostics:
                raise AssertionError(msg)
            warnings.warn(msg, RewardDiagnosticsWarning)

        if len(data) < 5000:
            sw_stat, sw_pval = stats.shapiro(data)
            diagnostics[f"{col}_shapiro_stat"] = float(sw_stat)
            diagnostics[f"{col}_shapiro_pval"] = float(sw_pval)
            diagnostics[f"{col}_is_normal_shapiro"] = bool(sw_pval > 0.05)

        ad_result = stats.anderson(data, dist="norm")
        diagnostics[f"{col}_anderson_stat"] = float(ad_result.statistic)
        diagnostics[f"{col}_anderson_critical_5pct"] = float(
            ad_result.critical_values[2]
        )
        diagnostics[f"{col}_is_normal_anderson"] = bool(
            ad_result.statistic < ad_result.critical_values[2]
        )

        from scipy.stats import probplot

        (_osm, _osr), (_slope, _intercept, r) = probplot(data, dist="norm", plot=None)
        diagnostics[f"{col}_qq_r_squared"] = float(r**2)

    _validate_distribution_diagnostics(
        diagnostics, strict_diagnostics=strict_diagnostics
    )
    return diagnostics


def _validate_distribution_diagnostics(
    diag: Dict[str, Any], *, strict_diagnostics: bool
) -> None:
    """Validate mathematical properties of distribution diagnostics.

    Ensures all reported statistics are finite and within theoretical bounds where applicable.
    Invoked automatically inside distribution_diagnostics(); raising AssertionError on violation
    enforces fail-fast semantics consistent with other validation helpers.
    """
    # Pre-compute zero-variance flags to allow graceful handling of undefined higher moments.
    zero_var_columns = set()
    for k, v in diag.items():
        if k.endswith("_std") and (not np.isfinite(v) or v == 0):
            prefix = k[: -len("_std")]
            zero_var_columns.add(prefix)

    for key, value in list(diag.items()):
        if any(suffix in key for suffix in ["_mean", "_std", "_skewness", "_kurtosis"]):
            if not np.isfinite(value):
                # Graceful degradation for constant distributions: skewness/kurtosis become NaN.
                constant_problem = any(
                    key.startswith(prefix)
                    and (key.endswith("_skewness") or key.endswith("_kurtosis"))
                    for prefix in zero_var_columns
                )
                if constant_problem and not strict_diagnostics:
                    fallback = INTERNAL_GUARDS.get(
                        "distribution_constant_fallback_moment", 0.0
                    )
                    diag[key] = fallback
                    warnings.warn(
                        f"Replaced undefined {key} (constant distribution) with {fallback}",
                        RewardDiagnosticsWarning,
                    )
                else:
                    raise AssertionError(
                        f"Distribution diagnostic {key} is not finite: {value}"
                    )
        if key.endswith("_shapiro_pval"):
            if not (0 <= value <= 1):
                raise AssertionError(
                    f"Shapiro p-value {key} must be in [0,1], got {value}"
                )
        if key.endswith("_anderson_stat") or key.endswith("_anderson_critical_5pct"):
            if not np.isfinite(value):
                prefix = key.rsplit("_", 2)[0]
                if prefix in zero_var_columns and not strict_diagnostics:
                    fallback = INTERNAL_GUARDS.get(
                        "distribution_constant_fallback_moment", 0.0
                    )
                    diag[key] = fallback
                    warnings.warn(
                        f"Replaced undefined Anderson diagnostic {key} (constant distribution) with {fallback}",
                        RewardDiagnosticsWarning,
                    )
                    continue
                raise AssertionError(
                    f"Anderson statistic {key} must be finite, got {value}"
                )
        if key.endswith("_qq_r_squared"):
            if not (
                isinstance(value, (int, float))
                and np.isfinite(value)
                and 0 <= value <= 1
            ):
                prefix = key[: -len("_qq_r_squared")]
                if prefix in zero_var_columns and not strict_diagnostics:
                    fallback_r2 = INTERNAL_GUARDS.get(
                        "distribution_constant_fallback_qq_r2", 1.0
                    )
                    diag[key] = fallback_r2
                    warnings.warn(
                        f"Replaced undefined Q-Q R^2 {key} (constant distribution) with {fallback_r2}",
                        RewardDiagnosticsWarning,
                    )
                else:
                    raise AssertionError(f"Q-Q R^2 {key} must be in [0,1], got {value}")


def build_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Synthetic stress-test of the ReforceXY reward shaping logic."
    )
    parser.add_argument(
        "--skip-feature-analysis",
        action="store_true",
        help="Skip feature importance and model-based analysis for all scenarios.",
    )
    parser.add_argument(
        "--num_samples",
        type=int,
        default=20000,
        help="Number of synthetic state/action samples to generate (default: 20000).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility (default: 42).",
    )
    parser.add_argument(
        "--skip_partial_dependence",
        action="store_true",
        help="Skip partial dependence computation to speed up analysis.",
    )
    parser.add_argument(
        "--stats_seed",
        type=int,
        default=None,
        help="Optional separate seed for statistical analyses (default: same as --seed).",
    )
    parser.add_argument(
        "--max_trade_duration",
        type=int,
        default=128,
        help="Configured trade timeout in candles (default: 128).",
    )
    parser.add_argument(
        "--base_factor",
        type=float,
        default=100.0,
        help="Base reward factor used inside the environment (default: 100).",
    )
    parser.add_argument(
        "--profit_target",
        type=float,
        default=0.03,
        help="Target profit threshold (default: 0.03).",
    )
    parser.add_argument(
        "--risk_reward_ratio",
        type=float,
        default=1.0,
        help="Risk reward ratio multiplier (default: 1.0).",
    )
    parser.add_argument(
        "--max_duration_ratio",
        type=float,
        default=2.5,
        help="Multiple of max duration used when sampling trade/idle durations.",
    )
    parser.add_argument(
        "--pnl_base_std",
        type=float,
        default=0.02,
        help="Base standard deviation for synthetic PnL generation (pre-scaling).",
    )
    parser.add_argument(
        "--pnl_duration_vol_scale",
        type=float,
        default=0.5,
        help="Scaling factor of additional PnL volatility proportional to trade duration ratio.",
    )
    parser.add_argument(
        "--trading_mode",
        type=str.lower,
        choices=["spot", "margin", "futures"],
        default="spot",
        help=("Trading mode to simulate (spot disables shorts). Default: spot."),
    )
    parser.add_argument(
        "--action_masking",
        type=str,
        choices=["true", "false", "1", "0", "yes", "no"],
        default="true",
        help="Enable action masking simulation (default: true).",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("reward_space_outputs"),
        help="Output directory for artifacts (default: reward_space_outputs).",
    )
    parser.add_argument(
        "--params",
        nargs="*",
        default=[],
        metavar="KEY=VALUE",
        help="Override reward parameters, e.g. hold_penalty_scale=0.5",
    )
    # Dynamically add CLI options for all tunables
    add_tunable_cli_args(parser)
    parser.add_argument(
        "--real_episodes",
        type=Path,
        default=None,
        help="Path to real episodes pickle for distribution shift analysis (optional).",
    )
    parser.add_argument(
        "--pvalue_adjust",
        type=str.lower,
        choices=["none", "benjamini_hochberg"],
        default="none",
        help="Multiple testing correction method for hypothesis tests (default: none).",
    )
    parser.add_argument(
        "--strict_diagnostics",
        action="store_true",
        help=(
            "Enable fail-fast mode for statistical diagnostics: raise on zero-width bootstrap CIs or undefined "
            "skewness/kurtosis/Anderson/Q-Q metrics produced by constant distributions instead of applying graceful replacements."
        ),
    )
    parser.add_argument(
        "--bootstrap_resamples",
        type=int,
        default=10000,
        metavar="N",
        help=(
            "Number of bootstrap resamples for confidence intervals (default: 10000). "
            "Lower this (e.g. 200-1000) for faster smoke tests; increase for more stable CI width estimates."
        ),
    )
    return parser


def write_complete_statistical_analysis(
    df: pd.DataFrame,
    output_dir: Path,
    max_trade_duration: int,
    profit_target: float,
    seed: int,
    real_df: Optional[pd.DataFrame] = None,
    *,
    adjust_method: str = "none",
    stats_seed: Optional[int] = None,
    strict_diagnostics: bool = False,
    bootstrap_resamples: int = 10000,
    skip_partial_dependence: bool = False,
    skip_feature_analysis: bool = False,
) -> None:
    """Generate a single comprehensive statistical analysis report with enhanced tests."""
    output_dir.mkdir(parents=True, exist_ok=True)
    report_path = output_dir / "statistical_analysis.md"

    # Helpers: consistent Markdown table renderers
    def _fmt_val(v: Any, ndigits: int = 6) -> str:
        try:
            if isinstance(v, (int, np.integer)):
                return f"{int(v)}"
            if isinstance(v, (float, np.floating)):
                if np.isnan(v):
                    return "NaN"
                return f"{float(v):.{ndigits}f}"
            return str(v)
        except Exception:
            return str(v)

    def _series_to_md(
        series: pd.Series, value_name: str = "value", ndigits: int = 6
    ) -> str:
        lines = [f"| Metric | {value_name} |", "|--------|-----------|"]
        for k, v in series.items():
            lines.append(f"| {k} | {_fmt_val(v, ndigits)} |")
        return "\n".join(lines) + "\n\n"

    def _df_to_md(df: pd.DataFrame, index_name: str = "index", ndigits: int = 6) -> str:
        if df is None or df.empty:
            return "_No data._\n\n"
        # Prepare header
        cols = list(df.columns)
        header = "| " + index_name + " | " + " | ".join(cols) + " |\n"
        sep = "|" + "-" * (len(index_name) + 2)
        for c in cols:
            sep += "|" + "-" * (len(str(c)) + 2)
        sep += "|\n"
        # Rows
        rows = []
        for idx, row in df.iterrows():
            vals = [_fmt_val(row[c], ndigits) for c in cols]
            rows.append("| " + str(idx) + " | " + " | ".join(vals) + " |")
        return header + sep + "\n".join(rows) + "\n\n"

    # Compute all statistics
    summary_stats = _compute_summary_stats(df)
    relationship_stats = _compute_relationship_stats(df, max_trade_duration)
    representativity_stats = _compute_representativity_stats(
        df, profit_target, max_trade_duration
    )

    # Model analysis: skip if requested or not enough samples
    importance_df = None
    analysis_stats = None
    partial_deps = {}
    if skip_feature_analysis or len(df) < 4:
        print("Skipping feature analysis: flag set or insufficient samples (<4).")
    else:
        importance_df, analysis_stats, partial_deps, _model = _perform_feature_analysis(
            df, seed, skip_partial_dependence=skip_partial_dependence
        )
        # Save feature importance CSV
        importance_df.to_csv(output_dir / "feature_importance.csv", index=False)
        # Save partial dependence CSVs
        if not skip_partial_dependence:
            for feature, pd_df in partial_deps.items():
                pd_df.to_csv(
                    output_dir / f"partial_dependence_{feature}.csv",
                    index=False,
                )

    # Enhanced statistics
    test_seed = (
        stats_seed
        if isinstance(stats_seed, int)
        else (seed if isinstance(seed, int) else 42)
    )
    hypothesis_tests = statistical_hypothesis_tests(
        df, adjust_method=adjust_method, seed=test_seed
    )
    metrics_for_ci = [
        "reward_total",
        "reward_idle",
        "reward_hold",
        "reward_exit",
        "pnl",
    ]
    bootstrap_ci = bootstrap_confidence_intervals(
        df,
        metrics_for_ci,
        n_bootstrap=int(bootstrap_resamples),
        seed=test_seed,
        strict_diagnostics=strict_diagnostics,
    )
    dist_diagnostics = distribution_diagnostics(
        df, seed=test_seed, strict_diagnostics=strict_diagnostics
    )

    distribution_shift = None
    if real_df is not None:
        distribution_shift = compute_distribution_shift_metrics(df, real_df)

    # Write comprehensive report
    with report_path.open("w", encoding="utf-8") as f:
        # Header
        f.write("# Reward Space Analysis Report\n\n")
        f.write("### Run Configuration\n\n")
        f.write("| Key | Value |\n")
        f.write("|-----|-------|\n")
        f.write(f"| Generated | {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')} |\n")
        f.write(f"| Total Samples | {len(df):,} |\n")
        f.write(f"| Random Seed | {seed} |\n")
        f.write(f"| Max Trade Duration | {max_trade_duration} |\n")
        # Blank separator to visually group core simulation vs PBRS parameters
        f.write("|  |  |\n")
        # Extra core PBRS parameters exposed in run configuration if present
        _rp = (
            df.attrs.get("reward_params")
            if isinstance(df.attrs.get("reward_params"), dict)
            else {}
        )
        exit_mode = _rp.get("exit_potential_mode", "canonical")
        potential_gamma = _rp.get(
            "potential_gamma",
            DEFAULT_MODEL_REWARD_PARAMETERS.get(
                "potential_gamma", POTENTIAL_GAMMA_DEFAULT
            ),
        )
        f.write(f"| exit_potential_mode | {exit_mode} |\n")
        f.write(f"| potential_gamma | {potential_gamma} |\n")
        # Blank separator before overrides block
        f.write("|  |  |\n")

        overrides_pairs = []
        if _rp:
            for k, default_v in DEFAULT_MODEL_REWARD_PARAMETERS.items():
                if k in ("exit_potential_mode", "potential_gamma"):
                    continue  # already printed explicitly
                try:
                    if k in _rp and _rp[k] != default_v:
                        overrides_pairs.append(f"{k}={_rp[k]}")
                except Exception:
                    continue
        if overrides_pairs:
            f.write(f"| Overrides | {', '.join(sorted(overrides_pairs))} |\n")
        else:
            f.write("| Overrides | (none) |\n")
        f.write("\n")

        f.write("---\n\n")

        # Section 1: Global Statistics
        f.write("## 1. Global Statistics\n\n")

        f.write("### 1.1 Reward Distribution\n\n")
        f.write(
            _series_to_md(
                summary_stats["global_stats"], value_name="reward_total", ndigits=6
            )
        )

        f.write("### 1.2 Reward Statistics by Action\n\n")
        action_df = summary_stats["action_summary"].copy()
        # Cast index to int for cleaner display (0,1,2,3,4)
        action_df.index = action_df.index.astype(int)
        if action_df.index.name is None:
            action_df.index.name = "action"
        f.write(_df_to_md(action_df, index_name=action_df.index.name, ndigits=6))

        f.write("### 1.3 Component Activation Rates\n\n")
        f.write("Percentage of samples where each reward component is non-zero:\n\n")
        comp_share = summary_stats["component_share"].copy()
        formatted_rows = [
            "| Component | Activation Rate |",
            "|-----------|----------------|",
        ]
        # Enforce deterministic logical ordering of components if present
        preferred_order = [
            "reward_invalid",
            "reward_idle",
            "reward_hold",
            "reward_exit",
            "reward_shaping",
            "reward_entry_additive",
            "reward_exit_additive",
        ]
        for comp in preferred_order:
            if comp in comp_share.index:
                val = comp_share.loc[comp]
                formatted_rows.append(f"| {comp} | {val:.1%} |")
        f.write("\n".join(formatted_rows) + "\n\n")

        f.write("### 1.4 Component Value Ranges\n\n")
        bounds_df = summary_stats["component_bounds"].copy()
        if bounds_df.index.name is None:
            bounds_df.index.name = "component"
        f.write(_df_to_md(bounds_df, index_name=bounds_df.index.name, ndigits=6))

        # Section 2: Representativity Analysis
        f.write("---\n\n")
        f.write("## 2. Sample Representativity\n\n")
        f.write(
            "This section evaluates whether the synthetic samples adequately represent "
        )
        f.write("the full reward space across different market scenarios.\n\n")

        f.write("### 2.1 Position Distribution\n\n")
        f.write(
            _series_to_md(
                representativity_stats["pos_counts"], value_name="count", ndigits=0
            )
        )

        f.write("### 2.2 Action Distribution\n\n")
        f.write(
            _series_to_md(
                representativity_stats["act_counts"], value_name="count", ndigits=0
            )
        )

        f.write("### 2.3 Critical Regime Coverage\n\n")
        f.write("| Regime | Coverage |\n")
        f.write("|--------|----------|\n")
        f.write(
            f"| PnL > target | {representativity_stats['pnl_above_target']:.1%} |\n"
        )
        f.write(
            f"| PnL near target (±20%) | {representativity_stats['pnl_near_target']:.1%} |\n"
        )
        f.write(
            f"| Duration overage (>1.0) | {representativity_stats['duration_overage_share']:.1%} |\n"
        )
        f.write(
            f"| Extreme PnL (\\|pnl\\|≥0.14) | {representativity_stats['pnl_extreme']:.1%} |\n"
        )
        f.write("\n")

        f.write("### 2.4 Component Activation Rates\n\n")
        f.write("| Component | Activation Rate |\n")
        f.write("|-----------|----------------|\n")
        f.write(f"| Idle penalty | {representativity_stats['idle_activated']:.1%} |\n")
        f.write(f"| Hold penalty | {representativity_stats['hold_activated']:.1%} |\n")
        f.write(f"| Exit reward | {representativity_stats['exit_activated']:.1%} |\n")
        f.write("\n")

        # Section 3: Reward Component Relationships
        f.write("---\n\n")
        f.write("## 3. Reward Component Analysis\n\n")
        f.write(
            "Analysis of how reward components behave under different conditions.\n\n"
        )

        f.write("### 3.1 Idle Penalty vs Duration\n\n")
        if relationship_stats["idle_stats"].empty:
            f.write("_No idle samples present._\n\n")
        else:
            idle_df = relationship_stats["idle_stats"].copy()
            if idle_df.index.name is None:
                idle_df.index.name = "bin"
            f.write(_df_to_md(idle_df, index_name=idle_df.index.name, ndigits=6))

        f.write("### 3.2 Hold Penalty vs Trade Duration\n\n")
        if relationship_stats["hold_stats"].empty:
            f.write("_No hold samples present._\n\n")
        else:
            hold_df = relationship_stats["hold_stats"].copy()
            if hold_df.index.name is None:
                hold_df.index.name = "bin"
            f.write(_df_to_md(hold_df, index_name=hold_df.index.name, ndigits=6))

        f.write("### 3.3 Exit Reward vs PnL\n\n")
        if relationship_stats["exit_stats"].empty:
            f.write("_No exit samples present._\n\n")
        else:
            exit_df = relationship_stats["exit_stats"].copy()
            if exit_df.index.name is None:
                exit_df.index.name = "bin"
            f.write(_df_to_md(exit_df, index_name=exit_df.index.name, ndigits=6))

        f.write("### 3.4 Correlation Matrix\n\n")
        f.write("Pearson correlation coefficients between key metrics:\n\n")
        corr_df = relationship_stats["correlation"].copy()
        if corr_df.index.name is None:
            corr_df.index.name = "feature"
        f.write(_df_to_md(corr_df, index_name=corr_df.index.name, ndigits=4))
        _dropped = relationship_stats.get("correlation_dropped") or []
        if _dropped:
            f.write(
                "\n_Constant features removed (no variance): "
                + ", ".join(_dropped)
                + "._\n\n"
            )

        # Section 3.5: PBRS Analysis
        f.write("### 3.5 PBRS (Potential-Based Reward Shaping) Analysis\n\n")

        # Check if PBRS components are present in the data
        pbrs_components = [
            "reward_shaping",
            "reward_entry_additive",
            "reward_exit_additive",
        ]
        pbrs_present = all(col in df.columns for col in pbrs_components)

        if pbrs_present:
            # PBRS activation rates
            pbrs_activation = {}
            for comp in pbrs_components:
                pbrs_activation[comp.replace("reward_", "")] = (df[comp] != 0).mean()

            f.write("**PBRS Component Activation Rates:**\n\n")
            f.write("| Component | Activation Rate | Description |\n")
            f.write("|-----------|-----------------|-------------|\n")
            f.write(
                f"| Shaping (Φ) | {pbrs_activation['shaping']:.1%} | Potential-based reward shaping |\n"
            )
            f.write(
                f"| Entry Additive | {pbrs_activation['entry_additive']:.1%} | Non-PBRS entry reward |\n"
            )
            f.write(
                f"| Exit Additive | {pbrs_activation['exit_additive']:.1%} | Non-PBRS exit reward |\n"
            )
            f.write("\n")

            # PBRS statistics
            f.write("**PBRS Component Statistics:**\n\n")
            pbrs_stats = df[pbrs_components].describe(
                percentiles=[0.1, 0.25, 0.5, 0.75, 0.9]
            )
            pbrs_stats_df = pbrs_stats.round(
                6
            ).T  # Transpose to make it DataFrame-compatible
            pbrs_stats_df.index.name = "component"
            f.write(_df_to_md(pbrs_stats_df, index_name="component", ndigits=6))

            # PBRS invariance check
            total_shaping = df["reward_shaping"].sum()
            entry_add_total = df.get("reward_entry_additive", pd.Series([0])).sum()
            exit_add_total = df.get("reward_exit_additive", pd.Series([0])).sum()

            # Get configuration for proper invariance assessment
            reward_params = (
                df.attrs.get("reward_params", {}) if hasattr(df, "attrs") else {}
            )
            exit_potential_mode = reward_params.get("exit_potential_mode", "canonical")
            entry_additive_enabled = reward_params.get("entry_additive_enabled", False)
            exit_additive_enabled = reward_params.get("exit_additive_enabled", False)

            # True invariance requires canonical mode AND no additives
            is_theoretically_invariant = exit_potential_mode == "canonical" and not (
                entry_additive_enabled or exit_additive_enabled
            )
            shaping_near_zero = abs(total_shaping) < PBRS_INVARIANCE_TOL

            # Prepare invariance summary markdown block
            if is_theoretically_invariant:
                if shaping_near_zero:
                    invariance_status = "✅ Canonical"
                    invariance_note = "Theoretical invariance preserved (canonical mode, no additives, Σ≈0)"
                else:
                    invariance_status = "⚠️ Canonical (with warning)"
                    invariance_note = f"Canonical mode but unexpected shaping sum = {total_shaping:.6f}"
            else:
                invariance_status = "❌ Non-canonical"
                reasons = []
                if exit_potential_mode != "canonical":
                    reasons.append(f"exit_potential_mode='{exit_potential_mode}'")
                if entry_additive_enabled or exit_additive_enabled:
                    additive_types = []
                    if entry_additive_enabled:
                        additive_types.append("entry")
                    if exit_additive_enabled:
                        additive_types.append("exit")
                    reasons.append(f"additives={additive_types}")
                invariance_note = f"Modified for flexibility: {', '.join(reasons)}"

            # Summarize PBRS invariance
            f.write("**PBRS Invariance Summary:**\n\n")
            f.write("| Field | Value |\n")
            f.write("|-------|-------|\n")
            f.write(f"| Invariance Status | {invariance_status} |\n")
            f.write(f"| Analysis Note | {invariance_note} |\n")
            f.write(f"| Exit Potential Mode | {exit_potential_mode} |\n")
            f.write(f"| Entry Additive Enabled | {entry_additive_enabled} |\n")
            f.write(f"| Exit Additive Enabled | {exit_additive_enabled} |\n")
            f.write(f"| Σ Shaping Reward | {total_shaping:.6f} |\n")
            f.write(f"| Abs Σ Shaping Reward | {abs(total_shaping):.6e} |\n")
            f.write(f"| Σ Entry Additive | {entry_add_total:.6f} |\n")
            f.write(f"| Σ Exit Additive | {exit_add_total:.6f} |\n\n")

        else:
            f.write("_PBRS components not present in this analysis._\n\n")

        # Section 4: Feature Importance Analysis
        f.write("---\n\n")
        f.write("## 4. Feature Importance\n\n")
        if skip_feature_analysis or len(df) < 4:
            reason = []
            if skip_feature_analysis:
                reason.append("flag --skip-feature-analysis set")
            if len(df) < 4:
                reason.append("insufficient samples <4")
            reason_str = "; ".join(reason) if reason else "skipped"
            f.write(f"_Skipped ({reason_str})._\n\n")
            if skip_partial_dependence:
                f.write(
                    "_Note: --skip_partial_dependence is redundant when feature analysis is skipped._\n\n"
                )
        else:
            f.write(
                "Machine learning analysis to identify which features most influence total reward.\n\n"
            )
            f.write("**Model:** Random Forest Regressor (400 trees)  \n")
            f.write(f"**R² Score:** {analysis_stats['r2_score']:.4f}\n\n")

            f.write("### 4.1 Top 10 Features by Importance\n\n")
            top_imp = importance_df.head(10).copy().reset_index(drop=True)
            # Render as markdown without index column
            header = "| feature | importance_mean | importance_std |\n"
            sep = "|---------|------------------|----------------|\n"
            rows = []
            for _, r in top_imp.iterrows():
                rows.append(
                    f"| {r['feature']} | {_fmt_val(r['importance_mean'], 6)} | {_fmt_val(r['importance_std'], 6)} |"
                )
            f.write(header + sep + "\n".join(rows) + "\n\n")
            f.write("**Exported Data:**\n")
            f.write("- Full feature importance: `feature_importance.csv`\n")
            if not skip_partial_dependence:
                f.write("- Partial dependence plots: `partial_dependence_*.csv`\n\n")
            else:
                f.write(
                    "- Partial dependence plots: (skipped via --skip_partial_dependence)\n\n"
                )

        # Section 5: Statistical Validation
        if hypothesis_tests:
            f.write("---\n\n")
            f.write("## 5. Statistical Validation\n\n")
            f.write(
                "Rigorous statistical tests to validate reward behavior and relationships.\n\n"
            )

            f.write("### 5.1 Hypothesis Tests\n\n")

            if "idle_correlation" in hypothesis_tests:
                h = hypothesis_tests["idle_correlation"]
                f.write("#### 5.1.1 Idle Duration → Idle Penalty Correlation\n\n")
                f.write(f"**Test Method:** {h['test']}\n\n")
                f.write(f"- Spearman ρ: **{h['rho']:.4f}**\n")
                f.write(f"- p-value: {h['p_value']:.4g}\n")
                if "p_value_adj" in h:
                    f.write(
                        f"- p-value (adj BH): {h['p_value_adj']:.4g} -> {'✅ Yes' if h['significant_adj'] else '❌ No'} (α=0.05)\n"
                    )
                f.write(f"- 95% CI: [{h['ci_95'][0]:.4f}, {h['ci_95'][1]:.4f}]\n")
                f.write(f"- Sample size: {h['n_samples']:,}\n")
                f.write(
                    f"- Significant (α=0.05): {'✅ Yes' if h['significant'] else '❌ No'}\n"
                )
                f.write(f"- **Interpretation:** {h['interpretation']}\n\n")

            if "position_reward_difference" in hypothesis_tests:
                h = hypothesis_tests["position_reward_difference"]
                f.write("#### 5.1.2 Position-Based Reward Differences\n\n")
                f.write(f"**Test Method:** {h['test']}\n\n")
                f.write(f"- H-statistic: **{h['statistic']:.4f}**\n")
                f.write(f"- p-value: {h['p_value']:.4g}\n")
                if "p_value_adj" in h:
                    f.write(
                        f"- p-value (adj BH): {h['p_value_adj']:.4g} -> {'✅ Yes' if h['significant_adj'] else '❌ No'} (α=0.05)\n"
                    )
                f.write(f"- Effect size (ε²): {h['effect_size_epsilon_sq']:.4f}\n")
                f.write(f"- Number of groups: {h['n_groups']}\n")
                f.write(
                    f"- Significant (α=0.05): {'✅ Yes' if h['significant'] else '❌ No'}\n"
                )
                f.write(f"- **Interpretation:** {h['interpretation']} effect\n\n")

            if "pnl_sign_reward_difference" in hypothesis_tests:
                h = hypothesis_tests["pnl_sign_reward_difference"]
                f.write("#### 5.1.3 Positive vs Negative PnL Comparison\n\n")
                f.write(f"**Test Method:** {h['test']}\n\n")
                f.write(f"- U-statistic: **{h['statistic']:.4f}**\n")
                f.write(f"- p-value: {h['p_value']:.4g}\n")
                if "p_value_adj" in h:
                    f.write(
                        f"- p-value (adj BH): {h['p_value_adj']:.4g} -> {'✅ Yes' if h['significant_adj'] else '❌ No'} (α=0.05)\n"
                    )
                f.write(f"- Median (PnL+): {h['median_pnl_positive']:.4f}\n")
                f.write(f"- Median (PnL-): {h['median_pnl_negative']:.4f}\n")
                f.write(
                    f"- Significant (α=0.05): {'✅ Yes' if h['significant'] else '❌ No'}\n\n"
                )

            # Bootstrap CI
            if bootstrap_ci:
                f.write("### 5.2 Confidence Intervals\n\n")
                f.write(
                    "Bootstrap confidence intervals (95%, 10,000 resamples) for key metrics:\n\n"
                )
                f.write("| Metric | Mean | 95% CI Lower | 95% CI Upper | Width |\n")
                f.write("|--------|------|--------------|--------------|-------|\n")
                for metric, (mean, ci_low, ci_high) in bootstrap_ci.items():
                    width = ci_high - ci_low
                    f.write(
                        f"| {metric} | {mean:.6f} | {ci_low:.6f} | {ci_high:.6f} | {width:.6f} |\n"
                    )
                f.write("\n")

            # Distribution diagnostics
            if dist_diagnostics:
                f.write("### 5.3 Distribution Normality Tests\n\n")
                f.write("Statistical tests for normality of key distributions:\n\n")
                for col in ["reward_total", "pnl", "trade_duration"]:
                    if f"{col}_mean" in dist_diagnostics:
                        f.write(f"#### {col.replace('_', ' ').title()}\n\n")
                        f.write("| Metric | Value |\n")
                        f.write("|--------|-------|\n")
                        f.write(f"| Mean | {dist_diagnostics[f'{col}_mean']:.4f} |\n")
                        f.write(f"| Std Dev | {dist_diagnostics[f'{col}_std']:.4f} |\n")
                        f.write(
                            f"| Skewness | {dist_diagnostics[f'{col}_skewness']:.4f} |\n"
                        )
                        f.write(
                            f"| Kurtosis | {dist_diagnostics[f'{col}_kurtosis']:.4f} |\n"
                        )
                        if f"{col}_shapiro_pval" in dist_diagnostics:
                            is_normal = (
                                "✅ Yes"
                                if dist_diagnostics[f"{col}_is_normal_shapiro"]
                                else "❌ No"
                            )
                            f.write(
                                f"| Normal? (Shapiro-Wilk) | {is_normal} (p={dist_diagnostics[f'{col}_shapiro_pval']:.4e}) |\n"
                            )
                        if f"{col}_qq_r_squared" in dist_diagnostics:
                            f.write(
                                f"| Q-Q Plot R² | {dist_diagnostics[f'{col}_qq_r_squared']:.4f} |\n"
                            )
                        f.write("\n")

            # Distribution shift (if real data provided)
            if distribution_shift:
                f.write("### 5.4 Distribution Shift Analysis\n\n")
                f.write("Comparison between synthetic and real data distributions:\n\n")
                f.write(
                    "| Feature | KL Div | JS Dist | Wasserstein | KS Stat | KS p-value |\n"
                )
                f.write(
                    "|---------|--------|---------|-------------|---------|------------|\n"
                )

                features = ["pnl", "trade_duration", "idle_duration"]
                for feature in features:
                    kl = distribution_shift.get(
                        f"{feature}_kl_divergence", float("nan")
                    )
                    js = distribution_shift.get(f"{feature}_js_distance", float("nan"))
                    ws = distribution_shift.get(f"{feature}_wasserstein", float("nan"))
                    ks_stat = distribution_shift.get(
                        f"{feature}_ks_statistic", float("nan")
                    )
                    ks_p = distribution_shift.get(f"{feature}_ks_pvalue", float("nan"))

                    f.write(
                        f"| {feature} | {kl:.4f} | {js:.4f} | {ws:.4f} | {ks_stat:.4f} | {ks_p:.4g} |\n"
                    )
                f.write("\n")
                f.write("**Interpretation Guide:**\n\n")
                f.write("| Metric | Threshold | Meaning |\n")
                f.write("|--------|-----------|--------|\n")
                f.write("| KL Divergence | < 0.3 | ✅ Yes: Good representativeness |\n")
                f.write("| JS Distance | < 0.2 | ✅ Yes: Similar distributions |\n")
                f.write(
                    "| KS p-value | > 0.05 | ✅ Yes: No significant difference |\n\n"
                )
            else:
                # Placeholder keeps numbering stable and explicit
                f.write("### 5.4 Distribution Shift Analysis\n\n")
                f.write("_Not performed (no real episodes provided)._\n\n")

        # Footer
        f.write("---\n\n")
        f.write("## Summary\n\n")
        f.write("This comprehensive report includes:\n\n")
        f.write(
            "1. **Global Statistics** - Overall reward distributions and component activation\n"
        )
        f.write(
            "2. **Sample Representativity** - Coverage of critical market scenarios\n"
        )
        f.write(
            "3. **Component Analysis** - Relationships between rewards and conditions (including PBRS)\n"
        )
        if skip_feature_analysis or len(df) < 4:
            f.write(
                "4. **Feature Importance** - (skipped) Machine learning analysis of key drivers\n"
            )
        else:
            f.write(
                "4. **Feature Importance** - Machine learning analysis of key drivers\n"
            )
        f.write(
            "5. **Statistical Validation** - Hypothesis tests and confidence intervals\n"
        )
        if distribution_shift:
            f.write("6. **Distribution Shift** - Comparison with real trading data\n")
        else:
            f.write(
                "6. **Distribution Shift** - Not performed (no real episodes provided)\n"
            )
        if "reward_shaping" in df.columns:
            _total_shaping = df["reward_shaping"].sum()
            _canonical = abs(_total_shaping) < 1e-6
            f.write(
                "7. **PBRS Invariance** - "
                + (
                    "Canonical (Σ shaping ≈ 0)"
                    if _canonical
                    else f"Non-canonical (Σ shaping = {_total_shaping:.6f})"
                )
                + "\n"
            )
        f.write("\n")
        f.write("**Generated Files:**\n")
        f.write("- `reward_samples.csv` - Raw synthetic samples\n")
        if not skip_feature_analysis and len(df) >= 4:
            f.write(
                "- `feature_importance.csv` - Complete feature importance rankings\n"
            )
            f.write(
                "- `partial_dependence_*.csv` - Partial dependence data for visualization\n"
            )


def main() -> None:
    parser = build_argument_parser()
    args = parser.parse_args()

    params = dict(DEFAULT_MODEL_REWARD_PARAMETERS)
    # Merge CLI tunables first (only those explicitly provided)
    _tunable_keys = set(DEFAULT_MODEL_REWARD_PARAMETERS.keys())
    for key, value in vars(args).items():
        if key in _tunable_keys and value is not None:
            params[key] = value
    # Then apply --params KEY=VALUE overrides (highest precedence)
    params.update(parse_overrides(args.params))

    # Early parameter validation (moved before simulation for alignment with docs)
    params_validated, adjustments = validate_reward_parameters(params)
    params = params_validated
    if adjustments:
        # Compact adjustments summary (param: original->adjusted [reason])
        adj_lines = [
            f"  - {k}: {v['original']} -> {v['adjusted']} ({v['reason']})"
            for k, v in adjustments.items()
        ]
        print("Parameter adjustments applied:\n" + "\n".join(adj_lines))
    # Normalize attenuation mode
    _normalize_and_validate_mode(params)

    base_factor = _get_float_param(params, "base_factor", float(args.base_factor))
    profit_target = _get_float_param(params, "profit_target", float(args.profit_target))
    risk_reward_ratio = _get_float_param(
        params, "risk_reward_ratio", float(args.risk_reward_ratio)
    )

    cli_action_masking = _to_bool(args.action_masking)
    if "action_masking" in params:
        params["action_masking"] = _to_bool(params["action_masking"])
    else:
        params["action_masking"] = cli_action_masking

    # Deterministic seeds cascade
    random.seed(args.seed)
    np.random.seed(args.seed)

    df = simulate_samples(
        num_samples=args.num_samples,
        seed=args.seed,
        params=params,
        max_trade_duration=args.max_trade_duration,
        base_factor=base_factor,
        profit_target=profit_target,
        risk_reward_ratio=risk_reward_ratio,
        max_duration_ratio=args.max_duration_ratio,
        trading_mode=args.trading_mode,
        pnl_base_std=args.pnl_base_std,
        pnl_duration_vol_scale=args.pnl_duration_vol_scale,
    )
    # Post-simulation critical NaN validation (non-PBRS structural columns)
    critical_cols = [
        "pnl",
        "trade_duration",
        "idle_duration",
        "position",
        "action",
        "reward_total",
        "reward_invalid",
        "reward_idle",
        "reward_hold",
        "reward_exit",
    ]
    nan_issues = {
        c: int(df[c].isna().sum())
        for c in critical_cols
        if c in df.columns and df[c].isna().any()
    }
    if nan_issues:
        raise AssertionError(
            "NaN values detected in critical simulated columns: "
            + ", ".join(f"{k}={v}" for k, v in nan_issues.items())
        )
    # Attach simulation parameters for downstream manifest
    df.attrs["simulation_params"] = {
        "num_samples": args.num_samples,
        "seed": args.seed,
        "max_trade_duration": args.max_trade_duration,
        "base_factor": base_factor,
        "profit_target": profit_target,
        "risk_reward_ratio": risk_reward_ratio,
        "max_duration_ratio": args.max_duration_ratio,
        "trading_mode": args.trading_mode,
        "action_masking": params.get("action_masking", True),
        "pnl_base_std": args.pnl_base_std,
        "pnl_duration_vol_scale": args.pnl_duration_vol_scale,
    }
    # Attach resolved reward parameters for inline overrides rendering in report
    df.attrs["reward_params"] = dict(params)

    args.output.mkdir(parents=True, exist_ok=True)
    csv_path = args.output / "reward_samples.csv"
    df.to_csv(csv_path, index=False)
    sample_output_message = f"Samples saved to {csv_path}"

    # Load real episodes if provided
    real_df = None
    if args.real_episodes and args.real_episodes.exists():
        print(f"Loading real episodes from {args.real_episodes}...")
        real_df = load_real_episodes(args.real_episodes)

    # Generate consolidated statistical analysis report (with enhanced tests)
    print("Generating complete statistical analysis...")

    write_complete_statistical_analysis(
        df,
        args.output,
        max_trade_duration=args.max_trade_duration,
        profit_target=float(profit_target * risk_reward_ratio),
        seed=args.seed,
        real_df=real_df,
        adjust_method=args.pvalue_adjust,
        stats_seed=args.stats_seed
        if getattr(args, "stats_seed", None) is not None
        else None,
        strict_diagnostics=bool(getattr(args, "strict_diagnostics", False)),
        bootstrap_resamples=getattr(args, "bootstrap_resamples", 10000),
        skip_partial_dependence=bool(getattr(args, "skip_partial_dependence", False)),
        skip_feature_analysis=bool(getattr(args, "skip_feature_analysis", False)),
    )
    print(
        f"Complete statistical analysis saved to: {args.output / 'statistical_analysis.md'}"
    )
    # Generate manifest summarizing key metrics
    try:
        manifest_path = args.output / "manifest.json"
        resolved_reward_params = dict(params)  # already validated/normalized upstream
        manifest = {
            "generated_at": pd.Timestamp.now().isoformat(),
            "num_samples": int(len(df)),
            "seed": int(args.seed),
            "max_trade_duration": int(args.max_trade_duration),
            "profit_target_effective": float(profit_target * risk_reward_ratio),
            "pvalue_adjust_method": args.pvalue_adjust,
            "parameter_adjustments": adjustments,
            "reward_params": resolved_reward_params,
        }
        sim_params = df.attrs.get("simulation_params", {})
        if isinstance(sim_params, dict) and sim_params:
            import hashlib as _hashlib
            import json as _json

            # Compose hash source from ALL simulation params and ALL resolved reward params for full reproducibility.
            _hash_source = {
                **{f"sim::{k}": sim_params[k] for k in sorted(sim_params)},
                **{
                    f"reward::{k}": resolved_reward_params[k]
                    for k in sorted(resolved_reward_params)
                },
            }
            serialized = _json.dumps(_hash_source, sort_keys=True)
            manifest["params_hash"] = _hashlib.sha256(
                serialized.encode("utf-8")
            ).hexdigest()
            manifest["simulation_params"] = sim_params
        with manifest_path.open("w", encoding="utf-8") as mh:
            import json as _json

            _json.dump(manifest, mh, indent=2)
        print(f"Manifest written to: {manifest_path}")
    except Exception as e:
        print(f"Manifest generation failed: {e}")

    print(f"Generated {len(df):,} synthetic samples.")
    print(sample_output_message)
    print(f"Artifacts saved to: {args.output.resolve()}")


# === PBRS TRANSFORM FUNCTIONS ===


def _apply_transform_tanh(value: float) -> float:
    """tanh(value) ∈ (-1,1)."""
    return float(np.tanh(value))


def _apply_transform_softsign(value: float) -> float:
    """softsign: value/(1+|value|)."""
    x = value
    return float(x / (1.0 + abs(x)))


def _apply_transform_softsign_sharp(value: float, sharpness: float = 1.0) -> float:
    """softsign_sharp: (sharpness*value)/(1+|sharpness*value|) - multiplicative sharpness."""
    xs = sharpness * value
    return float(xs / (1.0 + abs(xs)))


def _apply_transform_arctan(value: float) -> float:
    """arctan normalized: (2/pi)*atan(value) ∈ (-1,1)."""
    return float((2.0 / math.pi) * math.atan(value))


def _apply_transform_logistic(value: float) -> float:
    """Overflow‑safe logistic transform mapped to (-1,1): 2σ(x)−1."""
    x = value
    try:
        if x >= 0:
            z = math.exp(-x)  # z in (0,1]
            return float((1.0 - z) / (1.0 + z))
        else:
            z = math.exp(x)  # z in (0,1]
            return float((z - 1.0) / (z + 1.0))
    except OverflowError:
        return 1.0 if x > 0 else -1.0


def _apply_transform_asinh_norm(value: float) -> float:
    """Normalized asinh: value / sqrt(1 + value²) producing range (-1,1)."""
    return float(value / math.hypot(1.0, value))


def _apply_transform_clip(value: float) -> float:
    """clip(value) to [-1,1]."""
    return float(np.clip(value, -1.0, 1.0))


def apply_transform(transform_name: str, value: float, **kwargs: Any) -> float:
    """Apply named transform; unknown names fallback to tanh with warning."""
    transforms = {
        "tanh": _apply_transform_tanh,
        "softsign": _apply_transform_softsign,
        "softsign_sharp": _apply_transform_softsign_sharp,
        "arctan": _apply_transform_arctan,
        "logistic": _apply_transform_logistic,
        "asinh_norm": _apply_transform_asinh_norm,
        "clip": _apply_transform_clip,
    }

    if transform_name not in transforms:
        warnings.warn(
            f"Unknown potential transform '{transform_name}'; falling back to tanh",
            RewardDiagnosticsWarning,
            stacklevel=2,
        )
        return _apply_transform_tanh(value)
    return transforms[transform_name](value, **kwargs)


# === PBRS HELPER FUNCTIONS ===


def _get_potential_gamma(params: RewardParams) -> float:
    """Return validated potential_gamma.

    Process:
    - If NaN -> default POTENTIAL_GAMMA_DEFAULT with warning (missing or unparsable).
    - If outside [0,1] -> clamp + warning including original value.
    - Guarantee returned float ∈ [0,1].
    """
    gamma = _get_float_param(params, "potential_gamma", np.nan)
    if not np.isfinite(gamma):
        warnings.warn(
            f"potential_gamma not specified or invalid; defaulting to {POTENTIAL_GAMMA_DEFAULT}",
            RewardDiagnosticsWarning,
            stacklevel=2,
        )
        return POTENTIAL_GAMMA_DEFAULT
    if gamma < 0.0 or gamma > 1.0:
        original = gamma
        gamma = float(np.clip(gamma, 0.0, 1.0))
        warnings.warn(
            f"potential_gamma={original} outside [0,1]; clamped to {gamma}",
            RewardDiagnosticsWarning,
            stacklevel=2,
        )
        return gamma
    return float(gamma)


def _get_str_param(params: RewardParams, key: str, default: str) -> str:
    """Extract string parameter with type safety."""
    value = params.get(key, default)
    if isinstance(value, str):
        return value
    return default


def _get_bool_param(params: RewardParams, key: str, default: bool) -> bool:
    """Extract boolean parameter with type safety."""
    value = params.get(key, default)
    try:
        return _to_bool(value)
    except Exception:
        return bool(default)


# === PBRS IMPLEMENTATION ===


def _compute_hold_potential(
    pnl: float, duration_ratio: float, params: RewardParams
) -> float:
    """Compute PBRS hold potential Φ(s)."""
    if not _get_bool_param(params, "hold_potential_enabled", True):
        return _fail_safely("hold_potential_disabled")
    return _compute_bi_component(
        kind="hold_potential",
        pnl=pnl,
        duration_ratio=duration_ratio,
        params=params,
        scale_key="hold_potential_scale",
        gain_key="hold_potential_gain",
        transform_pnl_key="hold_potential_transform_pnl",
        transform_dur_key="hold_potential_transform_duration",
        non_finite_key="non_finite_hold_potential",
    )


def _compute_entry_additive(
    pnl: float, duration_ratio: float, params: RewardParams
) -> float:
    if not _get_bool_param(params, "entry_additive_enabled", False):
        return _fail_safely("entry_additive_disabled")
    return _compute_bi_component(
        kind="entry_additive",
        pnl=pnl,
        duration_ratio=duration_ratio,
        params=params,
        scale_key="entry_additive_scale",
        gain_key="entry_additive_gain",
        transform_pnl_key="entry_additive_transform_pnl",
        transform_dur_key="entry_additive_transform_duration",
        non_finite_key="non_finite_entry_additive",
    )


def _compute_exit_additive(
    pnl: float, duration_ratio: float, params: RewardParams
) -> float:
    if not _get_bool_param(params, "exit_additive_enabled", False):
        return _fail_safely("exit_additive_disabled")
    return _compute_bi_component(
        kind="exit_additive",
        pnl=pnl,
        duration_ratio=duration_ratio,
        params=params,
        scale_key="exit_additive_scale",
        gain_key="exit_additive_gain",
        transform_pnl_key="exit_additive_transform_pnl",
        transform_dur_key="exit_additive_transform_duration",
        non_finite_key="non_finite_exit_additive",
    )


def _compute_exit_potential(last_potential: float, params: RewardParams) -> float:
    """Compute next potential Φ(s') for closing/exit transitions.

    Semantics:
    - canonical: Φ' = 0.0 (preserves invariance, disables additives)
    - non-canonical: Φ' = 0.0 (allows additives, breaks invariance)
    - progressive_release: Φ' = Φ * (1 - decay) with decay clamped to [0,1]
    - spike_cancel: Φ' = Φ / γ (neutralizes shaping spike ≈ 0 net effect) if γ>0 else Φ
    - retain_previous: Φ' = Φ

    Invalid modes fall back to canonical. Any non-finite resulting potential is
    coerced to 0.0.
    """
    mode = _get_str_param(params, "exit_potential_mode", "canonical")
    if mode == "canonical" or mode == "non-canonical":
        return _fail_safely("canonical_exit_potential")

    if mode == "progressive_release":
        decay = _get_float_param(params, "exit_potential_decay", 0.5)
        if not np.isfinite(decay):
            warnings.warn(
                "exit_potential_decay invalid (NaN/inf); defaulting to 0.5",
                RewardDiagnosticsWarning,
                stacklevel=2,
            )
            decay = 0.5
        if decay < 0.0:
            warnings.warn(
                f"exit_potential_decay={decay} < 0; clamped to 0.0",
                RewardDiagnosticsWarning,
                stacklevel=2,
            )
            decay = 0.0
        if decay > 1.0:
            warnings.warn(
                f"exit_potential_decay={decay} > 1; clamped to 1.0",
                RewardDiagnosticsWarning,
                stacklevel=2,
            )
            decay = 1.0
        next_potential = last_potential * (1.0 - decay)
    elif mode == "spike_cancel":
        gamma = _get_potential_gamma(params)
        if gamma <= 0.0 or not np.isfinite(gamma):
            next_potential = last_potential
        else:
            next_potential = last_potential / gamma
    elif mode == "retain_previous":
        next_potential = last_potential
    else:
        next_potential = _fail_safely("invalid_exit_potential_mode")

    if not np.isfinite(next_potential):
        next_potential = _fail_safely("non_finite_next_exit_potential")
    return float(next_potential)


def apply_potential_shaping(
    base_reward: float,
    current_pnl: float,
    current_duration_ratio: float,
    next_pnl: float,
    next_duration_ratio: float,
    is_terminal: bool,
    last_potential: float,
    params: RewardParams,
) -> tuple[float, float, float]:
    """
    Apply PBRS potential-based reward shaping following Ng et al. (1999).

    Implements: R'(s,a,s') = R_base(s,a,s') + γΦ(s') - Φ(s)

    This function computes the complete PBRS transformation including:
    - Hold potential: Φ(s) based on current state features
    - Closing potential: Φ(s') with mode-specific terminal handling
    - Entry/exit potentials: Non-PBRS additive components
    - Gamma discounting: Standard RL discount factor
    - Invariance guarantees: Optimal policy preservation

    Theory:
    PBRS maintains optimal policy invariance by ensuring that the potential-based
    shaping reward is the difference of a potential function. Terminal states
    require special handling to preserve this property.

    Args:
        base_reward: Base environment reward R_base(s,a,s')
        current_pnl: Current state PnL ratio
        current_duration_ratio: Current state duration ratio
        next_pnl: Next state PnL ratio
        next_duration_ratio: Next state duration ratio
        is_terminal: Whether next state is terminal
        last_potential: Previous potential for closing mode calculations
        params: Reward parameters containing PBRS configuration

    Returns:
        tuple[total_reward, shaping_reward, next_potential]:
        - total_reward: R_base + R_shaping + additives
        - shaping_reward: Pure PBRS component γΦ(s') - Φ(s)
        - next_potential: Φ(s') for next iteration

    Raises:
        ValueError: If gamma is invalid or numerical issues detected
    """
    # Enforce PBRS invariance (auto-disable additives in canonical mode)
    params = _enforce_pbrs_invariance(params)

    # Resolve gamma
    gamma = _get_potential_gamma(params)

    # Compute current potential Φ(s)
    current_potential = _compute_hold_potential(
        current_pnl, current_duration_ratio, params
    )

    # Compute next potential Φ(s')
    if is_terminal:
        next_potential = _compute_exit_potential(last_potential, params)
    else:
        next_potential = _compute_hold_potential(next_pnl, next_duration_ratio, params)

    # PBRS shaping reward: γΦ(s') - Φ(s)
    shaping_reward = gamma * next_potential - current_potential

    # Compute additive components (non-PBRS)
    entry_additive = _compute_entry_additive(
        current_pnl, current_duration_ratio, params
    )
    exit_additive = (
        _compute_exit_additive(next_pnl, next_duration_ratio, params)
        if is_terminal
        else 0.0
    )

    # Total reward
    total_reward = base_reward + shaping_reward + entry_additive + exit_additive

    # Numerical validation & normalization of tiny shaping
    if not np.isfinite(total_reward):
        return float(base_reward), 0.0, 0.0
    if np.isclose(shaping_reward, 0.0):
        shaping_reward = 0.0
    return float(total_reward), float(shaping_reward), float(next_potential)


def _enforce_pbrs_invariance(params: RewardParams) -> RewardParams:
    """Enforce PBRS invariance by auto-disabling additives in canonical mode."""
    mode = _get_str_param(params, "exit_potential_mode", "canonical")
    if mode != "canonical":
        return params
    if params.get("_pbrs_invariance_applied"):
        return params

    entry_enabled = _get_bool_param(params, "entry_additive_enabled", False)
    exit_enabled = _get_bool_param(params, "exit_additive_enabled", False)
    if entry_enabled:
        warnings.warn(
            "Disabling entry additive to preserve PBRS invariance (canonical mode).",
            RewardDiagnosticsWarning,
            stacklevel=2,
        )
        params["entry_additive_enabled"] = False
    if exit_enabled:
        warnings.warn(
            "Disabling exit additive to preserve PBRS invariance (canonical mode).",
            RewardDiagnosticsWarning,
            stacklevel=2,
        )
        params["exit_additive_enabled"] = False
    params["_pbrs_invariance_applied"] = True
    return params


def _compute_bi_component(
    kind: str,
    pnl: float,
    duration_ratio: float,
    params: RewardParams,
    scale_key: str,
    gain_key: str,
    transform_pnl_key: str,
    transform_dur_key: str,
    non_finite_key: str,
) -> float:
    """Generic helper for (pnl, duration) bi-component transforms."""
    scale = _get_float_param(params, scale_key, 1.0)
    gain = _get_float_param(params, gain_key, 1.0)
    transform_pnl = _get_str_param(params, transform_pnl_key, "tanh")
    transform_duration = _get_str_param(params, transform_dur_key, "tanh")
    sharpness = _get_float_param(params, "potential_softsign_sharpness", 1.0)

    if transform_pnl == "softsign_sharp":
        t_pnl = apply_transform(transform_pnl, gain * pnl, sharpness=sharpness)
    else:
        t_pnl = apply_transform(transform_pnl, gain * pnl)
    if transform_duration == "softsign_sharp":
        t_dur = apply_transform(
            transform_duration, gain * duration_ratio, sharpness=sharpness
        )
    else:
        t_dur = apply_transform(transform_duration, gain * duration_ratio)
    value = scale * 0.5 * (t_pnl + t_dur)
    if not np.isfinite(value):
        return _fail_safely(non_finite_key)
    return float(value)


if __name__ == "__main__":
    main()
