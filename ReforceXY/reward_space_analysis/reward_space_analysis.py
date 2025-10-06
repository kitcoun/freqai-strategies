#!/usr/bin/env python3
"""Synthetic reward space analysis for the ReforceXY environment.

Capabilities:
- Hypothesis testing (Spearman, Kruskal-Wallis, Mann-Whitney).
- Percentile bootstrap confidence intervals (BCa not yet implemented).
- Distribution diagnostics (Shapiro, Anderson, skewness, kurtosis, Q-Q R²).
- Distribution shift metrics (KL divergence, JS distance, Wasserstein, KS test) with
    degenerate (constant) distribution safe‑guards.
- Unified RandomForest feature importance + partial dependence.
- Heteroscedastic PnL simulation (variance scales with duration).

Architecture principles:
- Single source of truth: `DEFAULT_MODEL_REWARD_PARAMETERS` for tunables + dynamic CLI.
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
from typing import Any, Dict, Iterable, List, Optional, Tuple

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


class ForceActions(IntEnum):
    Take_profit = 0
    Stop_loss = 1
    Timeout = 2


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


def _get_param_float(params: Dict[str, float | str], key: str, default: float) -> float:
    """Extract float parameter with type safety and default fallback."""
    value = params.get(key, default)
    if isinstance(value, (int, float)):
        return float(value)
    if isinstance(value, str):
        try:
            return float(value)
        except (ValueError, TypeError):
            return default
    return default


def _piecewise_duration_divisor(
    duration_ratio: float, params: Dict[str, float | str]
) -> float:
    """Compute divisor for piecewise attenuation (single source of truth).

    Ensures consistent fallback behaviour across the primary code path and
    exception fallback in ``_get_exit_factor`` without duplicating logic.
    """
    exit_piecewise_grace = _get_param_float(params, "exit_piecewise_grace", 1.0)
    # Only enforce a lower bound; values >1.0 extend the grace region beyond max duration ratio.
    if exit_piecewise_grace < 0.0:
        exit_piecewise_grace = 0.0
    exit_piecewise_slope = _get_param_float(params, "exit_piecewise_slope", 1.0)
    if exit_piecewise_slope < 0.0:  # sanitize slope sign
        exit_piecewise_slope = 1.0
    if duration_ratio <= exit_piecewise_grace:
        return 1.0
    return 1.0 + exit_piecewise_slope * (duration_ratio - exit_piecewise_grace)


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

DEFAULT_MODEL_REWARD_PARAMETERS: Dict[str, float | str] = {
    "invalid_action": -2.0,
    "base_factor": 100.0,
    # Idle penalty (env defaults)
    "idle_penalty_power": 1.0,
    "idle_penalty_scale": 1.0,
    # If <=0 or unset, falls back to max_trade_duration_candles at runtime
    "max_idle_duration_candles": 0,
    # Holding keys (env defaults)
    "holding_penalty_scale": 0.5,
    "holding_penalty_power": 1.0,
    # Exit factor configuration (env defaults)
    "exit_factor_mode": "piecewise",
    "exit_linear_slope": 1.0,
    "exit_piecewise_grace": 1.0,
    "exit_piecewise_slope": 1.0,
    "exit_power_tau": 0.5,
    "exit_half_life": 0.5,
    # Efficiency keys (env defaults)
    "efficiency_weight": 0.75,
    "efficiency_center": 0.75,
    # Profit factor params (env defaults)
    "win_reward_factor": 2.0,
    "pnl_factor_beta": 0.5,
    # Invariant / safety controls (env defaults)
    "check_invariants": True,
    "exit_factor_threshold": 10000.0,
}

DEFAULT_MODEL_REWARD_PARAMETERS_HELP: Dict[str, str] = {
    "invalid_action": "Penalty for invalid actions.",
    "base_factor": "Base reward factor used inside the environment.",
    "idle_penalty_power": "Power applied to idle penalty scaling.",
    "idle_penalty_scale": "Scale of idle penalty.",
    "max_idle_duration_candles": "Maximum idle duration candles before full idle penalty scaling; 0 = use max_trade_duration_candles.",
    "holding_penalty_scale": "Scale of holding penalty.",
    "holding_penalty_power": "Power applied to holding penalty scaling.",
    "exit_factor_mode": "Time attenuation mode for exit factor.",
    "exit_linear_slope": "Slope for linear exit attenuation.",
    # exit_piecewise_grace: duration ratio boundary; >1 extends full-strength region
    "exit_piecewise_grace": "Grace boundary (duration ratio; >1 extends no-attenuation region).",
    "exit_piecewise_slope": "Slope after grace for piecewise mode (0 = flat).",
    "exit_power_tau": "Tau in (0,1] to derive alpha for power mode.",
    "exit_half_life": "Half-life for exponential decay exit mode.",
    "efficiency_weight": "Weight for efficiency factor in exit reward.",
    "efficiency_center": "Center for efficiency factor sigmoid.",
    "win_reward_factor": "Amplification for pnl above target (no hard cap; asymptotic).",
    "pnl_factor_beta": "Sensitivity of amplification around target.",
    "check_invariants": "Boolean flag (true/false) to enable runtime invariant & safety checks.",
    "exit_factor_threshold": "If |exit factor| exceeds this threshold, emit warning.",
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
    "holding_penalty_scale": {"min": 0.0},
    "holding_penalty_power": {"min": 0.0},
    "exit_linear_slope": {"min": 0.0},
    "exit_piecewise_grace": {"min": 0.0},
    "exit_piecewise_slope": {"min": 0.0},
    "exit_power_tau": {"min": 1e-6, "max": 1.0},  # open (0,1] approximated
    "exit_half_life": {"min": 1e-6},
    "efficiency_weight": {"min": 0.0, "max": 2.0},
    "efficiency_center": {"min": 0.0, "max": 1.0},
    "win_reward_factor": {"min": 0.0},
    "pnl_factor_beta": {"min": 1e-6},
}


def validate_reward_parameters(
    params: Dict[str, float | str],
) -> Tuple[Dict[str, float | str], Dict[str, Dict[str, Any]]]:
    """Validate and clamp reward parameter values.

    Returns
    -------
    sanitized_params : dict
        Potentially adjusted copy of input params.
    adjustments : dict
        Mapping param -> {original, adjusted, reason} for every modification.
    """
    sanitized = dict(params)
    adjustments: Dict[str, Dict[str, Any]] = {}
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
        if not math.isfinite(adjusted):
            adjusted = bounds.get("min", 0.0)
            reason_parts.append("non_finite_reset")
        if not math.isclose(adjusted, original):
            sanitized[key] = adjusted
            adjustments[key] = {
                "original": original,
                "adjusted": adjusted,
                "reason": ",".join(reason_parts),  # textual reason directly
            }
    return sanitized, adjustments


def add_tunable_cli_args(parser: argparse.ArgumentParser) -> None:
    """Dynamically add CLI options for each tunable in DEFAULT_MODEL_REWARD_PARAMETERS.

    Rules:
    - Use the same underscored names as option flags (e.g., --idle_penalty_scale).
    - Defaults are None so only user-provided values override params.
    - For exit_factor_mode, enforce allowed choices and lowercase conversion.
    - Skip keys already managed as top-level options (e.g., base_factor) to avoid duplicates.
    """
    skip_keys = {"base_factor"}  # already defined as top-level
    for key, default in DEFAULT_MODEL_REWARD_PARAMETERS.items():
        if key in skip_keys:
            continue
        help_text = DEFAULT_MODEL_REWARD_PARAMETERS_HELP.get(
            key, f"Override tunable '{key}'."
        )
        if key == "exit_factor_mode":
            parser.add_argument(
                f"--{key}",
                type=str.lower,
                choices=["legacy", "sqrt", "linear", "power", "piecewise", "half_life"],
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
    force_action: Optional[ForceActions]


@dataclasses.dataclass
class RewardBreakdown:
    total: float = 0.0
    invalid_penalty: float = 0.0
    idle_penalty: float = 0.0
    holding_penalty: float = 0.0
    exit_component: float = 0.0


def _get_exit_factor(
    factor: float,
    pnl: float,
    pnl_factor: float,
    duration_ratio: float,
    params: Dict[str, float | str],
) -> float:
    """Compute the complete exit factor (time attenuation + PnL scaling).

    Synchronization
    ---------------
    Mirrors the environment implementation `ReforceXY._get_exit_factor` (parity date: 2025-10-06).
    Any upstream change MUST be ported here to keep analytical results consistent with live behavior.

    Processing steps
    ----------------
    1. Clamp negative ``duration_ratio`` to 0.
    2. Apply time-based attenuation according to ``exit_factor_mode``.
    3. Multiply by ``pnl_factor`` (already includes profit amplification & efficiency component).
    4. Enforce invariants (finite; non-negative when pnl>=0) and optionally emit a warning if
       ``|factor| > exit_factor_threshold`` (warning-only, no capping).

    Modes (attenuation formulas)
    ----------------------------
    legacy    : factor *= 1.5 if r <= 1 else *= 0.5 (step change)
    sqrt      : factor /= sqrt(1 + r)
    linear    : factor /= (1 + slope * r)
    power     : alpha = -log(tau)/log(2) with tau in (0,1]; factor /= (1 + r)^alpha (fallback alpha=1)
    piecewise : grace g in [0,1]; if r <= g then divisor=1 else divisor = 1 + slope * (r - g)
    half_life : factor *= 2^(-r / half_life)

    Fallback: Unknown modes default to piecewise.

    Invariants
    ----------
    - Factor set to 0 if non-finite.
    - Factor clamped to >=0 when pnl >= 0.
    - Threshold exceedance triggers RuntimeWarning only.

    Returns
    -------
    float : attenuated & pnl-scaled factor (reward exit component = pnl * factor).
    """
    # Basic finiteness checks
    if (
        not math.isfinite(factor)
        or not math.isfinite(pnl)
        or not math.isfinite(duration_ratio)
    ):
        return 0.0

    # Guard: duration ratio should never be negative
    if duration_ratio < 0.0:
        duration_ratio = 0.0

    exit_factor_mode = str(params.get("exit_factor_mode", "piecewise")).lower()

    try:
        if exit_factor_mode == "legacy":
            factor *= 1.5 if duration_ratio <= 1.0 else 0.5
        elif exit_factor_mode == "sqrt":
            factor /= math.sqrt(1.0 + duration_ratio)
        elif exit_factor_mode == "linear":
            slope = _get_param_float(params, "exit_linear_slope", 1.0)
            if slope < 0.0:
                slope = 1.0
            factor /= 1.0 + slope * duration_ratio
        elif exit_factor_mode == "power":
            exit_power_alpha = params.get("exit_power_alpha")
            if isinstance(exit_power_alpha, (int, float)) and exit_power_alpha < 0.0:
                exit_power_alpha = None
            if exit_power_alpha is None:
                exit_power_tau = params.get("exit_power_tau")
                if isinstance(exit_power_tau, (int, float)):
                    exit_power_tau = float(exit_power_tau)
                    if 0.0 < exit_power_tau <= 1.0:
                        exit_power_alpha = -math.log(exit_power_tau) / _LOG_2
            if not isinstance(exit_power_alpha, (int, float)):
                exit_power_alpha = 1.0
            else:
                exit_power_alpha = float(exit_power_alpha)
            factor /= math.pow(1.0 + duration_ratio, exit_power_alpha)
        elif exit_factor_mode == "piecewise" or exit_factor_mode not in {
            "legacy",
            "sqrt",
            "linear",
            "power",
            "half_life",
        }:
            # Default behaviour
            factor /= _piecewise_duration_divisor(duration_ratio, params)
        elif exit_factor_mode == "half_life":
            exit_half_life = _get_param_float(params, "exit_half_life", 0.5)
            if exit_half_life <= 0.0:
                exit_half_life = 0.5
            factor *= math.pow(2.0, -duration_ratio / exit_half_life)
    except Exception:
        # Safe fallback to piecewise logic if any unexpected error arises (centralized)
        factor /= _piecewise_duration_divisor(duration_ratio, params)

    # Apply pnl_factor after time attenuation
    factor *= pnl_factor

    # Invariant & safety checks
    if _to_bool(params.get("check_invariants", True)):
        if not math.isfinite(factor):
            return 0.0
        if factor < 0.0 and pnl >= 0.0:
            # Clamp: avoid negative amplification on non-negative pnl
            factor = 0.0
        thr = params.get("exit_factor_threshold")
        if isinstance(thr, (int, float)) and thr > 0 and math.isfinite(thr):
            if abs(factor) > thr:
                warnings.warn(
                    (
                        f"_get_exit_factor |factor|={abs(factor):.2f} exceeds threshold {thr:.2f}"
                    ),
                    RuntimeWarning,
                    stacklevel=2,
                )

    return factor


def _get_pnl_factor(
    params: Dict[str, float | str], context: RewardContext, profit_target: float
) -> float:
    """Env-aligned PnL factor combining profit amplification and exit efficiency."""
    pnl = context.pnl

    if not math.isfinite(pnl) or not math.isfinite(profit_target):
        return 0.0

    profit_target_factor = 1.0
    if profit_target > 0.0 and pnl > profit_target:
        win_reward_factor = float(params.get("win_reward_factor", 2.0))
        pnl_factor_beta = float(params.get("pnl_factor_beta", 0.5))
        pnl_ratio = pnl / profit_target
        profit_target_factor = 1.0 + win_reward_factor * math.tanh(
            pnl_factor_beta * (pnl_ratio - 1.0)
        )

    efficiency_factor = 1.0
    efficiency_weight = float(params.get("efficiency_weight", 0.75))
    efficiency_center = float(params.get("efficiency_center", 0.75))
    if efficiency_weight != 0.0 and pnl >= 0.0:
        max_pnl = max(context.max_unrealized_profit, pnl)
        min_pnl = min(context.min_unrealized_profit, pnl)
        range_pnl = max_pnl - min_pnl
        if math.isfinite(range_pnl) and not math.isclose(range_pnl, 0.0):
            efficiency_ratio = (pnl - min_pnl) / range_pnl
            efficiency_factor = 1.0 + efficiency_weight * (
                efficiency_ratio - efficiency_center
            )

    return max(0.0, profit_target_factor * efficiency_factor)


def _is_valid_action(
    position: Positions,
    action: Actions,
    *,
    short_allowed: bool,
    force_action: Optional[ForceActions],
) -> bool:
    if force_action is not None and position in (Positions.Long, Positions.Short):
        if position == Positions.Long:
            return action == Actions.Long_exit
        return action == Actions.Short_exit

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
    context: RewardContext, idle_factor: float, params: Dict[str, float | str]
) -> float:
    """Mirror the environment's idle penalty behaviour."""
    idle_penalty_scale = _get_param_float(params, "idle_penalty_scale", 1.0)
    idle_penalty_power = _get_param_float(params, "idle_penalty_power", 1.0)
    max_trade_duration = int(params.get("max_trade_duration_candles", 128))
    max_idle_duration_candles = params.get("max_idle_duration_candles")
    try:
        max_idle_duration = (
            int(max_idle_duration_candles)
            if max_idle_duration_candles is not None
            else max_trade_duration
        )
    except (TypeError, ValueError):
        max_idle_duration = max_trade_duration
    if max_idle_duration <= 0:
        max_idle_duration = max_trade_duration
    idle_duration_ratio = context.idle_duration / max(1, max_idle_duration)
    return -idle_factor * idle_penalty_scale * idle_duration_ratio**idle_penalty_power


def _holding_penalty(
    context: RewardContext, holding_factor: float, params: Dict[str, float | str]
) -> float:
    """Mirror the environment's holding penalty behaviour."""
    holding_penalty_scale = _get_param_float(params, "holding_penalty_scale", 0.5)
    holding_penalty_power = _get_param_float(params, "holding_penalty_power", 1.0)
    duration_ratio = _compute_duration_ratio(
        context.trade_duration, context.max_trade_duration
    )

    if duration_ratio < 1.0:
        return 0.0

    return (
        -holding_factor
        * holding_penalty_scale
        * (duration_ratio - 1.0) ** holding_penalty_power
    )


def _compute_exit_reward(
    base_factor: float,
    pnl_factor: float,
    context: RewardContext,
    params: Dict[str, float | str],
) -> float:
    """Compose the exit reward: pnl * exit_factor."""
    duration_ratio = _compute_duration_ratio(
        context.trade_duration, context.max_trade_duration
    )
    exit_factor = _get_exit_factor(
        base_factor, context.pnl, pnl_factor, duration_ratio, params
    )
    return context.pnl * exit_factor


def compute_exit_factor(
    base_factor: float,
    pnl: float,
    pnl_factor: float,
    duration_ratio: float,
    params: Dict[str, float | str],
) -> float:
    """Public wrapper to compute the time-attenuated + pnl-scaled exit factor."""
    return _get_exit_factor(base_factor, pnl, pnl_factor, duration_ratio, params)


def calculate_reward(
    context: RewardContext,
    params: Dict[str, float | str],
    base_factor: float,
    profit_target: float,
    risk_reward_ratio: float,
    *,
    short_allowed: bool,
    action_masking: bool,
) -> RewardBreakdown:
    breakdown = RewardBreakdown()

    is_valid = _is_valid_action(
        context.position,
        context.action,
        short_allowed=short_allowed,
        force_action=context.force_action,
    )
    if not is_valid and not action_masking:
        breakdown.invalid_penalty = _get_param_float(params, "invalid_action", -2.0)
        breakdown.total = breakdown.invalid_penalty
        return breakdown

    factor = _get_param_float(params, "base_factor", base_factor)

    profit_target_override = params.get("profit_target")
    if isinstance(profit_target_override, (int, float)):
        profit_target = float(profit_target_override)

    rr_override = params.get("rr")
    if not isinstance(rr_override, (int, float)):
        rr_override = params.get("risk_reward_ratio")
    if isinstance(rr_override, (int, float)):
        risk_reward_ratio = float(rr_override)

    # Scale profit target by risk-reward ratio (reward multiplier)
    # E.g., profit_target=0.03, RR=2.0 → profit_target_final=0.06
    profit_target_final = profit_target * risk_reward_ratio
    idle_factor = factor * profit_target_final / 3.0
    pnl_factor = _get_pnl_factor(params, context, profit_target_final)
    holding_factor = idle_factor

    if context.force_action in (
        ForceActions.Take_profit,
        ForceActions.Stop_loss,
        ForceActions.Timeout,
    ):
        exit_reward = _compute_exit_reward(
            factor,
            pnl_factor,
            context,
            params,
        )
        breakdown.exit_component = exit_reward
        breakdown.total = exit_reward
        return breakdown

    if context.action == Actions.Neutral and context.position == Positions.Neutral:
        breakdown.idle_penalty = _idle_penalty(context, idle_factor, params)
        breakdown.total = breakdown.idle_penalty
        return breakdown

    if (
        context.position in (Positions.Long, Positions.Short)
        and context.action == Actions.Neutral
    ):
        breakdown.holding_penalty = _holding_penalty(context, holding_factor, params)
        breakdown.total = breakdown.holding_penalty
        return breakdown

    if context.action == Actions.Long_exit and context.position == Positions.Long:
        exit_reward = _compute_exit_reward(
            factor,
            pnl_factor,
            context,
            params,
        )
        breakdown.exit_component = exit_reward
        breakdown.total = exit_reward
        return breakdown

    if context.action == Actions.Short_exit and context.position == Positions.Short:
        exit_reward = _compute_exit_reward(
            factor,
            pnl_factor,
            context,
            params,
        )
        breakdown.exit_component = exit_reward
        breakdown.total = exit_reward
        return breakdown

    breakdown.total = 0.0
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


def parse_overrides(overrides: Iterable[str]) -> Dict[str, float | str]:
    parsed: Dict[str, float | str] = {}
    for override in overrides:
        if "=" not in override:
            raise ValueError(f"Invalid override format: '{override}'")
        key, raw_value = override.split("=", 1)
        try:
            parsed[key] = float(raw_value)
        except ValueError:
            parsed[key] = raw_value
    return parsed


def simulate_samples(
    num_samples: int,
    seed: int,
    params: Dict[str, float | str],
    max_trade_duration: int,
    base_factor: float,
    profit_target: float,
    risk_reward_ratio: float,
    holding_max_ratio: float,
    trading_mode: str,
    pnl_base_std: float,
    pnl_duration_vol_scale: float,
) -> pd.DataFrame:
    rng = random.Random(seed)
    short_allowed = _is_short_allowed(trading_mode)
    action_masking = _to_bool(params.get("action_masking", True))
    samples: list[dict[str, float]] = []
    for _ in range(num_samples):
        if short_allowed:
            position_choices = [Positions.Neutral, Positions.Long, Positions.Short]
            position_weights = [0.45, 0.3, 0.25]
        else:
            position_choices = [Positions.Neutral, Positions.Long]
            position_weights = [0.6, 0.4]

        position = rng.choices(position_choices, weights=position_weights, k=1)[0]
        force_action: Optional[ForceActions]
        if position != Positions.Neutral and rng.random() < 0.08:
            force_action = rng.choice(
                [ForceActions.Take_profit, ForceActions.Stop_loss, ForceActions.Timeout]
            )
        else:
            force_action = None

        if (
            action_masking
            and force_action is not None
            and position != Positions.Neutral
        ):
            action = (
                Actions.Long_exit if position == Positions.Long else Actions.Short_exit
            )
        else:
            action = _sample_action(position, rng, short_allowed=short_allowed)

        if position == Positions.Neutral:
            trade_duration = 0
            idle_duration = int(rng.uniform(0, max_trade_duration * holding_max_ratio))
        else:
            trade_duration = int(rng.uniform(1, max_trade_duration * holding_max_ratio))
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

            # Force actions should correlate with PnL sign
            if force_action == ForceActions.Take_profit:
                # Take profit exits should have positive PnL
                pnl = abs(pnl) + rng.uniform(0.01, 0.05)
            elif force_action == ForceActions.Stop_loss:
                # Stop loss exits should have negative PnL
                pnl = -abs(pnl) - rng.uniform(0.01, 0.05)

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
            force_action=force_action,
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
                "force_action": float(
                    -1 if context.force_action is None else context.force_action.value
                ),
                "reward_total": breakdown.total,
                "reward_invalid": breakdown.invalid_penalty,
                "reward_idle": breakdown.idle_penalty,
                "reward_holding": breakdown.holding_penalty,
                "reward_exit": breakdown.exit_component,
                "is_force_exit": float(context.force_action is not None),
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
    exit_mask = df["reward_exit"] != 0
    exit_pnl_sum = df.loc[exit_mask, "pnl"].sum()

    pnl_diff = abs(total_pnl - exit_pnl_sum)
    if pnl_diff > 1e-10:
        raise AssertionError(
            f"PnL INVARIANT VIOLATION: Total PnL ({total_pnl:.6f}) != "
            f"Exit PnL sum ({exit_pnl_sum:.6f}), difference = {pnl_diff:.2e}"
        )

    # INVARIANT 2: PnL Exclusivity - Only exit actions should have non-zero PnL
    non_zero_pnl_actions = set(df[df["pnl"] != 0]["action"].unique())
    valid_exit_actions = {2.0, 4.0}  # Long_exit, Short_exit

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
        ["reward_invalid", "reward_idle", "reward_holding", "reward_exit"]
    ].apply(lambda col: (col != 0).mean())

    components = [
        "reward_invalid",
        "reward_idle",
        "reward_holding",
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


def write_summary(df: pd.DataFrame, output_dir: Path) -> None:
    """Legacy function - kept for backward compatibility."""
    output_dir.mkdir(parents=True, exist_ok=True)
    summary_path = output_dir / "reward_summary.md"
    stats = _compute_summary_stats(df)

    with summary_path.open("w", encoding="utf-8") as handle:
        handle.write("# Reward space summary\n\n")
        handle.write("## Global statistics\n\n")
        handle.write(stats["global_stats"].to_frame(name="reward_total").to_string())
        handle.write("\n\n")
        handle.write("## Action-wise reward statistics\n\n")
        handle.write(stats["action_summary"].to_string())
        handle.write("\n\n")
        handle.write("## Component activation ratio\n\n")
        handle.write(
            stats["component_share"].to_frame(name="activation_rate").to_string()
        )
        handle.write("\n\n")
        handle.write("## Component bounds (min/mean/max)\n\n")
        handle.write(stats["component_bounds"].to_string())
        handle.write("\n")


def _binned_stats(
    df: pd.DataFrame,
    column: str,
    target: str,
    bins: Iterable[float],
) -> pd.DataFrame:
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
    """Compute relationship statistics without writing to file."""
    idle_bins = np.linspace(0, max_trade_duration * 3.0, 13)
    trade_bins = np.linspace(0, max_trade_duration * 3.0, 13)
    pnl_min = float(df["pnl"].min())
    pnl_max = float(df["pnl"].max())
    if math.isclose(pnl_min, pnl_max):
        pnl_max = pnl_min + 1e-6
    pnl_bins = np.linspace(pnl_min, pnl_max, 13)

    idle_stats = _binned_stats(df, "idle_duration", "reward_idle", idle_bins)
    holding_stats = _binned_stats(df, "trade_duration", "reward_holding", trade_bins)
    exit_stats = _binned_stats(df, "pnl", "reward_exit", pnl_bins)

    idle_stats = idle_stats.round(6)
    holding_stats = holding_stats.round(6)
    exit_stats = exit_stats.round(6)

    correlation_fields = [
        "reward_total",
        "reward_invalid",
        "reward_idle",
        "reward_holding",
        "reward_exit",
        "pnl",
        "trade_duration",
        "idle_duration",
    ]
    correlation = df[correlation_fields].corr().round(4)

    return {
        "idle_stats": idle_stats,
        "holding_stats": holding_stats,
        "exit_stats": exit_stats,
        "correlation": correlation,
    }


def write_relationship_reports(
    df: pd.DataFrame,
    output_dir: Path,
    max_trade_duration: int,
) -> None:
    """Legacy function - kept for backward compatibility."""
    output_dir.mkdir(parents=True, exist_ok=True)
    relationships_path = output_dir / "reward_relationships.md"
    stats = _compute_relationship_stats(df, max_trade_duration)

    with relationships_path.open("w", encoding="utf-8") as handle:
        handle.write("# Reward component relationships\n\n")

        handle.write("## Idle penalty by idle duration bins\n\n")
        if stats["idle_stats"].empty:
            handle.write("_No idle samples present._\n\n")
        else:
            handle.write(stats["idle_stats"].to_string())
            handle.write("\n\n")

        handle.write("## Holding penalty by trade duration bins\n\n")
        if stats["holding_stats"].empty:
            handle.write("_No holding samples present._\n\n")
        else:
            handle.write(stats["holding_stats"].to_string())
            handle.write("\n\n")

        handle.write("## Exit reward by PnL bins\n\n")
        if stats["exit_stats"].empty:
            handle.write("_No exit samples present._\n\n")
        else:
            handle.write(stats["exit_stats"].to_string())
            handle.write("\n\n")

        handle.write("## Correlation matrix\n\n")
        handle.write(stats["correlation"].to_csv(sep="\t", float_format="%.4f"))
        handle.write("\n")


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
    holding_activated = float((df["reward_holding"] != 0).mean())
    exit_activated = float((df["reward_exit"] != 0).mean())
    force_exit_share = float(df["is_force_exit"].mean())

    return {
        "total": total,
        "pos_counts": pos_counts,
        "act_counts": act_counts,
        "pnl_above_target": pnl_above_target,
        "pnl_near_target": pnl_near_target,
        "pnl_extreme": pnl_extreme,
        "duration_overage_share": duration_overage_share,
        "idle_activated": idle_activated,
        "holding_activated": holding_activated,
        "exit_activated": exit_activated,
        "force_exit_share": force_exit_share,
    }


def write_representativity_report(
    df: pd.DataFrame,
    output_dir: Path,
    profit_target: float,
    max_trade_duration: int,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / "representativity.md"

    stats = _compute_representativity_stats(df, profit_target)

    with path.open("w", encoding="utf-8") as h:
        h.write("# Representativity diagnostics\n\n")
        h.write(f"Total samples: {stats['total']}\n\n")
        h.write("## Position distribution\n\n")
        h.write(stats["pos_counts"].to_frame(name="count").to_string())
        h.write("\n\n")
        h.write("## Action distribution\n\n")
        h.write(stats["act_counts"].to_frame(name="count").to_string())
        h.write("\n\n")
        h.write("## Key regime coverage\n\n")
        h.write(f"pnl > target fraction: {stats['pnl_above_target']:.4f}\n")
        h.write(f"pnl near target [0.8,1.2] fraction: {stats['pnl_near_target']:.4f}\n")
        h.write(
            f"duration overage (>1.0) fraction: {stats['duration_overage_share']:.4f}\n"
        )
        h.write(f"idle activated fraction: {stats['idle_activated']:.4f}\n")
        h.write(f"holding activated fraction: {stats['holding_activated']:.4f}\n")
        h.write(f"exit activated fraction: {stats['exit_activated']:.4f}\n")
        h.write(f"force exit fraction: {stats['force_exit_share']:.4f}\n")
        h.write(f"extreme pnl (|pnl|>=0.14) fraction: {stats['pnl_extreme']:.4f}\n")
        h.write("\n")
        h.write(
            "Notes: Coverage of critical regimes (pnl≈target, overage>1) and component activation\n"
        )
        h.write(
            "are indicators of sufficient reward space representativity for the analysis.\n"
        )


def _perform_feature_analysis(
    df: pd.DataFrame, seed: int
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
        "force_action",
        "is_force_exit",
        "is_invalid",
    ]
    X = df[feature_cols]
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

    # Compute partial dependence for key features
    partial_deps = {}
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


def model_analysis(df: pd.DataFrame, output_dir: Path, seed: int) -> None:
    """Legacy wrapper for backward compatibility."""
    importance_df, analysis_stats, partial_deps, model = _perform_feature_analysis(
        df, seed
    )

    # Save feature importance
    importance_df.to_csv(output_dir / "feature_importance.csv", index=False)

    # Save diagnostics
    diagnostics_path = output_dir / "model_diagnostics.md"
    top_features = importance_df.head(10)
    with diagnostics_path.open("w", encoding="utf-8") as handle:
        handle.write("# Random forest diagnostics\n\n")
        handle.write(f"R^2 score on hold-out set: {analysis_stats['r2_score']:.4f}\n\n")
        handle.write("## Feature importance (top 10)\n\n")
        handle.write(top_features.to_string(index=False))
        handle.write("\n\n")
        handle.write(
            "Partial dependence data exported to CSV files for trade_duration, "
            "idle_duration, and pnl.\n"
        )

    # Save partial dependence data
    for feature, pd_df in partial_deps.items():
        pd_df.to_csv(
            output_dir / f"partial_dependence_{feature}.csv",
            index=False,
        )


def load_real_episodes(path: Path) -> pd.DataFrame:
    """Load real episodes transitions from pickle file."""
    with path.open("rb") as f:
        episodes_data = pickle.load(f)

    if (
        isinstance(episodes_data, list)
        and episodes_data
        and isinstance(episodes_data[0], dict)
    ):
        if "transitions" in episodes_data[0]:
            all_transitions = []
            for episode in episodes_data:
                all_transitions.extend(episode["transitions"])
            return pd.DataFrame(all_transitions)

    return pd.DataFrame(episodes_data)


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
        if math.isclose(max_val, min_val, rel_tol=0, abs_tol=1e-12):
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

    # Test 3: Force vs regular exits
    force_exits = df[df["is_force_exit"] == 1]["reward_exit"].dropna()
    regular_exits = df[(df["is_force_exit"] == 0) & (df["reward_exit"] != 0)][
        "reward_exit"
    ].dropna()

    if len(force_exits) >= 30 and len(regular_exits) >= 30:
        u_stat, p_val = stats.mannwhitneyu(
            force_exits, regular_exits, alternative="two-sided"
        )
        n1, n2 = len(force_exits), len(regular_exits)
        # Rank-biserial correlation (directional): r = 2*U1/(n1*n2) - 1
        # Compute U1 from sum of ranks of the first group for a robust sign.
        combined = np.concatenate([force_exits.values, regular_exits.values])
        ranks = stats.rankdata(combined, method="average")
        R1 = float(ranks[:n1].sum())
        U1 = R1 - n1 * (n1 + 1) / 2.0
        r_rb = (2.0 * U1) / (n1 * n2) - 1.0

        results["force_vs_regular_exits"] = {
            "test": "Mann-Whitney U",
            "statistic": float(u_stat),
            "p_value": float(p_val),
            "significant": bool(p_val < alpha),
            "effect_size_rank_biserial": float(r_rb),
            "median_force": float(force_exits.median()),
            "median_regular": float(regular_exits.median()),
            "n_force": len(force_exits),
            "n_regular": len(regular_exits),
        }

    # Test 4: PnL sign differences
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
) -> Dict[str, Tuple[float, float, float]]:
    """Bootstrap confidence intervals for key metrics."""
    alpha = 1 - confidence_level
    lower_percentile = 100 * alpha / 2
    upper_percentile = 100 * (1 - alpha / 2)

    results = {}

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
    _validate_bootstrap_results(results)

    return results


def _validate_bootstrap_results(results: Dict[str, Tuple[float, float, float]]) -> None:
    """Validate mathematical properties of bootstrap confidence intervals."""
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
            raise AssertionError(
                f"Bootstrap CI for {metric}: non-positive width {width:.6f}"
            )


def distribution_diagnostics(
    df: pd.DataFrame, *, seed: int | None = None
) -> Dict[str, Any]:
    """Distribution diagnostics: normality tests, skewness, kurtosis.

    Parameters
    ----------
    df : pd.DataFrame
        Input samples.
    seed : int | None, optional
        Reserved for future stochastic diagnostic extensions; currently unused.
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
        diagnostics[f"{col}_skewness"] = float(stats.skew(data))
        diagnostics[f"{col}_kurtosis"] = float(stats.kurtosis(data, fisher=True))

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

    _validate_distribution_diagnostics(diagnostics)
    return diagnostics


def _validate_distribution_diagnostics(diag: Dict[str, Any]) -> None:
    """Validate mathematical properties of distribution diagnostics.

    Ensures all reported statistics are finite and within theoretical bounds where applicable.
    Invoked automatically inside distribution_diagnostics(); raising AssertionError on violation
    enforces fail-fast semantics consistent with other validation helpers.
    """
    for key, value in diag.items():
        if any(suffix in key for suffix in ["_mean", "_std", "_skewness", "_kurtosis"]):
            if not np.isfinite(value):
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
                raise AssertionError(
                    f"Anderson statistic {key} must be finite, got {value}"
                )
        if key.endswith("_qq_r_squared"):
            if not (0 <= value <= 1):
                raise AssertionError(f"Q-Q R^2 {key} must be in [0,1], got {value}")


def write_enhanced_statistical_report(
    df: pd.DataFrame,
    output_dir: Path,
    real_df: Optional[pd.DataFrame] = None,
    *,
    adjust_method: str = "none",
) -> None:
    """Generate enhanced statistical report with hypothesis tests and CI."""
    output_dir.mkdir(parents=True, exist_ok=True)
    report_path = output_dir / "enhanced_statistical_report.md"

    # Derive a deterministic seed for statistical tests: prefer provided seed if present in df attrs else fallback
    test_seed = 42
    if (
        hasattr(df, "attrs")
        and "seed" in df.attrs
        and isinstance(df.attrs["seed"], int)
    ):
        test_seed = int(df.attrs["seed"])
    hypothesis_tests = statistical_hypothesis_tests(
        df, adjust_method=adjust_method, seed=test_seed
    )

    metrics_for_ci = [
        "reward_total",
        "reward_idle",
        "reward_holding",
        "reward_exit",
        "pnl",
    ]
    confidence_intervals = bootstrap_confidence_intervals(df, metrics_for_ci)

    dist_diagnostics = distribution_diagnostics(df)

    shift_metrics = {}
    if real_df is not None:
        shift_metrics = compute_distribution_shift_metrics(df, real_df)

    with report_path.open("w", encoding="utf-8") as f:
        f.write("# Enhanced Statistical Report\n\n")
        f.write("**Generated with rigorous scientific methodology**\n\n")

        f.write("## 1. Statistical Hypothesis Tests\n\n")
        for test_name, test_result in hypothesis_tests.items():
            f.write(f"### {test_name.replace('_', ' ').title()}\n\n")
            f.write(f"- **Test:** {test_result['test']}\n")
            f.write(
                f"- **Statistic:** {test_result.get('statistic', test_result.get('rho', 'N/A')):.4f}\n"
            )
            f.write(f"- **p-value:** {test_result['p_value']:.4e}\n")
            if "p_value_adj" in test_result:
                f.write(
                    f"- **p-value (adj BH):** {test_result['p_value_adj']:.4e} -> {'✅' if test_result['significant_adj'] else '❌'} (α=0.05)\n"
                )
            f.write(
                f"- **Significant (α=0.05):** {'✅ Yes' if test_result['significant'] else '❌ No'}\n"
            )

            if "ci_95" in test_result:
                ci = test_result["ci_95"]
                f.write(f"- **95% CI:** [{ci[0]:.4f}, {ci[1]:.4f}]\n")

            if "effect_size_epsilon_sq" in test_result:
                f.write(
                    f"- **Effect Size (ε²):** {test_result['effect_size_epsilon_sq']:.4f}\n"
                )

            if "interpretation" in test_result:
                f.write(f"- **Interpretation:** {test_result['interpretation']}\n")

            f.write("\n")

        f.write("## 2. Bootstrap Confidence Intervals (95%)\n\n")
        f.write("| Metric | Point Estimate | CI Lower | CI Upper | Width |\n")
        f.write("|--------|----------------|----------|----------|-------|\n")

        for metric, (est, low, high) in confidence_intervals.items():
            width = high - low
            f.write(
                f"| {metric} | {est:.4f} | {low:.4f} | {high:.4f} | {width:.4f} |\n"
            )

        f.write("\n## 3. Distribution Diagnostics\n\n")

        for col in ["reward_total", "pnl", "trade_duration"]:
            if f"{col}_mean" in dist_diagnostics:
                f.write(f"### {col}\n\n")
                f.write(f"- **Mean:** {dist_diagnostics[f'{col}_mean']:.4f}\n")
                f.write(f"- **Std:** {dist_diagnostics[f'{col}_std']:.4f}\n")
                f.write(f"- **Skewness:** {dist_diagnostics[f'{col}_skewness']:.4f}\n")
                f.write(f"- **Kurtosis:** {dist_diagnostics[f'{col}_kurtosis']:.4f}\n")

                if f"{col}_shapiro_pval" in dist_diagnostics:
                    is_normal = (
                        "✅ Yes"
                        if dist_diagnostics[f"{col}_is_normal_shapiro"]
                        else "❌ No"
                    )
                    f.write(
                        f"- **Normal (Shapiro-Wilk):** {is_normal} (p={dist_diagnostics[f'{col}_shapiro_pval']:.4e})\n"
                    )

                if f"{col}_qq_r_squared" in dist_diagnostics:
                    f.write(
                        f"- **Q-Q R²:** {dist_diagnostics[f'{col}_qq_r_squared']:.4f}\n"
                    )

                f.write("\n")

        if shift_metrics:
            f.write("## 4. Distribution Shift Metrics (Synthetic vs Real)\n\n")
            f.write("| Feature | KL Div | JS Dist | Wasserstein | KS p-value |\n")
            f.write("|---------|--------|---------|-------------|------------|\n")

            for feature in ["pnl", "trade_duration", "idle_duration"]:
                kl_key = f"{feature}_kl_divergence"
                if kl_key in shift_metrics:
                    f.write(f"| {feature} | {shift_metrics[kl_key]:.4f} | ")
                    f.write(f"{shift_metrics[f'{feature}_js_distance']:.4f} | ")
                    f.write(f"{shift_metrics[f'{feature}_wasserstein']:.4f} | ")
                    f.write(f"{shift_metrics[f'{feature}_ks_pvalue']:.4e} |\n")

            f.write("\n**Interpretation:**\n")
            f.write("- KL/JS (distance) < 0.2: Acceptable similarity\n")
            f.write("- Wasserstein: Lower is better\n")
            f.write(
                "- KS p-value > 0.05: Distributions not significantly different\n\n"
            )

        f.write("## 5. Methodological Recommendations\n\n")

        has_issues = []

        if "reward_total_is_normal_shapiro" in dist_diagnostics:
            if not dist_diagnostics["reward_total_is_normal_shapiro"]:
                has_issues.append(
                    "⚠️ **Non-normal reward distribution:** Use non-parametric tests"
                )

        if shift_metrics:
            high_divergence = any(
                shift_metrics.get(f"{feat}_kl_divergence", 0) > 0.5
                for feat in ["pnl", "trade_duration", "idle_duration"]
            )
            if high_divergence:
                has_issues.append(
                    "🔴 **High distribution shift:** Consider real episode sampling"
                )

        if has_issues:
            f.write("**Issues identified:**\n\n")
            for issue in has_issues:
                f.write(f"- {issue}\n")
        else:
            f.write("✅ **No major methodological issues detected.**\n")

        f.write("\n---\n\n")
        f.write(
            "**References:** Efron & Tibshirani (1993), Henderson et al. (2018), Pineau et al. (2021)\n"
        )


def build_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Synthetic stress-test of the ReforceXY reward shaping logic."
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
        "--holding_max_ratio",
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
        help="Override reward parameters, e.g. holding_penalty_scale=0.5",
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

    # Model analysis
    importance_df, analysis_stats, partial_deps, _model = _perform_feature_analysis(
        df, seed
    )

    # Save feature importance CSV
    importance_df.to_csv(output_dir / "feature_importance.csv", index=False)

    # Save partial dependence CSVs
    for feature, pd_df in partial_deps.items():
        pd_df.to_csv(
            output_dir / f"partial_dependence_{feature}.csv",
            index=False,
        )

    # Enhanced statistics (always computed)
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
        "reward_holding",
        "reward_exit",
        "pnl",
    ]
    bootstrap_ci = bootstrap_confidence_intervals(
        df, metrics_for_ci, n_bootstrap=10000, seed=test_seed
    )
    dist_diagnostics = distribution_diagnostics(df, seed=test_seed)

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
        f.write(f"| Profit Target (effective) | {profit_target:.4f} |\n\n")

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
        if action_df.index.name is None:
            action_df.index.name = "action"
        f.write(_df_to_md(action_df, index_name=action_df.index.name, ndigits=6))

        f.write("### 1.3 Component Activation Rates\n\n")
        f.write("Percentage of samples where each reward component is non-zero:\n\n")
        f.write(
            _series_to_md(
                summary_stats["component_share"],
                value_name="activation_rate",
                ndigits=6,
            )
        )

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
            f"| Extreme PnL (|pnl|≥0.14) | {representativity_stats['pnl_extreme']:.1%} |\n"
        )
        f.write("\n")

        f.write("### 2.4 Component Activation Coverage\n\n")
        f.write("| Component | Activation Rate |\n")
        f.write("|-----------|----------------|\n")
        f.write(f"| Idle penalty | {representativity_stats['idle_activated']:.1%} |\n")
        f.write(
            f"| Holding penalty | {representativity_stats['holding_activated']:.1%} |\n"
        )
        f.write(f"| Exit reward | {representativity_stats['exit_activated']:.1%} |\n")
        f.write(f"| Force exit | {representativity_stats['force_exit_share']:.1%} |\n")
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

        f.write("### 3.2 Holding Penalty vs Trade Duration\n\n")
        if relationship_stats["holding_stats"].empty:
            f.write("_No holding samples present._\n\n")
        else:
            holding_df = relationship_stats["holding_stats"].copy()
            if holding_df.index.name is None:
                holding_df.index.name = "bin"
            f.write(_df_to_md(holding_df, index_name=holding_df.index.name, ndigits=6))

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

        # Section 4: Feature Importance Analysis
        f.write("---\n\n")
        f.write("## 4. Feature Importance\n\n")
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
        f.write("- Partial dependence plots: `partial_dependence_*.csv`\n\n")

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
                        f"- p-value (adj BH): {h['p_value_adj']:.4g} -> {'✅' if h['significant_adj'] else '❌'} (α=0.05)\n"
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
                        f"- p-value (adj BH): {h['p_value_adj']:.4g} -> {'✅' if h['significant_adj'] else '❌'} (α=0.05)\n"
                    )
                f.write(f"- Effect size (ε²): {h['effect_size_epsilon_sq']:.4f}\n")
                f.write(f"- Number of groups: {h['n_groups']}\n")
                f.write(
                    f"- Significant (α=0.05): {'✅ Yes' if h['significant'] else '❌ No'}\n"
                )
                f.write(f"- **Interpretation:** {h['interpretation']} effect\n\n")

            if "force_vs_regular_exits" in hypothesis_tests:
                h = hypothesis_tests["force_vs_regular_exits"]
                f.write("#### 5.1.3 Force vs Regular Exits Comparison\n\n")
                f.write(f"**Test Method:** {h['test']}\n\n")
                f.write(f"- U-statistic: **{h['statistic']:.4f}**\n")
                f.write(f"- p-value: {h['p_value']:.4g}\n")
                if "p_value_adj" in h:
                    f.write(
                        f"- p-value (adj BH): {h['p_value_adj']:.4g} -> {'✅' if h['significant_adj'] else '❌'} (α=0.05)\n"
                    )
                f.write(
                    f"- Effect size (rank-biserial): {h['effect_size_rank_biserial']:.4f}\n"
                )
                f.write(f"- Median (force): {h['median_force']:.4f}\n")
                f.write(f"- Median (regular): {h['median_regular']:.4f}\n")
                f.write(
                    f"- Significant (α=0.05): {'✅ Yes' if h['significant'] else '❌ No'}\n\n"
                )

            if "pnl_sign_reward_difference" in hypothesis_tests:
                h = hypothesis_tests["pnl_sign_reward_difference"]
                f.write("#### 5.1.4 Positive vs Negative PnL Comparison\n\n")
                f.write(f"**Test Method:** {h['test']}\n\n")
                f.write(f"- U-statistic: **{h['statistic']:.4f}**\n")
                f.write(f"- p-value: {h['p_value']:.4g}\n")
                if "p_value_adj" in h:
                    f.write(
                        f"- p-value (adj BH): {h['p_value_adj']:.4g} -> {'✅' if h['significant_adj'] else '❌'} (α=0.05)\n"
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
                f.write("| KL Divergence | < 0.3 | ✅ Good representativeness |\n")
                f.write("| JS Distance | < 0.2 | ✅ Similar distributions |\n")
                f.write("| KS p-value | > 0.05 | ✅ No significant difference |\n\n")

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
            "3. **Component Analysis** - Relationships between rewards and conditions\n"
        )
        f.write(
            "4. **Feature Importance** - Machine learning analysis of key drivers\n"
        )
        f.write(
            "5. **Statistical Validation** - Hypothesis tests and confidence intervals\n"
        )
        if distribution_shift:
            f.write("6. **Distribution Shift** - Comparison with real trading data\n")
        f.write("\n")
        f.write("**Generated Files:**\n")
        f.write("- `reward_samples.csv` - Raw synthetic samples\n")
        f.write("- `feature_importance.csv` - Complete feature importance rankings\n")
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

    base_factor = float(params.get("base_factor", args.base_factor))
    profit_target = float(params.get("profit_target", args.profit_target))
    rr_override = params.get("rr")
    if not isinstance(rr_override, (int, float)):
        rr_override = params.get("risk_reward_ratio", args.risk_reward_ratio)
    risk_reward_ratio = float(rr_override)

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
        holding_max_ratio=args.holding_max_ratio,
        trading_mode=args.trading_mode,
        pnl_base_std=args.pnl_base_std,
        pnl_duration_vol_scale=args.pnl_duration_vol_scale,
    )
    # Attach simulation parameters for downstream manifest
    df.attrs["simulation_params"] = {
        "num_samples": args.num_samples,
        "seed": args.seed,
        "max_trade_duration": args.max_trade_duration,
        "base_factor": base_factor,
        "profit_target": profit_target,
        "risk_reward_ratio": risk_reward_ratio,
        "holding_max_ratio": args.holding_max_ratio,
        "trading_mode": args.trading_mode,
        "action_masking": params.get("action_masking", True),
        "pnl_base_std": args.pnl_base_std,
        "pnl_duration_vol_scale": args.pnl_duration_vol_scale,
    }

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
    )
    print(
        f"Complete statistical analysis saved to: {args.output / 'statistical_analysis.md'}"
    )
    # Generate manifest summarizing key metrics
    try:
        manifest_path = args.output / "manifest.json"
        if (args.output / "feature_importance.csv").exists():
            fi_df = pd.read_csv(args.output / "feature_importance.csv")
            top_features = fi_df.head(5)["feature"].tolist()
        else:
            top_features = []
        # Detect reward parameter overrides vs defaults for traceability
        reward_param_overrides = {
            k: params[k]
            for k in DEFAULT_MODEL_REWARD_PARAMETERS
            if k in params and params[k] != DEFAULT_MODEL_REWARD_PARAMETERS[k]
        }

        manifest = {
            "generated_at": pd.Timestamp.now().isoformat(),
            "num_samples": int(len(df)),
            "seed": int(args.seed),
            "max_trade_duration": int(args.max_trade_duration),
            "profit_target_effective": float(profit_target * risk_reward_ratio),
            "top_features": top_features,
            "reward_param_overrides": reward_param_overrides,
            "pvalue_adjust_method": args.pvalue_adjust,
            "parameter_adjustments": adjustments,
        }
        sim_params = df.attrs.get("simulation_params", {})
        if isinstance(sim_params, dict) and sim_params:
            import hashlib as _hashlib
            import json as _json

            _hash_source = {
                **{f"sim::{k}": sim_params[k] for k in sorted(sim_params)},
                **{
                    f"reward::{k}": reward_param_overrides[k]
                    for k in sorted(reward_param_overrides)
                },
            }
            serialized = _json.dumps(_hash_source, sort_keys=True)
            manifest["params_hash"] = _hashlib.sha256(
                serialized.encode("utf-8")
            ).hexdigest()
            manifest["params"] = sim_params
        with manifest_path.open("w", encoding="utf-8") as mh:
            import json as _json

            _json.dump(manifest, mh, indent=2)
        print(f"Manifest written to: {manifest_path}")
    except Exception as e:
        print(f"Manifest generation failed: {e}")

    print(f"Generated {len(df):,} synthetic samples.")
    print(sample_output_message)
    print(f"Artifacts saved to: {args.output.resolve()}")


if __name__ == "__main__":
    main()
