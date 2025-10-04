#!/usr/bin/env python3
"""
Comprehensive regression test for reward_space_analysis.py

This script checks critical aspects of alignment with MyRLEnv.
Use this test to validate future changes and avoid regressions.

Usage:
    python test_reward_alignment.py
"""

import sys

from reward_space_analysis import (
    DEFAULT_MODEL_REWARD_PARAMETERS,
    Actions,
    ForceActions,
    Positions,
    RewardContext,
    _get_exit_factor,
    _get_pnl_factor,
    _is_valid_action,
    calculate_reward,
)


def test_enums():
    """Ensure enums have the expected values"""
    print("Testing enums...")
    assert Actions.Neutral.value == 0
    assert Actions.Long_enter.value == 1
    assert Actions.Long_exit.value == 2
    assert Actions.Short_enter.value == 3
    assert Actions.Short_exit.value == 4

    assert Positions.Short.value == 0
    assert Positions.Long.value == 1
    assert Positions.Neutral.value == 0.5

    assert ForceActions.Take_profit.value == 0
    assert ForceActions.Stop_loss.value == 1
    assert ForceActions.Timeout.value == 2
    print("  ✅ Enums OK")


def test_default_parameters():
    """Ensure default parameters are correct"""
    print("Testing default parameters...")
    assert DEFAULT_MODEL_REWARD_PARAMETERS["base_factor"] == 100.0
    assert DEFAULT_MODEL_REWARD_PARAMETERS["idle_penalty_scale"] == 1.0
    assert DEFAULT_MODEL_REWARD_PARAMETERS["idle_penalty_power"] == 1.0
    assert DEFAULT_MODEL_REWARD_PARAMETERS["holding_penalty_scale"] == 0.3
    assert DEFAULT_MODEL_REWARD_PARAMETERS["holding_penalty_power"] == 1.0
    assert DEFAULT_MODEL_REWARD_PARAMETERS["holding_duration_ratio_grace"] == 1.0
    # Ensure max_idle_duration_candles is NOT in defaults
    assert "max_idle_duration_candles" not in DEFAULT_MODEL_REWARD_PARAMETERS
    print("  ✅ Default parameters OK")


def test_invalid_action_penalty():
    """Ensure invalid action penalty is applied correctly"""
    print("Testing invalid action penalty...")
    ctx = RewardContext(
        pnl=0.02,
        trade_duration=50,
        idle_duration=0,
        max_trade_duration=100,
        max_unrealized_profit=0.03,
        min_unrealized_profit=0.01,
        position=Positions.Neutral,
        action=Actions.Long_exit,  # Invalid: Long_exit in Neutral
        force_action=None,
    )

    # Without masking → penalty
    bd_no_mask = calculate_reward(
        ctx,
        DEFAULT_MODEL_REWARD_PARAMETERS,
        base_factor=100.0,
        profit_target=0.03,
        risk_reward_ratio=2.0,
        short_allowed=True,
        action_masking=False,
    )
    assert bd_no_mask.invalid_penalty == -2.0
    assert bd_no_mask.total == -2.0

    # With masking → no penalty
    bd_with_mask = calculate_reward(
        ctx,
        DEFAULT_MODEL_REWARD_PARAMETERS,
        base_factor=100.0,
        profit_target=0.03,
        risk_reward_ratio=2.0,
        short_allowed=True,
        action_masking=True,
    )
    assert bd_with_mask.invalid_penalty == 0.0
    assert bd_with_mask.total == 0.0
    print("  ✅ Invalid action penalty OK")


def test_idle_penalty_no_capping():
    """Ensure idle penalty has NO capping"""
    print("Testing idle penalty (no capping)...")
    ctx = RewardContext(
        pnl=0.0,
        trade_duration=0,
        idle_duration=200,  # ratio = 2.0
        max_trade_duration=100,
        max_unrealized_profit=0.0,
        min_unrealized_profit=0.0,
        position=Positions.Neutral,
        action=Actions.Neutral,
        force_action=None,
    )
    bd = calculate_reward(
        ctx,
        DEFAULT_MODEL_REWARD_PARAMETERS,
        base_factor=100.0,
        profit_target=0.03,
        risk_reward_ratio=2.0,
        short_allowed=True,
        action_masking=True,
    )
    # idle_factor = (100.0 * 0.03 * 2.0) / 3.0 = 2.0
    # idle_penalty = -2.0 * 1.0 * 2.0^1.0 = -4.0
    assert abs(bd.idle_penalty - (-4.0)) < 0.001
    print("  ✅ Idle penalty (no capping) OK")


def test_force_exit():
    """Ensure force exits take priority"""
    print("Testing force exit...")
    ctx = RewardContext(
        pnl=0.04,
        trade_duration=50,
        idle_duration=0,
        max_trade_duration=100,
        max_unrealized_profit=0.05,
        min_unrealized_profit=0.03,
        position=Positions.Long,
        action=Actions.Long_exit,
        force_action=ForceActions.Take_profit,
    )
    bd = calculate_reward(
        ctx,
        DEFAULT_MODEL_REWARD_PARAMETERS,
        base_factor=100.0,
        profit_target=0.03,
        risk_reward_ratio=2.0,
        short_allowed=True,
        action_masking=True,
    )
    assert bd.exit_component > 0
    assert bd.total == bd.exit_component
    print("  ✅ Force exit OK")


def test_holding_penalty_grace():
    """Ensure grace period works correctly"""
    print("Testing holding penalty with grace period...")

    # Case 1: Below target, within grace → reward = 0
    ctx1 = RewardContext(
        pnl=0.04,  # Below target (0.06)
        trade_duration=50,
        idle_duration=0,
        max_trade_duration=100,
        max_unrealized_profit=0.05,
        min_unrealized_profit=0.03,
        position=Positions.Long,
        action=Actions.Neutral,
        force_action=None,
    )
    bd1 = calculate_reward(
        ctx1,
        DEFAULT_MODEL_REWARD_PARAMETERS,
        base_factor=100.0,
        profit_target=0.03,
        risk_reward_ratio=2.0,
        short_allowed=True,
        action_masking=True,
    )
    assert bd1.total == 0.0

    ctx2 = RewardContext(
        pnl=0.08,
        trade_duration=50,
        idle_duration=0,
        max_trade_duration=100,
        max_unrealized_profit=0.10,
        min_unrealized_profit=0.05,
        position=Positions.Long,
        action=Actions.Neutral,
        force_action=None,
    )
    bd2 = calculate_reward(
        ctx2,
        DEFAULT_MODEL_REWARD_PARAMETERS,
        base_factor=100.0,
        profit_target=0.03,
        risk_reward_ratio=2.0,
        short_allowed=True,
        action_masking=True,
    )
    assert bd2.holding_penalty < 0
    print("  ✅ Holding penalty with grace OK")


def test_exit_factor():
    """Ensure exit_factor is computed correctly"""
    print("Testing exit factor...")
    params = DEFAULT_MODEL_REWARD_PARAMETERS.copy()
    params["exit_reward_mode"] = "piecewise"
    factor = _get_exit_factor(
        factor=100.0, pnl=0.05, pnl_factor=1.5, duration_ratio=0.5, params=params
    )
    # Piecewise mode with duration_ratio=0.5 should yield 100.0 * 1.5 = 150.0
    assert abs(factor - 150.0) < 0.1
    print("  ✅ Exit factor OK")


def test_pnl_factor():
    """Ensure pnl_factor includes efficiency"""
    print("Testing PnL factor...")
    params = DEFAULT_MODEL_REWARD_PARAMETERS.copy()
    context = RewardContext(
        pnl=0.05,
        trade_duration=50,
        idle_duration=0,
        max_trade_duration=100,
        max_unrealized_profit=0.08,
        min_unrealized_profit=0.01,
        position=Positions.Long,
        action=Actions.Long_exit,
        force_action=None,
    )
    pnl_factor = _get_pnl_factor(params, context, profit_target=0.03)
    # Should include amplification + efficiency
    assert pnl_factor > 1.0  # Profit above target
    print("  ✅ PnL factor OK")


def test_short_exit_without_short_allowed():
    """Ensure Short_exit is valid even if short_allowed=False"""
    print("Testing short exit without short_allowed...")

    # Validation
    assert _is_valid_action(
        Positions.Short, Actions.Short_exit, short_allowed=False, force_action=None
    )

    # Reward
    ctx = RewardContext(
        pnl=0.03,
        trade_duration=40,
        idle_duration=0,
        max_trade_duration=100,
        max_unrealized_profit=0.04,
        min_unrealized_profit=0.02,
        position=Positions.Short,
        action=Actions.Short_exit,
        force_action=None,
    )
    bd = calculate_reward(
        ctx,
        DEFAULT_MODEL_REWARD_PARAMETERS,
        base_factor=100.0,
        profit_target=0.03,
        risk_reward_ratio=2.0,
        short_allowed=False,  # Short not allowed
        action_masking=True,
    )
    assert bd.invalid_penalty == 0.0
    assert bd.exit_component > 0  # Reward positif car profit
    print("  ✅ Short exit without short_allowed OK")


def run_all_tests():
    """Run all tests"""
    print("\n" + "=" * 60)
    print("REGRESSION TEST - REWARD SPACE ANALYSIS")
    print("=" * 60 + "\n")

    tests = [
        test_enums,
        test_default_parameters,
        test_invalid_action_penalty,
        test_idle_penalty_no_capping,
        test_force_exit,
        test_holding_penalty_grace,
        test_exit_factor,
        test_pnl_factor,
        test_short_exit_without_short_allowed,
    ]

    failed = []
    for test in tests:
        try:
            test()
        except AssertionError as e:
            print(f"  ❌ FAILED: {e}")
            failed.append(test.__name__)
        except Exception as e:
            print(f"  ❌ ERROR: {e}")
            failed.append(test.__name__)

    print("\n" + "=" * 60)
    if failed:
        print(f"❌ {len(failed)} failed test(s):")
        for name in failed:
            print(f"   - {name}")
        sys.exit(1)
    else:
        print("✅ ALL TESTS PASSED")
        print("\nCode is aligned with MyRLEnv and ready for production.")
    print("=" * 60 + "\n")


if __name__ == "__main__":
    run_all_tests()
