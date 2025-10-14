# 📊 Reward Space Analysis - User Guide

**Analyze and validate ReforceXY reward logic with synthetic data**

---

## 🎯 What is this?

This tool helps you understand and validate how the ReforceXY reinforcement learning environment calculates rewards. It generates synthetic trading scenarios to analyze reward behavior across different market conditions.

### Key Features

- ✅ Generate thousands of synthetic trading scenarios deterministically
- ✅ Analyze reward distribution, feature importance & partial dependence
- ✅ Built-in invariant & statistical validation layers (fail-fast)
- ✅ PBRS (Potential-Based Reward Shaping) integration with canonical invariance
- ✅ Export reproducible artifacts (parameter hash + execution manifest)
- ✅ Compare synthetic vs real trading data (distribution shift metrics)
- ✅ Parameter bounds validation & automatic sanitization

---

**New to this tool?** Start with [Common Use Cases](#-common-use-cases) then explore [CLI Parameters](#️-cli-parameters-reference).

## Table of contents

- [What is this?](#-what-is-this)
- [Key Features](#key-features)
- [Common Use Cases](#-common-use-cases)
    - [1. Validate Reward Logic](#1-validate-reward-logic)
    - [2. Analyze Parameter Sensitivity](#2-analyze-parameter-sensitivity)
    - [3. Debug Reward Issues](#3-debug-reward-issues)
    - [4. Compare Real vs Synthetic Data](#4-compare-real-vs-synthetic-data)
- [Prerequisites](#-prerequisites)
    - [System Requirements](#system-requirements)
    - [Virtual environment setup](#virtual-environment-setup)
- [CLI Parameters Reference](#️-cli-parameters-reference)
    - [Required Parameters](#required-parameters)
    - [Core Simulation Parameters](#core-simulation-parameters)
    - [Reward Configuration](#reward-configuration)
    - [PnL / Volatility Controls](#pnl--volatility-controls)
    - [Trading Environment](#trading-environment)
    - [Output Configuration](#output-configuration)
    - [Reproducibility Model](#reproducibility-model)
    - [Direct Tunable Overrides vs `--params`](#direct-tunable-overrides-vs---params)
- [Example Commands](#-example-commands)
- [Understanding Results](#-understanding-results)
    - [Main Report](#main-report)
    - [Data Exports](#data-exports)
    - [Manifest Structure (`manifest.json`)](#manifest-structure-manifestjson)
    - [Distribution Shift Metric Conventions](#distribution-shift-metric-conventions)
- [Advanced Usage](#-advanced-usage)
    - [Custom Parameter Testing](#custom-parameter-testing)
    - [Real Data Comparison](#real-data-comparison)
    - [Batch Analysis](#batch-analysis)
- [Validation & Testing](#-validation--testing)
    - [Run Tests](#run-tests)
    - [Test Categories](#test-categories)
    - [Test Architecture](#test-architecture)
    - [Code Coverage Analysis](#code-coverage-analysis)
    - [When to Run Tests](#when-to-run-tests)
    - [Run Specific Test Categories](#run-specific-test-categories)
- [Troubleshooting](#-troubleshooting)
    - [No Output Files Generated](#no-output-files-generated)
    - [Unexpected Reward Values](#unexpected-reward-values)
    - [Slow Execution](#slow-execution)
    - [Memory Errors](#memory-errors)

## 📦 Prerequisites

### System Requirements

- Python 3.8+
- 4GB RAM minimum (8GB recommended for large analyses)
- No GPU required

### Virtual environment setup

Keep the tooling self-contained by creating a virtual environment directly inside `ReforceXY/reward_space_analysis` and installing packages against it:

```shell
# From the repository root
cd ReforceXY/reward_space_analysis
python -m venv .venv
source .venv/bin/activate
pip install pandas numpy scipy scikit-learn
```

Whenever you need to run analyses, activate the environment first and execute:

```shell
source .venv/bin/activate
python reward_space_analysis.py --num_samples 20000 --output reward_space_outputs
```

> Deactivate the environment with `deactivate` when you're done.

Unless otherwise noted, the command examples below assume your current working directory is `ReforceXY/reward_space_analysis` and the virtual environment is activated.

---

## 💡 Common Use Cases

### 1. Validate Reward Logic

**Goal:** Ensure rewards behave as expected in different scenarios

```shell
python reward_space_analysis.py --num_samples 20000 --output reward_space_outputs
```

**Check in `statistical_analysis.md`:**

- Long/Short exits should have positive average rewards
- Invalid actions should have negative penalties (default: -2.0)
- Idle periods should reduce rewards progressively
- Validation layers report any invariant violations

### 2. Analyze Parameter Sensitivity

**Goal:** See how reward parameters affect trading behavior

```shell
# Test different win reward factors
python reward_space_analysis.py \
    --num_samples 30000 \
    --params win_reward_factor=2.0 \
    --output conservative_rewards

python reward_space_analysis.py \
    --num_samples 30000 \
    --params win_reward_factor=4.0 \
    --output aggressive_rewards

# Test PBRS potential shaping
python reward_space_analysis.py \
    --num_samples 30000 \
    --params hold_potential_enabled=true potential_gamma=0.9 exit_potential_mode=progressive_release \
    --output pbrs_analysis
```

**Compare:** Reward distributions between runs in `statistical_analysis.md`

### 3. Debug Reward Issues

**Goal:** Identify why your RL agent behaves unexpectedly

```shell
# Generate detailed analysis
python reward_space_analysis.py \
    --num_samples 50000 \
    --output debug_analysis
```

**Look at:**

- `statistical_analysis.md` - Comprehensive report with:
  - Feature importance and model diagnostics
  - Statistical significance of relationships
  - Hypothesis tests and confidence intervals

### 4. Compare Real vs Synthetic Data

**Goal:** Validate synthetic analysis against real trading

```shell
# First, collect real episodes (see Advanced Usage section)
# Then compare:
python reward_space_analysis.py \
    --num_samples 100000 \
    --real_episodes path/to/episode_rewards.pkl \
    --output real_vs_synthetic
```

---

## ⚙️ CLI Parameters Reference

### Required Parameters

None - all parameters have sensible defaults.

### Core Simulation Parameters

**`--num_samples`** (int, default: 20000)

- Number of synthetic trading scenarios to generate
- More samples = more accurate statistics but slower analysis
- Recommended: 10,000 (quick test), 50,000 (standard), 100,000+ (detailed)

**`--seed`** (int, default: 42)

- Random seed for reproducibility
- Use same seed to get identical results across runs

**`--max_trade_duration`** (int, default: 128)

- Maximum trade duration in candles (from environment config)
- Should match your actual trading environment setting
- Drives idle grace: when `max_idle_duration_candles` fallback = `2 * max_trade_duration`

### Reward Configuration

**`--base_factor`** (float, default: 100.0)

- Base reward scaling factor (from environment config)
- Should match your environment's base_factor

**`--profit_target`** (float, default: 0.03)

- Target profit threshold as decimal (e.g., 0.03 = 3%)
- Used for exit reward

**`--risk_reward_ratio`** (float, default: 1.0)

- Risk/reward ratio multiplier
- Affects profit target adjustment in reward calculations

**`--max_duration_ratio`** (float, default: 2.5)

- Multiple of max_trade_duration used for sampling trade/idle durations
- Higher = more variety in duration scenarios

### PnL / Volatility Controls

These parameters shape the synthetic PnL generation process and heteroscedasticity (variance increasing with duration):

**`--pnl_base_std`** (float, default: 0.02)
- Base standard deviation (volatility floor) for generated PnL before duration scaling.

**`--pnl_duration_vol_scale`** (float, default: 0.5)
- Multiplicative scaling of additional volatility proportional to (trade_duration / max_trade_duration).
- Higher values = stronger heteroscedastic effect.

### Trading Environment

**`--trading_mode`** (choice: spot|margin|futures, default: spot)

- **spot**: Disables short selling
- **margin**: Enables short positions
- **futures**: Enables short positions

**`--action_masking`** (choice: true|false|1|0|yes|no, default: true)

- Enable/disable action masking simulation
- Should match your environment configuration

### Output Configuration

**`--output`** (path, default: reward_space_outputs)

- Output directory for all generated files
- Will be created if it doesn't exist

**`--params`** (key=value pairs)

- Override any reward parameter from DEFAULT_MODEL_REWARD_PARAMETERS
- Format: `--params key1=value1 key2=value2`
- Example: `--params win_reward_factor=3.0 idle_penalty_scale=2.0`

**All tunable parameters (override with --params):**

_Invalid action penalty:_

- `invalid_action` (default: -2.0) - Penalty for invalid actions

_Idle penalty configuration:_

- `idle_penalty_scale` (default: 0.5) - Scale of idle penalty
- `idle_penalty_power` (default: 1.025) - Power applied to idle penalty scaling

_Hold penalty configuration:_

- `hold_penalty_scale` (default: 0.25) - Scale of hold penalty
- `hold_penalty_power` (default: 1.025) - Power applied to hold penalty scaling

_Exit attenuation configuration:_

- `exit_attenuation_mode` (default: linear) - Selects attenuation kernel (see table below: legacy|sqrt|linear|power|half_life). Fallback to linear.
- `exit_plateau` (default: true) - Enables plateau (no attenuation until `exit_plateau_grace`).
- `exit_plateau_grace` (default: 1.0) - Duration ratio boundary of full-strength region (may exceed 1.0).
- `exit_linear_slope` (default: 1.0) - Slope parameter used only when mode = linear.
- `exit_power_tau` (default: 0.5) - Tau ∈ (0,1]; internally mapped to alpha (see kernel table).
- `exit_half_life` (default: 0.5) - Half-life parameter for the half_life kernel.
- `exit_factor_threshold` (default: 10000.0) - Warning-only soft threshold (emits RuntimeWarning; no capping).

Attenuation kernels:

Let r be the raw duration ratio and grace = `exit_plateau_grace`.

```
effective_r = 0            if exit_plateau and r <= grace
effective_r = r - grace    if exit_plateau and r >  grace
effective_r = r            if not exit_plateau
```

| Mode | Multiplier (applied to base_factor * pnl * pnl_factor * efficiency_factor) | Monotonic decreasing (Yes/No) | Notes |
|------|---------------------------------------------------------------------|-------------------------------|-------|
| legacy | step: ×1.5 if r* ≤ 1 else ×0.5 | No | Historical discontinuity retained (not smoothed) |
| sqrt | 1 / sqrt(1 + r*) | Yes | Sub-linear decay |
| linear | 1 / (1 + slope * r*) | Yes | slope = `exit_linear_slope` (≥0) |
| power | (1 + r*)^(-alpha) | Yes | alpha = -ln(tau)/ln(2), tau = `exit_power_tau` ∈ (0,1]; tau=1 ⇒ alpha=0 (flat) |
| half_life | 2^(- r* / hl) | Yes | hl = `exit_half_life`; r* = hl ⇒ factor × 0.5 |

Where r* = `effective_r` above.

_Efficiency configuration:_

- `efficiency_weight` (default: 1.0) - Weight for efficiency factor in exit reward
- `efficiency_center` (default: 0.5) - Linear pivot in [0,1] for efficiency ratio. If efficiency_ratio > center ⇒ amplification (>1); if < center ⇒ attenuation (<1, floored at 0).

_Profit factor configuration:_

- `win_reward_factor` (default: 2.0) - Asymptotic bonus multiplier for PnL above target. Raw `profit_target_factor` ∈ [1, 1 + win_reward_factor] (tanh bounds it); overall amplification may exceed this once multiplied by `efficiency_factor`.
- `pnl_factor_beta` (default: 0.5) - Sensitivity of amplification around target

_PBRS (Potential-Based Reward Shaping) configuration:_

- `potential_gamma` (default: 0.95) - Discount factor γ for PBRS potential term (0 ≤ γ ≤ 1)
- `potential_softsign_sharpness` (default: 1.0) - Sharpness parameter for softsign_sharp transform (smaller = sharper)
- `exit_potential_mode` (default: canonical) - Exit potential mode: 'canonical' (Φ=0), 'progressive_release', 'spike_cancel', 'retain_previous'
- `exit_potential_decay` (default: 0.5) - Decay factor for progressive_release exit mode (0 ≤ decay ≤ 1)
- `hold_potential_enabled` (default: true) - Enable PBRS hold potential function Φ(s)
- `hold_potential_scale` (default: 1.0) - Scale factor for hold potential function
- `hold_potential_gain` (default: 1.0) - Gain factor applied before transforms in hold potential
- `hold_potential_transform_pnl` (default: tanh) - Transform function for PnL: tanh, softsign, softsign_sharp, arctan, logistic, asinh_norm, clip
- `hold_potential_transform_duration` (default: tanh) - Transform function for duration ratio
- `entry_additive_enabled` (default: false) - Enable entry additive reward (non-PBRS component)
- `entry_additive_scale` (default: 1.0) - Scale factor for entry additive reward
- `entry_additive_gain` (default: 1.0) - Gain factor for entry additive reward
- `entry_additive_transform_pnl` (default: tanh) - Transform function for PnL in entry additive
- `entry_additive_transform_duration` (default: tanh) - Transform function for duration ratio in entry additive
- `exit_additive_enabled` (default: false) - Enable exit additive reward (non-PBRS component)
- `exit_additive_scale` (default: 1.0) - Scale factor for exit additive reward
- `exit_additive_gain` (default: 1.0) - Gain factor for exit additive reward
- `exit_additive_transform_pnl` (default: tanh) - Transform function for PnL in exit additive
- `exit_additive_transform_duration` (default: tanh) - Transform function for duration ratio in exit additive

_Invariant / safety controls:_

- `check_invariants` (default: true) - Enable/disable runtime invariant & safety validations (simulation invariants, mathematical bounds, distribution checks). Set to `false` only for performance experiments; not recommended for production validation.

**`--real_episodes`** (path, optional)

- Path to real episode rewards pickle file for distribution comparison
- Enables distribution shift analysis (KL(synthetic‖real), JS distance, Wasserstein distance, KS test)
- Example: `path/to/episode_rewards.pkl`

**`--pvalue_adjust`** (choice: none|benjamini_hochberg, default: none)

- Apply multiple hypothesis testing correction (Benjamini–Hochberg) to p-values in statistical hypothesis tests.
- When set to `benjamini_hochberg`, adjusted p-values and adjusted significance flags are added to the reports.

**`--stats_seed`** (int, optional; default: inherit `--seed`)

- Dedicated seed for statistical analyses (hypothesis tests, bootstrap confidence intervals, distribution diagnostics).
- Use this if you want to generate multiple independent statistical analyses over the same synthetic dataset without re-simulating samples.
- If omitted, falls back to `--seed` for full run determinism.

**`--strict_diagnostics`** (flag, default: disabled)

Fail-fast switch controlling handling of degenerate statistical situations:

| Condition | Graceful (default) | Strict (`--strict_diagnostics`) |
|-----------|--------------------|---------------------------------|
| Zero-width bootstrap CI | Widen by epsilon (~1e-9) + warning | Abort (AssertionError) |
| NaN skewness/kurtosis (constant distribution) | Replace with 0.0 + warning | Abort |
| NaN Anderson statistic (constant distribution) | Replace with 0.0 + warning | Abort |
| NaN Q-Q R² (constant distribution) | Replace with 1.0 + warning | Abort |

Use strict mode in CI or research contexts requiring hard guarantees; keep default for exploratory analysis to avoid aborting entire runs on trivial constants.

**`--bootstrap_resamples`** (int, default: 10000)

- Number of bootstrap resamples used for confidence intervals (percentile method).
- Lower values (< 500) yield coarse intervals; a warning (RewardDiagnosticsWarning) is emitted if below internal recommended minimum (currently 200) to help with very fast exploratory runs.
- Increase for more stable interval endpoints (typical: 5000–20000). Runtime scales roughly linearly.

**`--skip_partial_dependence`** (flag, default: disabled)

- When set, skips computation and export of partial dependence CSV files, reducing runtime (often 30–60% faster for large sample sizes) at the cost of losing marginal response curve inspection.
- Feature importance (RandomForest Gini importance + permutation importance) is still computed.

### Reproducibility Model

| Component | Controlled By | Notes |
|-----------|---------------|-------|
| Sample simulation | `--seed` | Drives action sampling, PnL noise generation. |
| Statistical tests / bootstrap | `--stats_seed` (fallback `--seed`) | Local RNG; isolation prevents side-effects in user code. |
| RandomForest & permutation importance | `--seed` | Ensures identical splits and tree construction. |
| Partial dependence grids | Deterministic | Depends only on fitted model & data. |

Common patterns:
```shell
# Same synthetic data, two different statistical re-analysis runs
python reward_space_analysis.py --num_samples 50000 --seed 123 --stats_seed 9001 --output run_stats1
python reward_space_analysis.py --num_samples 50000 --seed 123 --stats_seed 9002 --output run_stats2

# Fully reproducible end-to-end (all aspects deterministic)
python reward_space_analysis.py --num_samples 50000 --seed 777
```

### Direct Tunable Overrides vs `--params`

All reward parameters are also available as individual CLI flags. You may choose either style:

```shell
# Direct flag style
python reward_space_analysis.py --win_reward_factor 3.0 --idle_penalty_scale 2.0 --num_samples 15000

# Equivalent using --params
python reward_space_analysis.py --params win_reward_factor=3.0 idle_penalty_scale=2.0 --num_samples 15000
```

If both a direct flag and the same key in `--params` are provided, the `--params` value takes highest precedence.

## 📝 Example Commands

```shell
# Quick test with defaults
python reward_space_analysis.py --num_samples 10000

# Full analysis with custom profit target
python reward_space_analysis.py \
    --num_samples 50000 \
    --profit_target 0.05 \
    --trading_mode futures \
    --output custom_analysis

# Parameter sensitivity testing
python reward_space_analysis.py \
    --num_samples 30000 \
    --params win_reward_factor=3.0 idle_penalty_scale=1.5 \
    --output sensitivity_test

# PBRS potential shaping analysis
python reward_space_analysis.py \
    --num_samples 40000 \
    --params hold_potential_enabled=true exit_potential_mode=spike_cancel potential_gamma=0.95 \
    --output pbrs_test

# Real vs synthetic comparison
python reward_space_analysis.py \
    --num_samples 100000 \
    --real_episodes path/to/episode_rewards.pkl \
    --output validation
```

---

## 📊 Understanding Results

The analysis generates the following output files:

### Main Report

**`statistical_analysis.md`** - Comprehensive statistical analysis containing:

1. **Global Statistics** - Reward distribution, per-action stats, component activation & ranges.
2. **Sample Representativity** - Position/action distributions, critical regime coverage, component activation recap.
3. **Reward Component Analysis** - Binned relationships (idle, hold, exit), correlation matrix (constant features removed), PBRS analysis (activation rates, component stats, invariance summary).
4. **Feature Importance** - Random Forest importance + partial dependence.
5. **Statistical Validation** - Hypothesis tests, bootstrap confidence intervals, normality diagnostics, optional distribution shift (5.4) when real episodes provided.
**Summary** - 7-point concise synthesis:
1. Reward distribution health (center, spread, tail asymmetry)
2. Action & position coverage (usage %, invalid rate, masking efficacy)
3. Component contributions (activation rates + mean / |mean| ranking)
4. Exit attenuation behavior (mode, continuity, effective decay characteristics)
5. Feature signal quality (model R², leading predictors, stability notes)
6. Statistical outcomes (significant correlations / tests, any multiple-testing adjustment applied, distribution shift if real data)
7. PBRS invariance verdict (|Σ shaping| < 1e-6 => canonical; otherwise non-canonical with absolute deviation)

### Data Exports

| File                       | Description                                          |
| -------------------------- | ---------------------------------------------------- |
| `reward_samples.csv`       | Raw synthetic samples for custom analysis            |
| `feature_importance.csv`   | Feature importance rankings from random forest model |
| `partial_dependence_*.csv` | Partial dependence data for key features             |
| `manifest.json`            | Runtime manifest (simulation + reward params + hash) |

### Manifest Structure (`manifest.json`)

| Field | Type | Description |
|-------|------|-------------|
| `generated_at` | string (ISO 8601) | Timestamp of generation (not part of hash). |
| `num_samples` | int | Number of synthetic samples generated. |
| `seed` | int | Master random seed driving simulation determinism. |
| `max_trade_duration` | int | Max trade duration used to scale durations. |
| `profit_target_effective` | float | Profit target after risk/reward scaling. |
| `pvalue_adjust_method` | string | Multiple testing correction mode (`none` or `benjamini_hochberg`). |
| `parameter_adjustments` | object | Map of any automatic bound clamps (empty if none). |
| `reward_params` | object | Full resolved reward parameter set (post-validation). |
| `simulation_params` | object | All simulation inputs (num_samples, seed, volatility knobs, etc.). |
| `params_hash` | string (sha256) | Hash over ALL `simulation_params` + ALL `reward_params` (lexicographically ordered). |

Reproducibility: two runs are input-identical iff their `params_hash` values match. Because defaults are included in the hash, modifying a default value (even if not overridden) changes the hash.

### Distribution Shift Metric Conventions

| Metric | Definition | Notes |
|--------|------------|-------|
| `*_kl_divergence` | KL(synthetic‖real) = Σ p_synth log(p_synth / p_real) | Asymmetric; 0 iff identical histograms (after binning). |
| `*_js_distance` | d_JS(p_synth, p_real) = √( 0.5 KL(p_synth‖m) + 0.5 KL(p_real‖m) ), m = 0.5 (p_synth + p_real) | Symmetric, bounded [0,1]; square-root of JS divergence; stable vs KL when supports differ. |
| `*_wasserstein` | 1D Earth Mover's Distance | Non-negative; same units as feature. |
| `*_ks_statistic` | KS two-sample statistic | ∈ [0,1]; higher = greater divergence. |
| `*_ks_pvalue` | KS test p-value | ∈ [0,1]; small ⇒ reject equality (at α). |

Implementation details:
- Histograms: 50 uniform bins spanning min/max across both samples.
- Probabilities: counts + ε (1e-10) then normalized ⇒ avoids log(0) and division by zero.
- Degenerate distributions short-circuit to zeros / p-value 1.0.
- JS distance instead of raw JS divergence for bounded interpretability and smooth interpolation.

---

## 🔬 Advanced Usage

### Custom Parameter Testing

Test different reward parameter configurations to understand their impact:

```shell
# Test power-based exit attenuation with custom tau
python reward_space_analysis.py \
    --num_samples 25000 \
    --params exit_attenuation_mode=power exit_power_tau=0.5 efficiency_weight=0.8 \
    --output custom_test

# Test aggressive hold penalties
python reward_space_analysis.py \
    --num_samples 25000 \
    --params hold_penalty_scale=0.5 \
    --output aggressive_hold

# Test PBRS configurations
python reward_space_analysis.py \
    --num_samples 25000 \
    --params hold_potential_enabled=true entry_additive_enabled=true exit_additive_enabled=false exit_potential_mode=canonical \
    --output pbrs_canonical

python reward_space_analysis.py \
    --num_samples 25000 \
    --params hold_potential_transform_pnl=softsign_sharp potential_softsign_sharpness=0.5 \
    --output pbrs_sharp_transforms
```

### Real Data Comparison

For production validation, compare synthetic analysis with real trading episodes:

```shell
python reward_space_analysis.py \
    --num_samples 100000 \
    --real_episodes path/to/episode_rewards.pkl \
    --output real_vs_synthetic
```

The report will include distribution shift metrics (KL divergence ≥ 0, JS distance ∈ [0,1], Wasserstein ≥ 0, KS statistic ∈ [0,1], KS p-value ∈ [0,1]) showing how well synthetic samples represent real trading. Degenerate (constant) distributions are auto-detected and produce zero divergence and KS p-value = 1.0 to avoid spurious instability.

### Batch Analysis

```shell
# Test multiple parameter combinations
for factor in 1.5 2.0 2.5 3.0; do
    python reward_space_analysis.py \
        --num_samples 20000 \
        --params win_reward_factor=$factor \
        --output analysis_factor_$factor
done
```

---

## 🧪 Validation & Testing

### Run Tests

```shell
# activate the venv first
source .venv/bin/activate
pip install pytest packaging
pytest -q
```

Always run the full suite after modifying reward logic or attenuation parameters.

### Test Categories

| Category | Class | Focus |
|----------|-------|-------|
| Integration | TestIntegration | CLI, artifacts, manifest reproducibility |
| Statistical Coherence | TestStatisticalCoherence | Distribution shift, diagnostics, hypothesis basics |
| Reward Alignment | TestRewardAlignment | Component correctness & exit factors |
| Public API | TestPublicAPI | Core API functions and interfaces |
| Statistical Validation | TestStatisticalValidation | Mathematical bounds, heteroscedasticity, invariants |
| Boundary Conditions | TestBoundaryConditions | Extreme params & unknown mode fallback |
| Helper Functions | TestHelperFunctions | Report writers, model analysis, utility conversions |
| Private Functions | TestPrivateFunctions | Idle / hold / invalid penalties, exit scenarios (accessed indirectly) |
| Robustness | TestRewardRobustness | Monotonic attenuation (where applicable), decomposition integrity, boundary regimes |
| Parameter Validation | TestParameterValidation | Bounds clamping, warning threshold, penalty power scaling |
| Continuity | TestContinuityPlateau | Plateau boundary continuity & small-epsilon attenuation scaling |
| PBRS Integration | TestPBRSIntegration | Potential-based reward shaping, transforms, exit modes, canonical invariance |
| Report Formatting | TestReportFormatting | Report section presence, ordering, PBRS invariance line, formatting integrity |
| Load Real Episodes | TestLoadRealEpisodes | Real episodes ingestion, column validation, distribution shift preparation |

### Test Architecture

- **Single test file**: `test_reward_space_analysis.py` (consolidates all testing)
- **Base class**: `RewardSpaceTestBase` with shared configuration and utilities
- **Reproducible**: Fixed seed (`seed = 42`) for consistent results

### Code Coverage Analysis

```shell
pip install pytest-cov
pytest -q --cov=. --cov-report=term-missing
pytest -q --cov=. --cov-report=html # open htmlcov/index.html
```

**Coverage Focus Areas:**
- ✅ **Core reward calculation logic** - Excellently covered (>95%)
- ✅ **Statistical functions** - Comprehensively covered (>90%)
- ✅ **Public API functions** - Thoroughly covered (>85%)
- ✅ **Report generation functions** - Well covered via dedicated tests
- ✅ **Utility functions** - Well covered via simulation tests
- ⚠️ **CLI interface and main()** - Partially covered (sufficient via integration tests)
- ⚠️ **Error handling paths** - Basic coverage (acceptable for robustness)

### When to Run Tests

- After modifying reward logic
- Before important analyses
- When results seem unexpected
- After updating dependencies or Python version
- When contributing new features (aim for >80% coverage on new code)

### Run Specific Test Categories

```shell
pytest -q test_reward_space_analysis.py::TestIntegration
pytest -q test_reward_space_analysis.py::TestStatisticalCoherence
pytest -q test_reward_space_analysis.py::TestRewardAlignment

```

---

## 🆘 Troubleshooting

### No Output Files Generated

**Symptom:** Script completes but no files in output directory

**Solution:**

- Check write permissions in output directory
- Ensure sufficient disk space (min 100MB free)
- Verify Python path is correct

### Unexpected Reward Values

**Symptom:** Rewards don't match expected behavior

**Solution:**

- Run `test_reward_space_analysis.py` to validate logic
- Review parameter overrides with `--params`
- Check trading mode settings (spot vs margin/futures)
- Verify `base_factor` matches your environment config
- Check PBRS settings: `hold_potential_enabled`, `exit_potential_mode`, and transform functions
- Review parameter adjustments in output logs for any automatic bound clamping

### Slow Execution

**Symptom:** Analysis takes excessive time to complete

**Solution:**

- Reduce `--num_samples` (start with 10,000)
- Use `--trading_mode spot` (fewer action combinations)
- Close other memory-intensive applications
- Use SSD storage for faster I/O
- Use `--skip_partial_dependence` to skip marginal response curves
- Temporarily lower `--bootstrap_resamples` (e.g. 1000) during iteration (expect wider CIs)

### Memory Errors

**Symptom:** `MemoryError` or system freeze

**Solution:**

- Reduce sample size to 10,000-20,000
- Use 64-bit Python installation
- Add more RAM or configure swap file
- Process data in batches for custom analyses

