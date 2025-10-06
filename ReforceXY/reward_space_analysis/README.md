# 📊 Reward Space Analysis - User Guide

**Analyze and validate ReforceXY reward logic with synthetic data**

---

## 🎯 What is this?

This tool helps you understand and validate how the ReforceXY reinforcement learning environment calculates rewards. It generates synthetic trading scenarios to analyze reward behavior across different market conditions.

### Key Features

- ✅ Generate thousands of synthetic trading scenarios deterministically
- ✅ Analyze reward distribution, feature importance & partial dependence
- ✅ Built‑in invariant & statistical validation layers (fail‑fast)
- ✅ Export reproducible artifacts (parameter hash + execution manifest)
- ✅ Compare synthetic vs real trading data (distribution shift metrics)
- ✅ Parameter bounds validation & automatic sanitization

---

**New to this tool?** Start with [Common Use Cases](#-common-use-cases) then explore [CLI Parameters](#️-cli-parameters-reference). For runtime guardrails see [Validation Layers](#-validation-layers-runtime). The exit factor attenuation logic is now centralized through a single internal helper ensuring analytical parity with the live environment (parity date: 2025‑10‑06).

---

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

Whenever you need to run analyses or tests, activate the environment first:

```shell
source .venv/bin/activate
python reward_space_analysis.py --num_samples 20000 --output reward_space_outputs
python test_reward_space_analysis.py
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
    --real_episodes ../user_data/transitions/*.pkl \
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
- Also used as fallback for `max_idle_duration_candles` when that tunable is ≤ 0 (idle penalty grace behaviour)

### Reward Configuration

**`--base_factor`** (float, default: 100.0)

- Base reward scaling factor (from environment config)
- Should match your environment's base_factor

**`--profit_target`** (float, default: 0.03)

- Target profit threshold as decimal (e.g., 0.03 = 3%)
- Used for efficiency calculations and holding penalties

**`--risk_reward_ratio`** (float, default: 1.0)

- Risk/reward ratio multiplier
- Affects profit target adjustment in reward calculations

**`--holding_max_ratio`** (float, default: 2.5)

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

- `idle_penalty_scale` (default: 1.0) - Scale of idle penalty
- `idle_penalty_power` (default: 1.0) - Power applied to idle penalty scaling

_Holding penalty configuration:_

- `holding_penalty_scale` (default: 0.5) - Scale of holding penalty
- `holding_penalty_power` (default: 1.0) - Power applied to holding penalty scaling

_Exit factor configuration:_

- `exit_factor_mode` (default: piecewise) - Time attenuation mode for exit factor (legacy|sqrt|linear|power|piecewise|half_life)
- `exit_linear_slope` (default: 1.0) - Slope for linear exit attenuation
- `exit_piecewise_grace` (default: 1.0) - Grace region boundary (duration ratio); values >1.0 extend no-attenuation period
- `exit_piecewise_slope` (default: 1.0) - Slope after grace for piecewise mode (0 ⇒ flat beyond grace)
- `exit_power_tau` (default: 0.5) - Tau in (0,1] mapped to alpha = -ln(tau)/ln(2)
- `exit_half_life` (default: 0.5) - Half-life for exponential decay exit mode (factor *= 2^(-r/half_life))
- `exit_factor_threshold` (default: 10000.0) - Warning-only threshold; no capping occurs (emits RuntimeWarning if |factor| exceeds)

_Efficiency configuration:_

- `efficiency_weight` (default: 0.75) - Weight for efficiency factor in exit reward
- `efficiency_center` (default: 0.75) - Center for efficiency factor sigmoid

_Profit factor configuration:_

- `win_reward_factor` (default: 2.0) - Amplification for PnL above target (no upper bound; effective profit_target_factor ∈ [1, 1 + win_reward_factor] because tanh ≤ 1)
- `pnl_factor_beta` (default: 0.5) - Sensitivity of amplification around target

_Invariant / safety controls:_

- `check_invariants` (default: true) - Enable/disable runtime invariant & safety validations (simulation invariants, mathematical bounds, distribution checks). Set to `false` only for performance experiments; not recommended for production validation.

**`--real_episodes`** (path, optional)

- Path to real episode rewards pickle file for distribution comparison
- Enables distribution shift analysis (KL(synthetic‖real), JS distance, Wasserstein distance, KS test)
- Example: `../user_data/models/ReforceXY-PPO/sub_train_SYMBOL_DATE/episode_rewards.pkl`

**`--pvalue_adjust`** (choice: none|benjamini_hochberg, default: none)

- Apply multiple hypothesis testing correction (Benjamini–Hochberg) to p-values in statistical hypothesis tests.
- When set to `benjamini_hochberg`, adjusted p-values and adjusted significance flags are added to the reports.

**`--stats_seed`** (int, optional; default: inherit `--seed`)

- Dedicated seed for statistical analyses (hypothesis tests, bootstrap confidence intervals, distribution diagnostics).
- Use this if you want to generate multiple independent statistical analyses over the same synthetic dataset without re-simulating samples.
- If omitted, falls back to `--seed` for full run determinism.

### Reproducibility Model

| Component | Controlled By | Notes |
|-----------|---------------|-------|
| Sample simulation | `--seed` | Drives action sampling, PnL noise, force actions. |
| Statistical tests / bootstrap | `--stats_seed` (fallback `--seed`) | Local RNG; isolation prevents side‑effects in user code. |
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

---

#### Direct Tunable Overrides vs `--params`

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

# Real vs synthetic comparison
python reward_space_analysis.py \
    --num_samples 100000 \
    --real_episodes ../user_data/models/path/to/episode_rewards.pkl \
    --output validation
```

---

## 📊 Understanding Results

The analysis generates the following output files:

### Main Report

**`statistical_analysis.md`** - Comprehensive statistical analysis containing:

- **Global Statistics** - Reward distributions and component activation rates
- **Sample Representativity** - Coverage of critical market scenarios
- **Component Analysis** - Relationships between rewards and conditions
- **Feature Importance** - Machine learning analysis of key drivers
- **Statistical Validation** - Hypothesis tests, confidence intervals, normality + effect sizes
- **Distribution Shift** - Real vs synthetic divergence (KL, JS, Wasserstein, KS)
- **Diagnostics Validation Summary**
  - Pass/fail snapshot of all runtime checks
  - Consolidated pass/fail state of every validation layer (invariants, parameter bounds, bootstrap CIs, distribution metrics, diagnostics, hypothesis tests)

### Data Exports

| File                       | Description                                          |
| -------------------------- | ---------------------------------------------------- |
| `reward_samples.csv`       | Raw synthetic samples for custom analysis            |
| `feature_importance.csv`   | Feature importance rankings from random forest model |
| `partial_dependence_*.csv` | Partial dependence data for key features             |
| `manifest.json`            | Run metadata (seed, params, top features, overrides) |

### Manifest Structure (`manifest.json`)

Key fields:

| Field | Description |
|-------|-------------|
| `generated_at` | ISO timestamp of run |
| `num_samples` | Number of synthetic samples generated |
| `seed` | Random seed used (deterministic cascade) |
| `profit_target_effective` | Profit target after risk/reward scaling |
| `top_features` | Top 5 features by permutation importance |
| `reward_param_overrides` | Subset of reward tunables whose values differ from defaults |
| `params_hash` | SHA-256 hash combining simulation params + overrides (reproducibility) |
| `params` | Echo of core simulation parameters (subset, for quick audit) |
| `parameter_adjustments` | Any automatic bound clamps applied by `validate_reward_parameters` |

Use `params_hash` to verify reproducibility across runs; identical seeds + identical overrides ⇒ identical hash.

#### Distribution Shift Metric Conventions

| Metric | Definition | Notes |
|--------|------------|-------|
| `*_kl_divergence` | KL(synthetic‖real) = Σ p_synth log(p_synth / p_real) | Asymmetric; 0 iff identical histograms (after binning). |
| `*_js_distance` | √(JS(p_synth, p_real)) | Symmetric, bounded [0,1]; distance form (sqrt of JS divergence). |
| `*_wasserstein` | 1D Earth Mover's Distance | Non-negative; same units as feature. |
| `*_ks_statistic` | KS two-sample statistic | ∈ [0,1]; higher = greater divergence. |
| `*_ks_pvalue` | KS test p-value | ∈ [0,1]; small ⇒ reject equality (at α). |

Implementation details:
- Histograms: 50 uniform bins spanning min/max across both samples.
- Probabilities: counts + ε (1e‑10) then normalized ⇒ avoids log(0) and division by zero.
- Degenerate (constant) distributions short‑circuit to zeros (divergences) / p-value 1.0.
- JS distance is reported (not raw divergence) for bounded interpretability.

---

## 🔬 Advanced Usage

### Custom Parameter Testing

Test different reward parameter configurations to understand their impact:

```shell
# Test power-based exit factor with custom tau
python reward_space_analysis.py \
    --num_samples 25000 \
    --params exit_factor_mode=power exit_power_tau=0.5 efficiency_weight=0.8 \
    --output custom_test

# Test aggressive holding penalties
python reward_space_analysis.py \
    --num_samples 25000 \
    --params holding_penalty_scale=0.5 \
    --output aggressive_holding
```

### Real Data Comparison

For production validation, compare synthetic analysis with real trading episodes:

1. **Enable logging** in your ReforceXY config
2. **Run training** to collect real episodes
3. **Compare distributions** using `--real_episodes`

```shell
python reward_space_analysis.py \
    --num_samples 100000 \
    --real_episodes ../user_data/models/ReforceXY-PPO/sub_train_BTCUSDT_20231201/episode_rewards.pkl \
    --output real_vs_synthetic
```

The report will include distribution shift metrics (KL divergence ≥ 0, JS distance ∈ [0,1], Wasserstein ≥ 0, KS statistic ∈ [0,1], KS p‑value ∈ [0,1]) showing how well synthetic samples represent real trading. Degenerate (constant) distributions are auto‑detected and produce zero divergence and KS p‑value = 1.0 to avoid spurious instability.

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
python test_reward_space_analysis.py
```

The suite currently contains 53 tests (current state; this number evolves as new invariants and attenuation modes are added). Always run the full suite after modifying reward logic or attenuation parameters.

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
| Private Functions (via public API) | TestPrivateFunctions | Idle / holding / invalid penalties, exit scenarios |
| Robustness | TestRewardRobustness | Monotonic attenuation (where applicable), decomposition integrity, boundary regimes |
| Parameter Validation | TestParameterValidation | Bounds clamping, warning threshold, penalty power scaling |

### Test Architecture

- **Single test file**: `test_reward_space_analysis.py` (consolidates all testing)
- **Base class**: `RewardSpaceTestBase` with shared configuration and utilities
- **Unified framework**: `unittest` with optional `pytest` configuration
- **Reproducible**: Fixed seed (`seed = 42`) for consistent results

### Code Coverage Analysis

```shell
pip install coverage
coverage run --source=. test_reward_space_analysis.py
coverage report -m
coverage html  # open htmlcov/index.html
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
# All tests (recommended)
python test_reward_space_analysis.py

# Individual test classes using unittest
python -m unittest test_reward_space_analysis.TestIntegration
python -m unittest test_reward_space_analysis.TestStatisticalCoherence
python -m unittest test_reward_space_analysis.TestRewardAlignment

# With pytest (if installed)
pytest test_reward_space_analysis.py -v
```

---

## 🆘 Troubleshooting

### Module Installation Issues

**Symptom:** `ModuleNotFoundError` or import errors

**Solution:**

```shell
pip install pandas numpy scipy scikit-learn
```

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

### Slow Execution

**Symptom:** Analysis takes excessive time to complete

**Solution:**

- Reduce `--num_samples` (start with 10,000)
- Use `--trading_mode spot` (fewer action combinations)
- Close other memory-intensive applications
- Use SSD storage for faster I/O

### Memory Errors

**Symptom:** `MemoryError` or system freeze

**Solution:**

- Reduce sample size to 10,000-20,000
- Use 64-bit Python installation
- Add more RAM or configure swap file
- Process data in batches for custom analyses

---

## 📞 Quick Reference & Best Practices

### Getting Started

```shell
# Setup virtual environment (first time only)
cd ReforceXY/reward_space_analysis
python -m venv .venv
source .venv/bin/activate
pip install pandas numpy scipy scikit-learn

# Basic analysis
python reward_space_analysis.py --num_samples 20000 --output reward_space_outputs

# Run validation tests
python test_reward_space_analysis.py
```

### Best Practices

**For Beginners:**

- Start with 10,000-20,000 samples for quick iteration
- Use default parameters initially
- Always run tests after modifying reward logic: `python test_reward_space_analysis.py`
- Review `statistical_analysis.md` for insights

**For Advanced Users:**

- Use 50,000+ samples for statistical significance
- Compare multiple parameter sets via batch analysis
- Validate synthetic analysis against real trading data with `--real_episodes`
- Export CSV files for custom statistical analysis

**Performance Optimization:**

- Use SSD storage for faster I/O
- Parallelize parameter sweeps across multiple runs
- Cache results for repeated analyses
- Use `--trading_mode spot` for faster exploratory runs

### Common Issues Quick Reference

For detailed troubleshooting, see [Troubleshooting](#-troubleshooting) section.

| Issue              | Quick Solution                                                |
| ------------------ | ------------------------------------------------------------- |
| Memory errors      | Reduce `--num_samples` to 10,000-20,000                       |
| Slow execution     | Use `--trading_mode spot` or reduce samples                   |
| Unexpected rewards | Run `test_reward_space_analysis.py` and check `--params` overrides |
| Import errors      | Activate venv: `source .venv/bin/activate`                    |
| No output files    | Check write permissions and disk space                        |
| Hash mismatch      | Confirm overrides + seed; compare `reward_param_overrides`    |

### Validation Layers (Runtime)

All runs execute a sequence of fail‑fast validations; a failure aborts with a clear message:

| Layer | Scope | Guarantees |
|-------|-------|------------|
| Simulation Invariants | Raw synthetic samples | PnL only on exit actions; sum PnL equals exit PnL; no exit reward without PnL. |
| Parameter Bounds | Tunables | Clamps values outside declared bounds; records adjustments in manifest. |
| Bootstrap CIs | Mean estimates | Finite means; ordered CI bounds; non‑NaN across metrics. |
| Distribution Metrics | Real vs synthetic shifts | Metrics within mathematical bounds (KL ≥0, JS ∈[0,1], Wasserstein ≥0, KS stats/p ≤[0,1]). Degenerate distributions handled safely (zeroed metrics). |
| Distribution Diagnostics | Normality & moments | Finite mean/std/skew/kurtosis; Shapiro p-value ∈[0,1]; variance non-negative. |
| Hypothesis Tests | Test result dicts | p-values & effect sizes within valid ranges; optional multiple-testing adjustment (Benjamini–Hochberg). |
| Exit Factor Attenuation | Time-based scaling | Centralized piecewise divisor helper ensures single source of truth; threshold is warning-only (no hard cap). |

### Statistical Method Notes

- Bootstrap CIs: percentile method (default 10k resamples in full runs; tests may use fewer). BCa not yet implemented (explicitly deferred).
- Multiple testing: Benjamini–Hochberg available via `--pvalue_adjust benjamini_hochberg`.
- JS distance reported as the square root of Jensen–Shannon divergence (hence bounded by 1).
- Degenerate distributions (all values identical) short‑circuit to stable zero metrics.
- Random Forest: 400 trees, `n_jobs=1` for determinism.
- Heteroscedasticity model: σ = `pnl_base_std * (1 + pnl_duration_vol_scale * duration_ratio)`.

### Parameter Validation & Sanitization

Before simulation (early in `main()`), `validate_reward_parameters` enforces numeric bounds (see `_PARAMETER_BOUNDS` in code). Adjusted values are:

1. Clamped to min/max if out of range.
2. Reset to min if non-finite.
3. Recorded in `manifest.json` under `parameter_adjustments` with fields: `original`, `adjusted`, `reason` (a comma‑separated list of clamp reasons like `min=0.0`, `max=1.0`, `non_finite_reset`).


#### Parameter Bounds Summary

| Parameter | Min | Max | Notes |
|-----------|-----|-----|-------|
| `invalid_action` | — | 0.0 | Must be ≤ 0 (penalty) |
| `base_factor` | 0.0 | — | Global scaling factor |
| `idle_penalty_power` | 0.0 | — | Power exponent ≥ 0 |
| `idle_penalty_scale` | 0.0 | — | Scale ≥ 0 |
| `holding_penalty_scale` | 0.0 | — | Scale ≥ 0 |
| `holding_penalty_power` | 0.0 | — | Power exponent ≥ 0 |
| `exit_linear_slope` | 0.0 | — | Slope ≥ 0 |
| `exit_piecewise_grace` | 0.0 | — | Grace boundary expressed in duration ratio units (can exceed 1.0 to extend full-strength region) |
| `exit_piecewise_slope` | 0.0 | — | Slope ≥ 0 |
| `exit_power_tau` | 1e-6 | 1.0 | Mapped to alpha = -ln(tau) |
| `exit_half_life` | 1e-6 | — | Half-life in duration ratio units |
| `efficiency_weight` | 0.0 | 2.0 | Blend weight |
| `efficiency_center` | 0.0 | 1.0 | Sigmoid center |
| `win_reward_factor` | 0.0 | — | Amplification ≥ 0 |
| `pnl_factor_beta` | 1e-6 | — | Sensitivity ≥ tiny positive |

Non-finite inputs are reset to the applicable minimum (or 0.0 if only a maximum is declared) and logged as adjustments.
