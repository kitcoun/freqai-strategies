# 📊 Reward Space Analysis - User Guide

**Analyze and validate ReforceXY reward logic with synthetic data**

---

## 🎯 What is this?

This tool helps you understand and validate how the ReforceXY reinforcement learning environment calculates rewards. It generates synthetic trading scenarios to analyze reward behavior across different market conditions.

### Key Features
- ✅ Generate thousands of trading scenarios instantly
- ✅ Analyze reward distribution and patterns
- ✅ Validate reward logic against expected behavior
- ✅ Export results for further analysis
- ✅ Compare synthetic vs real trading data

---

**New to this tool?** Start with [Common Use Cases](#-common-use-cases) to see practical examples, then explore [CLI Parameters](#️-cli-parameters-reference) for detailed configuration options.

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
python reward_space_analysis.py --num_samples 20000 --output demo
python test_reward_alignment.py
```

> Deactivate the environment with `deactivate` when you're done.

Unless otherwise noted, the command examples below assume your current working directory is `ReforceXY/reward_space_analysis` (and the optional virtual environment is activated).

---

## 💡 Common Use Cases

### 1. Validate Reward Logic
**Goal:** Ensure rewards behave as expected in different scenarios

```shell
python reward_space_analysis.py --num_samples 20000 --output validation
```

**Check in `statistical_analysis.md`:**
- Long/Short exits should have positive average rewards
- Invalid actions should have negative penalties
- Idle periods should reduce rewards

### 2. Analyze Parameter Sensitivity
**Goal:** See how reward parameters affect trading behavior

```shell
# Test different win reward factors
python reward_space_analysis.py \
    --num_samples 30000 \
    --params win_reward_factor=2.0 \
    --output conservative

python reward_space_analysis.py \
    --num_samples 30000 \
    --params win_reward_factor=4.0 \
    --output aggressive
```

**Compare:** Reward distributions between runs in `statistical_analysis.md`

### 3. Debug Reward Issues
**Goal:** Identify why your RL agent behaves unexpectedly

```shell
# Generate detailed analysis (statistical validation is now default)
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
# First, collect real episodes (see Advanced section)
# Then compare:
python reward_space_analysis.py \
    --num_samples 100000 \
    --real_episodes ../user_data/transitions/*.pkl \
    --output comparison
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

*Invalid action penalty:*
- `invalid_action` (default: -2.0) - Penalty for invalid actions

*Idle penalty configuration:*
- `idle_penalty_scale` (default: 1.0) - Scale of idle penalty
- `idle_penalty_power` (default: 1.0) - Power applied to idle penalty scaling

*Holding penalty configuration:*
- `holding_duration_ratio_grace` (default: 1.0) - Grace ratio (≤1) before holding penalty increases with duration ratio
- `holding_penalty_scale` (default: 0.3) - Scale of holding penalty
- `holding_penalty_power` (default: 1.0) - Power applied to holding penalty scaling

*Exit factor configuration:*
- `exit_factor_mode` (default: piecewise) - Time attenuation mode for exit factor (legacy|sqrt|linear|power|piecewise|half_life)
- `exit_linear_slope` (default: 1.0) - Slope for linear exit attenuation
- `exit_piecewise_grace` (default: 1.0) - Grace region for piecewise exit attenuation
- `exit_piecewise_slope` (default: 1.0) - Slope after grace for piecewise mode
- `exit_power_tau` (default: 0.5) - Tau in (0,1] to derive alpha for power mode
- `exit_half_life` (default: 0.5) - Half-life for exponential attenuation exit mode

*Efficiency configuration:*
- `efficiency_weight` (default: 0.75) - Weight for efficiency factor in exit reward
- `efficiency_center` (default: 0.75) - Center for efficiency factor sigmoid

*Profit factor configuration:*
- `win_reward_factor` (default: 2.0) - Amplification for PnL above target
- `pnl_factor_beta` (default: 0.5) - Sensitivity of amplification around target

**`--real_episodes`** (path, optional)
- Path to real episode rewards pickle file for distribution comparison
- Enables distribution shift analysis (KL divergence, JS distance, Wasserstein distance)
- Example: `../user_data/models/ReforceXY-PPO/sub_train_SYMBOL_DATE/episode_rewards.pkl`

---

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
- **Statistical Validation** - Hypothesis tests, confidence intervals, normality diagnostics
- **Distribution Shift** (optional) - Comparison with real trading data

### Data Exports

**`reward_samples.csv`** - Raw synthetic samples for custom analysis

**`feature_importance.csv`** - Feature importance rankings from random forest model

**`partial_dependence_*.csv`** - Partial dependence data for key features

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
    --params holding_penalty_scale=0.5 holding_duration_ratio_grace=0.8 \
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

The report will include distribution shift metrics (KL divergence, JS distance, Wasserstein distance) showing how well synthetic samples represent real trading.

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

### Run Regression Tests
```shell
python test_reward_alignment.py
```

**Expected output:**
```
✅ ENUMS_MATCH: True
✅ DEFAULT_PARAMS_MATCH: True
✅ TEST_INVALID_ACTION: PASS
✅ TEST_IDLE_PENALTY: PASS
✅ TEST_HOLDING_PENALTY: PASS
✅ TEST_EXIT_REWARD: PASS
```

```shell
python test_stat_coherence.py
```

### When to Run Tests
- After modifying reward logic
- Before important analyses
- When results seem unexpected

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
- Run `test_reward_alignment.py` to validate logic
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
python reward_space_analysis.py --num_samples 20000 --output my_analysis

# Run validation tests
python test_reward_alignment.py
python test_stat_coherence.py
```

### Best Practices

**For Beginners:**
- Start with 10,000-20,000 samples for quick iteration
- Use default parameters initially
- Always run tests after modifying reward logic
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

| Issue | Quick Solution |
|-------|----------------|
| Memory errors | Reduce `--num_samples` to 10,000-20,000 |
| Slow execution | Use `--trading_mode spot` or reduce samples |
| Unexpected rewards | Run `test_reward_alignment.py` and check `--params` overrides |
| Import errors | Activate venv: `source .venv/bin/activate` |
| No output files | Check write permissions and disk space |
