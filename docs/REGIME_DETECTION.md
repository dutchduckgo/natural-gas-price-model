# Regime Detection Implementation

## Overview

Regime detection is a critical component of our natural gas price forecasting model that identifies different market states (regimes) based on volatility patterns and price levels. This enables the models to adapt their predictions to current market conditions, improving accuracy during periods of high volatility, price shocks, or structural market changes.

## Why Regime Detection Matters

Natural gas markets exhibit distinct behavioral patterns across different market conditions:

- **Low Volatility Regimes**: Stable prices, predictable patterns, lower forecast uncertainty
- **High Volatility Regimes**: Rapid price movements, increased uncertainty, potential for extreme events
- **Price Level Regimes**: Different price dynamics at low vs. high price levels (e.g., $2/MMBtu vs. $8/MMBtu)

Models trained on all historical data equally may be biased toward average conditions and perform poorly during regime transitions or extreme periods. Regime detection addresses this by:

1. **Identifying market states** in real-time using only backward-looking information
2. **Weighting training samples** to emphasize recent and high-volatility periods
3. **Enabling regime-specific analysis** of model performance

---

## Implementation Architecture

### Core Components

1. **`RegimeFeatureEngineer`** (`src/feature_engineering/regime_features.py`)
   - Computes regime labels and features from price data
   - Implements three detection methods: volatility-based, price-level-based, and k-means clustering

2. **`RegimeAwareTrainer`** (`src/models/regime_models.py`)
   - Computes sample weights for training based on regimes
   - Integrates with model training pipelines

3. **Configuration** (`config.py`)
   - `REGIME_CONFIG` dictionary controls all regime-aware behavior

---

## Regime Detection Methods

### Method 1: Volatility-Based Regime Detection (Default)

**Purpose**: Identifies regimes based on the volatility of price returns.

**Algorithm**:

1. **Compute Returns**:
   ```python
   returns = price_series.pct_change()
   ```

2. **Calculate Rolling Volatility**:
   ```python
   rolling_vol = returns.rolling(window=20, min_periods=5).std()
   ```
   - Uses a 20-day rolling window (configurable via `volatility_window`)
   - Standard deviation of daily returns measures price volatility

3. **Determine Regime Thresholds**:
   For each time point `t`, the algorithm:
   - Uses **only historical data up to time `t`** (prevents look-ahead bias)
   - Computes the 33rd and 67th percentiles of historical volatility
   - Assigns regime labels:
     - **Regime 0 (Low Volatility)**: `volatility_t ≤ 33rd percentile`
     - **Regime 1 (Medium Volatility)**: `33rd percentile < volatility_t ≤ 67th percentile`
     - **Regime 2 (High Volatility)**: `volatility_t > 67th percentile`

**Example**:
```
If historical volatility percentiles are:
  - 33rd percentile = 0.02 (2% daily volatility)
  - 67th percentile = 0.05 (5% daily volatility)

Then:
  - Current volatility = 0.015 → Regime 0 (Low)
  - Current volatility = 0.04 → Regime 1 (Medium)
  - Current volatility = 0.08 → Regime 2 (High)
```

**Key Features**:
- **Adaptive thresholds**: Percentiles are recalculated at each point using only past data
- **No look-ahead bias**: Regime label at time `t` uses only data from `t=0` to `t`
- **Early period handling**: First 20 days default to Regime 1 (Medium)

**When to Use**: Best for identifying periods of market stress, uncertainty, or rapid price movements.

---

### Method 2: Price-Level-Based Regime Detection

**Purpose**: Identifies regimes based on whether prices are relatively low, medium, or high compared to recent history.

**Algorithm**:

1. **Compute Rolling Statistics**:
   ```python
   rolling_mean = price_series.rolling(window=60, min_periods=10).mean()
   rolling_std = price_series.rolling(window=60, min_periods=10).std()
   ```
   - Uses a 60-day rolling window (configurable via `price_window`)

2. **Calculate Z-Score**:
   ```python
   z_score = (price_series - rolling_mean) / rolling_std
   ```
   - Measures how many standard deviations the current price is from the rolling mean

3. **Assign Regime Labels**:
   - **Regime 0 (Low Price)**: `z_score < -0.5` (price is significantly below recent average)
   - **Regime 1 (Medium Price)**: `-0.5 ≤ z_score ≤ 0.5` (price is near recent average)
   - **Regime 2 (High Price)**: `z_score > 0.5` (price is significantly above recent average)

**Example**:
```
If rolling_mean = $3.50/MMBtu and rolling_std = $0.50/MMBtu:

  - Current price = $3.00 → z_score = -1.0 → Regime 0 (Low)
  - Current price = $3.50 → z_score = 0.0 → Regime 1 (Medium)
  - Current price = $4.00 → z_score = 1.0 → Regime 2 (High)
```

**Key Features**:
- **Relative pricing**: Identifies whether prices are high/low relative to recent history, not absolute levels
- **Symmetric thresholds**: ±0.5 standard deviations create balanced regime boundaries
- **Adaptive to trends**: Rolling window adapts to gradual price level changes

**When to Use**: Best for identifying periods when prices are at extreme levels relative to recent history, which may indicate supply/demand imbalances.

---

### Method 3: K-Means Clustering Regime Detection

**Purpose**: Uses unsupervised machine learning to identify natural clusters in price behavior.

**Algorithm**:

1. **Create Feature Matrix**:
   ```python
   features = {
       'price': price_series,
       'rolling_mean': price_series.rolling(window=60).mean()
   }
   ```
   - Uses both absolute price level and rolling mean as features

2. **Incremental K-Means Fitting**:
   For each time point `t` (starting after minimum samples):
   - Uses **only historical data up to time `t`**
   - Fits K-Means with `n_regimes` clusters (default: 3)
   - Assigns the current point to the cluster it belongs to

3. **Regime Assignment**:
   - Each cluster becomes a regime (0, 1, or 2)
   - Clusters are determined by the algorithm based on price patterns

**Key Features**:
- **Data-driven**: No fixed thresholds; clusters emerge from data patterns
- **No look-ahead bias**: K-Means is fit only on historical data at each point
- **Pattern discovery**: Can identify non-linear relationships between price and rolling mean

**When to Use**: Best when you want the algorithm to discover natural price regimes without imposing fixed thresholds.

**Limitations**:
- More computationally expensive (fits K-Means at each time point)
- Requires minimum samples (default: 50) before first regime assignment
- Cluster labels may not have intuitive meaning (e.g., "Regime 0" might not mean "low")

---

## Generated Features

The `RegimeFeatureEngineer.transform()` method adds the following features to the dataset:

### Primary Regime Labels

1. **`volatility_regime`** (int: 0, 1, or 2)
   - Volatility-based regime label
   - Always computed regardless of `regime_method`

2. **`price_level_regime`** (int: 0, 1, or 2)
   - Price-level-based regime label
   - Always computed regardless of `regime_method`

3. **`regime_label`** (int: 0, 1, or 2)
   - Primary regime label used for training
   - Set based on `regime_method`:
     - `"volatility"` → uses `volatility_regime`
     - `"price_level"` → uses `price_level_regime`
     - `"kmeans"` → uses k-means clustering result

### Supporting Features

4. **`volatility`** (float)
   - Rolling standard deviation of returns
   - Window: `volatility_window` (default: 20 days)
   - Units: dimensionless (percentage change)

5. **`price_zscore`** (float)
   - Z-score of price relative to rolling mean
   - Window: `price_window` (default: 60 days)
   - Formula: `(price - rolling_mean) / rolling_std`

6. **`regime_transition`** (int: 0 or 1)
   - Binary indicator for regime changes
   - `1` if `regime_label` changed from previous period
   - `0` otherwise
   - Useful for identifying periods of market instability

---

## Sample Weighting for Regime-Aware Training

Once regime labels are computed, models are trained with **sample weights** that emphasize certain periods:

### Weight Components

1. **Recency Weighting** (Exponential Decay):
   ```python
   weight[i] = decay_rate ^ (n - i)
   ```
   - Recent samples get higher weights
   - Default `decay_rate = 0.995` means the most recent sample gets weight 1.0, and samples from 100 days ago get weight ~0.61
   - Rationale: Recent market conditions are more relevant for near-term predictions

2. **Volatility Regime Weighting**:
   ```python
   multipliers = {0: 0.8, 1: 1.0, 2: 1.5}  # Low, Medium, High
   weight *= multipliers[volatility_regime]
   ```
   - High-volatility periods get 1.5x weight
   - Low-volatility periods get 0.8x weight
   - Rationale: Models should learn more from volatile periods where prediction errors are more costly

3. **Regime Transition Weighting**:
   ```python
   weight *= (1.0 + 0.2 * regime_transition)
   ```
   - Transition periods get 1.2x weight
   - Rationale: Regime changes are critical moments; models should learn to handle them

### Final Weight Calculation

```python
final_weight = recency_weight * volatility_weight * transition_weight
final_weight = final_weight / mean(final_weight)  # Normalize
```

**Example**:
```
Sample from 50 days ago, high volatility, no transition:
  recency = 0.995^50 = 0.78
  volatility = 1.5
  transition = 1.0
  weight = 0.78 * 1.5 * 1.0 = 1.17

Sample from yesterday, medium volatility, with transition:
  recency = 0.995^1 = 0.995
  volatility = 1.0
  transition = 1.2
  weight = 0.995 * 1.0 * 1.2 = 1.19
```

---

## Look-Ahead Bias Prevention

**Critical Design Principle**: All regime features are computed using **only backward-looking information**.

### Implementation Details

1. **Rolling Windows**:
   - All rolling statistics use `.rolling()` with no `center=True`
   - Window includes only past and current values, never future values

2. **Percentile Calculation**:
   - For volatility regimes, percentiles are computed using an **expanding window** up to the current point
   - At time `t`, percentiles use data from `t=0` to `t` only

3. **K-Means Fitting**:
   - K-Means is fit incrementally: at time `t`, only data from `t=0` to `t` is used
   - Each time point requires refitting, ensuring no future information leaks

4. **Regime Transitions**:
   - `regime_transition` uses `.diff()`, which compares current to previous period only

### Verification

You can verify no look-ahead bias by checking:
```python
# At time t, regime_label[t] should only depend on price[0:t+1]
# Never on price[t+1:] or future data
```

---

## Configuration

All regime detection behavior is controlled via `REGIME_CONFIG` in `config.py`:

```python
REGIME_CONFIG = {
    "enabled": True,                    # Enable/disable regime features
    "regime_method": "volatility",      # "volatility", "price_level", or "kmeans"
    "volatility_window": 20,            # Days for volatility calculation
    "price_window": 60,                 # Days for price level calculation
    "n_regimes": 3,                     # Number of regimes (for k-means)
    "use_sample_weighting": True,       # Enable sample weighting
    "recency_decay_rate": 0.995,        # Exponential decay for recency
    "volatility_weight_multipliers": {
        0: 0.8,   # Low volatility weight
        1: 1.0,   # Medium volatility weight
        2: 1.5    # High volatility weight
    },
    "transition_weight_multiplier": 1.2  # Weight for regime transitions
}
```

### Environment Variable Overrides

All settings can be overridden via environment variables:
```bash
export REGIME_ENABLED="True"
export REGIME_METHOD="volatility"
export REGIME_VOLATILITY_WINDOW=20
export REGIME_PRICE_WINDOW=60
export REGIME_USE_SAMPLE_WEIGHTING="True"
export REGIME_RECENCY_DECAY=0.995
export REGIME_VOL_WEIGHT_LOW=0.8
export REGIME_VOL_WEIGHT_MED=1.0
export REGIME_VOL_WEIGHT_HIGH=1.5
export REGIME_TRANSITION_WEIGHT=1.2
```

---

## Usage Examples

### Basic Usage

```python
from src.feature_engineering.regime_features import RegimeFeatureEngineer
import pandas as pd

# Initialize with default config
engineer = RegimeFeatureEngineer()

# Apply to dataframe with 'spot_price' column
df_with_regimes = engineer.transform(df)

# Check regime distribution
print(df_with_regimes['regime_label'].value_counts())
```

### Custom Configuration

```python
# Use price-level method with custom windows
config = {
    "regime_method": "price_level",
    "price_window": 90,  # Use 90-day window
    "volatility_window": 30
}
engineer = RegimeFeatureEngineer(config=config)
df_with_regimes = engineer.transform(df)
```

### With Sample Weighting

```python
from src.models.regime_models import RegimeAwareTrainer
from config import REGIME_CONFIG

# Initialize trainer
trainer = RegimeAwareTrainer(config=REGIME_CONFIG)

# Get sample weights
weights = trainer.get_sample_weights(df_with_regimes)

# Train model with weights
model.fit(X, y, sample_weight=weights)
```

---

## Integration with Training Pipeline

Regime detection is automatically integrated into the training pipeline:

1. **Feature Engineering** (`src/pipeline/train_baseline.py`):
   ```python
   # Regime features are added after price features
   df = self.price_engineer.transform(df)
   df = self.regime_engineer.transform(df)  # Adds regime features
   ```

2. **Model Training**:
   ```python
   # Sample weights computed from regime features
   sample_weight = self.regime_trainer.get_sample_weights(df)
   model.train_final_model(df, sample_weight=sample_weight)
   ```

3. **Backtesting**:
   ```python
   # Regime-specific performance tracking
   backtester = Backtester(track_regimes=True)
   results = backtester.walk_forward_validation(df, model)
   regime_summary = backtester.get_regime_performance_summary()
   ```

---

## Performance Analysis by Regime

The backtesting framework tracks performance separately for each regime:

```python
from src.evaluation.backtesting import Backtester

backtester = Backtester(track_regimes=True)
results = backtester.walk_forward_validation(df, model)

# Get regime-specific metrics
regime_summary = backtester.get_regime_performance_summary()
print(regime_summary)
```

Output example:
```
   regime  mae_mean  rmse_mean  total_samples
0       0      0.45       0.58            450
1       1      0.62       0.78            320
2       2      0.89       1.12            180
```

This reveals which regimes are more challenging to predict (typically high-volatility regimes).

---

## Mathematical Formulation

### Volatility Regime Detection

For time series `P(t)` (prices) and returns `R(t) = (P(t) - P(t-1)) / P(t-1)`:

1. **Rolling Volatility**:
   ```
   σ(t) = std(R(t-w+1), R(t-w+2), ..., R(t))
   ```
   where `w = volatility_window` (default: 20)

2. **Historical Percentiles** (at time `t`):
   ```
   q33(t) = percentile(σ(0), σ(1), ..., σ(t), 0.33)
   q67(t) = percentile(σ(0), σ(1), ..., σ(t), 0.67)
   ```

3. **Regime Assignment**:
   ```
   regime(t) = {
       0  if σ(t) ≤ q33(t)
       1  if q33(t) < σ(t) ≤ q67(t)
       2  if σ(t) > q67(t)
   }
   ```

### Price-Level Regime Detection

1. **Rolling Statistics**:
   ```
   μ(t) = mean(P(t-w+1), P(t-w+2), ..., P(t))
   s(t) = std(P(t-w+1), P(t-w+2), ..., P(t))
   ```
   where `w = price_window` (default: 60)

2. **Z-Score**:
   ```
   z(t) = (P(t) - μ(t)) / s(t)
   ```

3. **Regime Assignment**:
   ```
   regime(t) = {
       0  if z(t) < -0.5
       1  if -0.5 ≤ z(t) ≤ 0.5
       2  if z(t) > 0.5
   }
   ```

### Sample Weighting

For sample at index `i` (where `i=0` is oldest, `i=n-1` is newest):

```
w_recency(i) = decay_rate ^ (n - 1 - i)
w_volatility(i) = multipliers[volatility_regime[i]]
w_transition(i) = 1.0 + (transition_mult - 1.0) * regime_transition[i]

w_final(i) = w_recency(i) * w_volatility(i) * w_transition(i)
w_normalized(i) = w_final(i) / mean(w_final)
```

---

## Best Practices

1. **Choose Appropriate Method**:
   - **Volatility-based**: Best for identifying market stress periods
   - **Price-level-based**: Best for identifying relative price extremes
   - **K-means**: Best for discovering data-driven patterns

2. **Window Sizing**:
   - **Volatility window**: 20 days captures short-term volatility clusters
   - **Price window**: 60 days balances responsiveness with stability
   - Too short: Noisy, frequent regime changes
   - Too long: Slow to adapt, misses recent changes

3. **Weight Tuning**:
   - **Recency decay**: 0.995 gives ~61% weight to data from 100 days ago
   - **Volatility multipliers**: 1.5x for high-vol is a good starting point
   - **Transition multiplier**: 1.2x emphasizes transitions without over-weighting

4. **Validation**:
   - Always verify no look-ahead bias by checking regime labels use only past data
   - Monitor regime distribution: should see reasonable balance across regimes
   - Check regime-specific performance: high-volatility regimes typically have higher errors

---

## Troubleshooting

### Issue: All samples assigned to one regime

**Cause**: Window too short or thresholds too extreme

**Solution**: Increase `volatility_window` or `price_window`, or adjust percentile thresholds

### Issue: Regime labels change too frequently

**Cause**: Window too short or too sensitive thresholds

**Solution**: Increase window size or add smoothing (e.g., use regime label from 2-3 days ago)

### Issue: K-means fails or produces unstable labels

**Cause**: Insufficient data or high noise

**Solution**: Increase `min_samples` threshold or use volatility/price-level method instead

### Issue: Sample weights too extreme (very high or very low)

**Cause**: Extreme multipliers or decay rate

**Solution**: Normalize weights (enabled by default) or reduce multiplier values

---

## References

- **Regime-Switching Models**: Hamilton (1989) - "A New Approach to the Economic Analysis of Nonstationary Time Series"
- **Volatility Clustering**: Engle (1982) - "Autoregressive Conditional Heteroskedasticity"
- **Sample Weighting**: Schapire & Freund (2012) - "Boosting: Foundations and Algorithms"

---

## Summary

Regime detection enables our models to:

1. **Adapt to market conditions** by identifying volatility and price-level regimes
2. **Emphasize important periods** through sample weighting
3. **Improve prediction accuracy** during volatile or extreme periods
4. **Maintain scientific rigor** by preventing look-ahead bias

The implementation is flexible, configurable, and fully integrated into the training and evaluation pipelines.

