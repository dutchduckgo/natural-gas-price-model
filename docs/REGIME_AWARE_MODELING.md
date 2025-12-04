# Regime-Aware Modeling Implementation

## Overview

This document describes the regime-aware modeling implementation for the Natural Gas Price Model. Regime-aware modeling enables the system to adapt to different market conditions (e.g., high/low volatility, price level regimes) by:

1. **Feature Engineering**: Creating regime labels and features that identify market states
2. **Sample Weighting**: Weighting training samples based on recency, volatility regime, and regime transitions
3. **Performance Tracking**: Tracking model performance separately for each regime

## Key Components

### 1. Regime Feature Engineering (`src/feature_engineering/regime_features.py`)

The `RegimeFeatureEngineer` class creates regime-aware features:

- **`volatility_regime`**: Labels (0=low, 1=medium, 2=high) based on rolling volatility of returns
- **`price_level_regime`**: Labels based on z-score bands of price levels
- **`regime_label`**: Primary regime label (configurable: volatility, price_level, or kmeans)
- **`volatility`**: Rolling standard deviation of returns
- **`price_zscore`**: Z-score of price relative to rolling mean
- **`regime_transition`**: Binary indicator for regime changes

**Important**: All features use only backward-looking rolling windows to prevent look-ahead bias.

### 2. Regime-Aware Training (`src/models/regime_models.py`)

The `RegimeAwareTrainer` class provides:

- **`compute_regime_sample_weights()`**: Computes sample weights based on:
  - Recency (exponential decay)
  - Volatility regime (higher weight to high-volatility periods)
  - Regime transitions (higher weight to transition periods)

### 3. Configuration (`config.py`)

The `REGIME_CONFIG` dictionary controls regime-aware behavior:

```python
REGIME_CONFIG = {
    "enabled": False,  # Set to True to enable regime-aware modeling
    "regime_method": "volatility",  # "volatility", "price_level", or "kmeans"
    "volatility_window": 20,
    "price_window": 60,
    "n_regimes": 3,
    "use_sample_weighting": True,
    "recency_decay_rate": 0.995,
    "volatility_weight_multipliers": {0: 0.8, 1: 1.0, 2: 1.5},  # Low, Med, High
    "transition_weight_multiplier": 1.2
}
```

### 4. Model Updates

All model classes now support `sample_weight` parameter:

- **`BaselineModel.fit()`**: Accepts `sample_weight` for ElasticNet, LinearRegression, RandomForest
- **`XGBoostModel.fit()`**: Accepts `sample_weight` via DMatrix
- **`LightGBMModel.fit()`**: Accepts `sample_weight` via Dataset
- **`BaselinePipeline.train_final_model()`**: Accepts `sample_weight`
- **`TreeModelPipeline.train_final_model()`**: Accepts `sample_weight`

### 5. Training Pipeline Integration (`src/pipeline/train_baseline.py`)

The `ModelTrainingPipeline` class:

- Automatically applies regime feature engineering when `REGIME_CONFIG["enabled"] = True`
- Computes and applies sample weights during training when `use_sample_weighting = True`
- Maintains backward compatibility (when disabled, behavior is identical to original)

### 6. Backtesting Updates (`src/evaluation/backtesting.py`)

The `Backtester` class:

- Tracks regime-specific performance when `track_regimes=True`
- Provides `get_regime_performance_summary()` to report metrics by regime
- `ModelComparison` supports regime tracking via `track_regimes` parameter

## Usage

### Enable Regime-Aware Modeling

Set environment variable or modify `config.py`:

```python
# In config.py or via environment variable
REGIME_CONFIG["enabled"] = True
```

### Run Training

```python
from src.pipeline.train_baseline import ModelTrainingPipeline

pipeline = ModelTrainingPipeline()
df = pipeline.get_training_data(start_date="2020-01-01", end_date="2024-12-31")
pipeline.train_tree_models(df)  # Models will use regime-aware weighting if enabled
```

### Test Regime-Aware Modeling

```python
from src.pipeline.train_baseline import test_regime_aware_modeling

test_regime_aware_modeling()
```

This will:
1. Load recent data
2. Train models with regime features
3. Run a short backtest
4. Report overall and regime-specific metrics

## Backward Compatibility

The implementation is fully backward compatible:

- When `REGIME_CONFIG["enabled"] = False` (default), behavior is identical to the original pipeline
- All new parameters are optional with sensible defaults
- Existing code continues to work without modification

## Look-Ahead Bias Prevention

All regime features are computed using only backward-looking information:

- Rolling windows use `.rolling()` with no `center=True`
- Regime labels use expanding windows up to current point
- No future information is used in feature computation

## Performance Tracking

When regime tracking is enabled, the backtester reports:

- Overall metrics (MAE, RMSE, MAPE, Direction Accuracy)
- Regime-specific metrics (MAE and RMSE by regime)
- Sample counts per regime

This helps identify which regimes are more challenging to predict.

