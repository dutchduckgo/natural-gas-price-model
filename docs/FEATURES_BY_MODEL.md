# Features Used by Each Model Type

## Overview

**Important**: All model types (baseline, tree, deep learning) use the **same feature set**. The difference is how they process and use these features, not which features they receive.

All models automatically select all numeric columns from the input DataFrame (excluding the target variable `spot_price` and the `date` column). This means they all benefit from the same comprehensive feature engineering pipeline.

## Complete Feature List

### Core Raw Data Features
- `spot_price` (target variable - excluded from features)
- `hdd` - Heating degree days (base 65°F)
- `cdd` - Cooling degree days (base 65°F)
- `storage` - Working gas in storage (BCF)
- `temperature` - Average temperature
- `production` - Dry gas production (BCF/day)
- `working_gas` - Working gas storage level
- `five_year_avg` - 5-year average storage level

### Weather Features (from WeatherFeatureEngineer)

**Base Degree Days:**
- `HDD` - Heating degree days
- `CDD` - Cooling degree days
- `HDD_weighted` - Population-weighted HDD (if regional data)
- `CDD_weighted` - Population-weighted CDD (if regional data)

**Lagged Features** (lags: 1, 3, 7, 14, 30 days):
- `HDD_lag_1`, `HDD_lag_3`, `HDD_lag_7`, `HDD_lag_14`, `HDD_lag_30`
- `CDD_lag_1`, `CDD_lag_3`, `CDD_lag_7`, `CDD_lag_14`, `CDD_lag_30`
- `temperature_lag_1`, `temperature_lag_3`, `temperature_lag_7`, `temperature_lag_14`, `temperature_lag_30`

**Rolling Window Features** (windows: 7, 14, 30 days):
For each of HDD, CDD, temperature:
- `{col}_rolling_7` - 7-day rolling mean
- `{col}_rolling_7_std` - 7-day rolling standard deviation
- `{col}_rolling_7_max` - 7-day rolling maximum
- `{col}_rolling_7_min` - 7-day rolling minimum
- Same for 14-day and 30-day windows

**Forecast Features** (if forecast data available):
- `HDD_forecast` - Forecasted HDD
- `CDD_forecast` - Forecasted CDD
- `temp_forecast_error` - Forecast error (forecast - actual)

**Seasonal Features:**
- `month` - Month of year (1-12)
- `quarter` - Quarter (1-4)
- `day_of_year` - Day of year (1-365)
- `month_sin`, `month_cos` - Cyclical month encoding
- `day_sin`, `day_cos` - Cyclical day-of-year encoding
- `is_winter`, `is_spring`, `is_summer`, `is_fall` - Season indicators

**Interaction Features:**
- `HDD_CDD_interaction` - HDD × CDD
- `temp_wind_interaction` - Temperature × Wind speed
- `HDD_wind_interaction` - HDD × Wind speed (wind chill effect)

### Storage Features (from StorageFeatureEngineer)

**Base Metrics:**
- `storage_yoy_deviation` - Year-over-year deviation from 5-year avg
- `storage_yoy_deviation_pct` - YoY deviation as percentage
- `storage_cover_days` - Days of supply (storage / daily consumption)
- `storage_change` - Week-over-week change
- `storage_change_pct` - Week-over-week percentage change

**Lagged Features** (lags: 1, 7, 14, 30 days):
- `working_gas_lag_1`, `working_gas_lag_7`, `working_gas_lag_14`, `working_gas_lag_30`
- `storage_yoy_deviation_lag_1`, `storage_yoy_deviation_lag_7`, `storage_yoy_deviation_lag_14`, `storage_yoy_deviation_lag_30`
- `storage_cover_days_lag_1`, `storage_cover_days_lag_7`, `storage_cover_days_lag_14`, `storage_cover_days_lag_30`

**Rolling Window Features** (windows: 4, 8, 12 weeks):
For working_gas and storage_yoy_deviation:
- `{col}_rolling_4w` - 4-week rolling mean
- `{col}_rolling_4w_std` - 4-week rolling std
- `{col}_rolling_4w_min` - 4-week rolling min
- `{col}_rolling_4w_max` - 4-week rolling max
- Same for 8-week and 12-week windows

**Seasonal Features:**
- `month` - Month of year
- `quarter` - Quarter
- `week_of_year` - Week of year
- `month_sin`, `month_cos` - Cyclical month encoding
- `is_injection_season` - April-October indicator
- `is_withdrawal_season` - November-March indicator

**Tightness Features:**
- `storage_tightness` - Normalized tightness: (working_gas - 5yr_avg) / 5yr_avg
- `storage_very_tight` - Binary: tightness < -10%
- `storage_tight` - Binary: tightness -10% to -5%
- `storage_normal` - Binary: tightness -5% to +5%
- `storage_loose` - Binary: tightness > +5%
- `storage_velocity` - Rate of change (first difference)
- `storage_velocity_pct` - Percentage rate of change

**Forecast Features:**
- `storage_trend` - Linear trend from recent 4 weeks
- `storage_projected_4w` - Projected storage 4 weeks ahead
- `storage_projected_8w` - Projected storage 8 weeks ahead

**Interaction Features:**
- `storage_injection_interaction` - Storage × injection season indicator
- `storage_temp_interaction` - Storage × temperature (if available)

### Power Sector Features (when available)
- `gas_mwh` - Gas-fired generation (MWh)
- `total_load_mwh` - Total system load (MWh)
- `renewables_mwh` - Renewable generation (MWh)
- `pct_renewables` - Renewable percentage of load

### Market Structure Features (when available)
- `front_month` - Front-month futures price
- `m1_m2_spread` - Front-to-second month spread
- `implied_vol` - Implied volatility (when CME data integrated)

## How Each Model Type Uses Features

### Baseline Models (Linear Regression, Elastic Net, Random Forest)

**Feature Selection:**
- Uses ALL numeric features from the engineered feature set
- Automatically excludes: `spot_price` (target), `date` (non-numeric)

**Processing:**
- **Linear Regression & Elastic Net**: Features are standardized (mean=0, std=1) using StandardScaler before training
- **Random Forest**: No scaling needed (tree-based models are scale-invariant)

**Feature Usage:**
- Linear/Elastic Net: Each feature gets a coefficient; you can interpret which features drive price up/down
- Random Forest: Uses feature importance to show which features matter most for splits

**Example Features Used:**
- All HDD/CDD features, all storage features, all seasonal features, all interaction features
- Typically 50-70+ features depending on data availability

### Tree Models (XGBoost, LightGBM)

**Feature Selection:**
- Uses ALL numeric features from the engineered feature set (same as baseline)
- Automatically excludes: `spot_price` (target), `date` (non-numeric)

**Processing:**
- No scaling needed (tree models handle different scales naturally)
- Can apply monotonic constraints (e.g., HDD↑ → price↑, storage↑ → price↓)

**Feature Usage:**
- Automatically discovers non-linear relationships and interactions
- Can show feature importance (split importance, gain importance)
- More robust to irrelevant features (can ignore noise)

**Example Features Used:**
- Same 50-70+ features as baseline models
- Tree models excel at finding complex interactions (e.g., "cold weather + low storage + high power demand")

### Deep Learning Models (LSTM, Transformer)

**Feature Selection:**
- Uses ALL numeric features from the engineered feature set (same as baseline and tree)
- Automatically excludes: `spot_price` (target), `date` (non-numeric)

**Processing:**
- Features are standardized (mean=0, std=1) using StandardScaler
- **Key Difference**: Features are organized into **sequences** (windows of past values)
- Default sequence length: 30 days (configurable)

**Feature Usage:**
- LSTM: Processes sequences to learn temporal patterns (e.g., "if storage was low for 2 weeks, then price spikes")
- Transformer: Uses attention to focus on important time steps and features
- Can learn long-term dependencies (e.g., seasonal patterns, regime changes)

**Example Features Used:**
- Same 50-70+ features as other models
- But organized as: [features at t-30, t-29, ..., t-1] → predict price at t
- This allows the model to see how features evolved over the past month

## Feature Engineering Pipeline

All features are created in this order:

1. **Raw Data Collection**: EIA, weather, storage, power data
2. **Weather Feature Engineering**: HDD/CDD, lags, rolling windows, seasonal, interactions
3. **Storage Feature Engineering**: Tightness metrics, lags, rolling windows, seasonal, projections
4. **Combined Feature Matrix**: All features merged by date
5. **Model Training**: Each model type receives the same feature matrix

## Feature Count Summary

**Typical Feature Count:**
- Base features: ~10-15 (raw data columns)
- Weather features: ~40-50 (with lags, rolling windows, seasonal, interactions)
- Storage features: ~20-30 (with lags, rolling windows, tightness, projections)
- **Total: ~70-95 features** (depending on data availability)

**All models use the same feature set** - the difference is in how they process and learn from these features.

## Key Insights

1. **No Feature Selection by Model**: All models get the same comprehensive feature set. The feature engineering is done once, then shared.

2. **Model Differences are in Processing**:
   - Baseline: Linear relationships, interpretable coefficients
   - Tree: Non-linear relationships, automatic interaction discovery
   - Deep: Temporal patterns, sequence dependencies

3. **Feature Engineering is Critical**: The quality of features (HDD/CDD, storage tightness, lags, rolling windows) matters more than which model you use.

4. **Scalability**: Adding new features (e.g., LNG capacity, rig counts) automatically makes them available to all model types.

