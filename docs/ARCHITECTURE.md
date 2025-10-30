# Natural Gas Price Model — Architecture and Operations

This document explains how the repository works end-to-end: data sources, ingestion, storage schema, feature engineering, modeling, evaluation, leakage control, orchestration, and how to extend/operate it in production.

## 1) Goals and Scope

- Predict U.S. natural gas prices (starting with `Henry Hub` spot) over multiple horizons (7/14/30d) using fundamentals (weather, storage, power burn, LNG, rigs) and market structure (term structure).
- Provide a clean, modular pipeline from ingestion → features → models → evaluation → reporting.
- Avoid data leakage by using calendar-true timing and walk-forward evaluation.

Deliverables:
- Baseline (Elastic Net / Linear / RF)
- Tree models (XGBoost / LightGBM)
- Deep models (LSTM / Transformer; scaffolding provided)
- Backtesting framework and comparisons

## 2) Repository Map (What lives where)

- `config.py`: Global configuration (paths, API endpoints, hyperparameters, logging)
- `requirements.txt`: Python deps
- `demo.py` / `demo_simple.py`: End-to-end demos (the simple version avoids OpenMP deps)
- `src/data_ingestion/`
  - `eia_client.py`: EIA Open Data API client (spot, storage, production, LNG)
  - `weather_client.py`: NWS/CPC scaffolding; degree days from forecasts/historical (placeholders included)
  - `power_client.py`: EIA-930 / PJM / ERCOT / ISO-NE scaffolding (placeholders included)
  - `database.py`: DuckDB schema, inserts, and feature matrix builder
- `src/feature_engineering/`
  - `weather_features.py`: HDD/CDD, lags, rolling windows, seasonal and interaction features
  - `storage_features.py`: storage tightness, lags/rollings, seasonal flags, projections
- `src/models/`
  - `baseline.py`: Elastic Net, Linear, Random Forest + pipelines
  - `tree_models.py`: XGBoost/LightGBM + pipelines
  - `deep_learning.py`: LSTM/Transformer + sequence pipelines
- `src/evaluation/`
  - `backtesting.py`: Rolling-origin CV, metrics aggregation, model comparison, plotting
- `src/pipeline/`
  - `ingest_data.py`: Unified ingestion job → DB
  - `train_baseline.py`: End-to-end training/evaluation/report generation
- `notebooks/example_usage.ipynb`: Walkthrough notebook

## 3) Data Model (DuckDB schema)

Tables (grain and purpose):

- `prices` (daily)
  - `date, hub, spot_price, front_month, m1_m2_spread`
  - Source: EIA spot; CME (spreads) when integrated
- `storage_weekly` (weekly report date)
  - `report_date, working_gas, five_year_avg, yoy_deviation, wow_change`
  - Source: EIA storage weekly. Derived columns can be filled post-ingestion
- `production_monthly` (monthly)
  - `month, dry_gas_bcfpd` (EIA-914)
- `lng_monthly` (monthly)
  - `month, exports_bcf, avg_price, terminal_notes, lng_capacity_bcfpd`
- `power_burn` (daily)
  - `date, ba, gas_mwh, total_load_mwh, renewables_mwh`
- `weather_daily` (daily)
  - `date, region, hdd, cdd, hdd_norm_delta, cdd_norm_delta, temperature, wind_speed`
- `rigs_weekly` (weekly)
  - `date, total_rigs, oil_rigs, gas_rigs`
- `events` (daily)
  - `date, label, lng_cap_offline_bcfpd`
- `features` (daily, generic)
  - `date, feature_name, feature_value`
- `predictions` (daily by model + horizon)
  - `date, model_name, horizon_days, prediction, confidence_lower, confidence_upper, actual_value, error`

See: `src/data_ingestion/database.py` for CREATE TABLE statements and insert helpers.

## 4) Data Ingestion (APIs → DataFrames → DB)

- Entrypoint: `src/pipeline/ingest_data.py`
  - Uses `EIAClient` to fetch core series (spot, storage, production, LNG). Current implementation fetches and normalizes to the schema. Some series (e.g., 5y avg storage, CME spreads) are noted for follow-up integration.
  - `WeatherClient` fetches NWS forecast slices and (placeholder) CPC degree day backfills. You will extend this with CPC CSVs and NOMADS GRIB for historical/forecasted HDD/CDD.
  - `PowerGridClient` is scaffolded for EIA-930/PJM/ERCOT/ISO-NE. Add authenticated calls and parsers, then derive daily gas-burn proxies.
  - All dataframes are inserted to DuckDB using `GasModelDatabase` insert methods.

Best practices:
- Normalize at ingestion to consistent column names (`date`, numeric `value` columns) before inserting.
- Keep raw dumps as Parquet if you need full fidelity (optional; Parquet path exists in `config.py`).

## 5) Feature Engineering (domain-driven)

Weather-driven demand:
- Degree days: `HDD = max(65F - T, 0)`, `CDD = max(T - 65F, 0)`
- Lags: `t-1, t-3, t-7, t-14, t-30`
- Rolling windows: 7/14/30d means, std, extrema
- Forecast handling: merge future-known covariates for horizons (surprise = today’s D+7 HDD − yesterday’s D+7 HDD)
- Seasonal: month/quarter/day-of-year and their sin/cos cycles
- Interactions: HDD×CDD, Temp×Wind, HDD×Wind (proxy for wind chill)

Storage tightness:
- `(Working gas − 5yr avg) / 5yr avg`
- Injection/withdrawal flags and seasonality (injection: Apr–Oct; withdrawal: Nov–Mar)
- Storage velocity: diffs/pct-change; simple linear projection for 4/8 weeks

Power burn proxy:
- From EIA-930/ISOs: daily gas_mwh; interactions with renewables (`pct_renewables`) and load

Term structure (when added):
- M1−M2 spread, implied vol (CVOL), curve shape features

Implementation: `src/feature_engineering/weather_features.py`, `storage_features.py`

## 6) Modeling Layers

Level 0 — Baseline (`src/models/baseline.py`)
- Elastic Net: linear model with L1/L2 regularization; good for tabular with many correlated features
- Linear Regression: reference point / sanity check
- Random Forest: non-linear baseline, robust but less interpretable
- Pipeline: scaling (where appropriate), fit, predict, feature selection by numeric dtype; CV and walk-forward helpers

Level 1 — Trees (`src/models/tree_models.py`)
- XGBoost / LightGBM: gradient boosting with support for monotonic constraints and fast training
- Early stopping via validation folds; feature importance (split/gain)
- When enabling monotonicity: encode priors (e.g., HDD↑ → price↑; storage surplus↑ → price↓)

Level 2 — Deep (`src/models/deep_learning.py`)
- LSTM and Transformer scaffolds for sequence modeling
- Known/observed covariates separation and sequence windowing (`seq_length`)
- Pinball loss for quantiles can be added; TFT reference (Lim et al.) for production

Directional overlay (optional)
- Binary classifier for up/down over next K days using same features (+ thresholds for costs)

## 7) Evaluation and Backtesting

- `src/evaluation/backtesting.py`
  - Rolling-origin (walk-forward) validation with configurable window and step_size
  - Metrics per period: MAE, RMSE, MAPE, directional accuracy; aggregate mean/σ
  - ModelComparison: evaluate multiple models with a consistent protocol
  - Plotting utilities for diagnostics (MAE/RMSE/direction over time; preds vs actual)

Leakage control:
- Use only information that is truly available at the forecast timestamp
- Respect EIA release calendars (storage Thu 9:30a CT) and monthly releases
- For weather: use T−1 observations; T+1..T+H forecasts as of run time
- In DB feature matrix joins, align weekly/monthly series with appropriate carry-forward or MIDAS-like rules

## 8) Orchestration and Jobs

Run order (daily job suggestion):
1) Ingest (after storage release on Thu if targeting weekly refits)
2) Build/refresh features
3) Train/refit models (weekly cadence aligns with EIA storage)
4) Generate predictions and evaluation reports

Entry points:
- `python -m src.pipeline.ingest_data` — populate DB
- `python -m src.pipeline.train_baseline` — train/evaluate/report and save models
- `python demo_simple.py` — no external ML libs needed; quick E2E sanity
- `python demo.py` — full demo (requires OpenMP for XGBoost/LightGBM on macOS)

Schedulers:
- Local cron / launchd / systemd
- Cloud: GitHub Actions (nightly), Airflow, Prefect

## 9) Configuration and Secrets

- `config.py`: centralizes:
  - Paths: data, models, results, logs
  - API endpoints: EIA/NWS/CPC/EIA-930/PJM/ERCOT
  - Modeling params: lags/windows/hyperparameters
  - Logging config
- Secrets: use environment variables (`EIA_API_KEY`, etc.). Do not commit `.env`.

## 10) Extending the System

Add a new data source:
1) Create a client in `src/data_ingestion/` with a `get_*` method that returns a normalized DataFrame (`date`, numeric columns)
2) Add insert method or direct write to an existing table
3) Update `ingest_data.py` to call it; ensure type-safe joins in `database.py`

Add a new feature transform:
1) Add transform in `feature_engineering/` returning a DataFrame keyed on `date`
2) Merge into feature matrix builder or apply in model prep pipeline

Add a new model:
1) Implement a class with `prepare_features`, `fit`, `predict`, and optional `get_feature_importance`
2) Add to training pipeline and `ModelComparison`

Quantiles and intervals:
- Use pinball loss (quantile regression) in GBMs or deep models
- Calibrate intervals with residual bootstrap on walk-forward errors

## 11) Performance, Reliability, Monitoring

Performance:
- Columnar storage (DuckDB + Parquet) and vectorized Pandas/Polars operations
- Cache expensive API pulls locally; incremental updates by date

Reliability:
- Input validation on API responses
- Schema checks on DataFrames prior to insert
- Unit tests for feature math and date alignment (add `tests/`)

Monitoring:
- Track rolling error metrics (MAE/MAPE) and alert on drift
- Log feature drift (distribution shift) and model confidence calibration

## 12) Known Gaps and Next Steps

- Complete CPC HDD/CDD backfill and NOMADS forecast integration
- Implement EIA-930, PJM, ERCOT, ISO-NE data collectors with auth and pagination
- Add CME futures curve and CVOL features
- Enforce monotonic constraints in tree models; document priors explicitly
- Implement TFT with quantiles and holiday/event embeddings
- Productionize model persistence (save/load) and model registry

## 13) Quick FAQ

- Why DuckDB? Fast analytics, simple setup, no server.
- Why Elastic Net baseline? Strong tabular baseline and interpretable coefficients.
- How is leakage avoided? Calendar-true releases, future covariates restricted to forecast info, rolling-origin evaluation.
- What horizon is best? 7–14d tend to benefit most from HDD/CDD + storage; 30d needs curve structure and LNG capacity context.
