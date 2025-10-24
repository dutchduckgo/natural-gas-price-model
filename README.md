# U.S. LNG / Natural Gas Price Model

A comprehensive end-to-end pipeline for predicting U.S. natural gas prices using multiple data sources and machine learning approaches.

## Overview

This project implements a multi-level modeling approach for U.S. natural gas price prediction:

- **Level 0**: Baseline OLS/Elastic Net models
- **Level 1**: Tree-based models (XGBoost/LightGBM) with feature engineering
- **Level 2**: Deep learning models (LSTM/Temporal Fusion Transformer)

## Data Sources

### Core Price Data
- **EIA Open Data API**: Henry Hub spot prices, storage, production
- **CME Group**: NYMEX futures data
- **EIA-930**: Hourly electric grid monitor data

### Weather & Demand Drivers
- **NWS/NOAA**: Weather forecasts and observations
- **CPC**: Degree day data (HDD/CDD)
- **NOMADS**: GFS weather model data

### Supply & Infrastructure
- **Baker Hughes**: Rig count data
- **FERC**: LNG terminal status
- **DOE/FECM**: LNG export data

## Project Structure

```
├── data/                    # Raw and processed data storage
├── src/
│   ├── data_ingestion/      # API clients and data collectors
│   ├── feature_engineering/ # Feature creation and transformation
│   ├── models/             # ML model implementations
│   ├── evaluation/         # Backtesting and model evaluation
│   └── pipeline/           # End-to-end pipeline orchestration
├── notebooks/              # Jupyter notebooks for analysis
├── config/                 # Configuration files
└── tests/                  # Unit tests
```

## Quick Start

1. Install dependencies:
```bash
pip install -r requirements.txt
```

2. Set up environment variables:
```bash
cp .env.example .env
# Edit .env with your API keys
```

3. Run data ingestion:
```bash
python -m src.pipeline.ingest_data
```

4. Train baseline model:
```bash
python -m src.pipeline.train_baseline
```

## Features

- **Multi-source data integration**: EIA, weather, power grid, LNG infrastructure
- **Feature engineering**: Weather derivatives, storage metrics, power burn proxies
- **Temporal modeling**: Handles forecast horizons and data leakage prevention
- **Backtesting framework**: Walk-forward validation with realistic timing
- **Production ready**: Modular design with proper error handling and logging

## Model Performance

Expected performance benchmarks:
- **7-day horizon**: MAPE ~8-12%
- **14-day horizon**: MAPE ~12-18%
- **30-day horizon**: MAPE ~18-25%

Key drivers: HDD/CDD, storage levels, power burn, LNG capacity
