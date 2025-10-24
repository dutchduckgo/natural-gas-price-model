"""
Configuration settings for the natural gas price model.
"""
import os
from pathlib import Path
from typing import List

# Project paths
PROJECT_ROOT = Path(__file__).parent
DATA_DIR = PROJECT_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
MODEL_DIR = PROJECT_ROOT / "models"
RESULTS_DIR = PROJECT_ROOT / "results"
LOG_DIR = PROJECT_ROOT / "logs"

# Create directories if they don't exist
for directory in [DATA_DIR, RAW_DATA_DIR, PROCESSED_DATA_DIR, MODEL_DIR, RESULTS_DIR, LOG_DIR]:
    directory.mkdir(exist_ok=True)

# Database configuration
DATABASE_URL = os.getenv("DATABASE_URL", f"duckdb:///{DATA_DIR}/gas_model.db")

# API Configuration
EIA_API_KEY = os.getenv("EIA_API_KEY", "")
CME_API_KEY = os.getenv("CME_API_KEY", "")

# Model configuration
FORECAST_HORIZONS = [7, 14, 30]  # days
WALK_FORWARD_WINDOW = 252  # 1 year of trading days
MODEL_RETRAIN_FREQUENCY = "weekly"

# Data sources configuration
EIA_BASE_URL = "https://api.eia.gov/v2"
EIA_SERIES_IDS = {
    "henry_hub_spot": "NG.RNGWHHD.D",  # Henry Hub spot price
    "storage": "NG.NW2_EPG0_SWO_R48_BCF.W",  # Working gas in storage
    "production": "NG.N9010US2.M",  # Dry natural gas production
    "consumption": "NG.N3010US2.M",  # Natural gas consumption
    "lng_exports": "NG.N9130US2.M",  # LNG exports
}

# Weather data sources
NWS_BASE_URL = "https://api.weather.gov"
CPC_BASE_URL = "https://www.cpc.ncep.noaa.gov/products/analysis_monitoring/cdus/degree_days"

# Power grid data
EIA_930_BASE_URL = "https://api.eia.gov/v2/electricity/rto"
PJM_BASE_URL = "https://dataminer2.pjm.com"
ERCOT_BASE_URL = "https://www.ercot.com/api"

# Feature engineering parameters
WEATHER_LAGS = [1, 3, 7, 14, 30]  # days
STORAGE_LAGS = [1, 7, 14, 30]  # days
POWER_BURN_LAGS = [1, 7, 14]  # days

# Model hyperparameters
BASELINE_PARAMS = {
    "elastic_net": {
        "alpha": 0.01,
        "l1_ratio": 0.5,
        "max_iter": 1000
    }
}

XGBOOST_PARAMS = {
    "n_estimators": 1000,
    "max_depth": 6,
    "learning_rate": 0.1,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "random_state": 42
}

LIGHTGBM_PARAMS = {
    "n_estimators": 1000,
    "max_depth": 6,
    "learning_rate": 0.1,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "random_state": 42
}

# Evaluation metrics
EVALUATION_METRICS = ["mae", "mape", "rmse", "r2"]
QUANTILE_LEVELS = [0.1, 0.5, 0.9]  # P10, P50, P90

# Logging configuration
LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO")
LOG_FILE = LOG_DIR / "gas_model.log"
