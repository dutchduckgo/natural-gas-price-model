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
TARGET_HORIZON_DAYS = int(os.getenv("TARGET_HORIZON_DAYS", 7))
WALK_FORWARD_WINDOW = 252  # 1 year of trading days
MODEL_RETRAIN_FREQUENCY = "weekly"

# Data sources configuration
EIA_BASE_URL = "https://api.eia.gov/v2"
EIA_SERIES_IDS = {
    "henry_hub_spot": "NG.RNGWHHD.D",  # Henry Hub spot price (daily)
    "storage": ["NW2_EPG0_SWO_R31_BCF", "NW2_EPG0_SWO_R32_BCF", "NW2_EPG0_SWO_R33_BCF", 
                "NW2_EPG0_SWO_R34_BCF", "NW2_EPG0_SWO_R35_BCF", "NW2_EPG0_SWO_R48_BCF"],  # Working gas in storage (weekly) - all regions
    "production": "N9070US2",  # Dry natural gas production (monthly)
    "consumption": "NG.N3010US2.M",  # Natural gas consumption (monthly) - TBD
    "lng_exports": "N9133US2",  # LNG exports (monthly)
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

# Regime-aware modeling configuration
REGIME_CONFIG = {
    "enabled": os.getenv("REGIME_ENABLED", "True").lower() == "true",  # Enable by default for poster generation
    "regime_method": os.getenv("REGIME_METHOD", "volatility"),  # "volatility", "price_level", or "kmeans"
    "volatility_window": int(os.getenv("REGIME_VOLATILITY_WINDOW", 20)),
    "price_window": int(os.getenv("REGIME_PRICE_WINDOW", 60)),
    "n_regimes": int(os.getenv("REGIME_N_REGIMES", 3)),
    "use_sample_weighting": os.getenv("REGIME_USE_SAMPLE_WEIGHTING", "True").lower() == "true",
    "recency_decay_rate": float(os.getenv("REGIME_RECENCY_DECAY", 0.995)),
    "volatility_weight_multipliers": {
        0: float(os.getenv("REGIME_VOL_WEIGHT_LOW", 0.8)),   # Low volatility
        1: float(os.getenv("REGIME_VOL_WEIGHT_MED", 1.0)),   # Medium volatility
        2: float(os.getenv("REGIME_VOL_WEIGHT_HIGH", 1.5))   # High volatility
    },
    "transition_weight_multiplier": float(os.getenv("REGIME_TRANSITION_WEIGHT", 1.2))
}

# Logging configuration
LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO")
LOG_FILE = LOG_DIR / "gas_model.log"
