# 🔥 Natural Gas Price Model

[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://python.org)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

A comprehensive end-to-end pipeline for predicting U.S. natural gas prices using multiple data sources and machine learning approaches.

> Full system overview: see `docs/ARCHITECTURE.md`.

## 🌟 Overview

This project implements a multi-level modeling approach for U.S. natural gas price prediction:

- **Level 0**: Baseline OLS/Elastic Net models
- **Level 1**: Tree-based models (XGBoost/LightGBM) with feature engineering  
- **Level 2**: Deep learning models (LSTM/Temporal Fusion Transformer)

## 📊 Data Sources

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

## 🏗️ Project Structure

```
├── data/                    # Raw and processed data storage
├── src/
│   ├── data_ingestion/      # API clients and data collectors
│   ├── feature_engineering/ # Feature creation and transformation
│   ├── models/             # ML model implementations
│   ├── evaluation/         # Backtesting and model evaluation
│   └── pipeline/           # End-to-end pipeline orchestration
├── notebooks/              # Jupyter notebooks for analysis
├── docs/                   # Architecture and design docs
└── tests/                  # Unit tests
```

## 🚀 Quick Start

### 1. Installation

```bash
# Clone the repository
git clone https://github.com/YOUR_USERNAME/natural-gas-price-model.git
cd natural-gas-price-model

# Install dependencies
pip install -r requirements.txt
```

### 2. Configuration

```bash
# Set up environment variables
cp .env.example .env
# Edit .env with your API keys
```

### 3. Run Demo

```bash
# Run the complete demonstration
python demo.py

# Or explore the Jupyter notebook
jupyter notebook notebooks/example_usage.ipynb
```

### 4. Production Pipeline

```bash
# Run data ingestion
python -m src.pipeline.ingest_data

# Train models
python -m src.pipeline.train_baseline
```

## ✨ Features

- **🔗 Multi-source data integration**: EIA, weather, power grid, LNG infrastructure
- **⚙️ Feature engineering**: Weather derivatives, storage metrics, power burn proxies
- **📈 Temporal modeling**: Handles forecast horizons and data leakage prevention
- **🧪 Backtesting framework**: Walk-forward validation with realistic timing
- **🏭 Production ready**: Modular design with proper error handling and logging

## 📈 Model Performance

Expected performance benchmarks:
- **7-day horizon**: MAPE ~8-12%
- **14-day horizon**: MAPE ~12-18%
- **30-day horizon**: MAPE ~18-25%

**Key drivers**: HDD/CDD, storage levels, power burn, LNG capacity

## 🎯 Usage Examples

### Basic Usage

```python
from src.data_ingestion.eia_client import EIAClient
from src.models.baseline import BaselinePipeline
from src.evaluation.backtesting import Backtester

# Collect data
eia_client = EIAClient()
data = eia_client.get_all_core_data()

# Train model
pipeline = BaselinePipeline("elastic_net")
pipeline.train_final_model(data)

# Evaluate
backtester = Backtester()
results = backtester.walk_forward_validation(data, pipeline.model)
```

### Advanced Usage

```python
# Feature engineering
from src.feature_engineering.weather_features import WeatherFeatureEngineer
from src.feature_engineering.storage_features import StorageFeatureEngineer

weather_engineer = WeatherFeatureEngineer()
features = weather_engineer.engineer_all_weather_features(data)

# Model comparison
from src.evaluation.backtesting import ModelComparison

comparison = ModelComparison()
comparison.add_model("XGBoost", xgb_model)
comparison.add_model("LightGBM", lgb_model)
results = comparison.compare_models(data)
```

## 📚 Documentation

- **[Architecture & Operations](docs/ARCHITECTURE.md)**: Full technical overview and production guidance
- **[Implementation Summary](IMPLEMENTATION_SUMMARY.md)**: High-level changes and capabilities
- **[Example Notebook](notebooks/example_usage.ipynb)**: Step-by-step tutorial

## 🤝 Contributing

We welcome contributions! Please see our [Contributing Guidelines](CONTRIBUTING.md) for details.

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- U.S. Energy Information Administration for comprehensive energy data
- National Weather Service for weather forecasts and observations
- Climate Prediction Center for degree day data
- CME Group for futures market data

## 📞 Support

If you have any questions or need help, please:
- Open an issue on GitHub
- Check the documentation
- Review the example notebook

---

**⭐ Star this repository if you find it helpful!**
