#!/usr/bin/env python3
"""
Natural Gas Price Model - Demonstration Script

This script demonstrates the complete pipeline from data collection
to model training and evaluation.
"""

import sys
import os
import logging
from datetime import datetime, timedelta
import pandas as pd
import numpy as np

# Add src to path
sys.path.append('src')

# Import our modules
from src.data_ingestion.eia_client import EIAClient
from src.data_ingestion.weather_client import WeatherClient
from src.data_ingestion.database import GasModelDatabase
from src.feature_engineering.weather_features import WeatherFeatureEngineer
from src.feature_engineering.storage_features import StorageFeatureEngineer
from src.models.baseline import BaselinePipeline
from src.models.tree_models import TreeModelPipeline
from src.evaluation.backtesting import Backtester, ModelComparison

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def create_sample_data():
    """Create sample data for demonstration."""
    logger.info("Creating sample data for demonstration")
    
    np.random.seed(42)
    n_samples = 1000
    
    dates = pd.date_range('2020-01-01', periods=n_samples)
    
    # Create synthetic features with realistic patterns
    # Heating degree days (winter heating demand)
    hdd = 20 + 30 * np.sin(2 * np.pi * np.arange(n_samples) / 365) + np.random.normal(0, 5, n_samples)
    hdd = np.maximum(hdd, 0)  # HDD can't be negative
    
    # Cooling degree days (summer cooling demand)
    cdd = 10 + 20 * np.sin(2 * np.pi * np.arange(n_samples) / 365) + np.random.normal(0, 3, n_samples)
    cdd = np.maximum(cdd, 0)  # CDD can't be negative
    
    # Storage levels (seasonal pattern)
    storage = 3000 + 500 * np.sin(2 * np.pi * np.arange(n_samples) / 365) + np.random.normal(0, 100, n_samples)
    
    # Temperature (seasonal)
    temperature = 50 + 30 * np.sin(2 * np.pi * np.arange(n_samples) / 365) + np.random.normal(0, 5, n_samples)
    
    # Production (trending up over time)
    production = 100 + 0.1 * np.arange(n_samples) + np.random.normal(0, 5, n_samples)
    
    # Create target with realistic relationships
    spot_price = (
        3.0 +                           # Base price
        0.01 * hdd +                   # Heating demand increases price
        0.005 * cdd +                  # Cooling demand increases price
        -0.0001 * storage +            # High storage decreases price
        0.001 * production +           # Production increases price
        np.random.normal(0, 0.1, n_samples)  # Random noise
    )
    
    # Ensure prices are positive
    spot_price = np.maximum(spot_price, 0.5)
    
    df = pd.DataFrame({
        'date': dates,
        'spot_price': spot_price,
        'hdd': hdd,
        'cdd': cdd,
        'storage': storage,
        'temperature': temperature,
        'production': production
    })
    
    logger.info(f"Created sample data: {len(df)} samples from {df['date'].min()} to {df['date'].max()}")
    logger.info(f"Price range: ${df['spot_price'].min():.2f} to ${df['spot_price'].max():.2f}")
    
    return df


def demonstrate_feature_engineering(df):
    """Demonstrate feature engineering."""
    logger.info("Demonstrating feature engineering")
    
    # Weather feature engineering
    weather_engineer = WeatherFeatureEngineer()
    weather_features = weather_engineer.engineer_all_weather_features(df)
    logger.info(f"Weather features created: {len(weather_features.columns)} columns")
    
    # Storage feature engineering
    storage_engineer = StorageFeatureEngineer()
    storage_features = storage_engineer.engineer_all_storage_features(df)
    logger.info(f"Storage features created: {len(storage_features.columns)} columns")
    
    # Combine all features
    all_features = pd.concat([df, weather_features, storage_features], axis=1)
    logger.info(f"Total features: {len(all_features.columns)} columns")
    
    return all_features


def demonstrate_model_training(df):
    """Demonstrate model training and evaluation."""
    logger.info("Demonstrating model training and evaluation")
    
    # Train baseline models
    logger.info("Training baseline models...")
    
    # Elastic Net
    elastic_net = BaselinePipeline("elastic_net")
    elastic_net_scores = elastic_net.evaluate_model(df)
    elastic_net.train_final_model(df)
    logger.info(f"Elastic Net - MAE: {elastic_net_scores['mae']:.4f}, RMSE: {elastic_net_scores['rmse']:.4f}")
    
    # Random Forest
    random_forest = BaselinePipeline("random_forest")
    rf_scores = random_forest.evaluate_model(df)
    random_forest.train_final_model(df)
    logger.info(f"Random Forest - MAE: {rf_scores['mae']:.4f}, RMSE: {rf_scores['rmse']:.4f}")
    
    # Train tree-based models
    logger.info("Training tree-based models...")
    
    # XGBoost
    xgboost = TreeModelPipeline("xgboost")
    xgb_scores = xgboost.evaluate_model(df)
    xgboost.train_final_model(df)
    logger.info(f"XGBoost - MAE: {xgb_scores['mae']:.4f}, RMSE: {xgb_scores['rmse']:.4f}")
    
    # LightGBM
    lightgbm = TreeModelPipeline("lightgbm")
    lgb_scores = lightgbm.evaluate_model(df)
    lightgbm.train_final_model(df)
    logger.info(f"LightGBM - MAE: {lgb_scores['mae']:.4f}, RMSE: {lgb_scores['rmse']:.4f}")
    
    return {
        "elastic_net": elastic_net,
        "random_forest": random_forest,
        "xgboost": xgboost,
        "lightgbm": lightgbm
    }


def demonstrate_model_comparison(models, df):
    """Demonstrate model comparison."""
    logger.info("Demonstrating model comparison")
    
    # Create model comparison
    comparison = ModelComparison()
    
    # Add models
    for name, model in models.items():
        comparison.add_model(name, model.model)
    
    # Run comparison
    results = comparison.compare_models(df)
    
    # Display results
    logger.info("Model Comparison Results:")
    logger.info("=" * 50)
    
    for model_name, result in results.items():
        metrics = result["metrics"]
        logger.info(f"{model_name}:")
        logger.info(f"  MAE: {metrics['mae_mean']:.4f} ± {metrics['mae_std']:.4f}")
        logger.info(f"  RMSE: {metrics['rmse_mean']:.4f} ± {metrics['rmse_std']:.4f}")
        logger.info(f"  R²: {metrics['r2']:.4f}")
        logger.info(f"  Direction Accuracy: {metrics['direction_accuracy_mean']:.2f}% ± {metrics['direction_accuracy_std']:.2f}%")
        logger.info("")
    
    return results


def demonstrate_backtesting(models, df):
    """Demonstrate backtesting."""
    logger.info("Demonstrating backtesting")
    
    # Run backtesting for XGBoost model
    backtester = Backtester(window_size=200, step_size=50)
    backtest_results = backtester.walk_forward_validation(df, models["xgboost"].model)
    
    logger.info(f"Backtesting completed: {len(backtest_results)} validation periods")
    
    # Calculate performance metrics
    metrics = backtester.calculate_performance_metrics()
    logger.info("Backtesting Performance:")
    logger.info(f"  MAE: {metrics['mae_mean']:.4f} ± {metrics['mae_std']:.4f}")
    logger.info(f"  RMSE: {metrics['rmse_mean']:.4f} ± {metrics['rmse_std']:.4f}")
    logger.info(f"  MAPE: {metrics['mape_mean']:.2f}% ± {metrics['mape_std']:.2f}%")
    logger.info(f"  Direction Accuracy: {metrics['direction_accuracy_mean']:.2f}% ± {metrics['direction_accuracy_std']:.2f}%")
    
    return metrics


def demonstrate_feature_importance(models):
    """Demonstrate feature importance analysis."""
    logger.info("Demonstrating feature importance analysis")
    
    # Get feature importance from XGBoost model
    xgb_importance = models["xgboost"].get_feature_importance()
    
    logger.info("Top 10 Most Important Features (XGBoost):")
    for i, (_, row) in enumerate(xgb_importance.head(10).iterrows()):
        logger.info(f"  {i+1}. {row['feature']}: {row['importance']:.4f}")
    
    return xgb_importance


def main():
    """Main demonstration function."""
    logger.info("Natural Gas Price Model - Complete Demonstration")
    logger.info("=" * 60)
    
    try:
        # 1. Create sample data
        df = create_sample_data()
        
        # 2. Demonstrate feature engineering
        all_features = demonstrate_feature_engineering(df)
        
        # 3. Demonstrate model training
        models = demonstrate_model_training(all_features)
        
        # 4. Demonstrate model comparison
        comparison_results = demonstrate_model_comparison(models, all_features)
        
        # 5. Demonstrate backtesting
        backtest_metrics = demonstrate_backtesting(models, all_features)
        
        # 6. Demonstrate feature importance
        feature_importance = demonstrate_feature_importance(models)
        
        # Summary
        logger.info("Demonstration Summary:")
        logger.info("=" * 30)
        logger.info(f"✅ Sample data created: {len(df)} samples")
        logger.info(f"✅ Features engineered: {len(all_features.columns)} columns")
        logger.info(f"✅ Models trained: {len(models)} models")
        logger.info(f"✅ Backtesting completed: {len(backtest_metrics)} metrics")
        logger.info(f"✅ Feature importance analyzed: {len(feature_importance)} features")
        
        logger.info("\n🎉 Natural Gas Price Model demonstration completed successfully!")
        logger.info("\nNext steps:")
        logger.info("1. Replace sample data with real EIA, weather, and power data")
        logger.info("2. Implement additional data sources (CME, FERC, etc.)")
        logger.info("3. Add deep learning models (LSTM, Transformer)")
        logger.info("4. Implement real-time prediction pipeline")
        logger.info("5. Add trading strategy backtesting")
        
    except Exception as e:
        logger.error(f"Demonstration failed: {e}")
        raise


if __name__ == "__main__":
    main()
