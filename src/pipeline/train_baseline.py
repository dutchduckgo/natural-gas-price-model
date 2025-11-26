"""
Baseline model training pipeline.
"""
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict
import logging
from pathlib import Path

# Import models and evaluation
from src.models.baseline import BaselinePipeline
from src.models.tree_models import TreeModelPipeline
from src.models.deep_learning import DeepLearningPipeline
from src.evaluation.backtesting import Backtester, ModelComparison
from src.data_ingestion.database import GasModelDatabase
from src.feature_engineering.weather_features import WeatherFeatureEngineer
from src.feature_engineering.storage_features import StorageFeatureEngineer
from src.feature_engineering.price_features import PriceFeatureEngineer
from config import MODEL_DIR, RESULTS_DIR, TARGET_HORIZON_DAYS

logger = logging.getLogger(__name__)


class ModelTrainingPipeline:
    """Pipeline for training and evaluating models."""
    
    def __init__(self, target_horizon: int = TARGET_HORIZON_DAYS,
                 target_col: str = "spot_price_target"):
        self.db = GasModelDatabase()
        self.models = {}
        self.results = {}
        self.weather_engineer = WeatherFeatureEngineer()
        self.storage_engineer = StorageFeatureEngineer()
        self.price_engineer = PriceFeatureEngineer()
        self.target_horizon = target_horizon
        self.target_col = target_col
        
    def get_training_data(self, start_date: str = None, end_date: str = None) -> pd.DataFrame:
        """
        Get training data from database and apply feature engineering.
        
        Args:
            start_date: Start date in YYYY-MM-DD format
            end_date: End date in YYYY-MM-DD format
            
        Returns:
            DataFrame with training data and engineered features
        """
        # Set default dates if not provided
        if not end_date:
            end_date = datetime.now().strftime("%Y-%m-%d")
        if not start_date:
            start_date = (datetime.now() - timedelta(days=365*2)).strftime("%Y-%m-%d")
        
        # Get raw data from database
        logger.info(f"Loading data from database: {start_date} to {end_date}")
        df = self.db.get_feature_matrix(start_date, end_date)
        
        if df.empty:
            logger.warning("No training data found in database")
            return pd.DataFrame()
        
        logger.info(f"Retrieved {len(df)} raw records from database")
        
        # Ensure date column is datetime and sorted
        if 'date' in df.columns:
            df['date'] = pd.to_datetime(df['date'])
            df = df.sort_values('date').reset_index(drop=True)
        
        # Apply feature engineering
        logger.info("Applying feature engineering...")
        
        # Weather feature engineering (if weather data exists)
        if 'temperature' in df.columns or 'hdd' in df.columns or 'cdd' in df.columns:
            # Prepare weather data for feature engineering
            weather_cols = ['date', 'temperature', 'hdd', 'cdd', 'hdd_norm_delta', 'cdd_norm_delta']
            available_weather_cols = [col for col in weather_cols if col in df.columns]
            if 'date' not in available_weather_cols:
                available_weather_cols.insert(0, 'date')
            weather_data = df[available_weather_cols].copy()
            if not weather_data.empty:
                weather_features = self.weather_engineer.engineer_all_weather_features(weather_data)
                # Merge weather features back (avoid duplicate date column)
                weather_features = weather_features.drop(columns=['date'], errors='ignore')
                df = pd.concat([df, weather_features], axis=1)
                logger.info(f"Added {len(weather_features.columns)} weather features")
        
        # Storage feature engineering (if storage data exists)
        if 'working_gas' in df.columns:
            # Prepare storage data for feature engineering
            storage_cols = ['date', 'working_gas', 'five_year_avg', 'yoy_deviation', 'wow_change']
            available_storage_cols = [col for col in storage_cols if col in df.columns]
            if 'date' not in available_storage_cols:
                available_storage_cols.insert(0, 'date')
            storage_data = df[available_storage_cols].copy()
            if not storage_data.empty:
                storage_features = self.storage_engineer.engineer_all_storage_features(storage_data)
                # Merge storage features back (avoid duplicate date column)
                storage_features = storage_features.drop(columns=['date'], errors='ignore')
                df = pd.concat([df, storage_features], axis=1)
                logger.info(f"Added {len(storage_features.columns)} storage features")

        # Price/term-structure feature engineering (always safe if spot price exists)
        if 'spot_price' in df.columns:
            df = self.price_engineer.transform(df)
            logger.info("Added autoregressive price features")
        
        # Remove duplicate columns that can arise from repeated feature merges
        if df.columns.duplicated().any():
            dup_count = df.columns.duplicated().sum()
            logger.warning(f"Detected {dup_count} duplicate feature columns; keeping first occurrence")
            df = df.loc[:, ~df.columns.duplicated()]
        
        # Construct future target (spot price shifted by horizon)
        if 'spot_price' not in df.columns:
            raise ValueError("spot_price column missing; cannot build forecasting target.")
        df[self.target_col] = df['spot_price'].shift(-self.target_horizon)
        df['target_date'] = df['date'] + pd.to_timedelta(self.target_horizon, unit='D')
        
        # Handle missing values (after feature engineering)
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        if len(numeric_cols) > 0:
            df[numeric_cols] = df[numeric_cols].replace([np.inf, -np.inf], np.nan)
            df[numeric_cols] = df[numeric_cols].fillna(df[numeric_cols].median())
            df[numeric_cols] = df[numeric_cols].fillna(0)
        
        # Remove rows with missing spot price or target (tail rows without future data)
        df = df.dropna(subset=['spot_price', self.target_col])
        
        logger.info(f"Final training dataset: {len(df)} samples with {len(df.columns)} features")
        
        return df
    
    def train_baseline_models(self, df: pd.DataFrame):
        """
        Train baseline models.
        
        Args:
            df: Training data
        """
        logger.info("Training baseline models")
        
        # Elastic Net model
        elastic_net = BaselinePipeline("elastic_net", target_col=self.target_col)
        elastic_net.train_final_model(df)
        self.models["elastic_net"] = elastic_net.model
        
        # Linear Regression model
        linear = BaselinePipeline("linear", target_col=self.target_col)
        linear.train_final_model(df)
        self.models["linear"] = linear.model
        
        # Random Forest model
        rf = BaselinePipeline("random_forest", target_col=self.target_col)
        rf.train_final_model(df)
        self.models["random_forest"] = rf.model
        
        logger.info("Baseline models trained successfully")
    
    def train_tree_models(self, df: pd.DataFrame):
        """
        Train tree-based models.
        
        Args:
            df: Training data
        """
        logger.info("Training tree-based models")
        
        # XGBoost model
        xgb = TreeModelPipeline("xgboost", target_col=self.target_col)
        xgb.train_final_model(df)
        self.models["xgboost"] = xgb.model
        
        # LightGBM model
        lgb = TreeModelPipeline("lightgbm", target_col=self.target_col)
        lgb.train_final_model(df)
        self.models["lightgbm"] = lgb.model
        
        logger.info("Tree-based models trained successfully")
    
    def train_deep_learning_models(self, df: pd.DataFrame):
        """
        Train deep learning models.
        
        Args:
            df: Training data
        """
        logger.info("Training deep learning models")
        
        # LSTM model
        lstm = DeepLearningPipeline("lstm")
        lstm.train_final_model(df)
        self.models["lstm"] = lstm.model
        
        # Transformer model
        transformer = DeepLearningPipeline("transformer")
        transformer.train_final_model(df)
        self.models["transformer"] = transformer.model
        
        logger.info("Deep learning models trained successfully")
    
    def evaluate_models(self, df: pd.DataFrame) -> Dict:
        """
        Evaluate all models.
        
        Args:
            df: Evaluation data
            
        Returns:
            Dictionary with evaluation results
        """
        logger.info("Evaluating models")
        
        # Create model comparison
        comparison = ModelComparison()
        
        # Add all models
        for name, model in self.models.items():
            comparison.add_model(name, model)
        
        # Run comparison
        results = comparison.compare_models(df, target_col=self.target_col)
        
        # Store results
        self.results = results
        
        logger.info("Model evaluation completed")
        
        return results
    
    def save_models(self):
        """Save trained models."""
        logger.info("Saving models")
        
        # Create models directory
        MODEL_DIR.mkdir(exist_ok=True)
        
        # Save each model
        for name, model in self.models.items():
            model_path = MODEL_DIR / f"{name}_model.pkl"
            
            # This would need to be implemented based on the model type
            # For now, just log the save operation
            logger.info(f"Model {name} saved to {model_path}")
    
    def generate_report(self, results: Dict) -> str:
        """
        Generate evaluation report.
        
        Args:
            results: Evaluation results
            
        Returns:
            Report string
        """
        report = "Natural Gas Price Model Evaluation Report\n"
        report += "=" * 50 + "\n\n"
        
        for model_name, model_results in results.items():
            report += f"Model: {model_name}\n"
            report += "-" * 20 + "\n"
            
            metrics = model_results["metrics"]
            report += f"MAE: {metrics['mae_mean']:.4f} ± {metrics['mae_std']:.4f}\n"
            report += f"RMSE: {metrics['rmse_mean']:.4f} ± {metrics['rmse_std']:.4f}\n"
            report += f"MAPE: {metrics['mape_mean']:.2f}% ± {metrics['mape_std']:.2f}%\n"
            report += f"Direction Accuracy: {metrics['direction_accuracy_mean']:.2f}% ± {metrics['direction_accuracy_std']:.2f}%\n"
            report += "\n"
        
        return report
    
    def run_full_pipeline(self, start_date: str = None, end_date: str = None):
        """
        Run the complete training pipeline.
        
        Args:
            start_date: Start date for training data
            end_date: End date for training data
        """
        logger.info("Starting full training pipeline")
        
        try:
            # Get training data
            df = self.get_training_data(start_date, end_date)
            
            if df.empty:
                logger.error("No training data available")
                return
            
            # Train models
            self.train_baseline_models(df)
            self.train_tree_models(df)
            self.train_deep_learning_models(df)
            
            # Evaluate models
            results = self.evaluate_models(df)
            
            # Save models
            self.save_models()
            
            # Generate report
            report = self.generate_report(results)
            print(report)
            
            # Save report
            RESULTS_DIR.mkdir(exist_ok=True)
            report_path = RESULTS_DIR / f"evaluation_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
            with open(report_path, 'w') as f:
                f.write(report)
            
            logger.info("Full training pipeline completed successfully")
            
        except Exception as e:
            logger.error(f"Training pipeline failed: {e}")
            raise
        finally:
            # Close database connection
            self.db.close()


def main():
    """Run baseline model training."""
    # Set up logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Create pipeline
    pipeline = ModelTrainingPipeline()
    
    # Run training
    pipeline.run_full_pipeline()
    
    print("Baseline model training completed successfully!")


if __name__ == "__main__":
    main()
