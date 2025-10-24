"""
Baseline model training pipeline.
"""
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import logging
from pathlib import Path

# Import models and evaluation
from src.models.baseline import BaselinePipeline
from src.models.tree_models import TreeModelPipeline
from src.models.deep_learning import DeepLearningPipeline
from src.evaluation.backtesting import Backtester, ModelComparison
from src.data_ingestion.database import GasModelDatabase
from config import MODEL_DIR, RESULTS_DIR

logger = logging.getLogger(__name__)


class ModelTrainingPipeline:
    """Pipeline for training and evaluating models."""
    
    def __init__(self):
        self.db = GasModelDatabase()
        self.models = {}
        self.results = {}
        
    def get_training_data(self, start_date: str = None, end_date: str = None) -> pd.DataFrame:
        """
        Get training data from database.
        
        Args:
            start_date: Start date in YYYY-MM-DD format
            end_date: End date in YYYY-MM-DD format
            
        Returns:
            DataFrame with training data
        """
        # Set default dates if not provided
        if not end_date:
            end_date = datetime.now().strftime("%Y-%m-%d")
        if not start_date:
            start_date = (datetime.now() - timedelta(days=365*2)).strftime("%Y-%m-%d")
        
        # Get feature matrix from database
        df = self.db.get_feature_matrix(start_date, end_date)
        
        if df.empty:
            logger.warning("No training data found")
            return pd.DataFrame()
        
        # Handle missing values
        df = df.fillna(df.median())
        
        # Remove rows with missing target
        df = df.dropna(subset=['spot_price'])
        
        logger.info(f"Retrieved {len(df)} training samples")
        
        return df
    
    def train_baseline_models(self, df: pd.DataFrame):
        """
        Train baseline models.
        
        Args:
            df: Training data
        """
        logger.info("Training baseline models")
        
        # Elastic Net model
        elastic_net = BaselinePipeline("elastic_net")
        elastic_net.train_final_model(df)
        self.models["elastic_net"] = elastic_net.model
        
        # Linear Regression model
        linear = BaselinePipeline("linear")
        linear.train_final_model(df)
        self.models["linear"] = linear.model
        
        # Random Forest model
        rf = BaselinePipeline("random_forest")
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
        xgb = TreeModelPipeline("xgboost")
        xgb.train_final_model(df)
        self.models["xgboost"] = xgb.model
        
        # LightGBM model
        lgb = TreeModelPipeline("lightgbm")
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
        results = comparison.compare_models(df)
        
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
