"""
Backtesting framework for natural gas price models.
"""
import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple, Union
import logging
from config import WALK_FORWARD_WINDOW, FORECAST_HORIZONS

logger = logging.getLogger(__name__)


class Backtester:
    """Backtesting framework for natural gas price models."""
    
    def __init__(self, window_size: int = None, step_size: int = 21):
        self.window_size = window_size or WALK_FORWARD_WINDOW
        self.step_size = step_size
        self.results = []
        
    def walk_forward_validation(self, df: pd.DataFrame, model, 
                               target_col: str = "spot_price") -> List[Dict]:
        """
        Perform walk-forward validation.
        
        Args:
            df: DataFrame with features and target
            model: Model to test
            target_col: Name of target column
            
        Returns:
            List of validation results
        """
        results = []
        
        for i in range(self.window_size, len(df) - self.step_size, self.step_size):
            # Split data
            train_df = df.iloc[i-self.window_size:i].copy()
            test_df = df.iloc[i:i+self.step_size].copy()
            
            # Prepare features
            if hasattr(model, 'prepare_features'):
                X_train, y_train = model.prepare_features(train_df, target_col)
                X_test, y_test = model.prepare_features(test_df, target_col)
            else:
                # Assume model expects numpy arrays
                feature_cols = [col for col in df.columns if col not in [target_col, 'date']]
                X_train = train_df[feature_cols].values
                y_train = train_df[target_col].values
                X_test = test_df[feature_cols].values
                y_test = test_df[target_col].values
            
            # Fit model
            model.fit(X_train, y_train)
            
            # Make predictions
            y_pred = model.predict(X_test)
            
            # Calculate metrics
            mae = np.mean(np.abs(y_test - y_pred))
            rmse = np.sqrt(np.mean((y_test - y_pred) ** 2))
            mape = np.mean(np.abs((y_test - y_pred) / y_test)) * 100
            
            # Calculate directional accuracy
            direction_actual = np.sign(np.diff(y_test))
            direction_pred = np.sign(np.diff(y_pred))
            direction_accuracy = np.mean(direction_actual == direction_pred) * 100
            
            results.append({
                "train_start": train_df.index[0],
                "train_end": train_df.index[-1],
                "test_start": test_df.index[0],
                "test_end": test_df.index[-1],
                "mae": mae,
                "rmse": rmse,
                "mape": mape,
                "direction_accuracy": direction_accuracy,
                "predictions": y_pred,
                "actual": y_test
            })
        
        self.results = results
        return results
    
    def calculate_performance_metrics(self) -> Dict:
        """
        Calculate overall performance metrics.
        
        Returns:
            Dictionary with performance metrics
        """
        if not self.results:
            return {}
        
        # Aggregate metrics
        mae_scores = [result["mae"] for result in self.results]
        rmse_scores = [result["rmse"] for result in self.results]
        mape_scores = [result["mape"] for result in self.results]
        direction_scores = [result["direction_accuracy"] for result in self.results]
        
        metrics = {
            "mae_mean": np.mean(mae_scores),
            "mae_std": np.std(mae_scores),
            "rmse_mean": np.mean(rmse_scores),
            "rmse_std": np.std(rmse_scores),
            "mape_mean": np.mean(mape_scores),
            "mape_std": np.std(mape_scores),
            "direction_accuracy_mean": np.mean(direction_scores),
            "direction_accuracy_std": np.std(direction_scores)
        }
        
        return metrics
    
    def plot_performance(self, save_path: str = None):
        """
        Plot backtesting performance.
        
        Args:
            save_path: Path to save plot
        """
        import matplotlib.pyplot as plt
        
        if not self.results:
            logger.warning("No results to plot")
            return
        
        # Create subplots
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        # MAE over time
        mae_scores = [result["mae"] for result in self.results]
        axes[0, 0].plot(mae_scores)
        axes[0, 0].set_title("MAE Over Time")
        axes[0, 0].set_xlabel("Validation Period")
        axes[0, 0].set_ylabel("MAE")
        
        # RMSE over time
        rmse_scores = [result["rmse"] for result in self.results]
        axes[0, 1].plot(rmse_scores)
        axes[0, 1].set_title("RMSE Over Time")
        axes[0, 1].set_xlabel("Validation Period")
        axes[0, 1].set_ylabel("RMSE")
        
        # Direction accuracy over time
        direction_scores = [result["direction_accuracy"] for result in self.results]
        axes[1, 0].plot(direction_scores)
        axes[1, 0].set_title("Direction Accuracy Over Time")
        axes[1, 0].set_xlabel("Validation Period")
        axes[1, 0].set_ylabel("Direction Accuracy (%)")
        axes[1, 0].axhline(y=50, color='r', linestyle='--', alpha=0.5)
        
        # Prediction vs actual (last period)
        last_result = self.results[-1]
        axes[1, 1].scatter(last_result["actual"], last_result["predictions"])
        axes[1, 1].plot([min(last_result["actual"]), max(last_result["actual"])], 
                        [min(last_result["actual"]), max(last_result["actual"])], 'r--')
        axes[1, 1].set_title("Predictions vs Actual")
        axes[1, 1].set_xlabel("Actual")
        axes[1, 1].set_ylabel("Predicted")
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path)
        else:
            plt.show()
    
    def get_feature_importance(self, model) -> pd.DataFrame:
        """
        Get feature importance from model.
        
        Args:
            model: Trained model
            
        Returns:
            DataFrame with feature importance
        """
        if hasattr(model, 'get_feature_importance'):
            return model.get_feature_importance()
        else:
            logger.warning("Model does not support feature importance")
            return pd.DataFrame()


class ModelComparison:
    """Compare multiple models."""
    
    def __init__(self):
        self.models = {}
        self.results = {}
        
    def add_model(self, name: str, model):
        """Add model to comparison."""
        self.models[name] = model
        
    def compare_models(self, df: pd.DataFrame, target_col: str = "spot_price") -> Dict:
        """
        Compare all models.
        
        Args:
            df: DataFrame with features and target
            target_col: Name of target column
            
        Returns:
            Dictionary with comparison results
        """
        comparison_results = {}
        
        for name, model in self.models.items():
            logger.info(f"Evaluating model: {name}")
            
            # Create backtester
            backtester = Backtester()
            
            # Run backtesting
            results = backtester.walk_forward_validation(df, model, target_col)
            
            # Calculate metrics
            metrics = backtester.calculate_performance_metrics()
            
            comparison_results[name] = {
                "metrics": metrics,
                "results": results
            }
            
            self.results[name] = results
        
        return comparison_results
    
    def plot_comparison(self, save_path: str = None):
        """
        Plot model comparison.
        
        Args:
            save_path: Path to save plot
        """
        import matplotlib.pyplot as plt
        
        if not self.results:
            logger.warning("No results to plot")
            return
        
        # Create subplots
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        # MAE comparison
        mae_scores = {}
        for name, results in self.results.items():
            mae_scores[name] = [result["mae"] for result in results]
        
        for name, scores in mae_scores.items():
            axes[0, 0].plot(scores, label=name)
        axes[0, 0].set_title("MAE Comparison")
        axes[0, 0].set_xlabel("Validation Period")
        axes[0, 0].set_ylabel("MAE")
        axes[0, 0].legend()
        
        # RMSE comparison
        rmse_scores = {}
        for name, results in self.results.items():
            rmse_scores[name] = [result["rmse"] for result in results]
        
        for name, scores in rmse_scores.items():
            axes[0, 1].plot(scores, label=name)
        axes[0, 1].set_title("RMSE Comparison")
        axes[0, 1].set_xlabel("Validation Period")
        axes[0, 1].set_ylabel("RMSE")
        axes[0, 1].legend()
        
        # Direction accuracy comparison
        direction_scores = {}
        for name, results in self.results.items():
            direction_scores[name] = [result["direction_accuracy"] for result in results]
        
        for name, scores in direction_scores.items():
            axes[1, 0].plot(scores, label=name)
        axes[1, 0].set_title("Direction Accuracy Comparison")
        axes[1, 0].set_xlabel("Validation Period")
        axes[1, 0].set_ylabel("Direction Accuracy (%)")
        axes[1, 0].axhline(y=50, color='r', linestyle='--', alpha=0.5)
        axes[1, 0].legend()
        
        # Box plot of MAE
        mae_data = [scores for scores in mae_scores.values()]
        axes[1, 1].boxplot(mae_data, labels=list(mae_scores.keys()))
        axes[1, 1].set_title("MAE Distribution")
        axes[1, 1].set_ylabel("MAE")
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path)
        else:
            plt.show()


def main():
    """Test backtesting framework."""
    # Create sample data
    np.random.seed(42)
    n_samples = 1000
    
    dates = pd.date_range('2020-01-01', periods=n_samples)
    
    # Create synthetic features
    hdd = 20 + 30 * np.sin(2 * np.pi * np.arange(n_samples) / 365) + np.random.normal(0, 5, n_samples)
    cdd = 10 + 20 * np.sin(2 * np.pi * np.arange(n_samples) / 365) + np.random.normal(0, 3, n_samples)
    storage = 3000 + 500 * np.sin(2 * np.pi * np.arange(n_samples) / 365) + np.random.normal(0, 100, n_samples)
    
    # Create target with some relationship to features
    spot_price = 3.0 + 0.01 * hdd + 0.005 * cdd - 0.0001 * storage + np.random.normal(0, 0.1, n_samples)
    
    df = pd.DataFrame({
        'date': dates,
        'spot_price': spot_price,
        'hdd': hdd,
        'cdd': cdd,
        'storage': storage
    })
    
    # Test backtesting
    print("Testing backtesting framework...")
    
    # Create a simple model for testing
    class SimpleModel:
        def fit(self, X, y):
            self.coef_ = np.linalg.lstsq(X, y, rcond=None)[0]
        
        def predict(self, X):
            return X @ self.coef_
    
    model = SimpleModel()
    backtester = Backtester(window_size=200, step_size=50)
    results = backtester.walk_forward_validation(df, model)
    
    print(f"Backtesting completed: {len(results)} validation periods")
    
    # Calculate performance metrics
    metrics = backtester.calculate_performance_metrics()
    print(f"Performance metrics: {metrics}")


if __name__ == "__main__":
    main()
