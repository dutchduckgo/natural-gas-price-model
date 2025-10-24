"""
Storage feature engineering for natural gas price model.
"""
import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple
import logging
from config import STORAGE_LAGS

logger = logging.getLogger(__name__)


class StorageFeatureEngineer:
    """Feature engineer for storage data."""
    
    def __init__(self):
        self.storage_lags = STORAGE_LAGS
        
    def calculate_storage_metrics(self, storage_data: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate storage-related metrics.
        
        Args:
            storage_data: DataFrame with storage data
            
        Returns:
            DataFrame with storage metrics
        """
        df = storage_data.copy()
        df["date"] = pd.to_datetime(df["date"])
        
        # Calculate year-over-year deviation
        if "working_gas" in df.columns and "five_year_avg" in df.columns:
            df["storage_yoy_deviation"] = df["working_gas"] - df["five_year_avg"]
            df["storage_yoy_deviation_pct"] = (df["storage_yoy_deviation"] / df["five_year_avg"]) * 100
        
        # Calculate storage cover (days of supply)
        if "working_gas" in df.columns:
            # Estimate daily consumption (simplified)
            df["estimated_daily_consumption"] = 100  # BCF per day (rough estimate)
            df["storage_cover_days"] = df["working_gas"] / df["estimated_daily_consumption"]
        
        # Calculate storage injection/withdrawal rates
        if "working_gas" in df.columns:
            df["storage_change"] = df["working_gas"].diff()
            df["storage_change_pct"] = df["working_gas"].pct_change() * 100
        
        return df
    
    def create_storage_lags(self, df: pd.DataFrame, target_col: str) -> pd.DataFrame:
        """
        Create lagged storage features.
        
        Args:
            df: DataFrame with storage data
            target_col: Column to create lags for
            
        Returns:
            DataFrame with lagged features
        """
        result_df = df.copy()
        
        for lag in self.storage_lags:
            result_df[f"{target_col}_lag_{lag}"] = result_df[target_col].shift(lag)
        
        return result_df
    
    def create_storage_rolling_features(self, df: pd.DataFrame, target_col: str, 
                                      windows: List[int] = [4, 8, 12]) -> pd.DataFrame:
        """
        Create rolling storage features.
        
        Args:
            df: DataFrame with storage data
            target_col: Column to create rolling features for
            windows: List of rolling window sizes (in weeks)
            
        Returns:
            DataFrame with rolling features
        """
        result_df = df.copy()
        
        for window in windows:
            result_df[f"{target_col}_rolling_{window}w"] = result_df[target_col].rolling(window).mean()
            result_df[f"{target_col}_rolling_{window}w_std"] = result_df[target_col].rolling(window).std()
            result_df[f"{target_col}_rolling_{window}w_min"] = result_df[target_col].rolling(window).min()
            result_df[f"{target_col}_rolling_{window}w_max"] = result_df[target_col].rolling(window).max()
        
        return result_df
    
    def create_storage_seasonal_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Create seasonal storage features.
        
        Args:
            df: DataFrame with storage data
            
        Returns:
            DataFrame with seasonal features
        """
        result_df = df.copy()
        result_df["date"] = pd.to_datetime(result_df["date"])
        
        # Extract seasonal components
        result_df["month"] = result_df["date"].dt.month
        result_df["quarter"] = result_df["date"].dt.quarter
        result_df["week_of_year"] = result_df["date"].dt.isocalendar().week
        
        # Create seasonal indicators
        result_df["is_injection_season"] = result_df["month"].isin([4, 5, 6, 7, 8, 9, 10]).astype(int)
        result_df["is_withdrawal_season"] = result_df["month"].isin([11, 12, 1, 2, 3]).astype(int)
        
        # Create cyclical features
        result_df["month_sin"] = np.sin(2 * np.pi * result_df["month"] / 12)
        result_df["month_cos"] = np.cos(2 * np.pi * result_df["month"] / 12)
        
        return result_df
    
    def create_storage_tightness_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Create storage tightness features.
        
        Args:
            df: DataFrame with storage data
            
        Returns:
            DataFrame with tightness features
        """
        result_df = df.copy()
        
        # Calculate storage tightness indicators
        if "working_gas" in result_df.columns and "five_year_avg" in result_df.columns:
            result_df["storage_tightness"] = (result_df["working_gas"] - result_df["five_year_avg"]) / result_df["five_year_avg"]
            
            # Create tightness categories
            result_df["storage_very_tight"] = (result_df["storage_tightness"] < -0.1).astype(int)
            result_df["storage_tight"] = ((result_df["storage_tightness"] >= -0.1) & (result_df["storage_tightness"] < -0.05)).astype(int)
            result_df["storage_normal"] = ((result_df["storage_tightness"] >= -0.05) & (result_df["storage_tightness"] < 0.05)).astype(int)
            result_df["storage_loose"] = (result_df["storage_tightness"] >= 0.05).astype(int)
        
        # Calculate storage velocity (rate of change)
        if "working_gas" in result_df.columns:
            result_df["storage_velocity"] = result_df["working_gas"].diff()
            result_df["storage_velocity_pct"] = result_df["working_gas"].pct_change() * 100
        
        return result_df
    
    def create_storage_forecast_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Create storage forecast features.
        
        Args:
            df: DataFrame with storage data
            
        Returns:
            DataFrame with forecast features
        """
        result_df = df.copy()
        
        # Calculate projected end-of-season storage
        if "working_gas" in result_df.columns:
            # Simple linear projection based on recent trend
            result_df["storage_trend"] = result_df["working_gas"].rolling(4).apply(
                lambda x: np.polyfit(range(len(x)), x, 1)[0] if len(x) == 4 else np.nan
            )
            
            # Project storage levels
            result_df["storage_projected_4w"] = result_df["working_gas"] + (result_df["storage_trend"] * 4)
            result_df["storage_projected_8w"] = result_df["working_gas"] + (result_df["storage_trend"] * 8)
        
        return result_df
    
    def create_storage_interaction_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Create storage interaction features.
        
        Args:
            df: DataFrame with storage data
            
        Returns:
            DataFrame with interaction features
        """
        result_df = df.copy()
        
        # Storage * Season interaction
        if "working_gas" in result_df.columns and "is_injection_season" in result_df.columns:
            result_df["storage_injection_interaction"] = result_df["working_gas"] * result_df["is_injection_season"]
        
        # Storage * Temperature interaction (if temperature available)
        if "working_gas" in result_df.columns and "temperature" in result_df.columns:
            result_df["storage_temp_interaction"] = result_df["working_gas"] * result_df["temperature"]
        
        return result_df
    
    def engineer_all_storage_features(self, storage_data: pd.DataFrame) -> pd.DataFrame:
        """
        Engineer all storage features.
        
        Args:
            storage_data: Storage data
            
        Returns:
            DataFrame with all storage features
        """
        logger.info("Starting storage feature engineering")
        
        # Start with base data
        df = storage_data.copy()
        
        # Calculate storage metrics
        df = self.calculate_storage_metrics(df)
        
        # Create lags
        for col in ["working_gas", "storage_yoy_deviation", "storage_cover_days"]:
            if col in df.columns:
                df = self.create_storage_lags(df, col)
        
        # Create rolling features
        for col in ["working_gas", "storage_yoy_deviation"]:
            if col in df.columns:
                df = self.create_storage_rolling_features(df, col)
        
        # Create seasonal features
        df = self.create_storage_seasonal_features(df)
        
        # Create tightness features
        df = self.create_storage_tightness_features(df)
        
        # Create forecast features
        df = self.create_storage_forecast_features(df)
        
        # Create interaction features
        df = self.create_storage_interaction_features(df)
        
        logger.info(f"Storage feature engineering complete: {len(df.columns)} features")
        
        return df


def main():
    """Test storage feature engineering."""
    # Create sample data
    dates = pd.date_range('2024-01-01', periods=100, freq='W')
    working_gas = 3000 + 100 * np.sin(2 * np.pi * np.arange(100) / 52) + np.random.normal(0, 50, 100)
    five_year_avg = 3000 + 50 * np.sin(2 * np.pi * np.arange(100) / 52)
    
    storage_data = pd.DataFrame({
        'date': dates,
        'working_gas': working_gas,
        'five_year_avg': five_year_avg
    })
    
    # Test feature engineering
    engineer = StorageFeatureEngineer()
    features = engineer.engineer_all_storage_features(storage_data)
    
    print(f"Created {len(features.columns)} storage features")
    print(f"Feature names: {list(features.columns)}")


if __name__ == "__main__":
    main()
