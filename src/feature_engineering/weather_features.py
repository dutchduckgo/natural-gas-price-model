"""
Weather feature engineering for natural gas price model.
"""
import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple
import logging
from config import WEATHER_LAGS

logger = logging.getLogger(__name__)


class WeatherFeatureEngineer:
    """Feature engineer for weather data."""
    
    def __init__(self):
        self.weather_lags = WEATHER_LAGS
        
    def calculate_degree_days(self, temp_data: pd.DataFrame, base_temp: float = 65.0) -> pd.DataFrame:
        """
        Calculate heating and cooling degree days.
        
        Args:
            temp_data: DataFrame with 'date' and 'temperature' columns
            base_temp: Base temperature for degree day calculation
            
        Returns:
            DataFrame with HDD and CDD columns
        """
        df = temp_data.copy()
        df["HDD"] = np.maximum(base_temp - df["temperature"], 0)
        df["CDD"] = np.maximum(df["temperature"] - base_temp, 0)
        
        return df[["date", "HDD", "CDD"]]
    
    def calculate_regional_degree_days(self, weather_data: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate population-weighted degree days by region.
        
        Args:
            weather_data: DataFrame with regional weather data
            
        Returns:
            DataFrame with regional degree days
        """
        # Population weights for major gas-consuming regions
        region_weights = {
            "Northeast": 0.25,
            "Midwest": 0.20,
            "South": 0.30,
            "West": 0.25
        }
        
        df = weather_data.copy()
        df["date"] = pd.to_datetime(df["date"])
        
        # Calculate degree days for each region
        df["HDD"] = np.maximum(65.0 - df["temperature"], 0)
        df["CDD"] = np.maximum(df["temperature"] - 65.0, 0)
        
        # Calculate weighted averages
        weighted_hdd = []
        weighted_cdd = []
        
        for date in df["date"].unique():
            date_data = df[df["date"] == date]
            
            hdd_weighted = 0
            cdd_weighted = 0
            
            for _, row in date_data.iterrows():
                weight = region_weights.get(row["region"], 0.1)
                hdd_weighted += row["HDD"] * weight
                cdd_weighted += row["CDD"] * weight
            
            weighted_hdd.append(hdd_weighted)
            weighted_cdd.append(cdd_weighted)
        
        result = pd.DataFrame({
            "date": df["date"].unique(),
            "HDD_weighted": weighted_hdd,
            "CDD_weighted": weighted_cdd
        })
        
        return result
    
    def create_weather_lags(self, df: pd.DataFrame, target_col: str) -> pd.DataFrame:
        """
        Create lagged weather features.
        
        Args:
            df: DataFrame with weather data
            target_col: Column to create lags for
            
        Returns:
            DataFrame with lagged features
        """
        result_df = df.copy()
        
        for lag in self.weather_lags:
            result_df[f"{target_col}_lag_{lag}"] = result_df[target_col].shift(lag)
        
        return result_df
    
    def create_weather_rolling_features(self, df: pd.DataFrame, target_col: str, 
                                      windows: List[int] = [7, 14, 30]) -> pd.DataFrame:
        """
        Create rolling weather features.
        
        Args:
            df: DataFrame with weather data
            target_col: Column to create rolling features for
            windows: List of rolling window sizes
            
        Returns:
            DataFrame with rolling features
        """
        result_df = df.copy()
        
        for window in windows:
            result_df[f"{target_col}_rolling_{window}"] = result_df[target_col].rolling(window).mean()
            result_df[f"{target_col}_rolling_{window}_std"] = result_df[target_col].rolling(window).std()
            result_df[f"{target_col}_rolling_{window}_max"] = result_df[target_col].rolling(window).max()
            result_df[f"{target_col}_rolling_{window}_min"] = result_df[target_col].rolling(window).min()
        
        return result_df
    
    def create_weather_forecast_features(self, historical_df: pd.DataFrame, 
                                       forecast_df: pd.DataFrame) -> pd.DataFrame:
        """
        Create weather forecast features.
        
        Args:
            historical_df: Historical weather data
            forecast_df: Weather forecast data
            
        Returns:
            DataFrame with forecast features
        """
        # Calculate forecast surprises
        result_df = historical_df.copy()
        
        if not forecast_df.empty:
            # Merge forecast data
            forecast_merged = pd.merge(
                result_df, 
                forecast_df, 
                on="date", 
                how="left", 
                suffixes=("", "_forecast")
            )
            
            # Calculate forecast errors
            if "temperature" in forecast_merged.columns and "temperature_forecast" in forecast_merged.columns:
                forecast_merged["temp_forecast_error"] = (
                    forecast_merged["temperature_forecast"] - forecast_merged["temperature"]
                )
            
            # Calculate forecast HDD/CDD
            if "temperature_forecast" in forecast_merged.columns:
                forecast_merged["HDD_forecast"] = np.maximum(65.0 - forecast_merged["temperature_forecast"], 0)
                forecast_merged["CDD_forecast"] = np.maximum(forecast_merged["temperature_forecast"] - 65.0, 0)
            
            result_df = forecast_merged
        
        return result_df
    
    def create_seasonal_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Create seasonal weather features.
        
        Args:
            df: DataFrame with date column
            
        Returns:
            DataFrame with seasonal features
        """
        result_df = df.copy()
        result_df["date"] = pd.to_datetime(result_df["date"])
        
        # Extract seasonal components
        result_df["month"] = result_df["date"].dt.month
        result_df["quarter"] = result_df["date"].dt.quarter
        result_df["day_of_year"] = result_df["date"].dt.dayofyear
        
        # Create cyclical features
        result_df["month_sin"] = np.sin(2 * np.pi * result_df["month"] / 12)
        result_df["month_cos"] = np.cos(2 * np.pi * result_df["month"] / 12)
        result_df["day_sin"] = np.sin(2 * np.pi * result_df["day_of_year"] / 365)
        result_df["day_cos"] = np.cos(2 * np.pi * result_df["day_of_year"] / 365)
        
        # Create season indicators
        result_df["is_winter"] = result_df["month"].isin([12, 1, 2]).astype(int)
        result_df["is_spring"] = result_df["month"].isin([3, 4, 5]).astype(int)
        result_df["is_summer"] = result_df["month"].isin([6, 7, 8]).astype(int)
        result_df["is_fall"] = result_df["month"].isin([9, 10, 11]).astype(int)
        
        return result_df
    
    def create_weather_interaction_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Create weather interaction features.
        
        Args:
            df: DataFrame with weather data
            
        Returns:
            DataFrame with interaction features
        """
        result_df = df.copy()
        
        # HDD * CDD interaction
        if "HDD" in result_df.columns and "CDD" in result_df.columns:
            result_df["HDD_CDD_interaction"] = result_df["HDD"] * result_df["CDD"]
        
        # Temperature * Wind interaction
        if "temperature" in result_df.columns and "wind_speed" in result_df.columns:
            result_df["temp_wind_interaction"] = result_df["temperature"] * result_df["wind_speed"]
        
        # HDD * Wind interaction (wind chill effect)
        if "HDD" in result_df.columns and "wind_speed" in result_df.columns:
            result_df["HDD_wind_interaction"] = result_df["HDD"] * result_df["wind_speed"]
        
        return result_df
    
    def engineer_all_weather_features(self, weather_data: pd.DataFrame, 
                                   forecast_data: pd.DataFrame = None) -> pd.DataFrame:
        """
        Engineer all weather features.
        
        Args:
            weather_data: Historical weather data
            forecast_data: Weather forecast data
            
        Returns:
            DataFrame with all weather features
        """
        logger.info("Starting weather feature engineering")
        
        # Start with base data
        df = weather_data.copy()
        
        # Calculate degree days
        if "temperature" in df.columns:
            df = self.calculate_degree_days(df)
        
        # Create regional features if multiple regions
        if "region" in df.columns:
            df = self.calculate_regional_degree_days(df)
        
        # Create lags
        for col in ["HDD", "CDD", "temperature"]:
            if col in df.columns:
                df = self.create_weather_lags(df, col)
        
        # Create rolling features
        for col in ["HDD", "CDD", "temperature"]:
            if col in df.columns:
                df = self.create_weather_rolling_features(df, col)
        
        # Create forecast features
        if forecast_data is not None:
            df = self.create_weather_forecast_features(df, forecast_data)
        
        # Create seasonal features
        df = self.create_seasonal_features(df)
        
        # Create interaction features
        df = self.create_weather_interaction_features(df)
        
        logger.info(f"Weather feature engineering complete: {len(df.columns)} features")
        
        return df


def main():
    """Test weather feature engineering."""
    # Create sample data
    dates = pd.date_range('2024-01-01', periods=100)
    temps = 30 + 30 * np.sin(2 * np.pi * np.arange(100) / 365) + np.random.normal(0, 5, 100)
    
    weather_data = pd.DataFrame({
        'date': dates,
        'temperature': temps,
        'region': 'Northeast'
    })
    
    # Test feature engineering
    engineer = WeatherFeatureEngineer()
    features = engineer.engineer_all_weather_features(weather_data)
    
    print(f"Created {len(features.columns)} weather features")
    print(f"Feature names: {list(features.columns)}")


if __name__ == "__main__":
    main()
