"""
Weather data collection from NWS, CPC, and NOMADS.
"""
import requests
import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple
from datetime import datetime, timedelta
import logging
from config import NWS_BASE_URL, CPC_BASE_URL

logger = logging.getLogger(__name__)


class WeatherClient:
    """Client for weather data from multiple sources."""
    
    def __init__(self):
        self.nws_base_url = NWS_BASE_URL
        self.cpc_base_url = CPC_BASE_URL
        self.session = requests.Session()
        
    def _make_request(self, url: str, params: Dict = None) -> Dict:
        """Make API request with error handling."""
        try:
            response = self.session.get(url, params=params)
            response.raise_for_status()
            return response.json()
        except requests.exceptions.RequestException as e:
            logger.error(f"Weather API request failed: {e}")
            raise
    
    def get_degree_days_cpc(self, start_date: str = None, end_date: str = None) -> pd.DataFrame:
        """
        Get degree day data from Climate Prediction Center.
        
        Args:
            start_date: Start date in YYYY-MM-DD format
            end_date: End date in YYYY-MM-DD format
            
        Returns:
            DataFrame with date, HDD, CDD columns
        """
        # CPC degree day data is typically available as CSV files
        # This is a simplified implementation
        logger.info("CPC degree day data collection not yet implemented")
        return pd.DataFrame()
    
    def get_weather_forecast_nws(self, lat: float, lon: float, days: int = 14) -> pd.DataFrame:
        """
        Get weather forecast from NWS API.
        
        Args:
            lat: Latitude
            lon: Longitude
            days: Number of forecast days
            
        Returns:
            DataFrame with forecast data
        """
        # Get grid point for coordinates
        grid_url = f"{self.nws_base_url}/points/{lat},{lon}"
        
        try:
            grid_response = self._make_request(grid_url)
            forecast_url = grid_response["properties"]["forecast"]
            
            # Get forecast data
            forecast_response = self._make_request(forecast_url)
            
            # Parse forecast data
            periods = forecast_response["properties"]["periods"]
            data = []
            
            for period in periods[:days]:
                data.append({
                    "date": pd.to_datetime(period["startTime"]).date(),
                    "temperature": period["temperature"],
                    "wind_speed": period.get("windSpeed", ""),
                    "precipitation": period.get("probabilityOfPrecipitation", {}).get("value", 0)
                })
            
            df = pd.DataFrame(data)
            return df
            
        except Exception as e:
            logger.error(f"Error getting NWS forecast: {e}")
            return pd.DataFrame()
    
    def calculate_degree_days(self, temp_data: pd.DataFrame, base_temp: float = 65.0) -> pd.DataFrame:
        """
        Calculate heating and cooling degree days from temperature data.
        
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
    
    def get_weather_stations(self) -> List[Dict]:
        """
        Get list of weather stations for major gas-consuming regions.
        
        Returns:
            List of station dictionaries with lat/lon and region info
        """
        stations = [
            {"name": "New York", "lat": 40.7128, "lon": -74.0060, "region": "Northeast"},
            {"name": "Chicago", "lat": 41.8781, "lon": -87.6298, "region": "Midwest"},
            {"name": "Houston", "lat": 29.7604, "lon": -95.3698, "region": "South"},
            {"name": "Los Angeles", "lat": 34.0522, "lon": -118.2437, "region": "West"},
            {"name": "Boston", "lat": 42.3601, "lon": -71.0589, "region": "Northeast"},
            {"name": "Atlanta", "lat": 33.7490, "lon": -84.3880, "region": "South"},
        ]
        
        return stations
    
    def get_regional_weather_forecast(self, days: int = 14) -> pd.DataFrame:
        """
        Get weather forecast for multiple regions.
        
        Args:
            days: Number of forecast days
            
        Returns:
            DataFrame with regional weather data
        """
        stations = self.get_weather_stations()
        all_data = []
        
        for station in stations:
            try:
                forecast = self.get_weather_forecast_nws(
                    station["lat"], station["lon"], days
                )
                
                if not forecast.empty:
                    forecast["region"] = station["region"]
                    forecast["station"] = station["name"]
                    all_data.append(forecast)
                    
            except Exception as e:
                logger.warning(f"Failed to get forecast for {station['name']}: {e}")
                continue
        
        if all_data:
            df = pd.concat(all_data, ignore_index=True)
            return df
        else:
            return pd.DataFrame()
    
    def get_historical_weather(self, start_date: str, end_date: str) -> pd.DataFrame:
        """
        Get historical weather data.
        This would typically involve NOMADS or other historical data sources.
        
        Args:
            start_date: Start date in YYYY-MM-DD format
            end_date: End date in YYYY-MM-DD format
            
        Returns:
            DataFrame with historical weather data
        """
        logger.info("Historical weather data collection not yet implemented")
        return pd.DataFrame()
    
    def get_all_weather_data(self, start_date: str = None, end_date: str = None, 
                           forecast_days: int = 14) -> Dict[str, pd.DataFrame]:
        """
        Get all weather data (historical + forecast).
        
        Args:
            start_date: Start date for historical data
            end_date: End date for historical data
            forecast_days: Number of forecast days
            
        Returns:
            Dictionary with weather data
        """
        data = {}
        
        try:
            # Get forecast data
            data["forecast"] = self.get_regional_weather_forecast(forecast_days)
            
            # Get historical data if dates provided
            if start_date and end_date:
                data["historical"] = self.get_historical_weather(start_date, end_date)
            
            # Get degree days
            data["degree_days"] = self.get_degree_days_cpc(start_date, end_date)
            
            logger.info(f"Successfully collected {len(data)} weather datasets")
            
        except Exception as e:
            logger.error(f"Error collecting weather data: {e}")
            raise
            
        return data


def main():
    """Test the weather client."""
    client = WeatherClient()
    
    print("Testing weather client...")
    
    # Test forecast
    forecast = client.get_regional_weather_forecast(days=7)
    print(f"Forecast data: {len(forecast)} records")
    if not forecast.empty:
        print(f"Regions: {forecast['region'].unique()}")
        print(f"Date range: {forecast['date'].min()} to {forecast['date'].max()}")


if __name__ == "__main__":
    main()
