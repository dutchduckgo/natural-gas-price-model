"""
EIA Open Data API client for natural gas data.
"""
import requests
import pandas as pd
from typing import Dict, List, Optional, Union
from datetime import datetime, timedelta
import logging
from config import EIA_BASE_URL, EIA_API_KEY, EIA_SERIES_IDS

logger = logging.getLogger(__name__)


class EIAClient:
    """Client for EIA Open Data API."""
    
    def __init__(self, api_key: str = None):
        self.api_key = api_key or EIA_API_KEY
        self.base_url = EIA_BASE_URL
        self.session = requests.Session()
        
    def _make_request(self, endpoint: str, params: Dict = None) -> Dict:
        """Make API request with error handling."""
        url = f"{self.base_url}/{endpoint}"
        params = params or {}
        params["api_key"] = self.api_key
        
        try:
            response = self.session.get(url, params=params)
            response.raise_for_status()
            return response.json()
        except requests.exceptions.RequestException as e:
            logger.error(f"EIA API request failed: {e}")
            raise
    
    def get_series_data(self, series_id: str, start_date: str = None, end_date: str = None) -> pd.DataFrame:
        """
        Get time series data for a specific series ID.
        
        Args:
            series_id: EIA series identifier
            start_date: Start date in YYYY-MM-DD format
            end_date: End date in YYYY-MM-DD format
            
        Returns:
            DataFrame with date and value columns
        """
        # Use EIA v1 series endpoint for broader compatibility
        params = {
            "series_id": series_id,
            "api_key": self.api_key
        }
        
        response = self.session.get("https://api.eia.gov/series/", params=params)
        try:
            response.raise_for_status()
        except requests.exceptions.HTTPError as e:
            status = e.response.status_code if e.response is not None else None
            if status == 404:
                logger.warning(f"EIA series {series_id} not found (404). Skipping.")
                return pd.DataFrame()
            logger.error(f"EIA series API request failed: {e}")
            raise
        except requests.exceptions.RequestException as e:
            logger.error(f"EIA series API request failed: {e}")
            raise
        
        json_data = response.json()
        if "series" not in json_data or not json_data["series"]:
            logger.warning(f"No data found for series {series_id}")
            return pd.DataFrame()
        
        series_data = json_data["series"][0]["data"]
        df = pd.DataFrame(series_data, columns=["period", "value"])
        
        # Convert period strings to datetime (handles daily, weekly, monthly)
        df["date"] = pd.to_datetime(df["period"], errors="coerce")
        df = df.dropna(subset=["date"])
        df["value"] = pd.to_numeric(df["value"], errors="coerce")
        df = df.dropna(subset=["value"])
        
        # Apply date filters if provided
        if start_date:
            df = df[df["date"] >= pd.to_datetime(start_date)]
        if end_date:
            df = df[df["date"] <= pd.to_datetime(end_date)]
        
        df = df[["date", "value"]].sort_values("date").reset_index(drop=True)
        
        return df
    
    def get_henry_hub_spot(self, start_date: str = None, end_date: str = None) -> pd.DataFrame:
        """Get Henry Hub spot prices."""
        return self.get_series_data(EIA_SERIES_IDS["henry_hub_spot"], start_date, end_date)
    
    def get_storage_data(self, start_date: str = None, end_date: str = None) -> pd.DataFrame:
        """Get weekly storage data."""
        return self.get_series_data(EIA_SERIES_IDS["storage"], start_date, end_date)
    
    def get_production_data(self, start_date: str = None, end_date: str = None) -> pd.DataFrame:
        """Get monthly production data."""
        return self.get_series_data(EIA_SERIES_IDS["production"], start_date, end_date)
    
    def get_consumption_data(self, start_date: str = None, end_date: str = None) -> pd.DataFrame:
        """Get monthly consumption data."""
        return self.get_series_data(EIA_SERIES_IDS["consumption"], start_date, end_date)
    
    def get_lng_exports(self, start_date: str = None, end_date: str = None) -> pd.DataFrame:
        """Get monthly LNG export data."""
        return self.get_series_data(EIA_SERIES_IDS["lng_exports"], start_date, end_date)
    
    def get_eia_930_data(self, start_date: str = None, end_date: str = None) -> pd.DataFrame:
        """
        Get EIA-930 hourly electric grid monitor data.
        This requires a different endpoint structure.
        """
        # EIA-930 data structure is more complex
        # This is a placeholder for the actual implementation
        logger.info("EIA-930 data collection not yet implemented")
        return pd.DataFrame()
    
    def get_all_core_data(self, start_date: str = None, end_date: str = None) -> Dict[str, pd.DataFrame]:
        """
        Get all core natural gas data in one call.
        
        Returns:
            Dictionary with data type as key and DataFrame as value
        """
        data = {}
        
        try:
            data["henry_hub_spot"] = self.get_henry_hub_spot(start_date, end_date)
            data["storage"] = self.get_storage_data(start_date, end_date)
            data["production"] = self.get_production_data(start_date, end_date)
            data["consumption"] = self.get_consumption_data(start_date, end_date)
            data["lng_exports"] = self.get_lng_exports(start_date, end_date)
            
            logger.info(f"Successfully collected {len(data)} datasets from EIA")
            
        except Exception as e:
            logger.error(f"Error collecting EIA data: {e}")
            raise
            
        return data


def main():
    """Test the EIA client."""
    client = EIAClient()
    
    # Test with recent data
    end_date = datetime.now().strftime("%Y-%m-%d")
    start_date = (datetime.now() - timedelta(days=30)).strftime("%Y-%m-%d")
    
    print("Testing EIA client...")
    data = client.get_all_core_data(start_date, end_date)
    
    for name, df in data.items():
        print(f"{name}: {len(df)} records")
        if not df.empty:
            print(f"  Date range: {df['date'].min()} to {df['date'].max()}")
            print(f"  Sample values: {df['value'].head().tolist()}")


if __name__ == "__main__":
    main()
