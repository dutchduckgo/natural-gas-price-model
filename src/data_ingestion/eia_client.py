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
        params = {
            "frequency": "daily" if "D" in series_id else "monthly",
            "data[0]": "value",
            "sort[0][column]": "period",
            "sort[0][direction]": "asc"
        }
        
        if start_date:
            params["start"] = start_date
        if end_date:
            params["end"] = end_date
            
        response = self._make_request(f"natural-gas/pri/fut/{series_id}", params)
        
        if "data" not in response:
            logger.warning(f"No data found for series {series_id}")
            return pd.DataFrame()
            
        df = pd.DataFrame(response["data"])
        df["date"] = pd.to_datetime(df["period"])
        df = df[["date", "value"]].copy()
        df["value"] = pd.to_numeric(df["value"], errors="coerce")
        df = df.dropna()
        
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
