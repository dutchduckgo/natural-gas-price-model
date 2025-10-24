"""
Power grid data collection from EIA-930, PJM, ERCOT, and ISO-NE.
"""
import requests
import pandas as pd
import numpy as np
from typing import Dict, List, Optional
from datetime import datetime, timedelta
import logging
from config import EIA_930_BASE_URL, PJM_BASE_URL, ERCOT_BASE_URL

logger = logging.getLogger(__name__)


class PowerGridClient:
    """Client for power grid data from multiple ISOs."""
    
    def __init__(self):
        self.eia_930_url = EIA_930_BASE_URL
        self.pjm_url = PJM_BASE_URL
        self.ercot_url = ERCOT_BASE_URL
        self.session = requests.Session()
        
    def _make_request(self, url: str, params: Dict = None) -> Dict:
        """Make API request with error handling."""
        try:
            response = self.session.get(url, params=params)
            response.raise_for_status()
            return response.json()
        except requests.exceptions.RequestException as e:
            logger.error(f"Power grid API request failed: {e}")
            raise
    
    def get_eia_930_data(self, start_date: str = None, end_date: str = None) -> pd.DataFrame:
        """
        Get EIA-930 hourly electric grid monitor data.
        
        Args:
            start_date: Start date in YYYY-MM-DD format
            end_date: End date in YYYY-MM-DD format
            
        Returns:
            DataFrame with hourly power data
        """
        # EIA-930 data structure is complex and requires specific endpoints
        # This is a placeholder for the actual implementation
        logger.info("EIA-930 data collection not yet implemented")
        return pd.DataFrame()
    
    def get_pjm_data(self, start_date: str = None, end_date: str = None) -> pd.DataFrame:
        """
        Get PJM generation and LMP data.
        
        Args:
            start_date: Start date in YYYY-MM-DD format
            end_date: End date in YYYY-MM-DD format
            
        Returns:
            DataFrame with PJM power data
        """
        # PJM Data Miner 2 API implementation
        # This is a placeholder for the actual implementation
        logger.info("PJM data collection not yet implemented")
        return pd.DataFrame()
    
    def get_ercot_data(self, start_date: str = None, end_date: str = None) -> pd.DataFrame:
        """
        Get ERCOT fuel mix and load data.
        
        Args:
            start_date: Start date in YYYY-MM-DD format
            end_date: End date in YYYY-MM-DD format
            
        Returns:
            DataFrame with ERCOT power data
        """
        # ERCOT API implementation
        # This is a placeholder for the actual implementation
        logger.info("ERCOT data collection not yet implemented")
        return pd.DataFrame()
    
    def get_iso_ne_data(self, start_date: str = None, end_date: str = None) -> pd.DataFrame:
        """
        Get ISO-NE load and fuel mix data.
        
        Args:
            start_date: Start date in YYYY-MM-DD format
            end_date: End date in YYYY-MM-DD format
            
        Returns:
            DataFrame with ISO-NE power data
        """
        # ISO-NE Web Services implementation
        # This is a placeholder for the actual implementation
        logger.info("ISO-NE data collection not yet implemented")
        return pd.DataFrame()
    
    def calculate_gas_burn_proxy(self, power_data: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate gas burn proxy from power generation data.
        
        Args:
            power_data: DataFrame with generation by fuel type
            
        Returns:
            DataFrame with gas burn estimates
        """
        df = power_data.copy()
        
        # Estimate gas generation from total generation and fuel mix
        if "gas_generation" in df.columns:
            df["gas_burn_mwh"] = df["gas_generation"]
        else:
            # Estimate based on total load and typical gas share
            df["gas_burn_mwh"] = df["total_load"] * 0.4  # Assume 40% gas share
        
        # Convert to daily aggregates
        df["date"] = pd.to_datetime(df["date"])
        daily_gas_burn = df.groupby("date")["gas_burn_mwh"].sum().reset_index()
        
        return daily_gas_burn
    
    def get_all_power_data(self, start_date: str = None, end_date: str = None) -> Dict[str, pd.DataFrame]:
        """
        Get all power grid data from multiple sources.
        
        Args:
            start_date: Start date in YYYY-MM-DD format
            end_date: End date in YYYY-MM-DD format
            
        Returns:
            Dictionary with power data from different sources
        """
        data = {}
        
        try:
            # Get data from each ISO
            data["eia_930"] = self.get_eia_930_data(start_date, end_date)
            data["pjm"] = self.get_pjm_data(start_date, end_date)
            data["ercot"] = self.get_ercot_data(start_date, end_date)
            data["iso_ne"] = self.get_iso_ne_data(start_date, end_date)
            
            # Calculate gas burn proxies
            for source, df in data.items():
                if not df.empty:
                    data[f"{source}_gas_burn"] = self.calculate_gas_burn_proxy(df)
            
            logger.info(f"Successfully collected {len(data)} power datasets")
            
        except Exception as e:
            logger.error(f"Error collecting power data: {e}")
            raise
            
        return data


def main():
    """Test the power grid client."""
    client = PowerGridClient()
    
    print("Testing power grid client...")
    
    # Test with recent data
    end_date = datetime.now().strftime("%Y-%m-%d")
    start_date = (datetime.now() - timedelta(days=7)).strftime("%Y-%m-%d")
    
    data = client.get_all_power_data(start_date, end_date)
    
    for name, df in data.items():
        print(f"{name}: {len(df)} records")
        if not df.empty:
            print(f"  Columns: {list(df.columns)}")


if __name__ == "__main__":
    main()
