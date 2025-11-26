"""
EIA Open Data API client for natural gas data.
"""
import requests
import pandas as pd
from typing import Dict, List, Optional, Union
from datetime import datetime, timedelta
from pathlib import Path
import logging
from config import EIA_BASE_URL, EIA_API_KEY, EIA_SERIES_IDS, DATA_DIR

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
    
    def get_series_data(self, series_id: str, start_date: str = None, end_date: str = None, 
                       frequency: str = "daily", data_type: str = "prices", 
                       facets: Dict = None) -> pd.DataFrame:
        """
        Get time series data for a specific series ID using EIA v2 API.
        
        Args:
            series_id: EIA series identifier (full format, e.g., NG.RNGWHHD.D or just RNGWHHD)
            start_date: Start date in YYYY-MM-DD format
            end_date: End date in YYYY-MM-DD format
            frequency: Data frequency (daily, weekly, monthly, annual)
            data_type: Type of data (prices, storage, production, lng_exports, etc.) - determines endpoint
            facets: Additional facet filters (e.g., {"process": ["FPD"]})
            
        Returns:
            DataFrame with date and value columns
        """
        # Determine endpoint based on data type
        # Extract base series code (e.g., RNGWHHD from NG.RNGWHHD.D)
        series_code = series_id.split('.')[-2] if '.' in series_id else series_id
        
        # Map data types to v2 API endpoints
        endpoint_map = {
            "prices": "natural-gas/pri/fut/data/",
            "storage": "natural-gas/stor/wkly/data/",
            "production": "natural-gas/prod/sum/data/",
            "lng_exports": "natural-gas/move/expc/data/",
            "supply": "natural-gas/sum/snd/data/",
            "consumption": "natural-gas/sum/ngcons/data/",
        }
        
        # Default to prices endpoint if data_type not specified
        endpoint = endpoint_map.get(data_type, "natural-gas/pri/fut/data/")
        
        # Build query parameters for v2 API
        params = {
            "frequency": frequency,
            "data[0]": "value",
            "sort[0][column]": "period",
            "sort[0][direction]": "asc",
            "api_key": self.api_key,
            "offset": 0,
            "length": 5000
        }
        
        # Add date filters if provided
        if start_date:
            params["start"] = start_date
        if end_date:
            params["end"] = end_date
        
        # Add series filter - v2 API uses facets with series code
        # For storage, we don't filter by series (get all regions)
        # For other types, filter by series code if provided
        if data_type != "storage" and series_code:
            params["facets[series][]"] = series_code
        
        # Add additional facets if provided (e.g., process=FPD for production, process=ENG for LNG)
        if facets:
            for facet_key, facet_values in facets.items():
                if isinstance(facet_values, list):
                    for i, value in enumerate(facet_values):
                        params[f"facets[{facet_key}][]"] = value
                else:
                    params[f"facets[{facet_key}][]"] = facet_values
        
        url = f"{self.base_url}/{endpoint}"
        
        try:
            response = self.session.get(url, params=params)
            response.raise_for_status()
        except requests.exceptions.HTTPError as e:
            status = e.response.status_code if e.response is not None else None
            if status == 404:
                logger.warning(f"EIA series {series_id} not found (404). Skipping.")
                return pd.DataFrame()
            logger.error(f"EIA v2 API request failed: {e}")
            if e.response is not None:
                logger.error(f"Response: {e.response.text[:500]}")  # Limit response text
            raise
        except requests.exceptions.RequestException as e:
            logger.error(f"EIA v2 API request failed: {e}")
            raise
        
        json_data = response.json()
        
        # Parse v2 API response structure
        if "response" not in json_data:
            logger.warning(f"Unexpected response structure for series {series_id}")
            return pd.DataFrame()
        
        response_data = json_data["response"]
        
        # Check for data in response
        if "data" not in response_data:
            logger.warning(f"No data found for series {series_id}")
            return pd.DataFrame()
        
        data_records = response_data["data"]
        if not data_records:
            logger.warning(f"No data records found for series {series_id}")
            return pd.DataFrame()
        
        # Convert v2 API response to DataFrame
        df = pd.DataFrame(data_records)
        
        # Handle different possible column names in v2 API response
        # v2 API might return period/value or different column names
        if "period" in df.columns and "value" in df.columns:
            df = df.rename(columns={"period": "date", "value": "value"})
        elif len(df.columns) >= 2:
            # Assume first column is date, second is value
            df.columns = ["date", "value"] + list(df.columns[2:])
            df = df[["date", "value"]]
        else:
            logger.warning(f"Unexpected column structure for series {series_id}: {df.columns.tolist()}")
            return pd.DataFrame()
        
        # Convert period strings to datetime
        df["date"] = pd.to_datetime(df["date"], errors="coerce")
        df = df.dropna(subset=["date"])
        df["value"] = pd.to_numeric(df["value"], errors="coerce")
        df = df.dropna(subset=["value"])
        
        df = df[["date", "value"]].sort_values("date").reset_index(drop=True)
        
        return df
    
    def get_henry_hub_spot(self, start_date: str = None, end_date: str = None) -> pd.DataFrame:
        """
        Get Henry Hub spot prices (daily).
        Default start date is 2010-01-01 to align with storage data availability.
        """
        # Default to 2010-01-01 if no start date provided (align with storage data)
        if start_date is None:
            start_date = "2010-01-01"
        """
        Get Henry Hub spot prices (daily).
        Uses the natural-gas/pri/fut/data/ endpoint with daily frequency,
        filtered for RNGWHHD series and Y35NY duoarea using facets.
        Implements pagination to fetch all records (API limit is 5000 per request).
        """
        endpoint = "natural-gas/pri/fut/data/"
        
        all_records = []
        offset = 0
        page_size = 5000
        
        while True:
            params = {
                "frequency": "daily",
                "data[0]": "value",
                "facets[duoarea][]": "RGC",  # Filter for RGC duoarea (Henry Hub region)
                "sort[0][column]": "period",
                "sort[0][direction]": "asc",  # Ascending for chronological order
                "api_key": self.api_key,
                "offset": offset,
                "length": page_size
            }
            
            if start_date:
                params["start"] = start_date
            if end_date:
                params["end"] = end_date
            
            url = f"{self.base_url}/{endpoint}"
            
            try:
                response = self.session.get(url, params=params)
                response.raise_for_status()
            except requests.exceptions.HTTPError as e:
                status = e.response.status_code if e.response is not None else None
                if status == 404:
                    logger.warning(f"EIA price data not found (404). Skipping.")
                    break
                logger.error(f"EIA v2 API request failed: {e}")
                if e.response is not None:
                    logger.error(f"Response: {e.response.text[:500]}")
                raise
            except requests.exceptions.RequestException as e:
                logger.error(f"EIA v2 API request failed: {e}")
                raise
            
            json_data = response.json()
            
            if "response" not in json_data:
                logger.warning("Unexpected API response structure")
                break
            
            response_data = json_data["response"]
            
            # Get total count on first request
            if offset == 0:
                total = response_data.get("total", 0)
                logger.info(f"Total Henry Hub daily records available: {total}")
            
            if "data" not in response_data:
                break
            
            page_records = response_data["data"]
            if not page_records:
                break
            
            all_records.extend(page_records)
            
            # Check if we've fetched all records
            if len(page_records) < page_size:
                break
            
            offset += page_size
            logger.info(f"Fetched {len(all_records)} records so far...")
        
        if not all_records:
            logger.warning("No price data records found")
            return pd.DataFrame()
        
        df = pd.DataFrame(all_records)
        
        # Filter for RNGWHHD series if series column exists
        if "series" in df.columns:
            df = df[df["series"] == "RNGWHHD"].copy()
            if df.empty:
                logger.warning("No RNGWHHD series found in response")
                return pd.DataFrame()
        
        # Handle v2 API response structure
        if "period" in df.columns and "value" in df.columns:
            df = df.rename(columns={"period": "date"})
        else:
            logger.warning(f"Unexpected price data structure: {df.columns.tolist()}")
            return pd.DataFrame()
        
        # Convert to datetime and numeric
        df["date"] = pd.to_datetime(df["date"], errors="coerce")
        df = df.dropna(subset=["date"])
        df["value"] = pd.to_numeric(df["value"], errors="coerce")
        df = df.dropna(subset=["value"])
        
        df = df[["date", "value"]].sort_values("date").reset_index(drop=True)
        
        logger.info(f"Retrieved {len(df)} Henry Hub spot price records (daily)")
        
        return df
    
    def get_storage_data(self, start_date: str = None, end_date: str = None) -> pd.DataFrame:
        """
        Get weekly storage data for R48 region.
        Uses natural-gas/stor/wkly/data/ endpoint with process=SWO and duoarea=R48 facets.
        Default start date is 2010-01-01 to get all available historical data.
        Implements pagination to fetch all records.
        """
        endpoint = "natural-gas/stor/wkly/data/"
        
        # Default to 2010-01-01 if no start date provided (earliest available data)
        if start_date is None:
            start_date = "2010-01-01"
        
        all_records = []
        offset = 0
        page_size = 5000
        
        while True:
            params = {
                "frequency": "weekly",
                "data[0]": "value",
                "facets[process][]": "SWO",  # Storage working gas
                "facets[duoarea][]": "R48",  # Region 48
                "start": start_date,  # Always include start date
                "sort[0][column]": "period",
                "sort[0][direction]": "asc",  # Ascending for chronological order
                "api_key": self.api_key,
                "offset": offset,
                "length": page_size
            }
            
            if end_date:
                params["end"] = end_date
            
            url = f"{self.base_url}/{endpoint}"
            
            try:
                response = self.session.get(url, params=params)
                response.raise_for_status()
            except requests.exceptions.HTTPError as e:
                status = e.response.status_code if e.response is not None else None
                if status == 404:
                    logger.warning(f"EIA storage data not found (404). Skipping.")
                    break
                logger.error(f"EIA v2 API request failed: {e}")
                if e.response is not None:
                    logger.error(f"Response: {e.response.text[:500]}")
                raise
            except requests.exceptions.RequestException as e:
                logger.error(f"EIA v2 API request failed: {e}")
                raise
            
            json_data = response.json()
            
            if "response" not in json_data:
                logger.warning("Unexpected API response structure")
                break
            
            response_data = json_data["response"]
            
            # Get total count on first request
            if offset == 0:
                total = response_data.get("total", 0)
                logger.info(f"Total storage records available: {total}")
            
            if "data" not in response_data:
                break
            
            page_records = response_data["data"]
            if not page_records:
                break
            
            all_records.extend(page_records)
            
            # Check if we've fetched all records
            if len(page_records) < page_size:
                break
            
            offset += page_size
            logger.info(f"Fetched {len(all_records)} storage records so far...")
        
        if not all_records:
            logger.warning("No storage data records found")
            return pd.DataFrame()
        
        df = pd.DataFrame(all_records)
        
        # Filter for NW2_EPG0_SWO_R48_BCF series if series column exists
        if "series" in df.columns:
            df = df[df["series"] == "NW2_EPG0_SWO_R48_BCF"].copy()
            if df.empty:
                logger.warning("No NW2_EPG0_SWO_R48_BCF series found in response")
                return pd.DataFrame()
        
        # Handle v2 API response structure
        if "period" in df.columns and "value" in df.columns:
            df = df.rename(columns={"period": "date"})
        else:
            logger.warning(f"Unexpected storage data structure: {df.columns.tolist()}")
            return pd.DataFrame()
        
        # Convert to datetime and numeric
        df["date"] = pd.to_datetime(df["date"], errors="coerce")
        df = df.dropna(subset=["date"])
        df["value"] = pd.to_numeric(df["value"], errors="coerce")
        df = df.dropna(subset=["value"])
        
        # Filter out obviously invalid values (storage should be in reasonable BCF range, e.g., 0-5000 BCF)
        # Values larger than 10000 BCF are likely data errors
        df = df[df["value"] <= 10000]
        
        # Also filter out negative values
        df = df[df["value"] >= 0]
        
        df = df[["date", "value"]].sort_values("date").reset_index(drop=True)
        
        logger.info(f"Retrieved {len(df)} storage records (weekly)")
        
        return df
    
    def get_production_data(self, start_date: str = None, end_date: str = None) -> pd.DataFrame:
        """
        Get monthly production data with process=FPD facet.
        Default start date is 2010-01-01 to align with storage data availability.
        """
        # Default to 2010-01-01 if no start date provided (align with storage data)
        if start_date is None:
            start_date = "2010-01-01"
        return self.get_series_data(EIA_SERIES_IDS["production"], start_date, end_date, 
                                   frequency="monthly", data_type="production",
                                   facets={"process": ["FPD"]})
    
    def get_consumption_data(self, start_date: str = None, end_date: str = None) -> pd.DataFrame:
        """
        Get monthly consumption data from Excel file.
        Reads from NG_CONS_SUM_A_EPG0_VC0_MMCF_M.xls in the data directory.
        File structure: Sheet "Data 1", dates in column 0 starting row 2, values in column 1.
        Default start date is 2010-01-01 to align with storage data availability.
        """
        # Default to 2010-01-01 if no start date provided (align with storage data)
        if start_date is None:
            start_date = "2010-01-01"
        
        consumption_file = DATA_DIR / "NG_CONS_SUM_A_EPG0_VC0_MMCF_M.xls"
        
        if not consumption_file.exists():
            logger.warning(f"Consumption file not found: {consumption_file}")
            return pd.DataFrame()
        
        try:
            # Read from "Data 1" sheet
            # Data starts at row 2 (0-indexed), row 0 has metadata, row 1 has headers
            df = pd.read_excel(consumption_file, sheet_name="Data 1")
            
            # Extract date and value columns (first two columns)
            # Skip first two rows (metadata and header)
            result_df = pd.DataFrame({
                "date": pd.to_datetime(df.iloc[2:, 0], errors="coerce"),
                "value": pd.to_numeric(df.iloc[2:, 1], errors="coerce")
            })
            
            # Remove rows with invalid dates or values
            result_df = result_df.dropna(subset=["date", "value"])
            
            # Apply date filters (default to 2010-01-01)
            result_df = result_df[result_df["date"] >= pd.to_datetime(start_date)]
            if end_date:
                result_df = result_df[result_df["date"] <= pd.to_datetime(end_date)]
            
            # Sort by date
            result_df = result_df.sort_values("date").reset_index(drop=True)
            
            logger.info(f"Loaded {len(result_df)} consumption records from Excel file")
            return result_df
            
        except Exception as e:
            logger.error(f"Error reading consumption Excel file: {e}")
            import traceback
            logger.error(traceback.format_exc())
            return pd.DataFrame()
    
    def get_lng_exports(self, start_date: str = None, end_date: str = None) -> pd.DataFrame:
        """
        Get monthly LNG export data with process=ENG facet.
        Default start date is 2010-01-01 to align with storage data availability.
        """
        # Default to 2010-01-01 if no start date provided (align with storage data)
        if start_date is None:
            start_date = "2010-01-01"
        return self.get_series_data(EIA_SERIES_IDS["lng_exports"], start_date, end_date, 
                                   frequency="monthly", data_type="lng_exports",
                                   facets={"process": ["ENG"]})
    
    def get_eia_930_data(self, start_date: str = None, end_date: str = None) -> pd.DataFrame:
        """
        Get EIA-930 hourly electric grid monitor data for natural gas generation.
        Uses electricity/rto/fuel-type-data/data/ endpoint with fueltype=NG facet.
        Includes all major balancing authorities (BAs).
        Earliest data available is 2019-01-01.
        Aggregates hourly data to daily for consistency with other data sources.
        
        Args:
            start_date: Start date in YYYY-MM-DD format (defaults to 2019-01-01, earliest available)
            end_date: End date in YYYY-MM-DD format
            
        Returns:
            DataFrame with date, ba (respondent), and gas_mwh (gas generation in MWh) columns
        """
        endpoint = "electricity/rto/fuel-type-data/data/"
        
        # Default to 2019-01-01 if no start date provided (earliest available data)
        if start_date is None:
            start_date = "2019-01-01"
        
        # Convert date to datetime format with hour for hourly API (YYYY-MM-DDTHH)
        if isinstance(start_date, str) and 'T' not in start_date:
            start_date_dt = f"{start_date}T00"
        else:
            start_date_dt = start_date
        
        if end_date and isinstance(end_date, str) and 'T' not in end_date:
            end_date_dt = f"{end_date}T23"
        else:
            end_date_dt = end_date
        
        # List of all balancing authorities (respondents) to include
        respondents = ["CISO", "ERCO", "FLA", "FMPP", "ISNE", "MISO", "NW", "NYIS", "PJM", "SWPP"]
        
        all_records = []
        offset = 0
        page_size = 5000
        
        while True:
            params = {
                "frequency": "hourly",
                "data[0]": "value",
                "facets[fueltype][]": "NG",  # Natural gas fuel type
                "start": start_date_dt,
                "sort[0][column]": "period",
                "sort[0][direction]": "asc",
                "sort[1][column]": "respondent",
                "sort[1][direction]": "desc",
                "api_key": self.api_key,
                "offset": offset,
                "length": page_size
            }
            
            # Add all respondent facets as a list (requests will create multiple params with same key)
            params["facets[respondent][]"] = respondents
            
            if end_date_dt:
                params["end"] = end_date_dt
            
            url = f"{self.base_url}/{endpoint}"
            
            try:
                response = self.session.get(url, params=params)
                response.raise_for_status()
            except requests.exceptions.HTTPError as e:
                status = e.response.status_code if e.response is not None else None
                if status == 404:
                    logger.warning(f"EIA-930 data not found (404). Skipping.")
                    break
                logger.error(f"EIA v2 API request failed: {e}")
                if e.response is not None:
                    logger.error(f"Response: {e.response.text[:500]}")
                raise
            except requests.exceptions.RequestException as e:
                logger.error(f"EIA v2 API request failed: {e}")
                raise
            
            json_data = response.json()
            
            if "response" not in json_data:
                logger.warning("Unexpected API response structure")
                break
            
            response_data = json_data["response"]
            
            # Get total count on first request
            if offset == 0:
                total = response_data.get("total", 0)
                logger.info(f"Total EIA-930 hourly records available: {total}")
            
            if "data" not in response_data:
                break
            
            page_records = response_data["data"]
            if not page_records:
                break
            
            all_records.extend(page_records)
            
            # Check if we've fetched all records
            if len(page_records) < page_size:
                break
            
            offset += page_size
            logger.info(f"Fetched {len(all_records)} EIA-930 records so far...")
        
        if not all_records:
            logger.warning("No EIA-930 data records found")
            return pd.DataFrame()
        
        df = pd.DataFrame(all_records)
        
        # Handle v2 API response structure
        if "period" not in df.columns or "value" not in df.columns:
            logger.warning(f"Unexpected EIA-930 data structure: {df.columns.tolist()}")
            return pd.DataFrame()
        
        # Convert period to datetime (hourly format)
        df["period"] = pd.to_datetime(df["period"], errors="coerce")
        df = df.dropna(subset=["period"])
        df["value"] = pd.to_numeric(df["value"], errors="coerce")
        df = df.dropna(subset=["value"])
        
        # Extract date (remove hour component) and respondent (BA)
        df["date"] = df["period"].dt.date
        df["date"] = pd.to_datetime(df["date"])
        
        # Aggregate hourly data to daily by respondent (BA)
        if "respondent" in df.columns:
            # Group by date and respondent, sum the hourly values
            df_daily = df.groupby(["date", "respondent"])["value"].sum().reset_index()
            df_daily = df_daily.rename(columns={"value": "gas_mwh", "respondent": "ba"})
        else:
            # If no respondent column, just aggregate by date
            df_daily = df.groupby("date")["value"].sum().reset_index()
            df_daily = df_daily.rename(columns={"value": "gas_mwh"})
            df_daily["ba"] = "US48"  # Default to US48 if no respondent info
        
        df_daily = df_daily.sort_values(["date", "ba"]).reset_index(drop=True)
        
        logger.info(f"Retrieved {len(df_daily)} EIA-930 daily records (aggregated from {len(df)} hourly records)")
        
        return df_daily
    
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
