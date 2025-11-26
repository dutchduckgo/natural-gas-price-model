"""
NOAA NCEI CDO (Climate Data Online) API client for historical weather data.
Fetches daily temperature and wind speed from NOAA weather stations.
"""
import requests
import pandas as pd
import numpy as np
from typing import List, Optional, Dict
from datetime import datetime, timedelta
import logging
import os
import time

logger = logging.getLogger(__name__)

# NOAA CDO API base URL
NOAA_CDO_BASE_URL = "https://www.ncei.noaa.gov/cdo-web/api/v2"

# Representative weather stations for CONUS aggregate
# These are major stations in key gas-consuming regions
DEFAULT_STATIONS = [
    "USW00012960",  # Houston, TX (IAH)
    "USW00014732",  # New York, NY (LGA)
    "USW00023174",  # Los Angeles, CA (LAX)
    "USW00014819",  # Chicago, IL (ORD)
    "USW00013722",  # Boston, MA (BOS)
    "USW00013881",  # Atlanta, GA (ATL)
    "USW00013960",  # Dallas, TX (DFW)
    "USW00014925",  # Minneapolis, MN (MSP)
    "USW00024233",  # Seattle, WA (SEA)
    "USW00003017",  # Denver, CO (DEN)
]


class NOAAWeatherClient:
    """Client for fetching historical weather data from NOAA NCEI CDO API."""
    
    def __init__(self, token: Optional[str] = None, stations: Optional[List[str]] = None):
        """
        Initialize NOAA weather client.
        
        Args:
            token: NOAA CDO API token (defaults to NOAA_CDO_TOKEN env var)
            stations: List of station IDs to use (defaults to DEFAULT_STATIONS)
        """
        self.token = token or os.getenv("NOAA_CDO_TOKEN", "")
        if not self.token:
            logger.warning("NOAA_CDO_TOKEN not found in environment. API calls will fail.")
        
        self.stations = stations or DEFAULT_STATIONS
        self.session = requests.Session()
        self.session.headers.update({
            "token": self.token,
            "User-Agent": "Mozilla/5.0 (compatible; GasModel/1.0)"
        })
        
        # Rate limiting: NOAA API allows 5 requests per second
        self.last_request_time = 0
        self.min_request_interval = 0.2  # 200ms between requests
    
    def _rate_limit(self):
        """Enforce rate limiting between API requests."""
        current_time = time.time()
        time_since_last = current_time - self.last_request_time
        if time_since_last < self.min_request_interval:
            time.sleep(self.min_request_interval - time_since_last)
        self.last_request_time = time.time()
    
    def _request(self, endpoint: str, params: Dict) -> Dict:
        """
        Make GET request to NOAA CDO API with pagination.
        
        Args:
            endpoint: API endpoint (e.g., "data")
            params: Request parameters
            
        Returns:
            Combined JSON data from all pages
        """
        url = f"{NOAA_CDO_BASE_URL}/{endpoint}"
        all_results = []
        offset = 1
        limit = 1000
        
        while True:
            self._rate_limit()
            
            # Add pagination parameters
            page_params = params.copy()
            page_params["limit"] = limit
            page_params["offset"] = offset
            
            try:
                response = self.session.get(url, params=page_params, timeout=30)
                response.raise_for_status()
                json_data = response.json()
                
                # Check if we have results
                if "results" not in json_data or not json_data["results"]:
                    break
                
                all_results.extend(json_data["results"])
                
                # Check if we've fetched all records
                metadata = json_data.get("metadata", {})
                result_count = metadata.get("resultset", {}).get("count", 0)
                
                if len(all_results) >= result_count:
                    break
                
                offset += limit
                
                logger.debug(f"Fetched {len(all_results)} records, continuing...")
                
            except requests.exceptions.HTTPError as e:
                if e.response.status_code == 429:
                    # Rate limited - wait and retry
                    logger.warning("Rate limited (429), waiting 2 seconds before retry...")
                    time.sleep(2)
                    continue
                elif e.response.status_code == 401:
                    logger.error("Authentication failed (401). Check NOAA_CDO_TOKEN.")
                    raise
                elif e.response.status_code >= 500:
                    # Server error - retry once
                    logger.warning(f"Server error ({e.response.status_code}), retrying once...")
                    time.sleep(2)
                    continue
                logger.error(f"NOAA API request failed: {e}")
                if e.response is not None:
                    logger.error(f"Response: {e.response.text[:500]}")
                raise
            except requests.exceptions.Timeout as e:
                logger.warning(f"Request timeout, retrying once...")
                time.sleep(2)
                continue
            except requests.exceptions.RequestException as e:
                logger.error(f"NOAA API request failed: {e}")
                raise
        
        return {"results": all_results, "metadata": {"resultset": {"count": len(all_results)}}}
    
    def get_daily_station_weather(
        self, 
        station_id: str, 
        start_date: str, 
        end_date: str
    ) -> pd.DataFrame:
        """
        Get daily weather data for a specific station.
        
        Args:
            station_id: Station ID (e.g., "USW00012960")
            start_date: Start date in YYYY-MM-DD format
            end_date: End date in YYYY-MM-DD format
            
        Returns:
            DataFrame with columns: date, station_id, tmax, tmin, tavg, awnd
        """
        # Fetch each datatype separately (NOAA API doesn't handle multiple datatypes well in one call)
        all_results = []
        
        for datatype in ["TMAX", "TMIN", "AWND"]:
            params = {
                "datasetid": "GHCND",
                "stationid": f"GHCND:{station_id}",
                "startdate": start_date,
                "enddate": end_date,
                "datatypeid": datatype,
                "units": "standard",
            }
            
            try:
                data = self._request("data", params)
                results = data.get("results", [])
                all_results.extend(results)
            except Exception as e:
                logger.warning(f"Failed to fetch {datatype} for station {station_id}: {e}")
                continue
        
        logger.info(f"Fetching weather data for station {station_id} from {start_date} to {end_date}")
        
        try:
            if not all_results:
                logger.warning(f"No data found for station {station_id}")
                return pd.DataFrame()
            
            # Parse results into DataFrame
            records = []
            for record in all_results:
                datatype = record.get("datatype")
                date = record.get("date")
                value = record.get("value")
                
                # Skip records with missing date or value
                # Also check for quality flags in attributes (e.g., missing data indicators)
                if date and value is not None:
                    # Parse date (handle both "2024-01-01T00:00:00" and "2024-01-01" formats)
                    try:
                        if "T" in str(date):
                            date_parsed = pd.to_datetime(date.split("T")[0])
                        else:
                            date_parsed = pd.to_datetime(date)
                        
                        records.append({
                            "date": date_parsed,
                            "station_id": station_id,
                            "datatype": datatype,
                            "value": value
                        })
                    except (ValueError, TypeError) as e:
                        logger.debug(f"Skipping record with invalid date: {date}, error: {e}")
                        continue
            
            if not records:
                return pd.DataFrame()
            
            df = pd.DataFrame(records)
            
            # Pivot to get separate columns for each datatype
            df_pivot = df.pivot_table(
                index=["date", "station_id"],
                columns="datatype",
                values="value",
                aggfunc="first"
            ).reset_index()
            
            # Rename columns
            df_pivot.columns.name = None
            if "TMAX" in df_pivot.columns:
                df_pivot = df_pivot.rename(columns={"TMAX": "tmax"})
            if "TMIN" in df_pivot.columns:
                df_pivot = df_pivot.rename(columns={"TMIN": "tmin"})
            if "AWND" in df_pivot.columns:
                df_pivot = df_pivot.rename(columns={"AWND": "awnd"})
            
            # Convert temperatures from tenths of degrees C to degrees F
            # NOAA returns temperatures in tenths of degrees Celsius when units="standard"
            # Verified: raw value 65.0 = 6.5°C = 43.7°F
            # Formula: F = (C * 9/5) + 32, where C = raw_value / 10.0
            if "tmax" in df_pivot.columns:
                # Handle NaN values
                df_pivot["tmax"] = df_pivot["tmax"].apply(
                    lambda x: (x / 10.0) * 9/5 + 32 if pd.notna(x) else np.nan
                )
            if "tmin" in df_pivot.columns:
                df_pivot["tmin"] = df_pivot["tmin"].apply(
                    lambda x: (x / 10.0) * 9/5 + 32 if pd.notna(x) else np.nan
                )
            
            # Calculate average temperature
            if "tmax" in df_pivot.columns and "tmin" in df_pivot.columns:
                df_pivot["tavg"] = (df_pivot["tmax"] + df_pivot["tmin"]) / 2.0
            elif "tmax" in df_pivot.columns:
                df_pivot["tavg"] = df_pivot["tmax"]
            elif "tmin" in df_pivot.columns:
                df_pivot["tavg"] = df_pivot["tmin"]
            
            # Wind speed is already in m/s from NOAA (with units="standard")
            # Verified: raw value 7.4 m/s = 16.6 mph
            # Convert to mph for consistency: 1 m/s = 2.237 mph
            if "awnd" in df_pivot.columns:
                # Handle NaN values
                df_pivot["awnd"] = df_pivot["awnd"].apply(
                    lambda x: x * 2.237 if pd.notna(x) else np.nan
                )
            
            # Sort by date
            df_pivot = df_pivot.sort_values("date").reset_index(drop=True)
            
            logger.info(f"Retrieved {len(df_pivot)} daily records for station {station_id}")
            
            return df_pivot
            
        except Exception as e:
            logger.error(f"Error fetching weather data for station {station_id}: {e}")
            return pd.DataFrame()
    
    def get_aggregate_daily_weather(
        self, 
        start_date: str, 
        end_date: str,
        region: str = "CONUS"
    ) -> pd.DataFrame:
        """
        Get aggregated daily weather data across multiple stations.
        
        Args:
            start_date: Start date in YYYY-MM-DD format
            end_date: End date in YYYY-MM-DD format
            region: Region identifier (default: "CONUS")
            
        Returns:
            DataFrame with columns: date, region, temperature, wind_speed
        """
        logger.info(f"Fetching aggregate weather data from {len(self.stations)} stations")
        
        all_station_data = []
        
        for station_id in self.stations:
            try:
                station_df = self.get_daily_station_weather(station_id, start_date, end_date)
                if not station_df.empty:
                    all_station_data.append(station_df)
            except Exception as e:
                logger.warning(f"Failed to fetch data for station {station_id}: {e}")
                continue
        
        if not all_station_data:
            logger.warning("No station data retrieved")
            return pd.DataFrame()
        
        # Combine all station data
        combined_df = pd.concat(all_station_data, ignore_index=True)
        
        # Group by date and compute averages
        # Use mean() which automatically handles NaN values (excludes them from calculation)
        # This ensures that if some stations have missing data, we still get a valid average
        agg_df = combined_df.groupby("date").agg({
            "tavg": "mean",  # Average temperature across stations (excludes NaN)
            "awnd": "mean"   # Average wind speed across stations (excludes NaN)
        }).reset_index()
        
        # Drop rows where both temperature and wind_speed are NaN (no data for that date)
        agg_df = agg_df.dropna(subset=["tavg", "awnd"], how="all")
        
        # Rename columns
        agg_df = agg_df.rename(columns={
            "tavg": "temperature",
            "awnd": "wind_speed"
        })
        
        # Add region column
        agg_df["region"] = region
        
        # Reorder columns
        agg_df = agg_df[["date", "region", "temperature", "wind_speed"]]
        
        # Sort by date
        agg_df = agg_df.sort_values("date").reset_index(drop=True)
        
        logger.info(f"Retrieved {len(agg_df)} aggregated daily weather records")
        
        return agg_df


def main():
    """Test the NOAA weather client."""
    import logging
    logging.basicConfig(level=logging.INFO)
    
    client = NOAAWeatherClient()
    
    if not client.token:
        print("ERROR: NOAA_CDO_TOKEN not set in environment")
        print("Please set it with: export NOAA_CDO_TOKEN='WmnnsvLsydLYxTVhCqgOFKBNYKJDVQnO'")
        return
    
    print("Testing NOAA Weather Client...")
    
    # Test with a small date range
    start_date = "2024-01-01"
    end_date = "2024-01-10"
    
    print(f"\n1. Testing get_daily_station_weather ({start_date} to {end_date}):")
    station_df = client.get_daily_station_weather("USW00012960", start_date, end_date)
    print(f"   Retrieved {len(station_df)} records")
    if not station_df.empty:
        print(f"   Columns: {list(station_df.columns)}")
        print(f"   Sample data:")
        print(station_df.head())
    
    print(f"\n2. Testing get_aggregate_daily_weather ({start_date} to {end_date}):")
    agg_df = client.get_aggregate_daily_weather(start_date, end_date)
    print(f"   Retrieved {len(agg_df)} records")
    if not agg_df.empty:
        print(f"   Columns: {list(agg_df.columns)}")
        print(f"   Sample data:")
        print(agg_df.head())
        print(f"\n   Temperature range: {agg_df['temperature'].min():.1f} to {agg_df['temperature'].max():.1f} °F")
        print(f"   Wind speed range: {agg_df['wind_speed'].min():.1f} to {agg_df['wind_speed'].max():.1f} mph")


if __name__ == "__main__":
    main()

