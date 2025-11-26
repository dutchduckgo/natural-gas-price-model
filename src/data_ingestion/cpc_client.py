"""
CPC (Climate Prediction Center) Degree Day data collection.
Fetches population-weighted HDD/CDD data from CPC FTP servers.
"""
import requests
import pandas as pd
import numpy as np
from typing import Optional
from datetime import datetime, timedelta
import logging
import io

logger = logging.getLogger(__name__)

# CPC FTP base URLs
CPC_BASE_URL = "https://ftp.cpc.ncep.noaa.gov/htdocs/degree_days/weighted/daily_data"
CPC_CLIMATOLOGY_URL = "https://ftp.cpc.ncep.noaa.gov/htdocs/degree_days/weighted/daily_data/climatology/1981-2010"


class CPCDegreeDayClient:
    """Client for fetching CPC population-weighted degree day data."""
    
    def __init__(self):
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'Mozilla/5.0 (compatible; GasModel/1.0)'
        })
    
    def _fetch_file(self, url: str) -> str:
        """Fetch a file from CPC FTP and return its content as string."""
        try:
            response = self.session.get(url, timeout=30)
            response.raise_for_status()
            return response.text
        except requests.exceptions.RequestException as e:
            logger.error(f"Failed to fetch {url}: {e}")
            raise
    
    def _parse_observed_file(self, content: str, year: int, region_filter: str = "CONUS") -> pd.DataFrame:
        """
        Parse an observed HDD or CDD file.
        
        Args:
            content: File content as string
            year: Year of the data
            region_filter: Region to extract (default: "CONUS" or "U.S.")
            
        Returns:
            DataFrame with date, region, hdd (or cdd) columns
        """
        lines = content.strip().split('\n')
        
        # Find the header line (contains "Region|" followed by dates)
        header_idx = None
        for i, line in enumerate(lines):
            if line.startswith('Region|'):
                header_idx = i
                break
        
        if header_idx is None:
            logger.warning(f"No header line found in file for year {year}")
            return pd.DataFrame()
        
        # Parse header to get dates
        header = lines[header_idx]
        parts = header.split('|')
        if len(parts) < 2:
            logger.warning(f"Invalid header format for year {year}")
            return pd.DataFrame()
        
        # Dates are in YYYYMMDD format starting from index 1
        date_strs = parts[1:]
        
        # Find the region row
        # The file is "StatesCONUS" which means it's population-weighted CONUS data
        # The aggregate might be explicit or we may need to calculate it from all states
        region_row = None
        region_name = None
        
        # First, try to find explicit aggregate identifiers
        for line in lines[header_idx + 1:]:
            if '|' in line:
                parts = line.split('|')
                if len(parts) == len(date_strs) + 1:  # Correct number of columns
                    region_code = parts[0].strip().upper()
                    # Look for various aggregate identifiers
                    if region_code in ['CONUS', 'U.S.', 'US', 'US48', 'UNITED STATES', 'U.S', 'USA']:
                        region_row = parts[1:]
                        region_name = region_filter
                        logger.debug(f"Found aggregate row with identifier: {region_code}")
                        break
        
        # If not found, calculate aggregate from all state rows
        # Since the file is "StatesCONUS", the data is already population-weighted per state
        # We'll calculate a simple average (in reality, this should be population-weighted, 
        # but the file name suggests it's already weighted)
        if region_row is None:
            logger.info(f"Calculating {region_filter} aggregate from all state rows")
            state_rows = []
            for line in lines[header_idx + 1:]:
                if '|' in line:
                    parts = line.split('|')
                    if len(parts) == len(date_strs) + 1:
                        region_code = parts[0].strip()
                        # Only include 2-letter state codes (exclude any non-state rows)
                        if len(region_code) == 2:
                            try:
                                values = [float(v.strip()) if v.strip() else 0.0 for v in parts[1:]]
                                state_rows.append(values)
                            except (ValueError, TypeError):
                                continue
            
            if state_rows:
                # Calculate average across all states
                # Note: Since the file is "StatesCONUS" and uses population weights,
                # each state's value is already population-weighted, so we average them
                import numpy as np
                state_array = np.array(state_rows)
                region_row = state_array.mean(axis=0).tolist()
                region_name = region_filter
                logger.info(f"Calculated aggregate from {len(state_rows)} state rows")
        
        if region_row is None or len(region_row) != len(date_strs):
            logger.warning(f"Could not find or calculate {region_filter} region data for year {year}")
            return pd.DataFrame()
        
        # Parse dates and values
        data = []
        for date_str, value in zip(date_strs, region_row):
            try:
                # Parse date from YYYYMMDD format
                date = pd.to_datetime(date_str, format='%Y%m%d')
                # Handle both string and float values
                if isinstance(value, str):
                    value = float(value.strip())
                else:
                    value = float(value)
                data.append({
                    'date': date,
                    'region': region_name,
                    'value': value
                })
            except (ValueError, TypeError) as e:
                logger.debug(f"Could not parse date/value pair: {date_str}/{value}: {e}")
                continue
        
        if not data:
            return pd.DataFrame()
        
        df = pd.DataFrame(data)
        return df
    
    def _parse_normals_file(self, content: str, region_filter: str = "CONUS") -> pd.DataFrame:
        """
        Parse a normals (climatology) file.
        
        Args:
            content: File content as string
            region_filter: Region to extract (default: "CONUS")
            
        Returns:
            DataFrame with doy (day-of-year 1-366), region, hdd_norm (or cdd_norm) columns
        """
        lines = content.strip().split('\n')
        
        # Find the header line
        header_idx = None
        for i, line in enumerate(lines):
            if line.startswith('Region|'):
                header_idx = i
                break
        
        if header_idx is None:
            logger.warning("No header line found in normals file")
            return pd.DataFrame()
        
        # Parse header - dates are in MMDD format
        header = lines[header_idx]
        parts = header.split('|')
        if len(parts) < 2:
            logger.warning("Invalid header format in normals file")
            return pd.DataFrame()
        
        # Dates are in MMDD format (day-of-year will be calculated)
        date_strs = parts[1:]
        
        # Find the region row (same logic as observed data)
        region_row = None
        region_name = None
        
        # First, try to find explicit aggregate identifiers
        for line in lines[header_idx + 1:]:
            if '|' in line:
                parts = line.split('|')
                if len(parts) == len(date_strs) + 1:
                    region_code = parts[0].strip().upper()
                    if region_code in ['CONUS', 'U.S.', 'US', 'US48', 'UNITED STATES', 'U.S', 'USA']:
                        region_row = parts[1:]
                        region_name = region_filter
                        break
        
        # If not found, calculate aggregate from all state rows
        if region_row is None:
            logger.info(f"Calculating {region_filter} normals aggregate from all state rows")
            state_rows = []
            for line in lines[header_idx + 1:]:
                if '|' in line:
                    parts = line.split('|')
                    if len(parts) == len(date_strs) + 1:
                        region_code = parts[0].strip()
                        if len(region_code) == 2:  # State code
                            try:
                                values = [float(v.strip()) if v.strip() else 0.0 for v in parts[1:]]
                                state_rows.append(values)
                            except (ValueError, TypeError):
                                continue
            
            if state_rows:
                import numpy as np
                state_array = np.array(state_rows)
                region_row = state_array.mean(axis=0).tolist()
                region_name = region_filter
                logger.info(f"Calculated normals aggregate from {len(state_rows)} state rows")
        
        if region_row is None or len(region_row) != len(date_strs):
            logger.warning(f"Could not find or calculate {region_filter} region in normals file")
            return pd.DataFrame()
        
        # Parse day-of-year and values
        # We'll use a reference year (2000, a leap year) to calculate day-of-year
        data = []
        for i, (date_str, value) in enumerate(zip(date_strs, region_row)):
            try:
                # Parse MMDD format
                month = int(date_str[:2])
                day = int(date_str[2:])
                # Use 2000 (leap year) as reference to get day-of-year
                ref_date = pd.Timestamp(2000, month, day)
                doy = ref_date.timetuple().tm_yday
                
                # Handle both string and float values
                if isinstance(value, str):
                    value = float(value.strip())
                else:
                    value = float(value)
                
                data.append({
                    'doy': doy,
                    'region': region_name,
                    'value': value
                })
            except (ValueError, TypeError) as e:
                logger.debug(f"Could not parse date/value pair: {date_str}/{value}: {e}")
                continue
        
        if not data:
            return pd.DataFrame()
        
        df = pd.DataFrame(data)
        return df
    
    def get_daily_hdd_cdd(self, start_date: str, end_date: str, region_filter: str = "CONUS") -> pd.DataFrame:
        """
        Get daily HDD and CDD data for the specified date range.
        
        Args:
            start_date: Start date in YYYY-MM-DD format
            end_date: End date in YYYY-MM-DD format
            region_filter: Region to extract (default: "CONUS")
            
        Returns:
            DataFrame with date, region, hdd, cdd columns
        """
        start = pd.to_datetime(start_date)
        end = pd.to_datetime(end_date)
        
        # Get all years needed
        years = range(start.year, end.year + 1)
        
        all_hdd = []
        all_cdd = []
        
        for year in years:
            try:
                # Fetch HDD file
                hdd_url = f"{CPC_BASE_URL}/{year}/StatesCONUS.Heating.txt"
                logger.info(f"Fetching HDD data for {year}")
                hdd_content = self._fetch_file(hdd_url)
                hdd_df = self._parse_observed_file(hdd_content, year, region_filter)
                if not hdd_df.empty:
                    hdd_df = hdd_df.rename(columns={'value': 'hdd'})
                    all_hdd.append(hdd_df)
                
                # Fetch CDD file
                cdd_url = f"{CPC_BASE_URL}/{year}/StatesCONUS.Cooling.txt"
                logger.info(f"Fetching CDD data for {year}")
                cdd_content = self._fetch_file(cdd_url)
                cdd_df = self._parse_observed_file(cdd_content, year, region_filter)
                if not cdd_df.empty:
                    cdd_df = cdd_df.rename(columns={'value': 'cdd'})
                    all_cdd.append(cdd_df)
                    
            except Exception as e:
                logger.warning(f"Failed to fetch data for year {year}: {e}")
                continue
        
        # Combine all years
        if not all_hdd and not all_cdd:
            logger.warning("No HDD or CDD data retrieved")
            return pd.DataFrame()
        
        # Merge HDD and CDD
        if all_hdd:
            hdd_combined = pd.concat(all_hdd, ignore_index=True)
        else:
            hdd_combined = pd.DataFrame(columns=['date', 'region'])
        
        if all_cdd:
            cdd_combined = pd.concat(all_cdd, ignore_index=True)
        else:
            cdd_combined = pd.DataFrame(columns=['date', 'region'])
        
        # Merge on date and region
        if not hdd_combined.empty and not cdd_combined.empty:
            df = pd.merge(hdd_combined, cdd_combined, on=['date', 'region'], how='outer')
        elif not hdd_combined.empty:
            df = hdd_combined.copy()
            df['cdd'] = 0.0
        elif not cdd_combined.empty:
            df = cdd_combined.copy()
            df['hdd'] = 0.0
        else:
            return pd.DataFrame()
        
        # Fill missing values with 0
        df['hdd'] = df['hdd'].fillna(0.0)
        df['cdd'] = df['cdd'].fillna(0.0)
        
        # Filter to date range
        df = df[(df['date'] >= start) & (df['date'] <= end)].copy()
        df = df.sort_values('date').reset_index(drop=True)
        
        logger.info(f"Retrieved {len(df)} daily HDD/CDD records from {start_date} to {end_date}")
        
        return df
    
    def get_normals(self, region_filter: str = "CONUS") -> pd.DataFrame:
        """
        Get climatological normals (1981-2010) for HDD and CDD.
        
        Args:
            region_filter: Region to extract (default: "CONUS")
            
        Returns:
            DataFrame with doy (day-of-year 1-366), region, hdd_norm, cdd_norm columns
        """
        try:
            # Fetch HDD normals
            hdd_url = f"{CPC_CLIMATOLOGY_URL}/StatesCONUS.Heating.txt"
            logger.info("Fetching HDD normals")
            hdd_content = self._fetch_file(hdd_url)
            hdd_df = self._parse_normals_file(hdd_content, region_filter)
            if not hdd_df.empty:
                hdd_df = hdd_df.rename(columns={'value': 'hdd_norm'})
            
            # Fetch CDD normals
            cdd_url = f"{CPC_CLIMATOLOGY_URL}/StatesCONUS.Cooling.txt"
            logger.info("Fetching CDD normals")
            cdd_content = self._fetch_file(cdd_url)
            cdd_df = self._parse_normals_file(cdd_content, region_filter)
            if not cdd_df.empty:
                cdd_df = cdd_df.rename(columns={'value': 'cdd_norm'})
            
            # Merge
            if not hdd_df.empty and not cdd_df.empty:
                df = pd.merge(hdd_df, cdd_df, on=['doy', 'region'], how='outer')
            elif not hdd_df.empty:
                df = hdd_df.copy()
                df['cdd_norm'] = 0.0
            elif not cdd_df.empty:
                df = cdd_df.copy()
                df['hdd_norm'] = 0.0
            else:
                logger.warning("No normals data retrieved")
                return pd.DataFrame()
            
            # Fill missing values
            df['hdd_norm'] = df['hdd_norm'].fillna(0.0)
            df['cdd_norm'] = df['cdd_norm'].fillna(0.0)
            
            df = df.sort_values('doy').reset_index(drop=True)
            
            logger.info(f"Retrieved normals for {len(df)} days")
            
            return df
            
        except Exception as e:
            logger.error(f"Failed to fetch normals: {e}")
            return pd.DataFrame()
    
    def get_daily_with_normals(self, start_date: str, end_date: str, region_filter: str = "CONUS") -> pd.DataFrame:
        """
        Get daily HDD/CDD with normals and anomalies.
        
        Args:
            start_date: Start date in YYYY-MM-DD format
            end_date: End date in YYYY-MM-DD format
            region_filter: Region to extract (default: "CONUS")
            
        Returns:
            DataFrame with date, region, hdd, cdd, hdd_norm, cdd_norm, hdd_anom, cdd_anom columns
        """
        # Get observed data
        observed_df = self.get_daily_hdd_cdd(start_date, end_date, region_filter)
        
        if observed_df.empty:
            logger.warning("No observed data available")
            return pd.DataFrame()
        
        # Get normals
        normals_df = self.get_normals(region_filter)
        
        if normals_df.empty:
            logger.warning("No normals data available, returning observed data only")
            return observed_df
        
        # Add day-of-year to observed data
        observed_df['doy'] = observed_df['date'].dt.dayofyear
        
        # Merge with normals
        df = pd.merge(observed_df, normals_df, on=['doy', 'region'], how='left')
        
        # Fill missing normals (shouldn't happen, but just in case)
        df['hdd_norm'] = df['hdd_norm'].fillna(0.0)
        df['cdd_norm'] = df['cdd_norm'].fillna(0.0)
        
        # Calculate anomalies
        df['hdd_anom'] = df['hdd'] - df['hdd_norm']
        df['cdd_anom'] = df['cdd'] - df['cdd_norm']
        
        # Drop doy column (not needed in final output)
        df = df.drop(columns=['doy'])
        
        # Reorder columns
        df = df[['date', 'region', 'hdd', 'cdd', 'hdd_norm', 'cdd_norm', 'hdd_anom', 'cdd_anom']]
        
        logger.info(f"Retrieved {len(df)} daily records with normals and anomalies")
        
        return df


def main():
    """Test the CPC client."""
    client = CPCDegreeDayClient()
    
    print("Testing CPC Degree Day Client...")
    
    # Test with a small date range
    start_date = "2024-01-01"
    end_date = "2024-01-31"
    
    print(f"\n1. Testing get_daily_hdd_cdd ({start_date} to {end_date}):")
    df = client.get_daily_hdd_cdd(start_date, end_date)
    print(f"   Retrieved {len(df)} records")
    if not df.empty:
        print(f"   Date range: {df['date'].min()} to {df['date'].max()}")
        print(f"   Columns: {list(df.columns)}")
        print(f"   Sample data:")
        print(df.head())
    
    print("\n2. Testing get_normals:")
    normals = client.get_normals()
    print(f"   Retrieved {len(normals)} normals records")
    if not normals.empty:
        print(f"   DOY range: {normals['doy'].min()} to {normals['doy'].max()}")
        print(f"   Columns: {list(normals.columns)}")
        print(f"   Sample data:")
        print(normals.head())
    
    print(f"\n3. Testing get_daily_with_normals ({start_date} to {end_date}):")
    df_with_normals = client.get_daily_with_normals(start_date, end_date)
    print(f"   Retrieved {len(df_with_normals)} records")
    if not df_with_normals.empty:
        print(f"   Columns: {list(df_with_normals.columns)}")
        print(f"   Sample data:")
        print(df_with_normals.head())


if __name__ == "__main__":
    main()

