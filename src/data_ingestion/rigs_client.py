"""
Baker Hughes Rig Count data ingestion.
Reads weekly U.S. gas rig count data from Excel file.
"""
import pandas as pd
import numpy as np
from typing import Optional
from datetime import datetime
from pathlib import Path
import logging

logger = logging.getLogger(__name__)

# Path to Baker Hughes Excel file
EXCEL_PATH = Path(__file__).parent.parent.parent / "data" / "baker_hughes_weekly_rigs_2013_2025.xlsx"


class RigCountClient:
    """
    Client for loading and processing Baker Hughes weekly rig count data.
    
    Data source: Baker Hughes North America Weekly Rig Count Report (Excel)
    Filters applied in code (not relying on Excel slicers):
    - Country == "UNITED STATES"
    - DrillFor == "Gas"
    
    Returns weekly aggregated U.S. gas rig counts with land/offshore breakdown.
    """
    
    def __init__(self, excel_path: Optional[Path] = None):
        """
        Initialize rig count client.
        
        Args:
            excel_path: Optional path to Excel file (defaults to standard location)
        """
        self.excel_path = excel_path or EXCEL_PATH
    
    def load_raw(self) -> pd.DataFrame:
        """
        Load raw data from Baker Hughes Excel file.
        
        Returns:
            Raw DataFrame with all columns from the Excel file
        """
        if not self.excel_path.exists():
            logger.error(f"Rig count Excel file not found: {self.excel_path}")
            return pd.DataFrame()
        
        try:
            # Read Excel file - header is at row 10 (0-indexed)
            # Columns: Country, County, Basin, GOM, DrillFor, Location, State/Province, 
            #          Trajectory, Year, Month, US_PublishDate, Rig Count Value
            xl_file = pd.ExcelFile(self.excel_path)
            
            # Use first sheet (usually "NAM Weekly")
            sheet_name = xl_file.sheet_names[0]
            
            logger.info(f"Reading rig count data from sheet: {sheet_name} (header at row 10)")
            
            # Read the sheet with header at row 10
            df = pd.read_excel(self.excel_path, sheet_name=sheet_name, header=10)
            
            # Normalize column names: strip whitespace, replace spaces with underscores
            df.columns = [str(col).strip().replace(' ', '_') for col in df.columns]
            
            # Remove any rows that are completely empty
            df = df.dropna(how='all')
            
            logger.info(f"Loaded {len(df)} raw rows with columns: {list(df.columns)}")
            
            return df
            
        except Exception as e:
            logger.error(f"Error loading rig count Excel file: {e}")
            import traceback
            logger.error(traceback.format_exc())
            return pd.DataFrame()
    
    def get_us_gas_weekly(self) -> pd.DataFrame:
        """
        Get weekly U.S. gas rig counts aggregated by publish date.
        
        Filters applied:
        - Country == "UNITED STATES"
        - DrillFor == "Gas"
        
        Aggregates by US_PublishDate to compute:
        - gas_rigs: Total gas rigs for that week
        - gas_rigs_land: Gas rigs on land
        - gas_rigs_offshore: Gas rigs offshore or inland water
        
        Returns:
            DataFrame with columns: date, region, gas_rigs, gas_rigs_land, gas_rigs_offshore
        """
        # Load raw data
        df = self.load_raw()
        
        if df.empty:
            logger.warning("No raw data loaded, returning empty DataFrame")
            return pd.DataFrame()
        
        # Normalize string columns for filtering (handle case and whitespace)
        string_cols = ['Country', 'DrillFor', 'Location']
        for col in string_cols:
            if col in df.columns:
                df[col] = df[col].astype(str).str.strip()
        
        # Apply hard filters (not relying on Excel slicers)
        # Filter for United States
        if 'Country' in df.columns:
            df_filtered = df[df['Country'].str.upper() == 'UNITED STATES'].copy()
            logger.info(f"After Country='UNITED STATES' filter: {len(df_filtered)} rows (from {len(df)} total)")
        else:
            logger.error("'Country' column not found, cannot filter by country")
            return pd.DataFrame()
        
        # Filter for Gas rigs
        if 'DrillFor' in df_filtered.columns:
            df_filtered = df_filtered[df_filtered['DrillFor'].str.upper() == 'GAS'].copy()
            logger.info(f"After DrillFor='Gas' filter: {len(df_filtered)} rows")
        else:
            logger.error("'DrillFor' column not found, cannot filter by drill type")
            return pd.DataFrame()
        
        if df_filtered.empty:
            logger.warning("No data after applying filters")
            return pd.DataFrame()
        
        # Parse US_PublishDate to datetime
        date_col = 'US_PublishDate'
        if date_col not in df_filtered.columns:
            logger.error(f"'{date_col}' column not found. Available columns: {list(df_filtered.columns)}")
            return pd.DataFrame()
        
        logger.info(f"Using date column: {date_col}")
        
        # Parse date column - handle various formats (e.g., "1/4/13" -> datetime)
        df_filtered['date'] = pd.to_datetime(df_filtered[date_col], errors='coerce')
        
        # Remove rows with invalid dates
        invalid_dates = df_filtered['date'].isna().sum()
        if invalid_dates > 0:
            logger.warning(f"Dropping {invalid_dates} rows with invalid dates")
            df_filtered = df_filtered.dropna(subset=['date'])
        
        # Normalize to date only (remove time component)
        df_filtered['date'] = df_filtered['date'].dt.date
        
        # Find rig count value column
        rig_count_col = 'Rig_Count_Value'
        if rig_count_col not in df_filtered.columns:
            # Try alternative names
            for col in ['Rig_Count', 'Count', 'Value', 'Rigs']:
                if col in df_filtered.columns:
                    rig_count_col = col
                    break
        
        if rig_count_col not in df_filtered.columns:
            logger.error(f"Rig count value column not found. Available columns: {list(df_filtered.columns)}")
            return pd.DataFrame()
        
        logger.info(f"Using rig count column: {rig_count_col}")
        
        # Convert rig count to numeric
        df_filtered[rig_count_col] = pd.to_numeric(df_filtered[rig_count_col], errors='coerce')
        df_filtered = df_filtered.dropna(subset=[rig_count_col])
        
        # Identify Location column for land/offshore breakdown
        location_col = 'Location'
        if location_col not in df_filtered.columns:
            logger.warning("Location column not found, cannot compute land/offshore breakdown")
            # Still aggregate total gas rigs
            result_df = df_filtered.groupby('date').agg({
                rig_count_col: 'sum'
            }).reset_index()
            result_df = result_df.rename(columns={rig_count_col: 'gas_rigs'})
            result_df['gas_rigs_land'] = 0
            result_df['gas_rigs_offshore'] = 0
        else:
            # Normalize location values
            df_filtered[location_col] = df_filtered[location_col].astype(str).str.strip()
            
            # Aggregate by date and location
            location_agg = df_filtered.groupby(['date', location_col]).agg({
                rig_count_col: 'sum'
            }).reset_index()
            
            # Separate land and offshore
            # Land: Location == "Land"
            # Offshore: Location in {"Offshore", "Inland Water"}
            land_data = location_agg[location_agg[location_col].str.upper() == 'LAND'].groupby('date')[rig_count_col].sum().reset_index()
            land_data = land_data.rename(columns={rig_count_col: 'gas_rigs_land'})
            
            offshore_data = location_agg[location_agg[location_col].str.upper().isin(['OFFSHORE', 'INLAND WATER'])].groupby('date')[rig_count_col].sum().reset_index()
            offshore_data = offshore_data.rename(columns={rig_count_col: 'gas_rigs_offshore'})
            
            # Total gas rigs by date
            total_data = df_filtered.groupby('date')[rig_count_col].sum().reset_index()
            total_data = total_data.rename(columns={rig_count_col: 'gas_rigs'})
            
            # Merge all aggregations
            result_df = total_data.merge(land_data, on='date', how='left')
            result_df = result_df.merge(offshore_data, on='date', how='left')
            
            # Fill NaN with 0 (for dates that have no land or no offshore rigs)
            result_df['gas_rigs_land'] = result_df['gas_rigs_land'].fillna(0)
            result_df['gas_rigs_offshore'] = result_df['gas_rigs_offshore'].fillna(0)
        
        # Add region column
        result_df['region'] = 'US'
        
        # Convert date back to datetime for consistency
        result_df['date'] = pd.to_datetime(result_df['date'])
        
        # Ensure integer types for rig counts
        result_df['gas_rigs'] = result_df['gas_rigs'].fillna(0).astype(int)
        result_df['gas_rigs_land'] = result_df['gas_rigs_land'].fillna(0).astype(int)
        result_df['gas_rigs_offshore'] = result_df['gas_rigs_offshore'].fillna(0).astype(int)
        
        # Sort by date
        result_df = result_df.sort_values('date').reset_index(drop=True)
        
        # Reorder columns
        result_df = result_df[['date', 'region', 'gas_rigs', 'gas_rigs_land', 'gas_rigs_offshore']]
        
        # Sanity checks
        self._validate_rig_counts(result_df)
        
        logger.info(f"Aggregated {len(result_df)} weekly U.S. gas rig count records")
        logger.info(f"Date range: {result_df['date'].min()} to {result_df['date'].max()}")
        
        return result_df
    
    def _validate_rig_counts(self, df: pd.DataFrame) -> None:
        """
        Perform sanity checks on aggregated rig count data.
        
        Args:
            df: Aggregated rig count DataFrame
        """
        if df.empty:
            return
        
        # Check 1: gas_rigs >= gas_rigs_land + gas_rigs_offshore
        total_check = df['gas_rigs'] >= (df['gas_rigs_land'] + df['gas_rigs_offshore'])
        if not total_check.all():
            invalid_rows = df[~total_check]
            logger.warning(f"Found {len(invalid_rows)} rows where gas_rigs < gas_rigs_land + gas_rigs_offshore")
            logger.warning(f"Sample invalid rows:\n{invalid_rows.head()}")
        else:
            logger.debug("✓ Sanity check passed: gas_rigs >= gas_rigs_land + gas_rigs_offshore")
        
        # Check 2: Date range is within expected 2013-2025
        min_date = df['date'].min()
        max_date = df['date'].max()
        
        if min_date < pd.to_datetime('2013-01-01'):
            logger.warning(f"Earliest date {min_date} is before 2013")
        if max_date > pd.to_datetime('2025-12-31'):
            logger.warning(f"Latest date {max_date} is after 2025")
        
        logger.info(f"Date range validation: {min_date.date()} to {max_date.date()}")
        
        # Check 3: No negative rig counts
        negative_rigs = (df[['gas_rigs', 'gas_rigs_land', 'gas_rigs_offshore']] < 0).any(axis=1).sum()
        if negative_rigs > 0:
            logger.warning(f"Found {negative_rigs} rows with negative rig counts")
        else:
            logger.debug("✓ Sanity check passed: No negative rig counts")


def main():
    """Test the rig count client."""
    import logging
    logging.basicConfig(level=logging.INFO)
    
    client = RigCountClient()
    
    print("=" * 70)
    print("Testing Rig Count Client")
    print("=" * 70)
    
    # Test raw loading
    print("\n1. Testing load_raw()...")
    raw_df = client.load_raw()
    if not raw_df.empty:
        print(f"   ✓ Loaded {len(raw_df)} raw rows")
        print(f"   Columns: {list(raw_df.columns)[:10]}...")
    else:
        print("   ✗ Failed to load raw data")
        return
    
    # Test US gas weekly aggregation
    print("\n2. Testing get_us_gas_weekly()...")
    gas_df = client.get_us_gas_weekly()
    
    if not gas_df.empty:
        print(f"   ✓ Aggregated {len(gas_df)} weekly records")
        print(f"   Date range: {gas_df['date'].min()} to {gas_df['date'].max()}")
        print(f"\n   Sample data:")
        print(gas_df.head(10))
        print(f"\n   Summary statistics:")
        print(f"   Total gas rigs - Min: {gas_df['gas_rigs'].min()}, Max: {gas_df['gas_rigs'].max()}, Mean: {gas_df['gas_rigs'].mean():.1f}")
        print(f"   Land rigs - Min: {gas_df['gas_rigs_land'].min()}, Max: {gas_df['gas_rigs_land'].max()}, Mean: {gas_df['gas_rigs_land'].mean():.1f}")
        print(f"   Offshore rigs - Min: {gas_df['gas_rigs_offshore'].min()}, Max: {gas_df['gas_rigs_offshore'].max()}, Mean: {gas_df['gas_rigs_offshore'].mean():.1f}")
    else:
        print("   ✗ Failed to aggregate data")


if __name__ == "__main__":
    main()

