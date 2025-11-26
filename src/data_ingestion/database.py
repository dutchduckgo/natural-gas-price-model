"""
Database schema and storage layer for natural gas price model.
"""
import duckdb
import pandas as pd
from pathlib import Path
from typing import Dict, List, Optional, Union
import logging
from config import DATABASE_URL, RAW_DATA_DIR, PROCESSED_DATA_DIR

logger = logging.getLogger(__name__)


class GasModelDatabase:
    """Database manager for natural gas price model."""
    
    def __init__(self, db_url: str = None):
        self.db_url = db_url or DATABASE_URL
        self.conn = None
        self._connect()
        self._create_schema()
        
    def _connect(self):
        """Connect to database."""
        try:
            # Handle duckdb:// URL scheme - DuckDB expects just a file path
            if self.db_url.startswith("duckdb:///"):
                # Remove duckdb:/// prefix and use the path directly
                db_path = self.db_url.replace("duckdb:///", "")
            elif self.db_url.startswith("duckdb://"):
                # Handle duckdb:// format
                db_path = self.db_url.replace("duckdb://", "")
            else:
                db_path = self.db_url
            
            self.conn = duckdb.connect(db_path)
            logger.info(f"Connected to database: {db_path}")
        except Exception as e:
            logger.error(f"Failed to connect to database: {e}")
            raise
    
    def _create_schema(self):
        """Create database schema."""
        schema_sql = """
        -- Prices table (daily grain)
        CREATE TABLE IF NOT EXISTS prices (
            date DATE,
            hub VARCHAR(50),
            spot_price DECIMAL(10,4),
            front_month DECIMAL(10,4),
            m1_m2_spread DECIMAL(10,4),
            PRIMARY KEY (date, hub)
        );
        
        -- Storage table (weekly grain)
        CREATE TABLE IF NOT EXISTS storage_weekly (
            report_date DATE,
            working_gas DECIMAL(15,2),
            five_year_avg DECIMAL(15,2),
            yoy_deviation DECIMAL(15,2),
            wow_change DECIMAL(15,2),
            PRIMARY KEY (report_date)
        );
        
        -- Production table (monthly grain)
        CREATE TABLE IF NOT EXISTS production_monthly (
            month DATE,
            dry_gas_bcfpd DECIMAL(10,2),
            PRIMARY KEY (month)
        );
        
        -- LNG table (monthly grain)
        CREATE TABLE IF NOT EXISTS lng_monthly (
            month DATE,
            exports_bcf DECIMAL(10,2),
            avg_price DECIMAL(10,4),
            terminal_notes TEXT,
            lng_capacity_bcfpd DECIMAL(10,2),
            PRIMARY KEY (month)
        );
        
        -- Power burn table (daily grain)
        CREATE TABLE IF NOT EXISTS power_burn (
            date DATE,
            ba VARCHAR(50),
            gas_mwh DECIMAL(10,2),
            total_load_mwh DECIMAL(10,2),
            renewables_mwh DECIMAL(10,2),
            PRIMARY KEY (date, ba)
        );
        
        -- Weather table (daily grain)
        CREATE TABLE IF NOT EXISTS weather_daily (
            date DATE,
            region VARCHAR(50),
            hdd DECIMAL(8,2),
            cdd DECIMAL(8,2),
            hdd_norm DECIMAL(8,2),
            cdd_norm DECIMAL(8,2),
            hdd_anom DECIMAL(8,2),
            cdd_anom DECIMAL(8,2),
            hdd_norm_delta DECIMAL(8,2),
            cdd_norm_delta DECIMAL(8,2),
            temperature DECIMAL(6,2),
            wind_speed DECIMAL(6,2),
            PRIMARY KEY (date, region)
        );
        
        -- Add new columns if they don't exist (for existing databases)
        -- This is idempotent - will not error if columns already exist
        ALTER TABLE weather_daily ADD COLUMN IF NOT EXISTS hdd_norm DECIMAL(8,2);
        ALTER TABLE weather_daily ADD COLUMN IF NOT EXISTS cdd_norm DECIMAL(8,2);
        ALTER TABLE weather_daily ADD COLUMN IF NOT EXISTS hdd_anom DECIMAL(8,2);
        ALTER TABLE weather_daily ADD COLUMN IF NOT EXISTS cdd_anom DECIMAL(8,2);
        ALTER TABLE weather_daily ADD COLUMN IF NOT EXISTS temperature DECIMAL(6,2);
        ALTER TABLE weather_daily ADD COLUMN IF NOT EXISTS wind_speed DECIMAL(6,2);
        
        -- Rigs table (weekly grain)
        -- Create with new schema (date, region as composite primary key)
        CREATE TABLE IF NOT EXISTS rigs_weekly (
            date DATE,
            region TEXT,
            gas_rigs INTEGER,
            gas_rigs_land INTEGER,
            gas_rigs_offshore INTEGER,
            PRIMARY KEY (date, region)
        );
        
        -- Add legacy columns if they don't exist (for backward compatibility)
        -- These may be used by existing queries
        ALTER TABLE rigs_weekly ADD COLUMN IF NOT EXISTS total_rigs INTEGER;
        ALTER TABLE rigs_weekly ADD COLUMN IF NOT EXISTS oil_rigs INTEGER;
        
        -- Events table (daily grain)
        CREATE TABLE IF NOT EXISTS events (
            date DATE,
            label VARCHAR(200),
            lng_cap_offline_bcfpd DECIMAL(10,2),
            PRIMARY KEY (date, label)
        );
        
        -- Features table (daily grain)
        CREATE TABLE IF NOT EXISTS features (
            date DATE,
            feature_name VARCHAR(100),
            feature_value DECIMAL(15,6),
            PRIMARY KEY (date, feature_name)
        );
        
        -- Model predictions table
        CREATE TABLE IF NOT EXISTS predictions (
            date DATE,
            model_name VARCHAR(100),
            horizon_days INTEGER,
            prediction DECIMAL(10,4),
            confidence_lower DECIMAL(10,4),
            confidence_upper DECIMAL(10,4),
            actual_value DECIMAL(10,4),
            error DECIMAL(10,4),
            PRIMARY KEY (date, model_name, horizon_days)
        );
        """
        
        try:
            self.conn.execute(schema_sql)
            logger.info("Database schema created successfully")
        except Exception as e:
            logger.error(f"Failed to create schema: {e}")
            raise
    
    def insert_prices(self, df: pd.DataFrame):
        """Insert price data."""
        self._insert_data("prices", df)
    
    def insert_storage(self, df: pd.DataFrame):
        """Insert storage data."""
        self._insert_data("storage_weekly", df)
    
    def insert_production(self, df: pd.DataFrame):
        """Insert production data."""
        self._insert_data("production_monthly", df)
    
    def insert_lng(self, df: pd.DataFrame):
        """Insert LNG data."""
        self._insert_data("lng_monthly", df)
    
    def insert_power_burn(self, df: pd.DataFrame):
        """Insert power burn data."""
        self._insert_data("power_burn", df)
    
    def insert_weather(self, df: pd.DataFrame):
        """Insert weather data."""
        if df.empty:
            logger.warning("No data to insert into weather_daily")
            return
        
        # Remove existing rows with same primary keys to avoid constraint violations
        if {"date", "region"}.issubset(df.columns):
            unique_keys = df[["date", "region"]].drop_duplicates()
            for _, row in unique_keys.iterrows():
                date_value = pd.to_datetime(row["date"]).strftime("%Y-%m-%d")
                self.conn.execute(
                    "DELETE FROM weather_daily WHERE date = ? AND region = ?",
                    (date_value, row["region"])
                )
        
        self._insert_data("weather_daily", df)
    
    def insert_cpc_degree_days(self, df: pd.DataFrame):
        """
        Insert or upsert CPC daily HDD/CDD (+ normals/anomalies) into weather_daily.
        
        Args:
            df: DataFrame with columns: date, region, hdd, cdd, hdd_norm, cdd_norm, hdd_anom, cdd_anom
        """
        if df.empty:
            logger.warning("No CPC degree day data to insert")
            return
        
        # Ensure required columns exist
        required_cols = ['date', 'region', 'hdd', 'cdd']
        if not all(col in df.columns for col in required_cols):
            missing = [col for col in required_cols if col not in df.columns]
            logger.error(f"Missing required columns: {missing}")
            raise ValueError(f"DataFrame must contain columns: {required_cols}")
        
        # Remove existing rows with same primary keys
        if {"date", "region"}.issubset(df.columns):
            unique_keys = df[["date", "region"]].drop_duplicates()
            for _, row in unique_keys.iterrows():
                date_value = pd.to_datetime(row["date"]).strftime("%Y-%m-%d")
                self.conn.execute(
                    "DELETE FROM weather_daily WHERE date = ? AND region = ?",
                    (date_value, row["region"])
                )
        
        self._insert_data("weather_daily", df)
    
    def insert_historical_weather(self, df: pd.DataFrame):
        """
        Insert or upsert historical temperature and wind into weather_daily.
        
        Merges with existing rows on (date, region) without dropping other columns.
        
        Args:
            df: DataFrame with columns: date, region (optional, defaults to 'CONUS'), 
                temperature, wind_speed
        """
        if df.empty:
            logger.warning("No historical weather data to insert")
            return
        
        # Ensure required columns exist
        required_cols = ['date', 'temperature', 'wind_speed']
        if not all(col in df.columns for col in required_cols):
            missing = [col for col in required_cols if col not in df.columns]
            logger.error(f"Missing required columns: {missing}")
            raise ValueError(f"DataFrame must contain columns: {required_cols}")
        
        # Add region column if not present (default to CONUS)
        if 'region' not in df.columns:
            df = df.copy()
            df['region'] = 'CONUS'
        
        # For each (date, region) combination, update temperature and wind_speed
        # We'll use an UPDATE approach to preserve existing columns
        for _, row in df.iterrows():
            date_value = pd.to_datetime(row["date"]).strftime("%Y-%m-%d")
            region = row["region"]
            temperature = row["temperature"]
            wind_speed = row["wind_speed"]
            
            # Check if row exists
            existing = self.conn.execute(
                "SELECT COUNT(*) FROM weather_daily WHERE date = ? AND region = ?",
                (date_value, region)
            ).fetchone()[0]
            
            if existing > 0:
                # Update existing row
                self.conn.execute(
                    """
                    UPDATE weather_daily 
                    SET temperature = ?, wind_speed = ?
                    WHERE date = ? AND region = ?
                    """,
                    (temperature, wind_speed, date_value, region)
                )
            else:
                # Insert new row with minimal required columns
                # Other columns will be NULL
                self.conn.execute(
                    """
                    INSERT INTO weather_daily (date, region, temperature, wind_speed)
                    VALUES (?, ?, ?, ?)
                    """,
                    (date_value, region, temperature, wind_speed)
                )
        
        logger.info(f"Inserted/updated {len(df)} historical weather records")
    
    def insert_rig_counts(self, df: pd.DataFrame):
        """
        Insert or upsert weekly gas rig counts into rigs_weekly.
        
        Uses MERGE/upsert strategy keyed on (date, region).
        If a row exists for that date+region, updates gas_rigs/gas_rigs_land/gas_rigs_offshore.
        If not, inserts a new row.
        
        Args:
            df: DataFrame with columns: date, region, gas_rigs, gas_rigs_land, gas_rigs_offshore
        """
        if df.empty:
            logger.warning("No rig count data to insert")
            return
        
        # Ensure required columns exist
        required_cols = ['date', 'region', 'gas_rigs', 'gas_rigs_land', 'gas_rigs_offshore']
        if not all(col in df.columns for col in required_cols):
            missing = [col for col in required_cols if col not in df.columns]
            logger.error(f"Missing required columns: {missing}")
            raise ValueError(f"DataFrame must contain columns: {required_cols}")
        
        # Ensure date is datetime and region is string
        df = df.copy()
        df['date'] = pd.to_datetime(df['date'])
        df['region'] = df['region'].astype(str)
        
        # Ensure integer types for rig counts
        df['gas_rigs'] = df['gas_rigs'].fillna(0).astype(int)
        df['gas_rigs_land'] = df['gas_rigs_land'].fillna(0).astype(int)
        df['gas_rigs_offshore'] = df['gas_rigs_offshore'].fillna(0).astype(int)
        
        # Remove existing rows with same primary keys (date, region) to avoid constraint violations
        if {"date", "region"}.issubset(df.columns):
            unique_keys = df[["date", "region"]].drop_duplicates()
            for _, row in unique_keys.iterrows():
                date_value = pd.to_datetime(row["date"]).strftime("%Y-%m-%d")
                region_value = str(row["region"])
                self.conn.execute(
                    "DELETE FROM rigs_weekly WHERE date = ? AND region = ?",
                    (date_value, region_value)
                )
        
        # Insert new data
        self._insert_data("rigs_weekly", df)
        logger.info(f"Inserted/updated {len(df)} rig count records")
    
    def insert_rigs(self, df: pd.DataFrame):
        """Insert rigs data (alias for backward compatibility)."""
        self.insert_rig_counts(df)
    
    def insert_events(self, df: pd.DataFrame):
        """Insert events data."""
        self._insert_data("events", df)
    
    def insert_features(self, df: pd.DataFrame):
        """Insert features data."""
        self._insert_data("features", df)
    
    def insert_predictions(self, df: pd.DataFrame):
        """Insert predictions data."""
        self._insert_data("predictions", df)
    
    def _insert_data(self, table_name: str, df: pd.DataFrame):
        """Insert data into specified table using DuckDB native methods."""
        if df.empty:
            logger.warning(f"No data to insert into {table_name}")
            return
            
        try:
            # Use DuckDB's register and insert methods for better compatibility
            # Register the DataFrame as a temporary view
            self.conn.register(f"temp_{table_name}", df)
            
            # Get column names
            columns = ", ".join(df.columns.tolist())
            
            # Insert with conflict handling (ON CONFLICT DO NOTHING for primary key violations)
            # First, try to delete existing records with same primary keys
            if table_name == "storage_weekly" and "report_date" in df.columns:
                # Delete existing records for the same dates
                dates = df["report_date"].unique()
                for date in dates:
                    date_str = pd.to_datetime(date).strftime("%Y-%m-%d")
                    self.conn.execute(
                        f"DELETE FROM {table_name} WHERE report_date = ?",
                        [date_str]
                    )
            elif table_name == "prices" and "date" in df.columns and "hub" in df.columns:
                # Delete existing records for the same date/hub combinations
                for _, row in df.iterrows():
                    date_str = pd.to_datetime(row["date"]).strftime("%Y-%m-%d")
                    hub = row["hub"]
                    self.conn.execute(
                        f"DELETE FROM {table_name} WHERE date = ? AND hub = ?",
                        [date_str, hub]
                    )
            elif table_name == "production_monthly" and "month" in df.columns:
                dates = df["month"].unique()
                for date in dates:
                    date_str = pd.to_datetime(date).strftime("%Y-%m-%d")
                    self.conn.execute(
                        f"DELETE FROM {table_name} WHERE month = ?",
                        [date_str]
                    )
            elif table_name == "lng_monthly" and "month" in df.columns:
                dates = df["month"].unique()
                for date in dates:
                    date_str = pd.to_datetime(date).strftime("%Y-%m-%d")
                    self.conn.execute(
                        f"DELETE FROM {table_name} WHERE month = ?",
                        [date_str]
                    )
            elif table_name == "rigs_weekly" and "date" in df.columns and "region" in df.columns:
                # Delete existing records for the same date/region combinations
                unique_keys = df[["date", "region"]].drop_duplicates()
                for _, row in unique_keys.iterrows():
                    date_str = pd.to_datetime(row["date"]).strftime("%Y-%m-%d")
                    region = str(row["region"])
                    self.conn.execute(
                        f"DELETE FROM {table_name} WHERE date = ? AND region = ?",
                        [date_str, region]
                    )
            elif table_name == "power_burn" and {"date", "ba"}.issubset(df.columns):
                unique_keys = df[["date", "ba"]].drop_duplicates()
                for _, row in unique_keys.iterrows():
                    date_str = pd.to_datetime(row["date"]).strftime("%Y-%m-%d")
                    ba = str(row["ba"])
                    self.conn.execute(
                        f"DELETE FROM {table_name} WHERE date = ? AND ba = ?",
                        [date_str, ba]
                    )
            
            # Insert new data
            self.conn.execute(
                f"INSERT INTO {table_name} ({columns}) SELECT {columns} FROM temp_{table_name}"
            )
            
            # Unregister temporary view
            self.conn.unregister(f"temp_{table_name}")
            
            logger.info(f"Inserted {len(df)} records into {table_name}")
        except Exception as e:
            logger.error(f"Failed to insert data into {table_name}: {e}")
            # Try to unregister in case of error
            try:
                self.conn.unregister(f"temp_{table_name}")
            except:
                pass
            raise
    
    def get_data(self, table_name: str, start_date: str = None, end_date: str = None) -> pd.DataFrame:
        """Get data from specified table."""
        # Map table names to their date column names
        date_column_map = {
            "prices": "date",
            "storage_weekly": "report_date",
            "production_monthly": "month",
            "lng_monthly": "month",
            "weather_daily": "date",
            "power_burn": "date",
            "rigs_weekly": "date",
            "events": "date",
            "features": "date",
            "predictions": "date"
        }
        
        date_col = date_column_map.get(table_name, "date")
        
        query = f"SELECT * FROM {table_name}"
        params = []
        
        if start_date:
            query += f" WHERE {date_col} >= ?"
            params.append(start_date)
            
        if end_date:
            if start_date:
                query += f" AND {date_col} <= ?"
            else:
                query += f" WHERE {date_col} <= ?"
            params.append(end_date)
        
        query += f" ORDER BY {date_col}"
        
        try:
            df = pd.read_sql(query, self.conn, params=params)
            return df
        except Exception as e:
            logger.error(f"Failed to get data from {table_name}: {e}")
            return pd.DataFrame()
    
    def get_latest_data(self, table_name: str) -> pd.DataFrame:
        """Get latest data from specified table."""
        # Map table names to their date column names
        date_column_map = {
            "prices": "date",
            "storage_weekly": "report_date",
            "production_monthly": "month",
            "lng_monthly": "month",
            "weather_daily": "date",
            "power_burn": "date",
            "rigs_weekly": "date",
            "events": "date",
            "features": "date",
            "predictions": "date"
        }
        
        date_col = date_column_map.get(table_name, "date")
        
        query = f"""
        SELECT * FROM {table_name} 
        WHERE {date_col} = (SELECT MAX({date_col}) FROM {table_name})
        """
        
        try:
            df = pd.read_sql(query, self.conn)
            return df
        except Exception as e:
            logger.error(f"Failed to get latest data from {table_name}: {e}")
            return pd.DataFrame()
    
    def get_feature_matrix(self, start_date: str, end_date: str) -> pd.DataFrame:
        """Get feature matrix for model training."""
        query = """
        SELECT 
            p.date,
            p.spot_price,
            p.front_month,
            p.m1_m2_spread,
            s.working_gas,
            s.five_year_avg,
            s.yoy_deviation,
            s.wow_change,
            pr.dry_gas_bcfpd,
            l.exports_bcf,
            l.lng_capacity_bcfpd,
            pb.gas_mwh,
            pb.total_load_mwh,
            pb.renewables_mwh,
            w.hdd,
            w.cdd,
            w.hdd_norm,
            w.cdd_norm,
            w.hdd_anom,
            w.cdd_anom,
            w.temperature,
            w.wind_speed,
            r.gas_rigs,
            r.gas_rigs_land,
            r.gas_rigs_offshore
        FROM prices p
        LEFT JOIN storage_weekly s ON p.date = s.report_date
        LEFT JOIN production_monthly pr ON p.date = pr.month
        LEFT JOIN lng_monthly l ON p.date = l.month
        LEFT JOIN (
            SELECT date, 
                   SUM(gas_mwh) as gas_mwh,
                   SUM(total_load_mwh) as total_load_mwh,
                   SUM(renewables_mwh) as renewables_mwh
            FROM power_burn
            GROUP BY date
        ) pb ON p.date = pb.date
        LEFT JOIN weather_daily w ON p.date = w.date AND w.region = 'CONUS'
        LEFT JOIN rigs_weekly r ON p.date = r.date AND r.region = 'US'
        WHERE p.date BETWEEN ? AND ?
        ORDER BY p.date
        """
        
        try:
            df = pd.read_sql(query, self.conn, params=[start_date, end_date])
            return df
        except Exception as e:
            logger.error(f"Failed to get feature matrix: {e}")
            return pd.DataFrame()
    
    def close(self):
        """Close database connection."""
        if self.conn:
            self.conn.close()
            logger.info("Database connection closed")


def main():
    """Test the database."""
    db = GasModelDatabase()
    
    # Test schema creation
    print("Database schema created successfully")
    
    # Test data insertion with sample data
    sample_prices = pd.DataFrame({
        'date': pd.date_range('2024-01-01', periods=5),
        'hub': 'HenryHub',
        'spot_price': [3.50, 3.60, 3.45, 3.70, 3.55],
        'front_month': [3.55, 3.65, 3.50, 3.75, 3.60],
        'm1_m2_spread': [0.05, 0.05, 0.05, 0.05, 0.05]
    })
    
    db.insert_prices(sample_prices)
    print("Sample data inserted successfully")
    
    # Test data retrieval
    data = db.get_data("prices")
    print(f"Retrieved {len(data)} records from prices table")
    
    db.close()


if __name__ == "__main__":
    main()
