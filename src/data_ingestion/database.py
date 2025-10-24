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
            self.conn = duckdb.connect(self.db_url)
            logger.info(f"Connected to database: {self.db_url}")
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
            working_gas DECIMAL(10,2),
            five_year_avg DECIMAL(10,2),
            yoy_deviation DECIMAL(10,2),
            wow_change DECIMAL(10,2),
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
            hdd_norm_delta DECIMAL(8,2),
            cdd_norm_delta DECIMAL(8,2),
            temperature DECIMAL(6,2),
            wind_speed DECIMAL(6,2),
            PRIMARY KEY (date, region)
        );
        
        -- Rigs table (weekly grain)
        CREATE TABLE IF NOT EXISTS rigs_weekly (
            date DATE,
            total_rigs INTEGER,
            oil_rigs INTEGER,
            gas_rigs INTEGER,
            PRIMARY KEY (date)
        );
        
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
        self._insert_data("weather_daily", df)
    
    def insert_rigs(self, df: pd.DataFrame):
        """Insert rigs data."""
        self._insert_data("rigs_weekly", df)
    
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
        """Insert data into specified table."""
        if df.empty:
            logger.warning(f"No data to insert into {table_name}")
            return
            
        try:
            # Use pandas to_sql with duckdb
            df.to_sql(table_name, self.conn, if_exists='append', index=False)
            logger.info(f"Inserted {len(df)} records into {table_name}")
        except Exception as e:
            logger.error(f"Failed to insert data into {table_name}: {e}")
            raise
    
    def get_data(self, table_name: str, start_date: str = None, end_date: str = None) -> pd.DataFrame:
        """Get data from specified table."""
        query = f"SELECT * FROM {table_name}"
        params = []
        
        if start_date:
            query += " WHERE date >= ?"
            params.append(start_date)
            
        if end_date:
            if start_date:
                query += " AND date <= ?"
            else:
                query += " WHERE date <= ?"
            params.append(end_date)
            
        query += " ORDER BY date"
        
        try:
            df = pd.read_sql(query, self.conn, params=params)
            return df
        except Exception as e:
            logger.error(f"Failed to get data from {table_name}: {e}")
            return pd.DataFrame()
    
    def get_latest_data(self, table_name: str) -> pd.DataFrame:
        """Get latest data from specified table."""
        query = f"""
        SELECT * FROM {table_name} 
        WHERE date = (SELECT MAX(date) FROM {table_name})
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
            w.temperature,
            r.total_rigs,
            r.gas_rigs
        FROM prices p
        LEFT JOIN storage_weekly s ON p.date = s.report_date
        LEFT JOIN production_monthly pr ON p.date = pr.month
        LEFT JOIN lng_monthly l ON p.date = l.month
        LEFT JOIN power_burn pb ON p.date = pb.date
        LEFT JOIN weather_daily w ON p.date = w.date
        LEFT JOIN rigs_weekly r ON p.date = r.date
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
