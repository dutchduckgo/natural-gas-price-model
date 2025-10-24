"""
Data ingestion pipeline for natural gas price model.
"""
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import logging
from pathlib import Path

# Import data clients
from src.data_ingestion.eia_client import EIAClient
from src.data_ingestion.weather_client import WeatherClient
from src.data_ingestion.power_client import PowerGridClient
from src.data_ingestion.database import GasModelDatabase
from config import RAW_DATA_DIR, PROCESSED_DATA_DIR

logger = logging.getLogger(__name__)


class DataIngestionPipeline:
    """Pipeline for ingesting data from multiple sources."""
    
    def __init__(self):
        self.eia_client = EIAClient()
        self.weather_client = WeatherClient()
        self.power_client = PowerGridClient()
        self.db = GasModelDatabase()
        
    def ingest_eia_data(self, start_date: str = None, end_date: str = None):
        """
        Ingest EIA data.
        
        Args:
            start_date: Start date in YYYY-MM-DD format
            end_date: End date in YYYY-MM-DD format
        """
        logger.info("Starting EIA data ingestion")
        
        try:
            # Get all EIA data
            eia_data = self.eia_client.get_all_core_data(start_date, end_date)
            
            # Store in database
            for data_type, df in eia_data.items():
                if not df.empty:
                    if data_type == "henry_hub_spot":
                        # Transform for prices table
                        prices_df = pd.DataFrame({
                            'date': df['date'],
                            'hub': 'HenryHub',
                            'spot_price': df['value'],
                            'front_month': np.nan,  # Will be filled from CME data
                            'm1_m2_spread': np.nan
                        })
                        self.db.insert_prices(prices_df)
                        
                    elif data_type == "storage":
                        # Transform for storage table
                        storage_df = pd.DataFrame({
                            'report_date': df['date'],
                            'working_gas': df['value'],
                            'five_year_avg': np.nan,  # Will be calculated
                            'yoy_deviation': np.nan,
                            'wow_change': np.nan
                        })
                        self.db.insert_storage(storage_df)
                        
                    elif data_type == "production":
                        # Transform for production table
                        production_df = pd.DataFrame({
                            'month': df['date'],
                            'dry_gas_bcfpd': df['value']
                        })
                        self.db.insert_production(production_df)
                        
                    elif data_type == "lng_exports":
                        # Transform for LNG table
                        lng_df = pd.DataFrame({
                            'month': df['date'],
                            'exports_bcf': df['value'],
                            'avg_price': np.nan,
                            'terminal_notes': '',
                            'lng_capacity_bcfpd': np.nan
                        })
                        self.db.insert_lng(lng_df)
            
            logger.info("EIA data ingestion completed successfully")
            
        except Exception as e:
            logger.error(f"EIA data ingestion failed: {e}")
            raise
    
    def ingest_weather_data(self, start_date: str = None, end_date: str = None):
        """
        Ingest weather data.
        
        Args:
            start_date: Start date in YYYY-MM-DD format
            end_date: End date in YYYY-MM-DD format
        """
        logger.info("Starting weather data ingestion")
        
        try:
            # Get weather data
            weather_data = self.weather_client.get_all_weather_data(start_date, end_date)
            
            # Store in database
            for data_type, df in weather_data.items():
                if not df.empty:
                    if data_type == "forecast":
                        # Transform for weather table
                        weather_df = pd.DataFrame({
                            'date': df['date'],
                            'region': df['region'],
                            'hdd': np.nan,  # Will be calculated
                            'cdd': np.nan,
                            'hdd_norm_delta': np.nan,
                            'cdd_norm_delta': np.nan,
                            'temperature': df['temperature'],
                            'wind_speed': np.nan
                        })
                        self.db.insert_weather(weather_df)
            
            logger.info("Weather data ingestion completed successfully")
            
        except Exception as e:
            logger.error(f"Weather data ingestion failed: {e}")
            raise
    
    def ingest_power_data(self, start_date: str = None, end_date: str = None):
        """
        Ingest power grid data.
        
        Args:
            start_date: Start date in YYYY-MM-DD format
            end_date: End date in YYYY-MM-DD format
        """
        logger.info("Starting power data ingestion")
        
        try:
            # Get power data
            power_data = self.power_client.get_all_power_data(start_date, end_date)
            
            # Store in database
            for data_type, df in power_data.items():
                if not df.empty and "gas_burn" in data_type:
                    # Transform for power burn table
                    power_burn_df = pd.DataFrame({
                        'date': df['date'],
                        'ba': data_type.replace('_gas_burn', ''),
                        'gas_mwh': df['gas_burn_mwh'],
                        'total_load_mwh': np.nan,
                        'renewables_mwh': np.nan
                    })
                    self.db.insert_power_burn(power_burn_df)
            
            logger.info("Power data ingestion completed successfully")
            
        except Exception as e:
            logger.error(f"Power data ingestion failed: {e}")
            raise
    
    def ingest_all_data(self, start_date: str = None, end_date: str = None):
        """
        Ingest all data sources.
        
        Args:
            start_date: Start date in YYYY-MM-DD format
            end_date: End date in YYYY-MM-DD format
        """
        logger.info("Starting complete data ingestion")
        
        # Set default dates if not provided
        if not end_date:
            end_date = datetime.now().strftime("%Y-%m-%d")
        if not start_date:
            start_date = (datetime.now() - timedelta(days=365)).strftime("%Y-%m-%d")
        
        try:
            # Ingest EIA data
            self.ingest_eia_data(start_date, end_date)
            
            # Ingest weather data
            self.ingest_weather_data(start_date, end_date)
            
            # Ingest power data
            self.ingest_power_data(start_date, end_date)
            
            logger.info("Complete data ingestion finished successfully")
            
        except Exception as e:
            logger.error(f"Data ingestion failed: {e}")
            raise
        finally:
            # Close database connection
            self.db.close()


def main():
    """Run data ingestion pipeline."""
    # Set up logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Create pipeline
    pipeline = DataIngestionPipeline()
    
    # Run ingestion
    pipeline.ingest_all_data()
    
    print("Data ingestion completed successfully!")


if __name__ == "__main__":
    main()
