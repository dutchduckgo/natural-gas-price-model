#!/usr/bin/env python3
"""
Comprehensive test script for all ingestion pipelines.

Tests:
1. EIA data sources (prices, storage, production, consumption, LNG)
2. Weather data sources
3. Power grid data sources
4. Database operations
5. Full ingestion pipeline
"""

import sys
import os
from pathlib import Path
from datetime import datetime, timedelta
import logging

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.data_ingestion.eia_client import EIAClient
from src.data_ingestion.weather_client import WeatherClient
from src.data_ingestion.power_client import PowerGridClient
from src.data_ingestion.database import GasModelDatabase
from config import EIA_API_KEY, DATABASE_URL

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def test_eia_sources():
    """Test all EIA data sources."""
    print("\n" + "=" * 60)
    print("Testing EIA Data Sources")
    print("=" * 60)
    
    client = EIAClient(api_key=EIA_API_KEY)
    results = {}
    
    # Test 1: Henry Hub Spot Prices
    print("\n1. Henry Hub Spot Prices (daily):")
    try:
        df = client.get_henry_hub_spot()
        results["henry_hub"] = {"status": "✓", "count": len(df)}
        print(f"   ✓ Retrieved {len(df)} records")
        if len(df) > 0:
            print(f"   Date range: {df['date'].min()} to {df['date'].max()}")
            print(f"   Value range: ${df['value'].min():.2f} to ${df['value'].max():.2f}")
    except Exception as e:
        results["henry_hub"] = {"status": "✗", "error": str(e)}
        print(f"   ✗ Error: {e}")
    
    # Test 2: Storage Data
    print("\n2. Storage Data (weekly):")
    try:
        df = client.get_storage_data()
        results["storage"] = {"status": "✓", "count": len(df)}
        print(f"   ✓ Retrieved {len(df)} records")
        if len(df) > 0:
            print(f"   Date range: {df['date'].min()} to {df['date'].max()}")
            print(f"   Value range: {df['value'].min():.2f} to {df['value'].max():.2f} BCF")
        else:
            print(f"   ⚠ No data (API endpoint may need adjustment)")
    except Exception as e:
        results["storage"] = {"status": "✗", "error": str(e)}
        print(f"   ✗ Error: {e}")
    
    # Test 3: Production Data
    print("\n3. Production Data (monthly):")
    try:
        df = client.get_production_data()
        results["production"] = {"status": "✓", "count": len(df)}
        print(f"   ✓ Retrieved {len(df)} records")
        if len(df) > 0:
            print(f"   Date range: {df['date'].min()} to {df['date'].max()}")
            print(f"   Value range: {df['value'].min():.2f} to {df['value'].max():.2f} BCF/day")
    except Exception as e:
        results["production"] = {"status": "✗", "error": str(e)}
        print(f"   ✗ Error: {e}")
    
    # Test 4: Consumption Data
    print("\n4. Consumption Data (monthly, from Excel):")
    try:
        df = client.get_consumption_data()
        results["consumption"] = {"status": "✓", "count": len(df)}
        print(f"   ✓ Retrieved {len(df)} records")
        if len(df) > 0:
            print(f"   Date range: {df['date'].min()} to {df['date'].max()}")
            print(f"   Value range: {df['value'].min():.2f} to {df['value'].max():.2f} MMcf")
    except Exception as e:
        results["consumption"] = {"status": "✗", "error": str(e)}
        print(f"   ✗ Error: {e}")
    
    # Test 5: LNG Exports
    print("\n5. LNG Exports (monthly):")
    try:
        df = client.get_lng_exports()
        results["lng"] = {"status": "✓", "count": len(df)}
        print(f"   ✓ Retrieved {len(df)} records")
        if len(df) > 0:
            print(f"   Date range: {df['date'].min()} to {df['date'].max()}")
            print(f"   Value range: {df['value'].min():.2f} to {df['value'].max():.2f} BCF")
    except Exception as e:
        results["lng"] = {"status": "✗", "error": str(e)}
        print(f"   ✗ Error: {e}")
    
    return results


def test_weather_power():
    """Test weather and power data sources."""
    print("\n" + "=" * 60)
    print("Testing Weather and Power Data Sources")
    print("=" * 60)
    
    results = {}
    
    # Test Weather Client
    print("\n1. Weather Data:")
    try:
        weather_client = WeatherClient()
        weather_data = weather_client.get_all_weather_data()
        results["weather"] = {"status": "✓", "datasets": len(weather_data)}
        print(f"   ✓ Retrieved {len(weather_data)} datasets")
        for name, df in weather_data.items():
            if not df.empty:
                print(f"   - {name}: {len(df)} records")
            else:
                print(f"   - {name}: No data (placeholder implementation)")
    except Exception as e:
        results["weather"] = {"status": "✗", "error": str(e)}
        print(f"   ✗ Error: {e}")
    
    # Test Power Client
    print("\n2. Power Grid Data:")
    try:
        power_client = PowerGridClient()
        power_data = power_client.get_all_power_data()
        results["power"] = {"status": "✓", "datasets": len(power_data)}
        print(f"   ✓ Retrieved {len(power_data)} datasets")
        for name, df in power_data.items():
            if not df.empty:
                print(f"   - {name}: {len(df)} records")
            else:
                print(f"   - {name}: No data (placeholder implementation)")
    except Exception as e:
        results["power"] = {"status": "✗", "error": str(e)}
        print(f"   ✗ Error: {e}")
    
    return results


def test_database():
    """Test database operations."""
    print("\n" + "=" * 60)
    print("Testing Database Operations")
    print("=" * 60)
    
    results = {}
    
    try:
        db = GasModelDatabase()
        
        # Test feature matrix query
        print("\n1. Feature Matrix Query:")
        try:
            end_date = datetime.now().strftime("%Y-%m-%d")
            start_date = (datetime.now() - timedelta(days=365*2)).strftime("%Y-%m-%d")
            df = db.get_feature_matrix(start_date, end_date)
            results["feature_matrix"] = {"status": "✓", "rows": len(df), "cols": len(df.columns)}
            print(f"   ✓ Query successful")
            print(f"   Retrieved {len(df)} rows with {len(df.columns)} columns")
            if len(df) > 0:
                print(f"   Date range: {df['date'].min()} to {df['date'].max()}")
        except Exception as e:
            results["feature_matrix"] = {"status": "✗", "error": str(e)}
            print(f"   ✗ Error: {e}")
        
        # Test individual table queries
        print("\n2. Individual Table Queries:")
        tables = ["prices", "production_monthly", "lng_monthly", "storage_weekly", "weather_daily"]
        for table in tables:
            try:
                df = db.get_data(table)
                if len(df) > 0:
                    results[table] = {"status": "✓", "count": len(df)}
                    print(f"   ✓ {table}: {len(df)} records")
                else:
                    results[table] = {"status": "⚠", "count": 0}
                    print(f"   ⚠ {table}: {len(df)} records (empty)")
            except Exception as e:
                results[table] = {"status": "✗", "error": str(e)}
                print(f"   ✗ {table}: Error - {e}")
        
        db.close()
        
    except Exception as e:
        results["database"] = {"status": "✗", "error": str(e)}
        print(f"   ✗ Database connection error: {e}")
    
    return results


def print_summary(eia_results, weather_power_results, db_results):
    """Print summary of all tests."""
    print("\n" + "=" * 60)
    print("TEST SUMMARY")
    print("=" * 60)
    
    print("\nEIA Data Sources:")
    for source, result in eia_results.items():
        status = result.get("status", "?")
        if status == "✓":
            count = result.get("count", 0)
            print(f"  {status} {source}: {count} records")
        else:
            error = result.get("error", "Unknown error")
            print(f"  {status} {source}: {error[:50]}")
    
    print("\nWeather & Power:")
    for source, result in weather_power_results.items():
        status = result.get("status", "?")
        if status == "✓":
            datasets = result.get("datasets", 0)
            print(f"  {status} {source}: {datasets} datasets")
        else:
            error = result.get("error", "Unknown error")
            print(f"  {status} {source}: {error[:50]}")
    
    print("\nDatabase:")
    for table, result in db_results.items():
        status = result.get("status", "?")
        if status == "✓":
            count = result.get("count", result.get("rows", 0))
            print(f"  {status} {table}: {count} records")
        elif status == "⚠":
            print(f"  {status} {table}: Empty")
        else:
            error = result.get("error", "Unknown error")
            print(f"  {status} {table}: {error[:50]}")
    
    print("\n" + "=" * 60)


def main():
    """Run all ingestion tests."""
    print("=" * 60)
    print("COMPREHENSIVE INGESTION PIPELINE TEST")
    print("=" * 60)
    
    # Test EIA sources
    eia_results = test_eia_sources()
    
    # Test weather and power
    weather_power_results = test_weather_power()
    
    # Test database
    db_results = test_database()
    
    # Print summary
    print_summary(eia_results, weather_power_results, db_results)
    
    print("\n✓ All tests completed!")


if __name__ == "__main__":
    main()

