#!/usr/bin/env python3
"""
Test script to verify model training works with real data from the database.

This script:
1. Loads data from the database
2. Applies feature engineering
3. Trains baseline models
4. Evaluates performance
"""

import sys
import os
from pathlib import Path
import logging
from datetime import datetime, timedelta

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.pipeline.train_baseline import ModelTrainingPipeline

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def main():
    """Test model training with real data."""
    logger.info("=" * 60)
    logger.info("Testing Model Training with Real Data")
    logger.info("=" * 60)
    
    # Create training pipeline
    pipeline = ModelTrainingPipeline()
    
    # Set date range - use available data (last 2 years or whatever is available)
    end_date = datetime.now().strftime("%Y-%m-%d")
    start_date = (datetime.now() - timedelta(days=365*2)).strftime("%Y-%m-%d")
    
    logger.info(f"Date range: {start_date} to {end_date}")
    
    try:
        # Get training data (with feature engineering)
        logger.info("\nStep 1: Loading data from database...")
        df = pipeline.get_training_data(start_date, end_date)
        
        if df.empty:
            logger.error("No data available in database. Please run data ingestion first:")
            logger.error("  python -m src.pipeline.ingest_data")
            return
        
        logger.info(f"✓ Loaded {len(df)} samples with {len(df.columns)} features")
        logger.info(f"  Date range: {df['date'].min()} to {df['date'].max()}")
        logger.info(f"  Target (spot_price) range: ${df['spot_price'].min():.2f} to ${df['spot_price'].max():.2f}")
        
        # Train only baseline models (faster for testing)
        logger.info("\nStep 2: Training baseline models...")
        pipeline.train_baseline_models(df)
        logger.info("✓ Baseline models trained")
        
        # Evaluate models
        logger.info("\nStep 3: Evaluating models...")
        results = pipeline.evaluate_models(df)
        logger.info("✓ Model evaluation completed")
        
        # Print results
        logger.info("\n" + "=" * 60)
        logger.info("Model Performance Results")
        logger.info("=" * 60)
        
        for model_name, model_results in results.items():
            metrics = model_results.get("metrics", {})
            logger.info(f"\n{model_name.upper()}:")
            if 'mae_mean' in metrics:
                logger.info(f"  MAE:  ${metrics['mae_mean']:.4f} ± ${metrics.get('mae_std', 0):.4f}")
            if 'rmse_mean' in metrics:
                logger.info(f"  RMSE: ${metrics['rmse_mean']:.4f} ± ${metrics.get('rmse_std', 0):.4f}")
            if 'mape_mean' in metrics:
                logger.info(f"  MAPE: {metrics['mape_mean']:.2f}% ± {metrics.get('mape_std', 0):.2f}%")
            if 'direction_accuracy_mean' in metrics:
                logger.info(f"  Direction Accuracy: {metrics['direction_accuracy_mean']:.2f}% ± {metrics.get('direction_accuracy_std', 0):.2f}%")
        
        logger.info("\n" + "=" * 60)
        logger.info("✓ Test completed successfully!")
        logger.info("=" * 60)
        
    except Exception as e:
        logger.error(f"Test failed: {e}", exc_info=True)
        raise
    finally:
        pipeline.db.close()


if __name__ == "__main__":
    main()

