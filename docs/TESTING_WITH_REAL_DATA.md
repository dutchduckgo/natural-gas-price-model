# Testing Models with Real Data

## Overview

The pipeline is now configured to use real data from the EIA API and other sources. This document explains what's been set up and how to test the prediction models.

## What's Been Done

### 1. Data Ingestion ✅
- **EIA API v2 Integration**: All EIA data sources now use the v2 API
- **Henry Hub Spot Prices**: Daily prices (RNGWHHD series)
- **Production Data**: Monthly production (N9070US2 with process=FPD facet)
- **LNG Exports**: Monthly exports (N9133US2 with process=ENG facet)
- **Consumption Data**: Monthly consumption from Excel file
- **Storage Data**: Weekly storage (endpoint configured, may need API adjustments)
- **Weather Data**: Forecast data (placeholder implementation)
- **Database**: All data stored in DuckDB at `data/gas_model.db`

### 2. Feature Engineering Integration ✅
- **Training Pipeline Updated**: Now applies feature engineering to database data
- **Weather Features**: HDD/CDD, lags, rolling features, seasonal features
- **Storage Features**: Metrics, lags, rolling features, tightness indicators
- **Automatic Application**: Feature engineering is applied automatically when loading training data

### 3. Model Training Pipeline ✅
- **Database Integration**: Models load data directly from database
- **Feature Engineering**: Applied automatically before training
- **Baseline Models**: Elastic Net, Linear Regression, Random Forest
- **Tree Models**: XGBoost, LightGBM (if dependencies installed)
- **Deep Learning**: LSTM, Transformer (if dependencies installed)

## Current Data Status

Based on the last ingestion:
- **Prices**: 244 records (2024-11-25 to 2025-11-17)
- **Production**: 10 records (monthly)
- **LNG Exports**: 10 records (monthly)
- **Consumption**: 9 records (from Excel, filtered by date range)
- **Weather**: 28 records (forecast data)
- **Storage**: 0 records (API endpoint may need adjustment)

## How to Test

### Step 1: Verify Data Ingestion

```bash
# Activate virtual environment
source venv/bin/activate

# Run data ingestion (if not already done)
EIA_API_KEY=your_api_key DATABASE_URL="data/gas_model.db" python -m src.pipeline.ingest_data
```

### Step 2: Test Model Training

```bash
# Run the test script
DATABASE_URL="data/gas_model.db" python scripts/test_real_data_training.py
```

This will:
1. Load data from the database
2. Apply feature engineering
3. Train baseline models
4. Evaluate performance
5. Display results

### Step 3: Run Full Training Pipeline

```bash
# Train all models (baseline, tree, deep learning)
DATABASE_URL="data/gas_model.db" python -m src.pipeline.train_baseline
```

## What Still Needs Attention

### 1. Storage Data
- Storage API endpoint may need adjustment
- Currently returns 0 records
- May need different authentication or endpoint structure

### 2. Feature Engineering
- Some features may have many missing values due to sparse data
- Consider forward-filling monthly data (production, LNG) to daily
- Weather features may be limited until full weather data is implemented

### 3. Data Quality
- Monthly data (production, LNG, consumption) needs to be interpolated/forward-filled for daily predictions
- Consider adding data interpolation logic in the feature matrix query

### 4. Model Evaluation
- With limited data (244 price records), cross-validation may be challenging
- Consider using a longer date range or adjusting evaluation strategy

## Next Steps

1. **Fix Storage Data**: Investigate and fix storage API endpoint
2. **Data Interpolation**: Add logic to forward-fill monthly data to daily
3. **Extended Data Range**: Ingest more historical data for better training
4. **Feature Validation**: Verify all engineered features are working correctly
5. **Model Performance**: Evaluate if models are learning meaningful patterns

## Troubleshooting

### "No training data found"
- Check that data ingestion completed successfully
- Verify database file exists: `data/gas_model.db`
- Check date range in query matches available data

### "Feature engineering errors"
- Check that required columns exist in database
- Verify data types are correct
- Some features may fail gracefully if data is missing

### "Model training errors"
- Ensure all dependencies are installed
- Check that target variable (spot_price) is present
- Verify sufficient data for training (at least 30-50 samples)

## Files Modified

- `src/pipeline/train_baseline.py`: Added feature engineering integration
- `src/data_ingestion/eia_client.py`: Updated to v2 API
- `src/data_ingestion/database.py`: Fixed insertion methods
- `scripts/test_real_data_training.py`: New test script

## Summary

✅ **Ready to test**: The pipeline is configured to use real data
✅ **Feature engineering**: Automatically applied
✅ **Database integration**: Working
⚠️ **Limited data**: May need more historical data for robust training
⚠️ **Storage data**: Needs API endpoint fix

You can now test the models with real data using the test script!

