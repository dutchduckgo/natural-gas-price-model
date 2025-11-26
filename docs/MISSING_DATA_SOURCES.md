# Missing Data Sources

## Summary

Based on the database schema and codebase analysis, here are the data sources that still need implementation:

## ✅ Fully Implemented

1. **EIA Core Data** (All working with 2010-2025 data):
   - ✅ Henry Hub Spot Prices (daily) - 4,012 records
   - ✅ Storage Data (weekly) - 829 records
   - ✅ Production Data (monthly) - 188 records
   - ✅ Consumption Data (monthly) - 188 records
   - ✅ LNG Exports (monthly) - 188 records

## ⚠️ Partially Implemented

### Weather Data
- ✅ Weather Forecast (NWS) - Basic implementation working
- ❌ **CPC Degree Days** - Not yet implemented
  - Needed for: Historical HDD/CDD data
  - Source: Climate Prediction Center
  - Impact: Medium (can calculate from temperature, but historical data is valuable)

- ❌ **Historical Weather Data** - Not yet implemented
  - Needed for: Historical temperature, HDD/CDD calculations
  - Source: NWS, NOAA, or other weather APIs
  - Impact: Medium (forecast data is working, but historical is needed for training)

## ❌ Not Implemented

### Power Grid Data (All placeholders)
These are important for power burn calculations:

1. **EIA-930 Grid Monitor**
   - Status: Placeholder
   - Needed for: Hourly electric grid data, gas-fired generation
   - Source: EIA API v2
   - Impact: High (power burn is a key demand driver)

2. **PJM Data**
   - Status: Placeholder
   - Needed for: PJM region power generation and LMPs
   - Source: PJM Data Miner 2 API
   - Impact: Medium (regional power data)

3. **ERCOT Data**
   - Status: Placeholder
   - Needed for: ERCOT region fuel mix and load
   - Source: ERCOT API
   - Impact: Medium (regional power data)

4. **ISO-NE Data**
   - Status: Placeholder
   - Needed for: ISO-NE region load and fuel mix
   - Source: ISO-NE Web Services
   - Impact: Medium (regional power data)

### Rig Count Data
- **Status**: Database table exists but no ingestion method
- **Table**: `rigs_weekly`
- **Needed for**: Supply-side indicators (drilling activity)
- **Source**: Baker Hughes Rig Count API
- **Impact**: Medium (supply indicator, but production data may be sufficient)

### Events Data
- **Status**: Database table exists but no ingestion method
- **Table**: `events`
- **Needed for**: LNG terminal outages, pipeline disruptions
- **Source**: FERC, DOE, or manual tracking
- **Impact**: Low-Medium (important for specific events, but may be rare)

## Priority Recommendations

### High Priority (for model accuracy)
1. **EIA-930 Power Grid Data** - Power burn is a major demand driver
2. **Historical Weather Data** - Needed for training models with HDD/CDD features

### Medium Priority (nice to have)
3. **CPC Degree Days** - Historical degree day data
4. **Rig Count Data** - Supply-side indicator
5. **Regional Power Data** (PJM, ERCOT, ISO-NE) - Regional demand patterns

### Low Priority (can add later)
6. **Events Data** - Important for specific events but may be sparse

## Current Model Training Status

**Can train models now with:**
- ✅ Prices (4,012 daily records)
- ✅ Storage (829 weekly records)
- ✅ Production (188 monthly records)
- ✅ Consumption (188 monthly records)
- ✅ LNG Exports (188 monthly records)
- ⚠️ Weather Forecast (limited, forecast only)

**Missing for optimal performance:**
- ❌ Historical weather/degree days
- ❌ Power burn data (EIA-930)
- ❌ Rig count data

## Next Steps

If you want to improve model accuracy, prioritize:
1. **EIA-930 Power Grid Data** - Most impactful missing feature
2. **Historical Weather Data** - Needed for proper HDD/CDD features
3. **Rig Count Data** - Supply indicator

The current dataset (2010-2025) is sufficient to train working models, but adding power burn and historical weather would significantly improve accuracy.

