# Missing Data Sources

## Summary

Based on the database schema and codebase analysis, here are the data sources that still need implementation:

## ✅ Fully Implemented

1. **EIA Core Data** (All working with 2010-2025 data):
   - ✅ Henry Hub Spot Prices (daily) - 7,257 records (2010-2025)
   - ✅ Storage Data (weekly) - 829 records (2010-2025)
   - ✅ Production Data (monthly) - 188 records (2010-2025)
   - ✅ Consumption Data (monthly) - 188 records (2010-2025)
   - ✅ LNG Exports (monthly) - 188 records (2010-2025)

2. **EIA-930 Power Grid Data** (Working with 2019-2025 data):
   - ✅ EIA-930 Grid Monitor (daily, aggregated from hourly)
   - ✅ All 10 Balancing Authorities: CISO, ERCO, FLA, FMPP, ISNE, MISO, NW, NYIS, PJM, SWPP
   - ✅ Natural gas generation (gas_mwh) by BA
   - ✅ Earliest data: 2019-01-01 (earliest available from EIA)
   - ✅ Integrated into `power_burn` database table

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

### Power Grid Data (Regional ISOs)
Additional regional power data sources (EIA-930 covers all major BAs):

1. **PJM Data**
   - Status: Placeholder
   - Needed for: PJM region power generation and LMPs
   - Source: PJM Data Miner 2 API
   - Impact: Medium (regional power data, but PJM BA is already covered by EIA-930)

2. **ERCOT Data**
   - Status: Placeholder
   - Needed for: ERCOT region fuel mix and load
   - Source: ERCOT API
   - Impact: Medium (regional power data, but ERCO BA is already covered by EIA-930)

3. **ISO-NE Data**
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
1. **Historical Weather Data** - Needed for training models with HDD/CDD features
   - ✅ EIA-930 Power Grid Data - **NOW IMPLEMENTED** (covers all major BAs from 2019-2025)

### Medium Priority (nice to have)
2. **CPC Degree Days** - Historical degree day data
3. **Rig Count Data** - Supply-side indicator
4. **Regional Power Data** (PJM, ERCOT, ISO-NE) - Regional demand patterns (EIA-930 already covers these BAs)

### Low Priority (can add later)
5. **Events Data** - Important for specific events but may be sparse

## Current Model Training Status

**Can train models now with:**
- ✅ Prices (7,257 daily records, 2010-2025)
- ✅ Storage (829 weekly records, 2010-2025)
- ✅ Production (188 monthly records, 2010-2025)
- ✅ Consumption (188 monthly records, 2010-2025)
- ✅ LNG Exports (188 monthly records, 2010-2025)
- ✅ Power Burn Data (EIA-930, daily from 2019-2025, all 10 major BAs)
- ⚠️ Weather Forecast (limited, forecast only)

**Missing for optimal performance:**
- ❌ Historical weather/degree days (needed for HDD/CDD features in training)
- ❌ Rig count data (supply indicator)

## Next Steps

If you want to improve model accuracy, prioritize:
1. **Historical Weather Data** - Needed for proper HDD/CDD features in training data
2. **Rig Count Data** - Supply indicator
3. **CPC Degree Days** - Historical degree day data for feature engineering

**Current Status:**
- ✅ Core EIA data: 2010-2025 (15+ years)
- ✅ Power burn data: 2019-2025 (6+ years, all major BAs)
- The current dataset is sufficient to train working models with good accuracy
- Adding historical weather data would enable proper HDD/CDD feature engineering for the full training period

