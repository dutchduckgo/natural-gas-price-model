# Poster Visualization Guide

## Overview

The `generate_poster_visuals.py` script creates 8 publication-quality figures suitable for academic posters or presentations.

## Generated Figures

### 1. Data Overview (`1_data_overview.png`)
**Purpose**: Show the key input data series
**Content**:
- Top panel: Henry Hub natural gas spot price time series
- Middle panel: Heating (HDD) and Cooling (CDD) degree days
- Bottom panel: Natural gas storage levels over time

**Use for**: Introducing the problem and showing data characteristics

### 2. Model Comparison (`2_model_comparison.png`)
**Purpose**: Compare model performance across metrics
**Content**: Side-by-side bar charts showing:
- Mean Absolute Error (MAE)
- Root Mean Squared Error (RMSE)
- Mean Absolute Percentage Error (MAPE)

**Use for**: Demonstrating which models perform best

### 3. Prediction vs Actual (`3_prediction_vs_actual.png`)
**Purpose**: Show prediction accuracy
**Content**: Scatter plot of predicted vs actual prices with:
- Perfect prediction line (diagonal)
- R² score in title
- Points colored by prediction accuracy

**Use for**: Visualizing model fit quality

### 4. Time Series Predictions (`4_time_series_predictions.png`)
**Purpose**: Show predictions over time
**Content**: Time series plot with:
- Actual prices (solid line)
- Predicted prices (dashed line)
- Test period only (after train/test split)

**Use for**: Showing how well predictions track actual prices over time

### 5. Feature Importance (`5_feature_importance.png`)
**Purpose**: Identify most important features
**Content**: Horizontal bar chart showing:
- Top 15 most important features
- Importance scores from Random Forest model
- Features ranked by importance

**Use for**: Explaining which factors drive price predictions

### 6. Error Distribution (`6_error_distribution.png`)
**Purpose**: Analyze prediction errors
**Content**: Histogram showing:
- Distribution of prediction errors
- Mean error line
- Zero error reference line

**Use for**: Checking for bias and understanding error patterns

### 7. Correlation Heatmap (`7_correlation_heatmap.png`)
**Purpose**: Show relationships between features
**Content**: Correlation matrix heatmap of:
- Spot price
- HDD, CDD
- Storage
- Temperature
- Production

**Use for**: Understanding feature relationships and multicollinearity

### 8. Model Architecture (`8_model_architecture.png`)
**Purpose**: Illustrate the pipeline
**Content**: Flow diagram showing:
- Data sources (EIA, Weather, Power Grid)
- Feature engineering step
- Model types (Baseline, Tree Models)
- Final predictions

**Use for**: Explaining the overall system architecture

## Usage

```bash
# Activate virtual environment
source venv/bin/activate

# Run the script
python scripts/generate_poster_visuals.py
```

## Output

All figures are saved to `poster_figures/` directory with:
- High resolution (300 DPI) for printing
- White background for posters
- Publication-quality styling
- Consistent color scheme

## Customization

To customize figures:
1. Edit the plotting functions in `generate_poster_visuals.py`
2. Adjust colors, sizes, or layouts as needed
3. Modify `OUTPUT_DIR` to change save location

## Tips for Posters

1. **Figure 1 (Data Overview)**: Use as introduction/background
2. **Figure 2 (Model Comparison)**: Use in results section
3. **Figure 3 (Prediction vs Actual)**: Use to show model accuracy
4. **Figure 4 (Time Series)**: Use to show temporal performance
5. **Figure 5 (Feature Importance)**: Use to explain model drivers
6. **Figure 6 (Error Distribution)**: Use in methodology/results
7. **Figure 7 (Correlation)**: Use in data exploration section
8. **Figure 8 (Architecture)**: Use as system overview diagram

## Figure Sizes

All figures are optimized for:
- **Poster printing**: 300 DPI, suitable for large format
- **Presentation slides**: Can be scaled down as needed
- **Publications**: High enough resolution for journals

