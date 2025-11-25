#!/usr/bin/env python3
"""
Generate visualization graphics for poster presentation.

This script creates publication-quality figures showing:
- Model performance comparisons
- Feature importance rankings
- Prediction vs actual scatter plots
- Time series of predictions
- Error distributions
- Data visualizations (HDD/CDD, storage, prices)
"""

import sys
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta
from pathlib import Path

# Add src to path
sys.path.append(str(Path(__file__).parent.parent))

from src.models.baseline import BaselinePipeline
from src.feature_engineering.weather_features import WeatherFeatureEngineer
from src.feature_engineering.storage_features import StorageFeatureEngineer

# Set style for publication-quality figures
plt.style.use('seaborn-v0_8-whitegrid')
sns.set_palette("husl")
plt.rcParams['figure.dpi'] = 300
plt.rcParams['savefig.dpi'] = 300
plt.rcParams['font.size'] = 10
plt.rcParams['axes.labelsize'] = 11
plt.rcParams['axes.titlesize'] = 12
plt.rcParams['xtick.labelsize'] = 9
plt.rcParams['ytick.labelsize'] = 9
plt.rcParams['legend.fontsize'] = 9
plt.rcParams['figure.titlesize'] = 14

# Create output directory
OUTPUT_DIR = Path(__file__).parent.parent / "poster_figures"
OUTPUT_DIR.mkdir(exist_ok=True)


def create_sample_data():
    """Create realistic sample data for visualization."""
    np.random.seed(42)
    n_samples = 1000
    
    dates = pd.date_range('2020-01-01', periods=n_samples)
    
    # Create synthetic features with realistic patterns
    hdd = 20 + 30 * np.sin(2 * np.pi * np.arange(n_samples) / 365) + np.random.normal(0, 5, n_samples)
    hdd = np.maximum(hdd, 0)
    
    cdd = 10 + 20 * np.sin(2 * np.pi * np.arange(n_samples) / 365) + np.random.normal(0, 3, n_samples)
    cdd = np.maximum(cdd, 0)
    
    storage = 3000 + 500 * np.sin(2 * np.pi * np.arange(n_samples) / 365) + np.random.normal(0, 100, n_samples)
    temperature = 50 + 30 * np.sin(2 * np.pi * np.arange(n_samples) / 365) + np.random.normal(0, 5, n_samples)
    production = 100 + 0.1 * np.arange(n_samples) + np.random.normal(0, 5, n_samples)
    
    # Create target with realistic relationships
    spot_price = (
        3.0 + 0.01 * hdd + 0.005 * cdd - 0.0001 * storage + 
        0.001 * production + np.random.normal(0, 0.1, n_samples)
    )
    spot_price = np.maximum(spot_price, 0.5)
    
    df = pd.DataFrame({
        'date': dates,
        'spot_price': spot_price,
        'hdd': hdd,
        'cdd': cdd,
        'storage': storage,
        'temperature': temperature,
        'production': production
    })
    
    return df


def plot_1_data_overview(df):
    """Plot 1: Overview of key data series (prices, HDD/CDD, storage)."""
    fig, axes = plt.subplots(3, 1, figsize=(12, 10))
    
    # Price time series
    axes[0].plot(df['date'], df['spot_price'], color='#2E86AB', linewidth=1.5)
    axes[0].set_ylabel('Price ($/MMBtu)', fontweight='bold')
    axes[0].set_title('Henry Hub Natural Gas Spot Price', fontweight='bold', fontsize=12)
    axes[0].grid(True, alpha=0.3)
    axes[0].set_xlim(df['date'].min(), df['date'].max())
    
    # HDD/CDD
    axes[1].fill_between(df['date'], 0, df['hdd'], color='#A23B72', alpha=0.6, label='HDD')
    axes[1].fill_between(df['date'], 0, -df['cdd'], color='#F18F01', alpha=0.6, label='CDD')
    axes[1].axhline(y=0, color='black', linestyle='-', linewidth=0.5)
    axes[1].set_ylabel('Degree Days', fontweight='bold')
    axes[1].set_title('Heating and Cooling Degree Days', fontweight='bold', fontsize=12)
    axes[1].legend(loc='upper right')
    axes[1].grid(True, alpha=0.3)
    axes[1].set_xlim(df['date'].min(), df['date'].max())
    
    # Storage
    axes[2].plot(df['date'], df['storage'], color='#C73E1D', linewidth=1.5)
    axes[2].set_ylabel('Storage (BCF)', fontweight='bold')
    axes[2].set_xlabel('Date', fontweight='bold')
    axes[2].set_title('Natural Gas Storage Levels', fontweight='bold', fontsize=12)
    axes[2].grid(True, alpha=0.3)
    axes[2].set_xlim(df['date'].min(), df['date'].max())
    
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / '1_data_overview.png', bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"Saved: {OUTPUT_DIR / '1_data_overview.png'}")


def plot_2_model_comparison(df):
    """Plot 2: Model performance comparison (MAE, RMSE, MAPE)."""
    # Train models and get predictions
    weather_engineer = WeatherFeatureEngineer()
    storage_engineer = StorageFeatureEngineer()
    
    weather_features = weather_engineer.engineer_all_weather_features(df)
    storage_features = storage_engineer.engineer_all_storage_features(df)
    all_features = pd.concat([df, weather_features, storage_features], axis=1)
    
    # Split data
    train_size = int(len(all_features) * 0.8)
    train_df = all_features.iloc[:train_size]
    test_df = all_features.iloc[train_size:]
    
    models = {}
    predictions = {}
    
    # Train models
    for model_name in ['elastic_net', 'random_forest']:
        pipeline = BaselinePipeline(model_name)
        pipeline.train_final_model(train_df)
        models[model_name] = pipeline
        
        X_test, y_test = pipeline.model.prepare_features(test_df)
        pred = pipeline.model.predict(X_test)
        predictions[model_name] = {'pred': pred, 'actual': y_test.values}
    
    # Calculate metrics
    from sklearn.metrics import mean_absolute_error, mean_squared_error
    
    metrics_data = []
    for model_name, pred_data in predictions.items():
        mae = mean_absolute_error(pred_data['actual'], pred_data['pred'])
        rmse = np.sqrt(mean_squared_error(pred_data['actual'], pred_data['pred']))
        mape = np.mean(np.abs((pred_data['actual'] - pred_data['pred']) / pred_data['actual'])) * 100
        
        metrics_data.append({'Model': model_name.replace('_', ' ').title(), 'MAE': mae, 'RMSE': rmse, 'MAPE': mape})
    
    metrics_df = pd.DataFrame(metrics_data)
    
    # Create comparison plot
    fig, axes = plt.subplots(1, 3, figsize=(14, 4))
    
    x_pos = np.arange(len(metrics_df))
    width = 0.6
    
    # MAE
    axes[0].bar(x_pos, metrics_df['MAE'], width, color=['#2E86AB', '#A23B72'], alpha=0.8)
    axes[0].set_ylabel('MAE ($/MMBtu)', fontweight='bold')
    axes[0].set_title('Mean Absolute Error', fontweight='bold')
    axes[0].set_xticks(x_pos)
    axes[0].set_xticklabels(metrics_df['Model'], rotation=0)
    axes[0].grid(True, alpha=0.3, axis='y')
    
    # RMSE
    axes[1].bar(x_pos, metrics_df['RMSE'], width, color=['#2E86AB', '#A23B72'], alpha=0.8)
    axes[1].set_ylabel('RMSE ($/MMBtu)', fontweight='bold')
    axes[1].set_title('Root Mean Squared Error', fontweight='bold')
    axes[1].set_xticks(x_pos)
    axes[1].set_xticklabels(metrics_df['Model'], rotation=0)
    axes[1].grid(True, alpha=0.3, axis='y')
    
    # MAPE
    axes[2].bar(x_pos, metrics_df['MAPE'], width, color=['#2E86AB', '#A23B72'], alpha=0.8)
    axes[2].set_ylabel('MAPE (%)', fontweight='bold')
    axes[2].set_title('Mean Absolute Percentage Error', fontweight='bold')
    axes[2].set_xticks(x_pos)
    axes[2].set_xticklabels(metrics_df['Model'], rotation=0)
    axes[2].grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / '2_model_comparison.png', bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"Saved: {OUTPUT_DIR / '2_model_comparison.png'}")


def plot_3_prediction_vs_actual(df):
    """Plot 3: Prediction vs actual scatter plots."""
    # Train model and get predictions
    weather_engineer = WeatherFeatureEngineer()
    storage_engineer = StorageFeatureEngineer()
    
    weather_features = weather_engineer.engineer_all_weather_features(df)
    storage_features = storage_engineer.engineer_all_storage_features(df)
    all_features = pd.concat([df, weather_features, storage_features], axis=1)
    
    train_size = int(len(all_features) * 0.8)
    train_df = all_features.iloc[:train_size]
    test_df = all_features.iloc[train_size:]
    
    # Train Elastic Net
    pipeline = BaselinePipeline('elastic_net')
    pipeline.train_final_model(train_df)
    
    X_test, y_test = pipeline.model.prepare_features(test_df)
    predictions = pipeline.model.predict(X_test)
    actual = y_test.values
    
    # Create scatter plot
    fig, ax = plt.subplots(figsize=(8, 8))
    
    # Scatter plot
    ax.scatter(actual, predictions, alpha=0.6, s=30, color='#2E86AB', edgecolors='black', linewidth=0.5)
    
    # Perfect prediction line
    min_val = min(actual.min(), predictions.min())
    max_val = max(actual.max(), predictions.max())
    ax.plot([min_val, max_val], [min_val, max_val], 'r--', linewidth=2, label='Perfect Prediction')
    
    # Calculate R²
    from sklearn.metrics import r2_score
    r2 = r2_score(actual, predictions)
    
    ax.set_xlabel('Actual Price ($/MMBtu)', fontweight='bold')
    ax.set_ylabel('Predicted Price ($/MMBtu)', fontweight='bold')
    ax.set_title(f'Prediction vs Actual (R² = {r2:.3f})', fontweight='bold', fontsize=12)
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / '3_prediction_vs_actual.png', bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"Saved: {OUTPUT_DIR / '3_prediction_vs_actual.png'}")


def plot_4_time_series_predictions(df):
    """Plot 4: Time series showing actual vs predicted prices."""
    # Train model and get predictions
    weather_engineer = WeatherFeatureEngineer()
    storage_engineer = StorageFeatureEngineer()
    
    weather_features = weather_engineer.engineer_all_weather_features(df)
    storage_features = storage_engineer.engineer_all_storage_features(df)
    all_features = pd.concat([df, weather_features, storage_features], axis=1)
    
    # Remove duplicate columns (keep first occurrence)
    all_features = all_features.loc[:, ~all_features.columns.duplicated()]
    
    train_size = int(len(all_features) * 0.8)
    train_df = all_features.iloc[:train_size]
    test_df = all_features.iloc[train_size:]
    
    # Store original dates from the base dataframe
    test_indices = test_df.index
    test_dates = df.loc[test_indices, 'date'].values
    
    # Train Elastic Net
    pipeline = BaselinePipeline('elastic_net')
    pipeline.train_final_model(train_df)
    
    X_test, y_test = pipeline.model.prepare_features(test_df)
    predictions = pipeline.model.predict(X_test)
    actual = y_test.values
    
    # Ensure dates and predictions/actual have same length
    min_len = min(len(test_dates), len(actual), len(predictions))
    test_dates = test_dates[:min_len]
    actual = actual[:min_len]
    predictions = predictions[:min_len]
    
    # Create time series plot
    fig, ax = plt.subplots(figsize=(14, 6))
    
    ax.plot(test_dates, actual, label='Actual', color='#2E86AB', linewidth=2, alpha=0.8)
    ax.plot(test_dates, predictions, label='Predicted', color='#A23B72', linewidth=2, alpha=0.8, linestyle='--')
    
    ax.set_xlabel('Date', fontweight='bold')
    ax.set_ylabel('Price ($/MMBtu)', fontweight='bold')
    ax.set_title('Natural Gas Price: Actual vs Predicted', fontweight='bold', fontsize=12)
    ax.legend(loc='best')
    ax.grid(True, alpha=0.3)
    
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / '4_time_series_predictions.png', bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"Saved: {OUTPUT_DIR / '4_time_series_predictions.png'}")


def plot_5_feature_importance(df):
    """Plot 5: Feature importance from Random Forest model."""
    # Train model and get feature importance
    weather_engineer = WeatherFeatureEngineer()
    storage_engineer = StorageFeatureEngineer()
    
    weather_features = weather_engineer.engineer_all_weather_features(df)
    storage_features = storage_engineer.engineer_all_storage_features(df)
    all_features = pd.concat([df, weather_features, storage_features], axis=1)
    
    # Train Elastic Net (better for feature importance with many features)
    pipeline = BaselinePipeline('elastic_net')
    pipeline.train_final_model(all_features)
    
    # Get feature importance directly from model
    try:
        model = pipeline.model.model
        feature_names = pipeline.model.feature_names
        
        # Ensure feature_names is a flat list
        if isinstance(feature_names, (list, np.ndarray)):
            feature_names = list(feature_names)
        else:
            feature_names = [str(f) for f in feature_names]
        
        if hasattr(model, 'coef_'):
            # Elastic Net or Linear Regression
            coef = model.coef_
            # Flatten if needed
            if coef.ndim > 1:
                coef = coef.flatten()
            importance = np.abs(coef)
        elif hasattr(model, 'feature_importances_'):
            # Random Forest
            importance = model.feature_importances_
            if importance.ndim > 1:
                importance = importance.flatten()
        else:
            print("Warning: Model does not support feature importance, skipping plot 5")
            return
        
        # Ensure same length and flatten importance
        importance = importance.flatten() if importance.ndim > 1 else importance
        min_len = min(len(feature_names), len(importance))
        feature_names = feature_names[:min_len]
        importance = importance[:min_len]
        
        # Create DataFrame
        importance_df = pd.DataFrame({
            'feature': feature_names,
            'importance': importance
        }).sort_values('importance', ascending=False)
        
        # Get top features
        top_n = min(15, len(importance_df))
        top_features = importance_df.head(top_n)
    except Exception as e:
        print(f"Warning: Could not get feature importance ({e}), skipping plot 5")
        import traceback
        traceback.print_exc()
        return
    
    # Create horizontal bar plot
    fig, ax = plt.subplots(figsize=(10, 8))
    
    y_pos = np.arange(len(top_features))
    ax.barh(y_pos, top_features['importance'], color='#2E86AB', alpha=0.8)
    
    ax.set_yticks(y_pos)
    ax.set_yticklabels(top_features['feature'], fontsize=9)
    ax.set_xlabel('Importance', fontweight='bold')
    ax.set_title(f'Top {len(top_features)} Most Important Features (Elastic Net)', fontweight='bold', fontsize=12)
    ax.grid(True, alpha=0.3, axis='x')
    
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / '5_feature_importance.png', bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"Saved: {OUTPUT_DIR / '5_feature_importance.png'}")


def plot_6_error_distribution(df):
    """Plot 6: Distribution of prediction errors."""
    # Train model and get predictions
    weather_engineer = WeatherFeatureEngineer()
    storage_engineer = StorageFeatureEngineer()
    
    weather_features = weather_engineer.engineer_all_weather_features(df)
    storage_features = storage_engineer.engineer_all_storage_features(df)
    all_features = pd.concat([df, weather_features, storage_features], axis=1)
    
    train_size = int(len(all_features) * 0.8)
    train_df = all_features.iloc[:train_size]
    test_df = all_features.iloc[train_size:]
    
    # Train Elastic Net
    pipeline = BaselinePipeline('elastic_net')
    pipeline.train_final_model(train_df)
    
    X_test, y_test = pipeline.model.prepare_features(test_df)
    predictions = pipeline.model.predict(X_test)
    actual = y_test.values
    
    errors = actual - predictions
    
    # Flatten errors if needed
    if errors.ndim > 1:
        errors = errors.flatten()
    
    # Create histogram
    fig, ax = plt.subplots(figsize=(10, 6))
    
    ax.hist(errors, bins=30, color='#2E86AB', alpha=0.7, edgecolor='black', linewidth=0.5)
    ax.axvline(x=0, color='red', linestyle='--', linewidth=2, label='Zero Error')
    ax.axvline(x=np.mean(errors), color='green', linestyle='--', linewidth=2, label=f'Mean: {np.mean(errors):.4f}')
    
    ax.set_xlabel('Prediction Error ($/MMBtu)', fontweight='bold')
    ax.set_ylabel('Frequency', fontweight='bold')
    ax.set_title('Distribution of Prediction Errors', fontweight='bold', fontsize=12)
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / '6_error_distribution.png', bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"Saved: {OUTPUT_DIR / '6_error_distribution.png'}")


def plot_7_correlation_heatmap(df):
    """Plot 7: Correlation heatmap of key features."""
    # Select key features for correlation
    key_features = ['spot_price', 'hdd', 'cdd', 'storage', 'temperature', 'production']
    available_features = [f for f in key_features if f in df.columns]
    
    corr_data = df[available_features].corr()
    
    # Create heatmap
    fig, ax = plt.subplots(figsize=(10, 8))
    
    sns.heatmap(corr_data, annot=True, fmt='.2f', cmap='coolwarm', center=0,
                square=True, linewidths=1, cbar_kws={"shrink": 0.8}, ax=ax,
                vmin=-1, vmax=1)
    
    ax.set_title('Feature Correlation Matrix', fontweight='bold', fontsize=12)
    
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / '7_correlation_heatmap.png', bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"Saved: {OUTPUT_DIR / '7_correlation_heatmap.png'}")


def plot_8_model_architecture():
    """Plot 8: Diagram showing model architecture/pipeline."""
    fig, ax = plt.subplots(figsize=(14, 8))
    ax.axis('off')
    
    # Define boxes
    boxes = [
        ('Data Sources', 1, 7, 2, 1),
        ('EIA API', 0.5, 6, 1, 0.8),
        ('Weather API', 1.5, 6, 1, 0.8),
        ('Power Grid', 2.5, 6, 1, 0.8),
        ('Feature Engineering', 1, 4.5, 2, 1),
        ('Weather Features', 0.5, 3.5, 1, 0.8),
        ('Storage Features', 1.5, 3.5, 1, 0.8),
        ('Models', 1, 2, 2, 1),
        ('Baseline', 0.5, 1, 1, 0.8),
        ('Tree Models', 1.5, 1, 1, 0.8),
        ('Predictions', 1, 0, 2, 0.8),
    ]
    
    # Draw boxes
    for text, x, y, w, h in boxes:
        if 'API' in text or 'Features' in text or 'Baseline' in text or 'Tree' in text:
            color = '#E8E8E8'
        elif 'Engineering' in text or 'Models' in text:
            color = '#D4E4F7'
        else:
            color = '#B8D4E3'
        
        rect = plt.Rectangle((x - w/2, y - h/2), w, h, 
                            facecolor=color, edgecolor='black', linewidth=1.5)
        ax.add_patch(rect)
        ax.text(x, y, text, ha='center', va='center', fontweight='bold', fontsize=9)
    
    # Draw arrows
    arrows = [
        (1, 6.4, 1, 5.1),
        (1, 4.1, 1, 2.6),
        (1, 1.6, 1, 0.4),
    ]
    
    for x1, y1, x2, y2 in arrows:
        ax.annotate('', xy=(x2, y2), xytext=(x1, y1),
                   arrowprops=dict(arrowstyle='->', lw=2, color='black'))
    
    ax.set_xlim(-0.5, 3)
    ax.set_ylim(-0.5, 8)
    ax.set_title('Natural Gas Price Model Pipeline', fontweight='bold', fontsize=14, pad=20)
    
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / '8_model_architecture.png', bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"Saved: {OUTPUT_DIR / '8_model_architecture.png'}")


def main():
    """Generate all poster visualizations."""
    print("Generating poster visualizations...")
    print(f"Output directory: {OUTPUT_DIR}")
    print()
    
    # Create sample data
    print("Creating sample data...")
    df = create_sample_data()
    
    # Generate all plots
    print("\nGenerating plots...")
    plot_1_data_overview(df)
    plot_2_model_comparison(df)
    plot_3_prediction_vs_actual(df)
    plot_4_time_series_predictions(df)
    plot_5_feature_importance(df)
    plot_6_error_distribution(df)
    plot_7_correlation_heatmap(df)
    plot_8_model_architecture()
    
    print("\n" + "="*50)
    print("All visualizations generated successfully!")
    print(f"Figures saved to: {OUTPUT_DIR}")
    print("="*50)


if __name__ == "__main__":
    main()

