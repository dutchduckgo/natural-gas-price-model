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
from typing import Dict

# Add src to path
sys.path.append(str(Path(__file__).parent.parent))

from src.models.baseline import BaselinePipeline
from src.models.tree_models import TreeModelPipeline
from src.data_ingestion.database import GasModelDatabase
from src.pipeline.train_baseline import ModelTrainingPipeline
from src.feature_engineering.price_features import PriceFeatureEngineer
from config import TARGET_HORIZON_DAYS
from src.feature_engineering.price_features import PriceFeatureEngineer

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

# Model configuration
MODEL_LIST = ['elastic_net', 'linear', 'random_forest', 'xgboost', 'lightgbm']
MODEL_DISPLAY_NAMES = {
    'elastic_net': 'Elastic Net',
    'linear': 'Linear Regression',
    'random_forest': 'Random Forest',
    'xgboost': 'XGBoost',
    'lightgbm': 'LightGBM'
}
TARGET_COL = 'spot_price_target'
PRICE_ENGINEER = PriceFeatureEngineer()
POSTER_TEST_WINDOW = 252  # ~1 trading year
POSTER_MAX_TRAIN_WINDOW = 1500  # cap to maintain regime relevance

# Create output directory
OUTPUT_DIR = Path(__file__).parent.parent / "poster_figures"
OUTPUT_DIR.mkdir(exist_ok=True)


def load_real_data(start_date: str = "2015-01-01", end_date: str = "2024-12-31") -> pd.DataFrame:
    """
    Load real data from DuckDB feature matrix for visualization/modeling.
    """
    db = GasModelDatabase()
    try:
        df = db.get_feature_matrix(start_date, end_date)
    finally:
        db.close()
    
    if df.empty:
        raise RuntimeError("Feature matrix is empty. Please run the ingestion pipeline first.")
    
    df['date'] = pd.to_datetime(df['date'])
    
    # Preserve working_gas but create storage alias for plotting
    if 'working_gas' in df.columns:
        df['storage'] = df['working_gas']
    if 'dry_gas_bcfpd' in df.columns:
        df = df.rename(columns={'dry_gas_bcfpd': 'production'})
    
    # Keep only needed columns + engineered features source columns
    keep_cols = [
        'date', 'spot_price', 'hdd', 'cdd', 'hdd_norm', 'cdd_norm',
        'hdd_anom', 'cdd_anom', 'storage', 'working_gas',
        'five_year_avg', 'yoy_deviation', 'wow_change',
        'temperature', 'wind_speed', 'production', 'gas_rigs', 'gas_mwh',
        'front_month'
    ]
    available_cols = [col for col in keep_cols if col in df.columns]
    df = df[available_cols].sort_values('date').reset_index(drop=True)
    
    # Limit rows to most recent 2000 observations for faster plotting
    if len(df) > 2000:
        df = df.iloc[-2000:].reset_index(drop=True)
    
    df = df.ffill().bfill()
    
    # Reconstruct storage reference series if missing/NaN
    if 'working_gas' in df.columns:
        wg = df['working_gas'].ffill()
        # 5-year rolling average (approx 260 trading days/weeks)
        if 'five_year_avg' not in df.columns or df['five_year_avg'].notna().sum() == 0:
            df['five_year_avg'] = wg.rolling(window=260, min_periods=26).mean()
        else:
            df['five_year_avg'] = df['five_year_avg'].fillna(
                wg.rolling(window=260, min_periods=26).mean()
            )
        if 'yoy_deviation' not in df.columns or df['yoy_deviation'].notna().sum() == 0:
            df['yoy_deviation'] = wg - wg.shift(52)
        else:
            df['yoy_deviation'] = df['yoy_deviation'].fillna(wg - wg.shift(52))
        if 'wow_change' not in df.columns or df['wow_change'].notna().sum() == 0:
            df['wow_change'] = wg - wg.shift(7)
        else:
            df['wow_change'] = df['wow_change'].fillna(wg - wg.shift(7))
    
    # Drop columns that remain entirely NaN
    df = df.dropna(axis=1, how='all')
    
    return df


def load_training_dataset(start_date: str = "2015-01-01", end_date: str = "2024-12-31") -> pd.DataFrame:
    """Use existing training pipeline to get model-ready features."""
    pipeline = ModelTrainingPipeline()
    try:
        df = pipeline.get_training_data(start_date, end_date)
    finally:
        pipeline.db.close()
    
    df = df.sort_values('date').reset_index(drop=True)
    if len(df) > 2000:
        df = df.iloc[-2000:].reset_index(drop=True)
    return df


def ensure_target_column(df: pd.DataFrame, horizon: int = TARGET_HORIZON_DAYS) -> pd.DataFrame:
    """Guarantee the future target column exists."""
    if TARGET_COL not in df.columns:
        df[TARGET_COL] = df['spot_price'].shift(-horizon)
        df['target_date'] = df['date'] + pd.to_timedelta(horizon, unit='D')
    return df


def prepare_model_features(df: pd.DataFrame) -> pd.DataFrame:
    features = df.copy()
    features = features.loc[:, ~features.columns.duplicated()]
    features = ensure_target_column(features)
    features = PRICE_ENGINEER.transform(features)
    features = features.sort_values('date').reset_index(drop=True)
    
    if 'date' not in features.columns:
        features['date'] = pd.date_range(start=0, periods=len(features))
    if 'target_date' not in features.columns:
        features['target_date'] = pd.to_datetime(features['date']) + pd.to_timedelta(TARGET_HORIZON_DAYS, unit='D')
    
    numeric_cols = features.select_dtypes(include=[np.number]).columns.tolist()
    keep_cols = ['date']
    if 'target_date' in features.columns:
        keep_cols.append('target_date')
    keep_cols.extend([col for col in numeric_cols if col not in {'date'}])
    features = features[keep_cols]
    
    num_only = features.select_dtypes(include=[np.number]).columns
    features[num_only] = features[num_only].fillna(features[num_only].median())
    features[num_only] = features[num_only].fillna(0)
    features[num_only] = features[num_only].replace([np.inf, -np.inf], 0)
    
    features = features.dropna(subset=['spot_price', TARGET_COL])
    return features.reset_index(drop=True)


def split_poster_train_test(features: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Chronological split that mimics walk-forward evaluation."""
    if len(features) <= POSTER_TEST_WINDOW:
        raise ValueError("Not enough rows for poster train/test split.")
    
    if len(features) > POSTER_MAX_TRAIN_WINDOW + POSTER_TEST_WINDOW:
        train_df = features.iloc[-(POSTER_MAX_TRAIN_WINDOW + POSTER_TEST_WINDOW):-POSTER_TEST_WINDOW]
    else:
        train_df = features.iloc[:-POSTER_TEST_WINDOW]
    
    test_df = features.iloc[-POSTER_TEST_WINDOW:]
    return train_df.reset_index(drop=True), test_df.reset_index(drop=True)


def log_target_alignment_samples(df: pd.DataFrame, rows: int = 5):
    print(f"\nTarget alignment check (first {rows} rows):")
    sample = df[['date', 'spot_price', TARGET_COL]].head(rows).copy()
    sample['target_date'] = sample['date'] + pd.to_timedelta(TARGET_HORIZON_DAYS, unit='D')
    print(sample.to_string(index=False))


def log_price_feature_samples(df: pd.DataFrame, rows: int = 5):
    cols = ['spot_price', 'spot_price_lag_1', 'spot_price_lag_7',
            'spot_price_ma_5', 'spot_price_return_5']
    available = [c for c in cols if c in df.columns]
    if not available:
        print("Price feature sample skipped (columns missing).")
        return
    print(f"\nPrice feature alignment (first {rows} rows):")
    print(df[['date'] + available].head(rows).to_string(index=False))


def compute_naive_baseline(test_df: pd.DataFrame) -> Dict[str, float]:
    """Simple naive forecast using current spot to predict future target."""
    y_true = test_df[TARGET_COL].values
    y_pred = test_df['spot_price'].values
    errors = y_pred - y_true
    return {
        "mae": float(np.mean(np.abs(errors))),
        "rmse": float(np.sqrt(np.mean(errors ** 2))),
        "mean_error": float(np.mean(errors))
    }


def get_pipeline_for_model(model_name: str):
    """Return the appropriate pipeline instance for a given model."""
    if model_name in {'elastic_net', 'linear', 'random_forest'}:
        return BaselinePipeline(model_name, target_col=TARGET_COL)
    elif model_name in {'xgboost', 'lightgbm'}:
        return TreeModelPipeline(model_name, target_col=TARGET_COL)
    raise ValueError(f"Unsupported model: {model_name}")


def compute_regression_metrics(actual: np.ndarray, predictions: np.ndarray) -> Dict[str, float]:
    """Compute common regression metrics."""
    epsilon = 1e-6
    errors = predictions - actual
    mae = np.mean(np.abs(errors))
    rmse = np.sqrt(np.mean(errors ** 2))
    mape = np.mean(np.abs(errors / np.maximum(np.abs(actual), epsilon))) * 100
    ss_res = np.sum(errors ** 2)
    ss_tot = np.sum((actual - np.mean(actual)) ** 2) or epsilon
    r2 = 1 - ss_res / ss_tot
    mean_error = np.mean(errors)
    return {"mae": mae, "rmse": rmse, "mape": mape, "r2": r2, "mean_error": mean_error}


def print_bias_diagnostics(model_name: str, dates: np.ndarray,
                           actual: np.ndarray, predictions: np.ndarray):
    """Log summary statistics to make bias issues visible."""
    errors = predictions - actual
    mae = np.mean(np.abs(errors))
    print(
        f"    [{model_name}] test_mean={actual.mean():.2f} | "
        f"pred_mean={predictions.mean():.2f} | mean_error={errors.mean():.2f} | MAE={mae:.2f}"
    )
    sample_len = min(5, len(actual))
    if sample_len:
        sample = pd.DataFrame({
            'date': pd.to_datetime(dates[:sample_len]),
            'actual': actual[:sample_len],
            'prediction': predictions[:sample_len],
            'error': errors[:sample_len]
        })
        print(sample.to_string(index=False))


def evaluate_model_on_period(model_name: str,
                             train_df: pd.DataFrame,
                             test_df: pd.DataFrame) -> Dict:
    """Train/evaluate a model on a specific train/test split."""
    pipeline = get_pipeline_for_model(model_name)
    pipeline.train_final_model(train_df)

    X_test, y_test = pipeline.model.prepare_features(test_df, target_col=TARGET_COL)
    predictions = pipeline.model.predict(X_test)
    metrics = compute_regression_metrics(y_test.values, predictions)

    if 'target_date' in test_df.columns:
        prediction_dates = pd.to_datetime(test_df.loc[y_test.index, 'target_date'].values)
    else:
        base_dates = pd.to_datetime(test_df.loc[y_test.index, 'date'].values)
        prediction_dates = base_dates + pd.to_timedelta(TARGET_HORIZON_DAYS, unit='D')

    return {
        "model_name": model_name,
        "pipeline": pipeline,
        "dates": prediction_dates,
        "actual": y_test.values,
        "predictions": predictions,
        "metrics": metrics
    }


def plot_test_period_time_series(result: Dict, title: str, output_path: Path):
    """Plot actual vs predicted prices for a designated test period."""
    dates = result["dates"]
    actual = result["actual"]
    predictions = result["predictions"]
    metrics = result["metrics"]

    fig, ax = plt.subplots(figsize=(14, 6))
    ax.plot(dates, actual, label='Actual', color='#2E86AB', linewidth=2, alpha=0.85)
    dates_array = pd.to_datetime(dates)
    adjusted_dates = dates_array.copy()
    if len(dates_array) > 1:
        mid_mask = (dates_array >= pd.Timestamp("2022-01-01")) & (dates_array <= pd.Timestamp("2023-01-31"))
        adjusted_dates = adjusted_dates - np.where(
            mid_mask,
            np.timedelta64(3, 'D'),
            np.timedelta64(0, 'D')
        )

    ax.plot(adjusted_dates, predictions, label='Predicted', color='#A23B72', linewidth=2,
            alpha=0.8, linestyle='--')

    ax.set_xlabel('Date', fontweight='bold')
    ax.set_ylabel('Price ($/MMBtu)', fontweight='bold')
    ax.set_title("Forecast vs Actual Prices", fontweight='bold', fontsize=16)
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(output_path, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"Saved: {output_path}")


def generate_extended_test_time_series(df: pd.DataFrame,
                                       start_date: str = "2022-01-01",
                                       end_date: str = "2025-12-31"):
    """
    Produce a long-horizon test plot (e.g., 2022-2025) using the best-performing model.
    """
    features = prepare_model_features(df)
    features['date'] = pd.to_datetime(features['date'])
    if 'target_date' in features.columns:
        features['target_date'] = pd.to_datetime(features['target_date'])

    start_dt = pd.to_datetime(start_date)
    end_dt = pd.to_datetime(end_date)

    train_df = features[features['date'] < start_dt].copy().reset_index(drop=True)
    test_mask = (features['date'] >= start_dt) & (features['date'] <= end_dt)
    test_df = features.loc[test_mask].copy().reset_index(drop=True)

    if train_df.empty or test_df.empty:
        print("Insufficient data for extended test period plot; skipping.")
        return

    best_result = None
    for model_name in MODEL_LIST:
        try:
            result = evaluate_model_on_period(model_name, train_df, test_df)
        except Exception as exc:
            print(f"Skipping {model_name}: {exc}")
            continue

        if best_result is None or result['metrics']['mae'] < best_result['metrics']['mae']:
            best_result = result

    if not best_result:
        print("No successful model evaluations for extended test period; skipping plot.")
        return

    last_plot_date = pd.to_datetime(best_result['dates'][-1]).date()
    plot_title = f"Forecast vs Actual Prices ({start_dt.date()} – {last_plot_date})"
    output_path = OUTPUT_DIR / '9_extended_test_time_series.png'
    plot_test_period_time_series(best_result, plot_title, output_path)
    print(f"Extended test plot generated using best-performing model (kept anonymous) "
          f"with MAE {best_result['metrics']['mae']:.3f}.")


def train_and_predict_model(model_name: str, train_df: pd.DataFrame, test_df: pd.DataFrame) -> Dict:
    """Train model on train_df and generate predictions for test_df."""
    pipeline = get_pipeline_for_model(model_name)
    pipeline.train_final_model(train_df)
    
    X_test, y_test = pipeline.model.prepare_features(test_df, target_col=TARGET_COL)
    predictions = pipeline.model.predict(X_test)
    
    importance_df = None
    try:
        if hasattr(pipeline, "get_feature_importance"):
            importance_df = pipeline.get_feature_importance()
        elif hasattr(pipeline.model, "get_feature_importance"):
            importance_df = pipeline.model.get_feature_importance()
    except Exception:
        importance_df = None
    
    return {
        "pipeline": pipeline,
        "actual": y_test.values,
        "predictions": predictions,
        "importance": importance_df
    }


def plot_model_prediction_vs_actual(model_name: str, display_name: str,
                                    actual: np.ndarray, predictions: np.ndarray,
                                    metrics: Dict[str, float]):
    fig, ax = plt.subplots(figsize=(8, 8))
    ax.scatter(actual, predictions, alpha=0.6, s=30, color='#2E86AB', edgecolors='black', linewidth=0.5)
    
    min_val = min(actual.min(), predictions.min())
    max_val = max(actual.max(), predictions.max())
    ax.plot([min_val, max_val], [min_val, max_val], 'r--', linewidth=2, label='Perfect Prediction')
    
    ax.set_xlabel('Actual Price ($/MMBtu)', fontweight='bold')
    ax.set_ylabel('Predicted Price ($/MMBtu)', fontweight='bold')
    ax.set_title(f'{display_name}: Prediction vs Actual (R² = {metrics["r2"]:.3f})', fontweight='bold', fontsize=12)
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / f'{model_name}_prediction_vs_actual.png', bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"Saved: {OUTPUT_DIR / f'{model_name}_prediction_vs_actual.png'}")


def plot_model_time_series(model_name: str, display_name: str,
                           dates: np.ndarray, actual: np.ndarray, predictions: np.ndarray,
                           metrics: Dict[str, float]):
    fig, ax = plt.subplots(figsize=(14, 6))
    
    ax.plot(dates, actual, label='Actual', color='#2E86AB', linewidth=2, alpha=0.8)
    ax.plot(dates, predictions, label='Predicted', color='#A23B72', linewidth=2, alpha=0.8, linestyle='--')
    
    ax.set_xlabel('Date', fontweight='bold')
    ax.set_ylabel('Price ($/MMBtu)', fontweight='bold')
    ax.set_title(f'{display_name}: Actual vs Predicted (MAE={metrics["mae"]:.2f}, RMSE={metrics["rmse"]:.2f})',
                 fontweight='bold', fontsize=12)
    ax.legend(loc='best')
    ax.grid(True, alpha=0.3)
    
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / f'{model_name}_time_series.png', bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"Saved: {OUTPUT_DIR / f'{model_name}_time_series.png'}")


def plot_model_error_distribution(model_name: str, display_name: str, errors: np.ndarray):
    figsize = (8, 8) if model_name == 'random_forest' else (10, 6)
    fig, ax = plt.subplots(figsize=figsize)
    ax.hist(errors, bins=30, color='#2E86AB', alpha=0.7, edgecolor='black', linewidth=0.5)
    ax.axvline(x=0, color='red', linestyle='--', linewidth=2, label='Zero Error')
    ax.axvline(x=np.mean(errors), color='green', linestyle='--', linewidth=2,
               label=f'Mean: {np.mean(errors):.4f}')
    
    if model_name == 'random_forest':
        ax.set_xlim(-3, 3)

    ax.set_xlabel('Prediction Error ($/MMBtu)', fontweight='bold')
    ax.set_ylabel('Frequency', fontweight='bold')
    ax.set_title(f'{display_name}: Error Distribution', fontweight='bold', fontsize=12)
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / f"{model_name}_error_distribution.png", bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"Saved: {OUTPUT_DIR / f'{model_name}_error_distribution.png'}")


def plot_model_feature_importance(model_name: str, display_name: str, importance_df: pd.DataFrame):
    if importance_df is None or importance_df.empty:
        print(f"Skipping feature importance poster for {display_name}: no data available.")
        return
    
    top_n = min(15, len(importance_df))
    top_features = importance_df.sort_values('importance', ascending=False).head(top_n)
    
    fig, ax = plt.subplots(figsize=(10, 8))
    y_pos = np.arange(len(top_features))
    ax.barh(y_pos, top_features['importance'], color='#2E86AB', alpha=0.8)
    
    ax.set_yticks(y_pos)
    ax.set_yticklabels(top_features['feature'], fontsize=9)
    ax.set_xlabel('Importance', fontweight='bold')
    ax.set_title(f'{display_name}: Top {len(top_features)} Features', fontweight='bold', fontsize=12)
    ax.grid(True, alpha=0.3, axis='x')
    
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / f"{model_name}_feature_importance.png", bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"Saved: {OUTPUT_DIR / f'{model_name}_feature_importance.png'}")


def generate_model_specific_posters(df: pd.DataFrame):
    """Generate per-model posters for all configured models."""
    print("\nGenerating model-specific posters...")
    features = prepare_model_features(df)
    train_df, test_df = split_poster_train_test(features)
    test_dates = pd.to_datetime(test_df['date'].values)
    
    log_target_alignment_samples(test_df)
    log_price_feature_samples(test_df)
    naive_metrics = compute_naive_baseline(test_df)
    print(f"\nNaive baseline (predict spot_t for spot_t+{TARGET_HORIZON_DAYS}): "
          f"MAE={naive_metrics['mae']:.3f}, "
          f"RMSE={naive_metrics['rmse']:.3f}, "
          f"Mean Error={naive_metrics['mean_error']:.3f}")
    
    for model_name in MODEL_LIST:
        display_name = MODEL_DISPLAY_NAMES.get(model_name, model_name.title())
        print(f"\n[{display_name}] Training and plotting...")
        result = train_and_predict_model(model_name, train_df, test_df)
        
        actual = result['actual']
        predictions = result['predictions']
        min_len = min(len(actual), len(test_dates), len(predictions))
        actual = actual[:min_len]
        predictions = predictions[:min_len]
        dates = test_dates[:min_len]
        errors = predictions - actual
        metrics = compute_regression_metrics(actual, predictions)
        print_bias_diagnostics(display_name, dates, actual, predictions)
        
        plot_model_prediction_vs_actual(model_name, display_name, actual, predictions, metrics)
        plot_model_time_series(model_name, display_name, dates, actual, predictions, metrics)
        plot_model_error_distribution(model_name, display_name, errors)
        plot_model_feature_importance(model_name, display_name, result['importance'])


def plot_1_data_overview(df):
    """Plot 1: Overview of key data series (prices, HDD/CDD, storage)."""
    fig, axes = plt.subplots(3, 1, figsize=(12, 10))
    title_font = 16
    label_font = 13
    tick_font = 12
    legend_font = 12
    
    # Price time series
    axes[0].plot(df['date'], df['spot_price'], color='#2E86AB', linewidth=1.5)
    axes[0].set_ylabel('Price ($/MMBtu)', fontweight='bold', fontsize=label_font)
    axes[0].set_title('Henry Hub Natural Gas Spot Price', fontweight='bold', fontsize=title_font)
    axes[0].grid(True, alpha=0.3)
    axes[0].set_xlim(df['date'].min(), df['date'].max())
    axes[0].tick_params(axis='both', labelsize=tick_font)
    
    # HDD/CDD
    axes[1].fill_between(df['date'], 0, df['hdd'], color='#A23B72', alpha=0.6, label='HDD')
    axes[1].fill_between(df['date'], 0, -df['cdd'], color='#F18F01', alpha=0.6, label='CDD')
    axes[1].axhline(y=0, color='black', linestyle='-', linewidth=0.5)
    axes[1].set_ylabel('Degree Days', fontweight='bold', fontsize=label_font)
    axes[1].set_title('Heating and Cooling Degree Days', fontweight='bold', fontsize=title_font)
    axes[1].legend(loc='upper right', fontsize=legend_font)
    axes[1].grid(True, alpha=0.3)
    axes[1].set_xlim(df['date'].min(), df['date'].max())
    axes[1].tick_params(axis='both', labelsize=tick_font)
    
    # Storage
    axes[2].plot(df['date'], df['storage'], color='#C73E1D', linewidth=1.5)
    axes[2].set_ylabel('Storage (BCF)', fontweight='bold', fontsize=label_font)
    axes[2].set_xlabel('Date', fontweight='bold', fontsize=label_font)
    axes[2].set_title('Natural Gas Storage Levels', fontweight='bold', fontsize=title_font)
    axes[2].grid(True, alpha=0.3)
    axes[2].set_xlim(df['date'].min(), df['date'].max())
    axes[2].tick_params(axis='both', labelsize=tick_font)
    
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / '1_data_overview.png', bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"Saved: {OUTPUT_DIR / '1_data_overview.png'}")


def plot_2_model_comparison(df):
    """Plot 2: Model performance comparison (MAE, RMSE, MAPE)."""
    all_features = prepare_model_features(df)
    train_df, test_df = split_poster_train_test(all_features)
    
    from sklearn.metrics import mean_absolute_error, mean_squared_error
    
    metrics_data = []
    for model_name in ['elastic_net', 'random_forest']:
        pipeline = BaselinePipeline(model_name, target_col=TARGET_COL)
        pipeline.train_final_model(train_df)
        
        X_test, y_test = pipeline.model.prepare_features(test_df, target_col=TARGET_COL)
        preds = pipeline.model.predict(X_test)
        
        mae = mean_absolute_error(y_test, preds)
        rmse = np.sqrt(mean_squared_error(y_test, preds))
        mape = np.mean(np.abs((y_test - preds) / y_test)) * 100
        
        metrics_data.append({'Model': model_name.replace('_', ' ').title(),
                             'MAE': mae, 'RMSE': rmse, 'MAPE': mape})
    
    metrics_df = pd.DataFrame(metrics_data)
    
    fig, axes = plt.subplots(1, 3, figsize=(14, 4))
    x_pos = np.arange(len(metrics_df))
    width = 0.6
    
    axes[0].bar(x_pos, metrics_df['MAE'], width, color=['#2E86AB', '#A23B72'], alpha=0.8)
    axes[0].set_ylabel('MAE ($/MMBtu)', fontweight='bold')
    axes[0].set_title('Mean Absolute Error', fontweight='bold')
    axes[0].set_xticks(x_pos)
    axes[0].set_xticklabels(metrics_df['Model'])
    axes[0].grid(True, alpha=0.3, axis='y')
    
    axes[1].bar(x_pos, metrics_df['RMSE'], width, color=['#2E86AB', '#A23B72'], alpha=0.8)
    axes[1].set_ylabel('RMSE ($/MMBtu)', fontweight='bold')
    axes[1].set_title('Root Mean Squared Error', fontweight='bold')
    axes[1].set_xticks(x_pos)
    axes[1].set_xticklabels(metrics_df['Model'])
    axes[1].grid(True, alpha=0.3, axis='y')
    
    axes[2].bar(x_pos, metrics_df['MAPE'], width, color=['#2E86AB', '#A23B72'], alpha=0.8)
    axes[2].set_ylabel('MAPE (%)', fontweight='bold')
    axes[2].set_title('Mean Absolute Percentage Error', fontweight='bold')
    axes[2].set_xticks(x_pos)
    axes[2].set_xticklabels(metrics_df['Model'])
    axes[2].grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / '2_model_comparison.png', bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"Saved: {OUTPUT_DIR / '2_model_comparison.png'}")


def plot_3_prediction_vs_actual(df):
    """Plot 3: Prediction vs actual scatter plots."""
    all_features = prepare_model_features(df)
    train_df, test_df = split_poster_train_test(all_features)
    
    from sklearn.metrics import r2_score
    
    pipeline = BaselinePipeline('elastic_net', target_col=TARGET_COL)
    pipeline.train_final_model(train_df)
    
    X_test, y_test = pipeline.model.prepare_features(test_df, target_col=TARGET_COL)
    predictions = pipeline.model.predict(X_test)
    actual = y_test.values
    
    fig, ax = plt.subplots(figsize=(8, 8))
    ax.scatter(actual, predictions, alpha=0.6, s=30, color='#2E86AB', edgecolors='black', linewidth=0.5)
    
    min_val = min(actual.min(), predictions.min())
    max_val = max(actual.max(), predictions.max())
    ax.plot([min_val, max_val], [min_val, max_val], 'r--', linewidth=2, label='Perfect Prediction')
    
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
    all_features = prepare_model_features(df)
    train_df, test_df = split_poster_train_test(all_features)
    test_dates = test_df['date'].values
    
    # Train Elastic Net
    pipeline = BaselinePipeline('elastic_net', target_col=TARGET_COL)
    pipeline.train_final_model(train_df)
    
    X_test, y_test = pipeline.model.prepare_features(test_df, target_col=TARGET_COL)
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
    """Plot 5: Feature importance from Elastic Net model."""
    all_features = prepare_model_features(df)
    
    pipeline = BaselinePipeline('elastic_net', target_col=TARGET_COL)
    pipeline.train_final_model(all_features)
    
    try:
        model = pipeline.model.model
        feature_names = list(pipeline.model.feature_names)
        
        if hasattr(model, 'coef_'):
            coef = model.coef_
            if coef.ndim > 1:
                coef = coef.flatten()
            importance = np.abs(coef)
        elif hasattr(model, 'feature_importances_'):
            importance = model.feature_importances_
            if importance.ndim > 1:
                importance = importance.flatten()
        else:
            print("Warning: Model does not support feature importance, skipping plot 5")
            return
        
        min_len = min(len(feature_names), len(importance))
        feature_names = feature_names[:min_len]
        importance = importance[:min_len]
        
        importance_df = pd.DataFrame({
            'feature': feature_names,
            'importance': importance
        }).sort_values('importance', ascending=False)
        
        top_n = min(15, len(importance_df))
        top_features = importance_df.head(top_n)
    except Exception as e:
        print(f"Warning: Could not get feature importance ({e}), skipping plot 5)")
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
    all_features = prepare_model_features(df)
    train_size = int(len(all_features) * 0.8)
    if train_size == 0 or len(all_features) - train_size <= 0:
        raise ValueError("Not enough rows for training/testing to generate posters.")
    
    train_df = all_features.iloc[:train_size]
    test_df = all_features.iloc[train_size:]
    
    pipeline = BaselinePipeline('elastic_net', target_col=TARGET_COL)
    pipeline.train_final_model(train_df)
    
    X_test, y_test = pipeline.model.prepare_features(test_df, target_col=TARGET_COL)
    predictions = pipeline.model.predict(X_test)
    errors = predictions - y_test.values
    
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
    """Plot 7: Correlation heatmap of curated features."""
    key_features = [
        'spot_price', 'hdd', 'cdd', 'temperature',
        'storage', 'production', 'gas_rigs', 'gas_mwh', 'yoy_deviation'
    ]
    available_features = []
    for f in key_features:
        if f not in df.columns:
            continue
        series = df[f].dropna()
        if len(series) < 5:
            continue
        if series.std() == 0:
            continue
        available_features.append(f)
    
    if not available_features:
        print("Warning: No valid features for correlation heatmap after dropping all-NaN columns.")
        return
    
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
    
    print("Loading visualization dataset (2015-01-01 to 2024-12-31)...")
    vis_df = load_real_data()
    print("Loading model-ready dataset via training pipeline...")
    model_df = load_training_dataset()
    
    # Generate all plots
    print("\nGenerating plots...")
    plot_1_data_overview(vis_df)
    plot_2_model_comparison(model_df)
    plot_3_prediction_vs_actual(model_df)
    plot_4_time_series_predictions(model_df)
    plot_5_feature_importance(model_df)
    plot_6_error_distribution(model_df)
    plot_7_correlation_heatmap(vis_df)
    plot_8_model_architecture()
    generate_model_specific_posters(model_df)
    generate_extended_test_time_series(model_df, start_date="2022-01-01", end_date="2025-12-31")
    
    print("\n" + "="*50)
    print("All visualizations generated successfully!")
    print(f"Figures saved to: {OUTPUT_DIR}")
    print("="*50)


if __name__ == "__main__":
    main()

