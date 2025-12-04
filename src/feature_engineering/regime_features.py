"""
Regime-aware feature engineering for natural gas price model.

This module creates features that identify different market regimes (e.g., 
high/low volatility, price level states) to enable regime-aware modeling.
All features are computed using only backward-looking information to prevent
look-ahead bias.
"""
import pandas as pd
import numpy as np
from typing import Dict, Optional, Any
import logging
from sklearn.cluster import KMeans

logger = logging.getLogger(__name__)


class RegimeFeatureEngineer:
    """
    Engineer regime-aware features for natural gas price modeling.
    
    Creates features that identify market regimes based on:
    - Volatility levels (low/medium/high)
    - Price levels (low/medium/high)
    - Regime transitions
    
    All features use only backward-looking rolling windows to prevent
    look-ahead bias.
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize regime feature engineer.
        
        Args:
            config: Configuration dict with keys:
                - volatility_window: Window for volatility calculation (default: 20)
                - price_window: Window for price level calculation (default: 60)
                - regime_method: "volatility", "price_level", or "kmeans" (default: "volatility")
                - n_regimes: Number of regimes to identify (default: 3)
        """
        if config is None:
            config = {}
        
        self.volatility_window = config.get("volatility_window", 20)
        self.price_window = config.get("price_window", 60)
        self.regime_method = config.get("regime_method", "volatility")
        self.n_regimes = config.get("n_regimes", 3)
        
    def _compute_volatility_regime(self, price_series: pd.Series) -> pd.Series:
        """
        Compute volatility regime labels based on rolling standard deviation of returns.
        
        Uses backward-looking rolling windows only. Returns 0=low, 1=medium, 2=high.
        """
        # Compute returns (backward-looking)
        returns = price_series.pct_change()
        
        # Rolling volatility (backward-looking)
        rolling_vol = returns.rolling(window=self.volatility_window, min_periods=5).std()
        
        # Compute percentiles for regime thresholds (using expanding window to avoid look-ahead)
        # Use expanding window up to current point
        vol_regime = pd.Series(index=price_series.index, dtype=int)
        
        for i in range(len(price_series)):
            if i < self.volatility_window:
                # Not enough data yet
                vol_regime.iloc[i] = 1  # Default to medium
            else:
                # Use historical data up to current point
                historical_vol = rolling_vol.iloc[:i+1]
                if historical_vol.notna().sum() < 10:
                    vol_regime.iloc[i] = 1
                else:
                    current_vol = rolling_vol.iloc[i]
                    if pd.isna(current_vol):
                        vol_regime.iloc[i] = 1
                    else:
                        # Use percentiles from historical data
                        q33 = historical_vol.quantile(0.33)
                        q67 = historical_vol.quantile(0.67)
                        
                        if current_vol <= q33:
                            vol_regime.iloc[i] = 0  # Low volatility
                        elif current_vol <= q67:
                            vol_regime.iloc[i] = 1  # Medium volatility
                        else:
                            vol_regime.iloc[i] = 2  # High volatility
        
        return vol_regime
    
    def _compute_price_level_regime(self, price_series: pd.Series) -> pd.Series:
        """
        Compute price level regime labels based on z-score bands or k-means.
        
        Uses backward-looking rolling windows only. Returns 0=low, 1=medium, 2=high.
        """
        # Compute rolling mean and std (backward-looking)
        rolling_mean = price_series.rolling(window=self.price_window, min_periods=10).mean()
        rolling_std = price_series.rolling(window=self.price_window, min_periods=10).std()
        
        # Z-score (backward-looking)
        z_score = (price_series - rolling_mean) / rolling_std
        
        # Regime based on z-score bands
        price_regime = pd.Series(index=price_series.index, dtype=int)
        
        for i in range(len(price_series)):
            if i < self.price_window:
                price_regime.iloc[i] = 1  # Default to medium
            else:
                current_z = z_score.iloc[i]
                if pd.isna(current_z):
                    price_regime.iloc[i] = 1
                else:
                    if current_z < -0.5:
                        price_regime.iloc[i] = 0  # Low price
                    elif current_z <= 0.5:
                        price_regime.iloc[i] = 1  # Medium price
                    else:
                        price_regime.iloc[i] = 2  # High price
        
        return price_regime
    
    def _compute_kmeans_regime(self, price_series: pd.Series) -> pd.Series:
        """
        Compute regime labels using k-means clustering on price features.
        
        Uses only backward-looking information by fitting k-means on historical
        data up to each point.
        """
        # Create features: price level and rolling mean
        rolling_mean = price_series.rolling(window=self.price_window, min_periods=10).mean()
        
        # Prepare feature matrix
        features = pd.DataFrame({
            'price': price_series,
            'rolling_mean': rolling_mean
        }).dropna()
        
        regime_labels = pd.Series(index=price_series.index, dtype=int)
        regime_labels[:] = 1  # Default to medium
        
        # Fit k-means incrementally (using only past data)
        min_samples = max(50, self.n_regimes * 10)
        
        for i in range(min_samples, len(features)):
            # Use only data up to current point
            historical_features = features.iloc[:i+1]
            
            if len(historical_features) < min_samples:
                continue
            
            # Fit k-means on historical data
            try:
                kmeans = KMeans(n_clusters=self.n_regimes, random_state=42, n_init=10)
                historical_labels = kmeans.fit_predict(historical_features[['price', 'rolling_mean']].fillna(0))
                
                # Assign label for current point
                current_idx = historical_features.index[i]
                regime_labels.loc[current_idx] = historical_labels[-1]
            except Exception as e:
                logger.warning(f"K-means fitting failed at index {i}: {e}")
                regime_labels.iloc[i] = 1
        
        return regime_labels
    
    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Apply regime feature engineering to dataframe.
        
        Args:
            df: DataFrame with 'spot_price' column and 'date' index/column
            
        Returns:
            DataFrame with added regime features:
            - volatility_regime: 0=low, 1=medium, 2=high volatility
            - price_level_regime: 0=low, 1=medium, 2=high price level
            - regime_label: Primary regime label (based on config)
            - volatility: Rolling volatility (std of returns)
            - price_zscore: Z-score of price relative to rolling mean
        """
        if 'spot_price' not in df.columns:
            logger.warning("spot_price not found; skipping regime features")
            return df
        
        result = df.copy()
        price_series = pd.to_numeric(result['spot_price'], errors='coerce')
        
        if price_series.isna().all():
            logger.warning("All spot_price values are NaN; skipping regime features")
            return result
        
        # Compute returns and volatility
        returns = price_series.pct_change()
        rolling_vol = returns.rolling(window=self.volatility_window, min_periods=5).std()
        
        # Compute volatility regime
        volatility_regime = self._compute_volatility_regime(price_series)
        result['volatility_regime'] = volatility_regime
        result['volatility'] = rolling_vol
        
        # Compute price level regime
        price_level_regime = self._compute_price_level_regime(price_series)
        result['price_level_regime'] = price_level_regime
        
        # Compute primary regime label based on method
        if self.regime_method == "volatility":
            result['regime_label'] = volatility_regime
        elif self.regime_method == "price_level":
            result['regime_label'] = price_level_regime
        elif self.regime_method == "kmeans":
            result['regime_label'] = self._compute_kmeans_regime(price_series)
        else:
            # Default to volatility
            result['regime_label'] = volatility_regime
            logger.warning(f"Unknown regime_method '{self.regime_method}'; using volatility")
        
        # Additional features
        rolling_mean = price_series.rolling(window=self.price_window, min_periods=10).mean()
        rolling_std = price_series.rolling(window=self.price_window, min_periods=10).std()
        result['price_zscore'] = (price_series - rolling_mean) / rolling_std
        
        # Regime transition indicator (1 if regime changed from previous period)
        result['regime_transition'] = (result['regime_label'].diff().abs() > 0).astype(int)
        
        logger.info(f"Added regime features: volatility_regime, price_level_regime, regime_label, "
                   f"volatility, price_zscore, regime_transition")
        
        return result

