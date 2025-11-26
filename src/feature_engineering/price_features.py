"""
Price feature engineering utilities for the natural gas model.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterable, List

import pandas as pd


@dataclass
class PriceFeatureEngineer:
    """
    Create autoregressive price signals so the models can track changing regimes.

    The baseline implementation only used fundamentals (storage, weather, rigs, etc.)
    which left the models anchored to historical means whenever the test period
    experienced a rapid price drop.  By supplying price history, moving averages,
    and basis information we allow every model type to self-calibrate to the most
    recent market level and reduce systematic bias.
    """

    lag_periods: Iterable[int] = (1, 3, 5, 7, 14, 21, 30)
    moving_windows: Iterable[int] = (5, 10, 20, 60)
    volatility_windows: Iterable[int] = (5, 20)
    _generated_columns: List[str] = field(default_factory=list, init=False)

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """Return a copy of df with autoregressive price features appended."""
        if "spot_price" not in df.columns:
            return df

        result = df.sort_values("date").copy()
        price_series = pd.to_numeric(result["spot_price"], errors="coerce")

        for lag in self.lag_periods:
            col = f"spot_price_lag_{lag}"
            result[col] = price_series.shift(lag)
            self._generated_columns.append(col)

        for window in self.moving_windows:
            col_ma = f"spot_price_ma_{window}"
            # use .shift(1) to ensure only historical information feeds the avg
            result[col_ma] = price_series.shift(1).rolling(window=window, min_periods=3).mean()
            self._generated_columns.append(col_ma)

            col_ret = f"spot_price_return_{window}"
            # shift to guarantee returns only use information available prior to t
            result[col_ret] = price_series.shift(1).pct_change(periods=window)
            self._generated_columns.append(col_ret)

        for window in self.volatility_windows:
            col_vol = f"spot_price_vol_{window}"
            result[col_vol] = price_series.shift(1).rolling(window=window, min_periods=3).std()
            self._generated_columns.append(col_vol)

        if "front_month" in result.columns:
            fm = pd.to_numeric(result["front_month"], errors="coerce").shift(1)
            basis_col = "front_month_basis"
            result[basis_col] = fm - price_series.shift(1)
            self._generated_columns.append(basis_col)

        return result


__all__ = ["PriceFeatureEngineer"]

