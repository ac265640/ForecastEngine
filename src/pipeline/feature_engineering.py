"""
feature_engineering.py — Calendar Feature Engineering

Encodes SNAP days, holidays (Thanksgiving, Black Friday), week-of-year
cyclical features, and rolling lag features from the M5 calendar.

Usage:
    from src.pipeline.feature_engineering import FeatureEngineer
    fe = FeatureEngineer(calendar_df)
    enriched = fe.transform(weekly_sales_df)
"""

import logging
from typing import Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


class FeatureEngineer:
    """
    Adds calendar-derived and lag features to the weekly aggregated sales DataFrame.

    Expected input columns (from M5DataLoader):
        series_id, store_id, cat_id, week_id, week_start, total_sales,
        snap_CA, snap_TX, snap_WI, is_holiday, is_thanksgiving, is_black_friday
    """

    # State → SNAP column mapping
    STATE_SNAP = {"CA": "snap_CA", "TX": "snap_TX", "WI": "snap_WI"}

    def __init__(self, calendar_df: Optional[pd.DataFrame] = None):
        """
        Parameters
        ----------
        calendar_df : optional raw calendar DataFrame (from M5DataLoader.load_calendar).
                      Needed only if you want to pull extra event metadata.
        """
        self.calendar_df = calendar_df

    # ── Public API ──────────────────────────────────────────────────────────────

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Full feature engineering pass. Returns enriched DataFrame with:
            - Cyclical week-of-year (sin/cos)
            - Cyclical month (sin/cos)
            - State-specific SNAP flag per series
            - Lag features: sales_lag_52 (same week last year)
            - Rolling features: rolling_4w_mean, rolling_4w_std
            - Year and quarter
        """
        df = df.copy()
        df = self._ensure_datetime(df)
        df = self._add_cyclical_features(df)
        df = self._add_snap_for_state(df)
        df = self._add_temporal_features(df)
        df = self._add_lag_features(df)
        logger.info(
            "Feature engineering complete — %d columns total", len(df.columns)
        )
        return df

    # ── Private helpers ─────────────────────────────────────────────────────────

    @staticmethod
    def _ensure_datetime(df: pd.DataFrame) -> pd.DataFrame:
        if not pd.api.types.is_datetime64_any_dtype(df["week_start"]):
            df["week_start"] = pd.to_datetime(df["week_start"])
        return df

    @staticmethod
    def _add_cyclical_features(df: pd.DataFrame) -> pd.DataFrame:
        """
        Encodes week-of-year and month as sine/cosine pairs so the model
        sees the circular nature of annual seasonality.
        """
        week_num = df["week_start"].dt.isocalendar().week.astype(float)
        month_num = df["week_start"].dt.month.astype(float)

        df["week_sin"] = np.sin(2 * np.pi * week_num / 52.0)
        df["week_cos"] = np.cos(2 * np.pi * week_num / 52.0)
        df["month_sin"] = np.sin(2 * np.pi * month_num / 12.0)
        df["month_cos"] = np.cos(2 * np.pi * month_num / 12.0)
        return df

    @staticmethod
    def _add_temporal_features(df: pd.DataFrame) -> pd.DataFrame:
        """Adds year and quarter columns."""
        df["year"] = df["week_start"].dt.year
        df["quarter"] = df["week_start"].dt.quarter
        df["week_of_year"] = df["week_start"].dt.isocalendar().week.astype(int)
        return df

    @staticmethod
    def _add_snap_for_state(df: pd.DataFrame) -> pd.DataFrame:
        """
        Creates a unified 'is_snap_week' column: 1 if the store's state had
        SNAP benefits that week. Each store's state is encoded in store_id
        (e.g., CA_1 → CA).
        """
        state_snap_map = {"CA": "snap_CA", "TX": "snap_TX", "WI": "snap_WI"}

        def get_snap(row):
            state = row["store_id"].split("_")[0]
            col = state_snap_map.get(state)
            return row[col] if col and col in row.index else 0

        df["is_snap_week"] = df.apply(get_snap, axis=1).astype(np.int8)
        return df

    @staticmethod
    def _add_lag_features(df: pd.DataFrame) -> pd.DataFrame:
        """
        Adds per-series lag features:
            - sales_lag_52: sales from 52 weeks ago (same week last year)
            - rolling_4w_mean: 4-week rolling mean of sales
            - rolling_4w_std:  4-week rolling std of sales
        All computed per series_id to avoid cross-series leakage.
        """
        df = df.sort_values(["series_id", "week_start"])

        df["sales_lag_52"] = (
            df.groupby("series_id")["total_sales"]
            .shift(52)
        )

        df["rolling_4w_mean"] = (
            df.groupby("series_id")["total_sales"]
            .transform(lambda x: x.shift(1).rolling(window=4, min_periods=1).mean())
        )

        df["rolling_4w_std"] = (
            df.groupby("series_id")["total_sales"]
            .transform(lambda x: x.shift(1).rolling(window=4, min_periods=1).std())
        )

        return df


# ── Convenience function ────────────────────────────────────────────────────────

def engineer_features(df: pd.DataFrame, calendar_df: Optional[pd.DataFrame] = None) -> pd.DataFrame:
    """Shortcut: applies full feature engineering pipeline."""
    return FeatureEngineer(calendar_df=calendar_df).transform(df)
