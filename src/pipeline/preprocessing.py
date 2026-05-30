"""
preprocessing.py — Temporal Splits & Scalers

Enforces strict temporal train/val/test splits and fits per-series
MinMaxScalers on training windows only (no data leakage).

Usage:
    from src.pipeline.preprocessing import TemporalSplitter, SeriesScaler

    splitter = TemporalSplitter(train_frac=0.8, val_frac=0.1)
    train, val, test = splitter.split(series_df)

    scaler = SeriesScaler()
    train_scaled = scaler.fit_transform(train)
    val_scaled   = scaler.transform(val)
    test_scaled  = scaler.transform(test)
"""

import logging
from typing import Dict, Tuple

import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import joblib
from pathlib import Path

logger = logging.getLogger(__name__)

# Final 13 weeks are the test set (as specified)
TEST_WEEKS = 13


class TemporalSplitter:
    """
    Splits a time-series DataFrame into train / val / test using strict
    temporal ordering. **Never shuffles the data.**

    The test set is always the last TEST_WEEKS (13) weeks.
    The val set is the next val_frac portion before test.
    The remainder is train.
    """

    def __init__(self, val_frac: float = 0.10, test_weeks: int = TEST_WEEKS):
        """
        Parameters
        ----------
        val_frac   : fraction of total length allocated to validation
        test_weeks : fixed number of final weeks for test set
        """
        self.val_frac = val_frac
        self.test_weeks = test_weeks

    def split(
        self, df: pd.DataFrame, date_col: str = "week_start"
    ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """
        Splits df (already sorted by date_col) into (train, val, test).

        Parameters
        ----------
        df       : single-series or multi-series DataFrame sorted by date_col
        date_col : name of the datetime column

        Returns
        -------
        train, val, test DataFrames (no overlapping dates)
        """
        df = df.sort_values(date_col).reset_index(drop=True)
        n = len(df)

        test_start = n - self.test_weeks
        val_size = max(1, int(n * self.val_frac))
        val_start = test_start - val_size

        train = df.iloc[:val_start].copy()
        val = df.iloc[val_start:test_start].copy()
        test = df.iloc[test_start:].copy()

        logger.info(
            "Temporal split → train: %d rows, val: %d rows, test: %d rows",
            len(train), len(val), len(test),
        )
        return train, val, test

    def split_series(
        self, df: pd.DataFrame, series_col: str = "series_id", date_col: str = "week_start"
    ) -> Dict[str, Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]]:
        """
        Splits a multi-series DataFrame independently per series.

        Returns
        -------
        dict: {series_id: (train_df, val_df, test_df)}
        """
        result = {}
        for sid, group in df.groupby(series_col):
            result[sid] = self.split(group, date_col=date_col)
        logger.info("Split %d series into train/val/test.", len(result))
        return result


class SeriesScaler:
    """
    Per-series MinMaxScaler fitted **only on the training window**.

    Prevents data leakage by ensuring val/test are transformed using
    the scaler parameters derived from training data only.
    """

    def __init__(self, feature_range: Tuple[float, float] = (0.0, 1.0)):
        self.feature_range = feature_range
        self._scalers: Dict[str, MinMaxScaler] = {}

    def fit(self, series_id: str, train_values: np.ndarray) -> "SeriesScaler":
        """Fit scaler for a specific series on training data."""
        scaler = MinMaxScaler(feature_range=self.feature_range)
        scaler.fit(train_values.reshape(-1, 1))
        self._scalers[series_id] = scaler
        return self

    def transform(self, series_id: str, values: np.ndarray) -> np.ndarray:
        """Transform using the previously fitted scaler for this series."""
        if series_id not in self._scalers:
            raise KeyError(
                f"No scaler fitted for series '{series_id}'. Call fit() first."
            )
        return self._scalers[series_id].transform(values.reshape(-1, 1)).flatten()

    def fit_transform(self, series_id: str, train_values: np.ndarray) -> np.ndarray:
        """Fit on train_values and return the scaled train values."""
        self.fit(series_id, train_values)
        return self.transform(series_id, train_values)

    def inverse_transform(self, series_id: str, scaled_values: np.ndarray) -> np.ndarray:
        """Undo scaling — converts back to original scale."""
        if series_id not in self._scalers:
            raise KeyError(f"No scaler fitted for series '{series_id}'.")
        return self._scalers[series_id].inverse_transform(
            scaled_values.reshape(-1, 1)
        ).flatten()

    def save(self, path: str):
        """Persist all scalers to disk via joblib."""
        joblib.dump(self._scalers, path)
        logger.info("Saved %d scalers → %s", len(self._scalers), path)

    @classmethod
    def load(cls, path: str) -> "SeriesScaler":
        """Load scalers from a joblib file."""
        obj = cls()
        obj._scalers = joblib.load(path)
        logger.info("Loaded %d scalers from %s", len(obj._scalers), path)
        return obj


def make_lstm_windows(
    series: np.ndarray, window_size: int = 52
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Creates sliding-window (X, y) pairs for LSTM training.

    Parameters
    ----------
    series      : 1-D array of scaled values (training split only)
    window_size : number of past weeks used as input (default 52 = 1 year)

    Returns
    -------
    X : shape (n_samples, window_size, 1)
    y : shape (n_samples,)
    """
    X, y = [], []
    for i in range(window_size, len(series)):
        X.append(series[i - window_size: i])
        y.append(series[i])
    X = np.array(X).reshape(-1, window_size, 1)
    y = np.array(y)
    return X, y
