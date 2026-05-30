"""
metrics.py — Forecast Evaluation Metrics

Implements MASE (primary), RMSE, and MAE evaluation for all 3 models.
The naïve seasonal baseline for MASE is: y_hat[t] = y[t-52] (same week last year).
Test set = final 13 weeks, held out completely.

Usage:
    from src.evaluation.metrics import compute_mase, compute_rmse, compute_mae, build_results_table
"""

import logging
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


# ── Core metric functions ───────────────────────────────────────────────────────

def compute_mase(
    actual: np.ndarray,
    forecast: np.ndarray,
    train_actual: np.ndarray,
    seasonal_period: int = 52,
) -> float:
    """
    Mean Absolute Scaled Error — primary evaluation metric.

    MASE = MAE(forecast) / MAE(naïve seasonal baseline)
    Naïve seasonal baseline: y_hat[t] = y[t - seasonal_period]

    MASE < 1.0 means the model beats the naïve seasonal baseline.

    Parameters
    ----------
    actual          : test actuals   (n_test,)
    forecast        : model forecasts (n_test,)
    train_actual    : training actuals used to compute naïve MAE denominator
    seasonal_period : lag for naïve baseline (52 = same week last year)

    Returns
    -------
    float : MASE score
    """
    actual = np.asarray(actual, dtype=float)
    forecast = np.asarray(forecast, dtype=float)
    train_actual = np.asarray(train_actual, dtype=float)

    mae_forecast = np.mean(np.abs(actual - forecast))

    # Naïve seasonal MAE on training set
    if len(train_actual) <= seasonal_period:
        logger.warning(
            "Training series shorter than seasonal_period (%d). "
            "Falling back to lag-1 naïve baseline.",
            seasonal_period,
        )
        naive_errors = np.abs(np.diff(train_actual))
    else:
        naive_errors = np.abs(
            train_actual[seasonal_period:] - train_actual[:-seasonal_period]
        )

    mae_naive = np.mean(naive_errors)

    if mae_naive == 0:
        logger.warning("Naïve MAE is zero — returning NaN for MASE.")
        return float("nan")

    return float(mae_forecast / mae_naive)


def compute_rmse(actual: np.ndarray, forecast: np.ndarray) -> float:
    """Root Mean Squared Error."""
    actual = np.asarray(actual, dtype=float)
    forecast = np.asarray(forecast, dtype=float)
    return float(np.sqrt(np.mean((actual - forecast) ** 2)))


def compute_mae(actual: np.ndarray, forecast: np.ndarray) -> float:
    """Mean Absolute Error."""
    actual = np.asarray(actual, dtype=float)
    forecast = np.asarray(forecast, dtype=float)
    return float(np.mean(np.abs(actual - forecast)))


def compute_mean_error(actual: np.ndarray, forecast: np.ndarray) -> float:
    """
    Mean Error (ME) — signed bias metric.
    Positive = over-forecast, negative = under-forecast.
    """
    actual = np.asarray(actual, dtype=float)
    forecast = np.asarray(forecast, dtype=float)
    return float(np.mean(forecast - actual))


# ── Results table builder ───────────────────────────────────────────────────────

def build_results_table(
    results: List[Dict],
) -> pd.DataFrame:
    """
    Assembles a tidy results DataFrame from a list of evaluation dicts.

    Each dict should have:
        series_id, model, MASE, RMSE, MAE
    (optional: ME, horizon)

    Returns
    -------
    pd.DataFrame with columns: [series_id, model, MASE, RMSE, MAE]
    sorted by (series_id, MASE ascending).
    """
    df = pd.DataFrame(results)
    expected_cols = {"series_id", "model", "MASE", "RMSE", "MAE"}
    missing = expected_cols - set(df.columns)
    if missing:
        raise ValueError(f"Results dicts missing columns: {missing}")

    df = df.sort_values(["series_id", "MASE"]).reset_index(drop=True)
    # Round for readability
    for col in ["MASE", "RMSE", "MAE"]:
        df[col] = df[col].round(4)
    return df


def evaluate_all_models(
    series_id: str,
    actual_test: np.ndarray,
    actual_train: np.ndarray,
    forecasts: Dict[str, np.ndarray],
    seasonal_period: int = 52,
) -> List[Dict]:
    """
    Evaluates multiple model forecasts against the same test actuals.

    Parameters
    ----------
    series_id      : identifier string for the series
    actual_test    : test set actuals (last 13 weeks)
    actual_train   : training set actuals (for MASE denominator)
    forecasts      : dict mapping model_name → forecast array
    seasonal_period: for MASE calculation (52 = weekly annual)

    Returns
    -------
    list of dicts ready for build_results_table()
    """
    rows = []
    for model_name, forecast in forecasts.items():
        mase = compute_mase(actual_test, forecast, actual_train, seasonal_period)
        rmse = compute_rmse(actual_test, forecast)
        mae = compute_mae(actual_test, forecast)
        me = compute_mean_error(actual_test, forecast)
        rows.append(
            {
                "series_id": series_id,
                "model": model_name,
                "MASE": mase,
                "RMSE": rmse,
                "MAE": mae,
                "ME": round(me, 4),
            }
        )
        logger.info(
            "[%s] %s → MASE=%.4f  RMSE=%.2f  MAE=%.2f",
            series_id, model_name, mase, rmse, mae,
        )
    return rows


def compute_rolling_mase(
    actual: pd.Series,
    forecast: pd.Series,
    train_actual: np.ndarray,
    window: int = 4,
    seasonal_period: int = 52,
) -> pd.Series:
    """
    Computes rolling MASE over a sliding window (used for drift monitoring).

    Parameters
    ----------
    actual         : time-ordered actual values
    forecast       : time-ordered forecast values (aligned with actual)
    train_actual   : training actuals for naïve denominator
    window         : rolling window size in weeks
    seasonal_period: for MASE denominator

    Returns
    -------
    pd.Series of rolling MASE values (NaN for first window-1 periods)
    """
    errors = np.abs(np.asarray(actual) - np.asarray(forecast))
    mae_rolling = pd.Series(errors).rolling(window=window, min_periods=window).mean()

    # Naïve MAE denominator (fixed from training set)
    train_arr = np.asarray(train_actual, dtype=float)
    if len(train_arr) > seasonal_period:
        naive_errors = np.abs(train_arr[seasonal_period:] - train_arr[:-seasonal_period])
    else:
        naive_errors = np.abs(np.diff(train_arr))
    mae_naive = float(np.mean(naive_errors)) if len(naive_errors) > 0 else 1.0

    return mae_rolling / (mae_naive if mae_naive > 0 else 1.0)
