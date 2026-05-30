"""
eda.py — Exploratory Data Analysis Module

Provides reusable, importable functions for:
    - ADF stationarity testing
    - STL decomposition (period=52, robust=True)
    - ACF / PACF computation and plotting
    - Anomaly detection: STL residuals > 3σ, classified as
      "explainable" (event-linked) or "unexplained"
    - Rolling 4-week mean and std

All functions return structured data (dicts / DataFrames) so they can
be consumed by both the pipeline runner and the Streamlit dashboard.

Usage:
    from src.pipeline.eda import (
        run_adf_test, run_stl_decomposition, compute_acf_pacf,
        detect_anomalies, rolling_stats
    )
"""

import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from statsmodels.tsa.stattools import adfuller, acf, pacf
from statsmodels.tsa.seasonal import STL
import matplotlib
matplotlib.use("Agg")  # headless backend — safe in pipeline context
import matplotlib.pyplot as plt

logger = logging.getLogger(__name__)

# ── ADF Stationarity Test ───────────────────────────────────────────────────────

def run_adf_test(
    series: pd.Series,
    significance: float = 0.05,
) -> Dict:
    """
    Augmented Dickey-Fuller test for unit root (non-stationarity).

    Parameters
    ----------
    series       : time-ordered sales values (no NaNs)
    significance : p-value threshold; default 0.05

    Returns
    -------
    dict with keys:
        status        : "STATIONARY" | "NON-STATIONARY"
        p_value       : float
        test_stat     : float
        critical_values : dict  (1%, 5%, 10% critical values)
        lags_used     : int
    """
    series = series.dropna()
    result = adfuller(series, autolag="AIC")
    p_value = float(result[1])
    status = "STATIONARY" if p_value < significance else "NON-STATIONARY"
    return {
        "status": status,
        "p_value": round(p_value, 6),
        "test_stat": round(float(result[0]), 4),
        "critical_values": {k: round(v, 4) for k, v in result[4].items()},
        "lags_used": int(result[2]),
    }


# ── STL Decomposition ──────────────────────────────────────────────────────────

def run_stl_decomposition(
    series: pd.Series,
    period: int = 52,
    robust: bool = True,
) -> Dict:
    """
    STL (Seasonal and Trend decomposition using Loess).

    Parameters
    ----------
    series : time-ordered numeric values (weekly, typically)
    period : seasonal period — 52 for weekly annual seasonality
    robust : use robust fitting to reduce outlier influence

    Returns
    -------
    dict with keys:
        trend      : pd.Series
        seasonal   : pd.Series
        residual   : pd.Series
        observed   : pd.Series
    """
    series = series.dropna().reset_index(drop=True)
    stl = STL(series, period=period, robust=robust)
    result = stl.fit()
    return {
        "observed": pd.Series(result.observed, name="observed"),
        "trend": pd.Series(result.trend, name="trend"),
        "seasonal": pd.Series(result.seasonal, name="seasonal"),
        "residual": pd.Series(result.resid, name="residual"),
    }


# ── ACF / PACF ─────────────────────────────────────────────────────────────────

def compute_acf_pacf(
    series: pd.Series,
    nlags: int = 60,
    alpha: float = 0.05,
) -> Dict:
    """
    Computes ACF and PACF values for SARIMA order selection.

    Returns
    -------
    dict with keys:
        acf_values    : np.ndarray
        acf_confint   : np.ndarray  shape (nlags+1, 2)
        pacf_values   : np.ndarray
        pacf_confint  : np.ndarray  shape (nlags+1, 2)
        lags          : np.ndarray  (0 … nlags)
    """
    series = series.dropna()
    acf_vals, acf_ci = acf(series, nlags=nlags, alpha=alpha)
    pacf_vals, pacf_ci = pacf(series, nlags=nlags, alpha=alpha, method="ywm")
    lags = np.arange(len(acf_vals))
    return {
        "acf_values": acf_vals,
        "acf_confint": acf_ci,
        "pacf_values": pacf_vals,
        "pacf_confint": pacf_ci,
        "lags": lags,
    }


def plot_acf_pacf(
    series: pd.Series,
    series_id: str = "series",
    nlags: int = 60,
    save_dir: Optional[Path] = None,
) -> plt.Figure:
    """
    Plots ACF and PACF side-by-side. Saves to PNG if save_dir is provided.

    Returns the matplotlib Figure object (usable in Streamlit via st.pyplot).
    """
    data = compute_acf_pacf(series, nlags=nlags)
    lags = data["lags"]

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle(f"ACF / PACF — {series_id}", fontsize=14)

    for ax, vals, ci, label in [
        (axes[0], data["acf_values"],  data["acf_confint"],  "ACF"),
        (axes[1], data["pacf_values"], data["pacf_confint"], "PACF"),
    ]:
        ax.bar(lags, vals, color="#4C72B0", alpha=0.7, width=0.4)
        # Confidence bands
        lower = ci[:, 0] - vals
        upper = ci[:, 1] - vals
        ax.fill_between(lags, lower, upper, alpha=0.2, color="gray", label="95% CI")
        ax.axhline(0, color="black", linewidth=0.8)
        ax.set_xlabel("Lag (weeks)")
        ax.set_ylabel(label)
        ax.set_title(label)
        ax.legend()

    plt.tight_layout()

    if save_dir:
        save_dir = Path(save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)
        path = save_dir / f"{series_id}_acf_pacf.png"
        fig.savefig(path, dpi=120, bbox_inches="tight")
        logger.info("Saved ACF/PACF plot → %s", path)

    return fig


# ── Anomaly Detection ──────────────────────────────────────────────────────────

def detect_anomalies(
    series: pd.Series,
    residuals: pd.Series,
    calendar_df: Optional[pd.DataFrame] = None,
    sigma_threshold: float = 3.0,
    index: Optional[pd.Series] = None,
) -> pd.DataFrame:
    """
    Detects anomalies in STL residuals (> sigma_threshold × σ).
    Cross-references calendar_df to classify each anomaly as:
        "explainable"  — anomaly occurs on or within 1 week of a calendar event
        "unexplained"  — no matching calendar event

    Parameters
    ----------
    series           : original sales series (for context)
    residuals        : STL residual component
    calendar_df      : processed calendar DataFrame with 'week_start' + 'is_holiday' / event columns
    sigma_threshold  : number of σ above which a residual is flagged
    index            : optional week_start DatetimeIndex aligned to series

    Returns
    -------
    DataFrame with columns:
        week_index, sales, residual, is_anomaly,
        anomaly_class ("explainable" | "unexplained" | "normal"),
        event_label (name of event if explainable, else "")
    """
    residuals = residuals.reset_index(drop=True)
    series = series.reset_index(drop=True)

    sigma = residuals.std()
    mean = residuals.mean()
    is_anomaly = (residuals - mean).abs() > sigma_threshold * sigma

    result = pd.DataFrame({
        "week_index": index.values if index is not None else np.arange(len(series)),
        "sales": series.values,
        "residual": residuals.values,
        "is_anomaly": is_anomaly.values,
        "anomaly_class": "normal",
        "event_label": "",
    })

    anomaly_idx = result[result["is_anomaly"]].index

    if calendar_df is not None and len(anomaly_idx) > 0:
        cal = calendar_df.copy()
        if "week_start" not in cal.columns:
            cal["week_start"] = pd.to_datetime(cal["date"]) - pd.to_timedelta(
                pd.to_datetime(cal["date"]).dt.dayofweek, unit="D"
            )
        cal["week_start"] = pd.to_datetime(cal["week_start"])

        # Build set of event weeks
        event_weeks = cal[cal["is_holiday"] == 1][["week_start", "event_name_1"]].dropna()
        event_weeks = event_weeks.drop_duplicates("week_start")
        event_week_set = set(event_weeks["week_start"])

        event_name_map = dict(
            zip(event_weeks["week_start"], event_weeks["event_name_1"])
        )

        if index is not None:
            idx_dates = pd.to_datetime(index.values)
        else:
            idx_dates = pd.Series([pd.NaT] * len(series))

        for i in anomaly_idx:
            if index is not None:
                anom_week = idx_dates[i]
                # Check ±1 week window
                window = {
                    anom_week,
                    anom_week - pd.Timedelta(weeks=1),
                    anom_week + pd.Timedelta(weeks=1),
                }
                matched = window & event_week_set
                if matched:
                    nearest = min(matched, key=lambda w: abs((w - anom_week).days))
                    result.at[i, "anomaly_class"] = "explainable"
                    result.at[i, "event_label"] = event_name_map.get(nearest, "event")
                    continue

            result.at[i, "anomaly_class"] = "unexplained"

    else:
        # No calendar available — all anomalies are unexplained
        result.loc[anomaly_idx, "anomaly_class"] = "unexplained"

    return result


# ── Rolling Stats ──────────────────────────────────────────────────────────────

def rolling_stats(
    series: pd.Series,
    window: int = 4,
) -> pd.DataFrame:
    """
    Computes rolling mean and std for trend visualization.

    Parameters
    ----------
    series : time-ordered numeric values
    window : rolling window size in weeks (default 4)

    Returns
    -------
    DataFrame with columns: value, rolling_mean, rolling_std
    """
    df = pd.DataFrame({"value": series.values})
    df["rolling_mean"] = df["value"].rolling(window=window, min_periods=1).mean()
    df["rolling_std"] = df["value"].rolling(window=window, min_periods=1).std().fillna(0)
    return df


# ── Convenience: full EDA report for one series ─────────────────────────────────

def run_full_eda(
    series: pd.Series,
    series_id: str,
    calendar_df: Optional[pd.DataFrame] = None,
    save_dir: Optional[Path] = None,
) -> Dict:
    """
    Runs the complete EDA suite on one series and returns a structured report.

    Returns
    -------
    dict with keys: adf, stl, acf_pacf, anomalies, rolling
    """
    logger.info("Running EDA for series: %s", series_id)

    adf_result = run_adf_test(series)
    stl_result = run_stl_decomposition(series)
    acf_pacf_data = compute_acf_pacf(series)
    plot_acf_pacf(series, series_id=series_id, save_dir=save_dir)
    anomaly_df = detect_anomalies(
        series,
        stl_result["residual"],
        calendar_df=calendar_df,
    )
    roll = rolling_stats(series)

    return {
        "series_id": series_id,
        "adf": adf_result,
        "stl": stl_result,
        "acf_pacf": acf_pacf_data,
        "anomalies": anomaly_df,
        "rolling": roll,
    }
