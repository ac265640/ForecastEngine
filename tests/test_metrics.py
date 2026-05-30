"""
test_metrics.py — Unit Tests for Evaluation Metrics

Tests MASE correctness, RMSE, MAE, results table builder,
and rolling MASE computation.
"""

import pytest
import numpy as np
import pandas as pd
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.evaluation.metrics import (
    compute_mase,
    compute_rmse,
    compute_mae,
    compute_mean_error,
    build_results_table,
    evaluate_all_models,
    compute_rolling_mase,
)


# ── Fixtures ──────────────────────────────────────────────────────────────────

@pytest.fixture
def perfect_forecast():
    """Forecast equals actual — MASE, RMSE, MAE should all be 0."""
    actual = np.array([100.0, 110.0, 105.0, 120.0, 115.0])
    return actual, actual.copy()


@pytest.fixture
def naïve_forecast():
    """Forecast equals naïve baseline — MASE should be exactly 1.0."""
    # 60-week training series with seasonal period 52
    n_train = 60
    rng = np.random.default_rng(0)
    train = 100 + rng.normal(0, 5, n_train)
    actual = train[52:60]   # weeks 52–59
    # Naïve forecast = same week last year (lag-52)
    naïve = train[:8]       # weeks 0–7 (the lag-52 values)
    return actual, naïve, train


@pytest.fixture
def sample_train():
    rng = np.random.default_rng(42)
    return 200 + rng.normal(0, 20, 104)


# ── MASE tests ────────────────────────────────────────────────────────────────

class TestMASE:
    def test_perfect_forecast_mase_is_zero(self, perfect_forecast, sample_train):
        actual, forecast = perfect_forecast
        mase = compute_mase(actual, forecast, sample_train)
        assert mase == pytest.approx(0.0, abs=1e-10)

    def test_naive_forecast_mase_is_one(self, naïve_forecast):
        actual, naïve, train = naïve_forecast
        mase = compute_mase(actual, naïve, train, seasonal_period=52)
        assert mase == pytest.approx(1.0, rel=0.05), (
            f"MASE of naïve forecast should be ≈1.0, got {mase:.4f}"
        )

    def test_better_than_naive_mase_below_one(self, sample_train):
        actual = sample_train[-13:]
        train = sample_train[:-13]
        # Forecast = actuals with tiny noise → MASE < 1
        rng = np.random.default_rng(0)
        forecast = actual + rng.normal(0, 1, len(actual))
        mase = compute_mase(actual, forecast, train)
        assert mase < 1.0, f"Good forecast should have MASE < 1.0, got {mase:.4f}"

    def test_mase_nan_when_naive_zero(self):
        """If all training values are identical, naïve MAE = 0 → MASE is NaN."""
        train = np.full(60, 100.0)
        actual = np.array([100.0, 100.0])
        forecast = np.array([110.0, 90.0])
        mase = compute_mase(actual, forecast, train)
        assert np.isnan(mase)

    def test_mase_short_train_fallback(self):
        """If train shorter than seasonal_period, uses lag-1 fallback gracefully."""
        train = np.array([10, 12, 14, 16, 18, 20], dtype=float)
        actual = np.array([22.0, 24.0])
        forecast = np.array([21.0, 23.0])
        mase = compute_mase(actual, forecast, train, seasonal_period=52)
        assert not np.isnan(mase)
        assert mase >= 0


# ── RMSE / MAE tests ──────────────────────────────────────────────────────────

class TestRMSEMAE:
    def test_rmse_perfect(self, perfect_forecast):
        actual, forecast = perfect_forecast
        assert compute_rmse(actual, forecast) == pytest.approx(0.0, abs=1e-10)

    def test_mae_perfect(self, perfect_forecast):
        actual, forecast = perfect_forecast
        assert compute_mae(actual, forecast) == pytest.approx(0.0, abs=1e-10)

    def test_rmse_known_value(self):
        actual = np.array([1.0, 2.0, 3.0])
        forecast = np.array([2.0, 2.0, 2.0])
        # errors = [-1, 0, 1], MSE = 2/3, RMSE = sqrt(2/3)
        expected = np.sqrt(2 / 3)
        assert compute_rmse(actual, forecast) == pytest.approx(expected)

    def test_mae_known_value(self):
        actual = np.array([1.0, 2.0, 3.0])
        forecast = np.array([2.0, 2.0, 2.0])
        # errors = [1, 0, 1], MAE = 2/3
        assert compute_mae(actual, forecast) == pytest.approx(2 / 3)

    def test_rmse_always_geq_mae(self):
        rng = np.random.default_rng(99)
        actual = rng.normal(100, 10, 50)
        forecast = rng.normal(100, 10, 50)
        assert compute_rmse(actual, forecast) >= compute_mae(actual, forecast)

    def test_mean_error_sign(self):
        actual = np.array([100.0, 100.0, 100.0])
        forecast = np.array([110.0, 110.0, 110.0])   # over-forecast
        me = compute_mean_error(actual, forecast)
        assert me > 0, "Over-forecast should have positive ME"

    def test_mean_error_negative_when_under(self):
        actual = np.array([100.0, 100.0])
        forecast = np.array([90.0, 90.0])             # under-forecast
        me = compute_mean_error(actual, forecast)
        assert me < 0


# ── Results table tests ───────────────────────────────────────────────────────

class TestResultsTable:
    def test_build_results_table_basic(self):
        rows = [
            {"series_id": "CA_1__FOODS", "model": "SARIMA", "MASE": 0.8, "RMSE": 50.0, "MAE": 40.0},
            {"series_id": "CA_1__FOODS", "model": "Prophet", "MASE": 0.9, "RMSE": 55.0, "MAE": 44.0},
        ]
        df = build_results_table(rows)
        assert len(df) == 2
        assert list(df.columns) >= ["series_id", "model", "MASE", "RMSE", "MAE"]
        # Should be sorted by MASE ascending
        assert df.iloc[0]["MASE"] <= df.iloc[1]["MASE"]

    def test_build_results_table_missing_column(self):
        rows = [{"series_id": "X", "model": "SARIMA", "MASE": 0.8}]
        with pytest.raises(ValueError, match="missing columns"):
            build_results_table(rows)

    def test_evaluate_all_models_returns_correct_structure(self, sample_train):
        actual = sample_train[-13:]
        train = sample_train[:-13]
        rng = np.random.default_rng(7)
        forecasts = {
            "SARIMA": actual + rng.normal(0, 5, 13),
            "Prophet": actual + rng.normal(0, 8, 13),
        }
        results = evaluate_all_models("TEST__SERIES", actual, train, forecasts)
        assert len(results) == 2
        for r in results:
            assert "series_id" in r
            assert "model" in r
            assert "MASE" in r
            assert r["MASE"] >= 0


# ── Rolling MASE tests ────────────────────────────────────────────────────────

class TestRollingMASE:
    def test_rolling_mase_output_length(self, sample_train):
        actual = pd.Series(sample_train[-13:])
        forecast = pd.Series(sample_train[-13:] * 1.05)
        train = sample_train[:-13]
        rolling = compute_rolling_mase(actual, forecast, train, window=4)
        assert len(rolling) == len(actual)

    def test_rolling_mase_first_values_nan(self, sample_train):
        actual = pd.Series(sample_train[-13:])
        forecast = pd.Series(sample_train[-13:] * 1.1)
        train = sample_train[:-13]
        rolling = compute_rolling_mase(actual, forecast, train, window=4)
        # First (window-1) values should be NaN
        assert rolling.iloc[:3].isna().all()

    def test_rolling_mase_perfect_forecast_is_zero(self, sample_train):
        actual = pd.Series(sample_train[-13:])
        train = sample_train[:-13]
        rolling = compute_rolling_mase(actual, actual, train, window=4)
        # After warmup, rolling MASE of perfect forecast should be 0
        assert rolling.dropna().max() == pytest.approx(0.0, abs=1e-10)
