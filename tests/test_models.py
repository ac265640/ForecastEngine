"""
test_models.py — Unit Tests for SARIMA, Prophet, and LSTM Forecasters

Verifies initialization, fitting, prediction shapes, diagnostic tests,
and save/load functionality for all three model classes.
Uses conditional test skipping for heavy model dependencies (Prophet, TensorFlow).
"""

import sys
from pathlib import Path
import pytest
import numpy as np
import pandas as pd

# Add project root to path
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.models.sarima_model import SARIMAForecaster


# ── Fixtures ──────────────────────────────────────────────────────────────────

@pytest.fixture
def mock_series():
    """Generates a 60-week synthetic weekly sales series."""
    dates = pd.date_range("2011-01-01", periods=60, freq="W")
    sales = 100.0 + 10.0 * np.sin(2 * np.pi * np.arange(60) / 52) + np.random.default_rng(42).normal(0, 2, 60)
    return pd.Series(sales, index=dates)


@pytest.fixture
def mock_calendar():
    """Generates a mock calendar aligned with mock_series."""
    dates = pd.date_range("2011-01-01", periods=60, freq="W")
    return pd.DataFrame({
        "date": dates,
        "is_holiday": np.zeros(60, dtype=np.int8),
        "event_name_1": [None] * 60,
        "snap_CA": np.zeros(60, dtype=np.int8),
        "snap_TX": np.zeros(60, dtype=np.int8),
        "snap_WI": np.zeros(60, dtype=np.int8),
    })


# ── SARIMA Tests ──────────────────────────────────────────────────────────────

class TestSARIMAForecaster:
    def test_fit_and_predict(self, mock_series):
        # Fit on 60 weeks, use seasonal_order s=4 to make it fast and avoid statsmodels errors with s=52 on short series
        model = SARIMAForecaster(
            series_id="CA_1__FOODS",
            order=(1, 1, 0),
            seasonal_order=(0, 0, 0, 0)
        )
        model.fit(mock_series)
        
        # Predict 4 steps
        preds = model.predict(steps=4)
        assert len(preds) == 4
        assert list(preds.columns) == ["predicted_value", "lower_bound", "upper_bound"]
        # Ensure non-negativity clip
        assert (preds >= 0).all().all()

    def test_diagnostics(self, mock_series):
        model = SARIMAForecaster(
            series_id="CA_1__FOODS",
            order=(1, 1, 0),
            seasonal_order=(0, 0, 0, 0)
        )
        model.fit(mock_series)
        diag = model.diagnostics(lags=5)
        
        assert "lb_stat" in diag
        assert "lb_pvalue" in diag
        assert "passed" in diag
        assert "aic" in diag
        assert "bic" in diag
        assert len(diag["residuals"]) > 0

    def test_save_load_roundtrip(self, mock_series, tmp_path):
        model = SARIMAForecaster(
            series_id="CA_1__FOODS",
            order=(1, 1, 0),
            seasonal_order=(0, 0, 0, 0)
        )
        model.fit(mock_series)
        
        save_path = model.save(str(tmp_path), version="test")
        
        # Load and verify
        loaded = SARIMAForecaster.load(save_path)
        assert loaded.series_id == "CA_1__FOODS"
        assert loaded.order == (1, 1, 0)
        
        # Compare predictions
        orig_preds = model.predict(steps=3)
        load_preds = loaded.predict(steps=3)
        assert np.allclose(orig_preds.values, load_preds.values)


# ── Prophet Tests ─────────────────────────────────────────────────────────────

class TestProphetForecaster:
    def test_prophet_model(self, mock_series, mock_calendar, tmp_path):
        # Skip test if prophet is not installed
        prophet = pytest.importorskip("prophet")
        from src.models.prophet_model import ProphetForecaster
        
        model = ProphetForecaster(
            series_id="CA_1__FOODS",
            state="CA",
            weekly_seasonality=False,
            yearly_seasonality=False
        )
        
        model.fit(mock_series, calendar_df=mock_calendar)
        
        # Predict
        preds = model.predict(steps=4)
        assert len(preds) == 4
        assert list(preds.columns) == ["ds", "predicted_value", "lower_bound", "upper_bound"]
        
        # Components
        comps = model.get_components()
        assert "trend" in comps
        
        # Save & Load roundtrip
        save_path = model.save(str(tmp_path), version="test")
        loaded = ProphetForecaster.load(save_path)
        assert loaded.series_id == "CA_1__FOODS"
        assert loaded.state == "CA"


# ── LSTM Tests ────────────────────────────────────────────────────────────────

class TestLSTMForecaster:
    def test_lstm_model(self, mock_series, tmp_path):
        # Skip test if tensorflow is not installed
        tf = pytest.importorskip("tensorflow")
        from src.models.lstm_model import LSTMForecaster
        
        train_vals = mock_series.iloc[:-10].values
        val_vals = mock_series.iloc[-10:].values
        
        # Small window size and epochs for fast unit test
        model = LSTMForecaster(
            series_id="CA_1__FOODS",
            window_size=12,
            lstm_units=8
        )
        
        model.fit(train_vals, val_vals, epochs=2, batch_size=4, verbose=0)
        
        # Predict (needs last 12 values)
        last_window = train_vals[-12:]
        preds = model.predict(steps=3, last_window=last_window)
        assert len(preds) == 3
        
        # Save & Load roundtrip
        paths = model.save(str(tmp_path), version="test")
        loaded = LSTMForecaster.load(
            model_path=paths["model_path"],
            scaler_path=paths["scaler_path"],
            series_id="CA_1__FOODS"
        )
        
        loaded_preds = loaded.predict(steps=3, last_window=last_window)
        assert len(loaded_preds) == 3
