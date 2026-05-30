"""
test_pipeline.py — Unit Tests for Data Pipeline & Preprocessing

Tests M5DataLoader missing files, FeatureEngineer features/lags,
TemporalSplitter logic, SeriesScaler transforms, and make_lstm_windows shapes.
"""

import sys
from pathlib import Path
import pytest
import numpy as np
import pandas as pd

# Add project root to path
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.pipeline.data_loader import M5DataLoader
from src.pipeline.feature_engineering import FeatureEngineer
from src.pipeline.preprocessing import TemporalSplitter, SeriesScaler, make_lstm_windows
from src.pipeline.eda import run_adf_test, run_stl_decomposition, detect_anomalies, rolling_stats


# ── M5 Data Loader Tests ──────────────────────────────────────────────────────

class TestM5DataLoader:
    def test_missing_files_error(self, tmp_path):
        """M5DataLoader should raise FileNotFoundError if any M5 file is missing."""
        loader = M5DataLoader(raw_dir=tmp_path)
        with pytest.raises(FileNotFoundError, match="Missing M5 dataset files"):
            loader.load_and_aggregate()


# ── Feature Engineering Tests ─────────────────────────────────────────────────

class TestFeatureEngineer:
    @pytest.fixture
    def mock_weekly_sales(self):
        """Create a mock weekly sales dataset for 60 weeks for two series."""
        dates = pd.date_range("2011-01-01", periods=60, freq="W")
        
        # Series 1: CA_1__FOODS
        s1 = pd.DataFrame({
            "series_id": "CA_1__FOODS",
            "store_id": "CA_1",
            "cat_id": "FOODS",
            "week_id": [d.strftime("%G-W%V") for d in dates],
            "week_start": dates,
            "total_sales": np.arange(10, 70, dtype=float), # trend from 10 to 69
            "snap_CA": np.ones(60, dtype=np.int8),
            "snap_TX": np.zeros(60, dtype=np.int8),
            "snap_WI": np.zeros(60, dtype=np.int8),
            "is_holiday": np.zeros(60, dtype=np.int8),
            "is_thanksgiving": np.zeros(60, dtype=np.int8),
            "is_black_friday": np.zeros(60, dtype=np.int8),
        })
        
        # Series 2: TX_2__HOBBIES
        s2 = pd.DataFrame({
            "series_id": "TX_2__HOBBIES",
            "store_id": "TX_2",
            "cat_id": "HOBBIES",
            "week_id": [d.strftime("%G-W%V") for d in dates],
            "week_start": dates,
            "total_sales": np.arange(100, 160, dtype=float), # trend from 100 to 159
            "snap_CA": np.zeros(60, dtype=np.int8),
            "snap_TX": np.ones(60, dtype=np.int8),
            "snap_WI": np.zeros(60, dtype=np.int8),
            "is_holiday": np.zeros(60, dtype=np.int8),
            "is_thanksgiving": np.zeros(60, dtype=np.int8),
            "is_black_friday": np.zeros(60, dtype=np.int8),
        })
        
        return pd.concat([s1, s2], ignore_index=True)

    def test_cyclical_features(self, mock_weekly_sales):
        fe = FeatureEngineer()
        enriched = fe.transform(mock_weekly_sales)
        
        assert "week_sin" in enriched.columns
        assert "week_cos" in enriched.columns
        assert "month_sin" in enriched.columns
        assert "month_cos" in enriched.columns
        assert enriched["week_sin"].min() >= -1.0
        assert enriched["week_sin"].max() <= 1.0

    def test_state_specific_snap(self, mock_weekly_sales):
        fe = FeatureEngineer()
        enriched = fe.transform(mock_weekly_sales)
        
        # CA_1__FOODS should look at snap_CA (which is 1)
        ca_row = enriched[enriched["series_id"] == "CA_1__FOODS"].iloc[0]
        assert ca_row["is_snap_week"] == 1
        
        # TX_2__HOBBIES should look at snap_TX (which is 1)
        tx_row = enriched[enriched["series_id"] == "TX_2__HOBBIES"].iloc[0]
        assert tx_row["is_snap_week"] == 1

    def test_lag_features_grouped_correctly(self, mock_weekly_sales):
        fe = FeatureEngineer()
        enriched = fe.transform(mock_weekly_sales)
        
        # 52 weeks lag test
        ca_series = enriched[enriched["series_id"] == "CA_1__FOODS"].sort_values("week_start")
        # index 52 should have lag value of index 0 (which has total_sales = 10)
        assert ca_series.iloc[52]["sales_lag_52"] == pytest.approx(10.0)
        assert pd.isna(ca_series.iloc[0]["sales_lag_52"])
        
        # Rolling stats test (rolling mean of shift 1)
        # For index 4, sales are [10, 11, 12, 13], mean = 11.5
        assert ca_series.iloc[4]["rolling_4w_mean"] == pytest.approx(11.5)


# ── Preprocessing & Splitter Tests ─────────────────────────────────────────────

class TestPreprocessing:
    def test_temporal_splitter(self):
        dates = pd.date_range("2011-01-01", periods=100, freq="W")
        df = pd.DataFrame({
            "week_start": dates,
            "total_sales": np.arange(100),
        })
        
        splitter = TemporalSplitter(val_frac=0.10, test_weeks=13)
        train, val, test = splitter.split(df)
        
        assert len(test) == 13
        assert len(val) == 10
        assert len(train) == 77
        
        # Verify no overlap and strict temporal sorting
        assert train["week_start"].max() < val["week_start"].min()
        assert val["week_start"].max() < test["week_start"].min()

    def test_scaler_no_leakage(self):
        scaler = SeriesScaler()
        train_vals = np.array([10.0, 20.0, 30.0, 40.0])
        val_vals = np.array([5.0, 50.0]) # outside train range
        
        # Fit transform on train
        scaled_train = scaler.fit_transform("CA_1", train_vals)
        assert scaled_train.min() == 0.0
        assert scaled_train.max() == 1.0
        
        # Transform val (should be [ -0.166, 1.33 ] relative to train min=10, max=40)
        scaled_val = scaler.transform("CA_1", val_vals)
        assert scaled_val[0] < 0.0
        assert scaled_val[1] > 1.0
        
        # Inverse transform
        inverted = scaler.inverse_transform("CA_1", scaled_train)
        assert np.allclose(inverted, train_vals)

    def test_make_lstm_windows(self):
        series = np.arange(100, dtype=float)
        window_size = 52
        X, y = make_lstm_windows(series, window_size=window_size)
        
        # 100 - 52 = 48 samples
        assert X.shape == (48, 52, 1)
        assert y.shape == (48,)
        
        # Verify first window elements
        assert np.allclose(X[0, :, 0], np.arange(52))
        assert y[0] == 52.0


# ── EDA Tests ─────────────────────────────────────────────────────────────────

class TestEDA:
    def test_rolling_stats(self):
        series = pd.Series([10.0, 20.0, 30.0, 40.0])
        res = rolling_stats(series, window=2)
        assert len(res) == 4
        assert res.loc[1, "rolling_mean"] == pytest.approx(15.0)

    def test_run_adf_test(self):
        # A simple linear trend should be non-stationary
        series = pd.Series(np.arange(50, dtype=float))
        adf = run_adf_test(series)
        assert "status" in adf
        assert "p_value" in adf

    def test_stl_decomposition(self):
        # Weekly data of 2 years (104 weeks) with strong seasonality
        t = np.arange(104)
        seasonal = 10 * np.sin(2 * np.pi * t / 52)
        trend = 0.5 * t
        noise = np.random.default_rng(0).normal(0, 1, 104)
        series = pd.Series(100 + trend + seasonal + noise)
        
        stl = run_stl_decomposition(series, period=52)
        assert "trend" in stl
        assert "seasonal" in stl
        assert "residual" in stl
        assert len(stl["trend"]) == 104

    def test_detect_anomalies(self):
        dates = pd.date_range("2011-01-01", periods=10, freq="W")
        sales = pd.Series([10, 12, 11, 100, 13, 11, 10, 12, 11, 12]) # index 3 is anomaly
        residuals = pd.Series([0, 1, 0, 80, 2, 0, -1, 1, 0, 1], dtype=float)
        
        # Calendar events mock
        calendar_df = pd.DataFrame({
            "date": dates,
            "is_holiday": [0, 0, 0, 1, 0, 0, 0, 0, 0, 0],
            "event_name_1": ["", "", "", "MockEvent", "", "", "", "", "", ""],
            "week_start": dates,
        })
        
        anom = detect_anomalies(sales, residuals, calendar_df=calendar_df, index=dates, sigma_threshold=1.5)
        assert len(anom) == 10
        # Index 3 anomaly should be flagged as explainable due to MockEvent
        assert anom.iloc[3]["is_anomaly"] == True
        assert anom.iloc[3]["anomaly_class"] == "explainable"
        assert anom.iloc[3]["event_label"] == "MockEvent"
