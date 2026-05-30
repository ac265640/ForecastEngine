"""
prophet_model.py — Prophet Forecaster

Additive model with:
    - Piecewise linear trend
    - Fourier seasonality (weekly + yearly)
    - Custom holiday effects: SNAP days, Thanksgiving, Black Friday

Usage:
    from src.models.prophet_model import ProphetForecaster

    model = ProphetForecaster(series_id="CA_1__FOODS", state="CA")
    model.fit(train_df, calendar_df)
    forecast_df = model.predict(steps=13)
    components = model.get_components()
    model.save("artifacts/models/")
"""

import logging
import warnings
from pathlib import Path
from typing import Dict, Optional

import joblib
import mlflow
import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)
warnings.filterwarnings("ignore")


class ProphetForecaster:
    """
    Wraps Facebook/Meta Prophet for one (store_id × cat_id) series.

    Prophet requires a DataFrame with columns 'ds' (datetime) and 'y' (target).

    Produces:
        - Point forecasts
        - Uncertainty intervals (yhat_lower, yhat_upper)
        - Component decomposition (trend, weekly, yearly, holiday effects)
    """

    # State → SNAP event name prefix (for holiday dataframe)
    SNAP_STATE_MAP = {"CA": "SNAP_CA", "TX": "SNAP_TX", "WI": "SNAP_WI"}

    def __init__(
        self,
        series_id: str,
        state: Optional[str] = None,
        seasonality_mode: str = "additive",
        weekly_seasonality: bool = True,
        yearly_seasonality: bool = True,
        changepoint_prior_scale: float = 0.05,
        seasonality_prior_scale: float = 10.0,
        holidays_prior_scale: float = 10.0,
    ):
        self.series_id = series_id
        self.state = state  # "CA", "TX", or "WI"
        self.seasonality_mode = seasonality_mode
        self.weekly_seasonality = weekly_seasonality
        self.yearly_seasonality = yearly_seasonality
        self.changepoint_prior_scale = changepoint_prior_scale
        self.seasonality_prior_scale = seasonality_prior_scale
        self.holidays_prior_scale = holidays_prior_scale
        self._model = None
        self._future_df = None

    # ── Fit ─────────────────────────────────────────────────────────────────────

    def fit(
        self,
        train_series: pd.Series,
        calendar_df: Optional[pd.DataFrame] = None,
    ) -> "ProphetForecaster":
        """
        Fit Prophet model.

        Parameters
        ----------
        train_series : time-ordered pd.Series with DatetimeIndex (week_start)
        calendar_df  : processed calendar DataFrame (for holiday encoding)
        """
        try:
            from prophet import Prophet
        except ImportError:
            raise ImportError("prophet not installed. Run: pip install prophet")

        holidays_df = self._build_holidays_df(train_series, calendar_df)

        m = Prophet(
            seasonality_mode=self.seasonality_mode,
            weekly_seasonality=self.weekly_seasonality,
            yearly_seasonality=self.yearly_seasonality,
            changepoint_prior_scale=self.changepoint_prior_scale,
            seasonality_prior_scale=self.seasonality_prior_scale,
            holidays_prior_scale=self.holidays_prior_scale,
            holidays=holidays_df,
        )

        prophet_df = self._to_prophet_df(train_series)
        logger.info(
            "Fitting Prophet on %s (%d obs, holidays=%s)",
            self.series_id,
            len(prophet_df),
            "yes" if holidays_df is not None else "no",
        )
        m.fit(prophet_df)
        self._model = m
        logger.info("Prophet fit complete for %s", self.series_id)
        return self

    # ── Predict ─────────────────────────────────────────────────────────────────

    def predict(self, steps: int = 13) -> pd.DataFrame:
        """
        Generate forecasts for the next `steps` weeks.

        Returns
        -------
        pd.DataFrame with columns:
            ds, predicted_value, lower_bound, upper_bound
        """
        self._check_fitted()
        future = self._model.make_future_dataframe(periods=steps, freq="W")
        forecast = self._model.predict(future)
        self._future_df = forecast

        result = forecast[["ds", "yhat", "yhat_lower", "yhat_upper"]].tail(steps).copy()
        result = result.rename(columns={
            "yhat": "predicted_value",
            "yhat_lower": "lower_bound",
            "yhat_upper": "upper_bound",
        })
        result[["predicted_value", "lower_bound", "upper_bound"]] = result[
            ["predicted_value", "lower_bound", "upper_bound"]
        ].clip(lower=0.0)
        return result.reset_index(drop=True)

    def get_components(self) -> Dict[str, pd.Series]:
        """
        Returns component decomposition after predict() has been called.

        Returns
        -------
        dict with keys: trend, weekly, yearly, holidays (if applicable)
        """
        self._check_fitted()
        if self._future_df is None:
            raise RuntimeError("Call .predict() before .get_components()")

        components = {"trend": self._future_df["trend"]}
        for comp in ["weekly", "yearly", "holidays"]:
            if comp in self._future_df.columns:
                components[comp] = self._future_df[comp]
        return components

    def predict_in_sample(self) -> pd.DataFrame:
        """Returns in-sample fitted values (for training residual analysis)."""
        self._check_fitted()
        future = self._model.make_future_dataframe(periods=0, freq="W")
        forecast = self._model.predict(future)
        return forecast[["ds", "yhat", "yhat_lower", "yhat_upper"]].rename(columns={
            "yhat": "predicted_value",
            "yhat_lower": "lower_bound",
            "yhat_upper": "upper_bound",
        })

    # ── Save / Load ─────────────────────────────────────────────────────────────

    def save(self, save_dir: str, version: str = "v1") -> str:
        """Saves the fitted Prophet model via joblib."""
        self._check_fitted()
        path = Path(save_dir) / f"prophet_{self.series_id}_{version}.pkl"
        path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(
            {
                "model": self._model,
                "series_id": self.series_id,
                "state": self.state,
            },
            path,
        )
        logger.info("Saved Prophet model → %s", path)
        return str(path)

    @classmethod
    def load(cls, path: str) -> "ProphetForecaster":
        """Loads a saved Prophet model from disk."""
        data = joblib.load(path)
        obj = cls(series_id=data["series_id"], state=data.get("state"))
        obj._model = data["model"]
        logger.info("Loaded Prophet model from %s", path)
        return obj

    def log_to_mlflow(
        self,
        val_mase: float,
        run_id: Optional[str] = None,
    ) -> str:
        """Logs Prophet model params and metrics to MLflow."""
        with mlflow.start_run(run_id=run_id) as run:
            mlflow.log_params({
                "series_id": self.series_id,
                "state": self.state,
                "model_type": "Prophet",
                "seasonality_mode": self.seasonality_mode,
                "changepoint_prior_scale": self.changepoint_prior_scale,
            })
            mlflow.log_metrics({"val_mase": val_mase})
            return run.info.run_id

    # ── Private helpers ─────────────────────────────────────────────────────────

    @staticmethod
    def _to_prophet_df(series: pd.Series) -> pd.DataFrame:
        """Converts a pd.Series with DatetimeIndex to Prophet's {ds, y} format."""
        if isinstance(series.index, pd.DatetimeIndex):
            df = series.reset_index()
            df.columns = ["ds", "y"]
        else:
            df = pd.DataFrame({"ds": series.index, "y": series.values})
        df["ds"] = pd.to_datetime(df["ds"])
        df["y"] = df["y"].astype(float)
        return df

    def _build_holidays_df(
        self,
        train_series: pd.Series,
        calendar_df: Optional[pd.DataFrame],
    ) -> Optional[pd.DataFrame]:
        """
        Builds a Prophet-compatible holidays DataFrame from calendar.csv.

        Includes:
            - Thanksgiving
            - Black Friday
            - SNAP days (for this series' state if known)
            - Other named events from event_name_1
        """
        if calendar_df is None:
            logger.warning("No calendar_df provided — no holiday effects in Prophet.")
            return None

        cal = calendar_df.copy()
        cal["date"] = pd.to_datetime(cal["date"])

        holiday_rows = []

        # Named holidays from event_name_1
        events = cal[cal["event_name_1"].notna()][["date", "event_name_1"]].copy()
        events = events.rename(columns={"date": "ds", "event_name_1": "holiday"})
        holiday_rows.append(events)

        # SNAP days for this series' state
        if self.state and self.state in self.SNAP_STATE_MAP:
            snap_col = f"snap_{self.state}"
            if snap_col in cal.columns:
                snap_days = cal[cal[snap_col] == 1][["date"]].copy()
                snap_days["holiday"] = self.SNAP_STATE_MAP[self.state]
                snap_days = snap_days.rename(columns={"date": "ds"})
                holiday_rows.append(snap_days)

        if not holiday_rows:
            return None

        holidays = pd.concat(holiday_rows, ignore_index=True)
        holidays["ds"] = pd.to_datetime(holidays["ds"])
        holidays = holidays.drop_duplicates(subset=["ds", "holiday"])
        return holidays

    def _check_fitted(self):
        if self._model is None:
            raise RuntimeError(
                f"Prophet model '{self.series_id}' not fitted. Call .fit() first."
            )
