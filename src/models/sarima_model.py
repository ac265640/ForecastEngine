"""
sarima_model.py — SARIMA Forecaster

SARIMA(p,d,q)(P,D,Q)[s=52] — one model per store-category series.
Orders guided by ACF/PACF analysis; defaults to (1,1,1)(1,1,1)[52].
Includes Ljung-Box residual diagnostic.

Usage:
    from src.models.sarima_model import SARIMAForecaster

    model = SARIMAForecaster(series_id="CA_1__FOODS", order=(1,1,1), seasonal_order=(1,1,1,52))
    model.fit(train_series)
    forecast_df = model.predict(steps=13)
    diagnostics = model.diagnostics()
    model.save("artifacts/models/")
"""

import logging
import warnings
from pathlib import Path
from typing import Dict, Optional, Tuple

import joblib
import mlflow
import numpy as np
import pandas as pd
from statsmodels.stats.diagnostic import acorr_ljungbox
from statsmodels.tsa.statespace.sarimax import SARIMAX

logger = logging.getLogger(__name__)
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

# Sensible defaults derived from ACF/PACF analysis of M5 weekly data:
#   - d=1 differencing achieves stationarity on most non-stationary series
#   - p=1, q=1 captures short-term AR/MA dynamics
#   - P=1, D=1, Q=1, s=52 captures annual weekly seasonality
DEFAULT_ORDER = (1, 1, 1)
DEFAULT_SEASONAL_ORDER = (1, 1, 1, 52)


class SARIMAForecaster:
    """
    Wraps statsmodels SARIMAX for one (store_id × cat_id) series.

    Produces:
        - Point forecasts
        - Analytical confidence intervals (from SARIMAX)
        - Ljung-Box diagnostic (pass/fail on residuals)
    """

    def __init__(
        self,
        series_id: str,
        order: Tuple[int, int, int] = DEFAULT_ORDER,
        seasonal_order: Tuple[int, int, int, int] = DEFAULT_SEASONAL_ORDER,
        enforce_stationarity: bool = False,
        enforce_invertibility: bool = False,
    ):
        self.series_id = series_id
        self.order = order
        self.seasonal_order = seasonal_order
        self.enforce_stationarity = enforce_stationarity
        self.enforce_invertibility = enforce_invertibility
        self._fitted_model = None
        self._train_series = None

    # ── Fit ─────────────────────────────────────────────────────────────────────

    def fit(self, train_series: pd.Series) -> "SARIMAForecaster":
        """
        Fit SARIMA on the training series.

        Parameters
        ----------
        train_series : time-ordered pd.Series (index = week_start datetimes)
        """
        self._train_series = train_series.astype(float)
        logger.info(
            "Fitting SARIMA%s × %s on %s (%d obs)",
            self.order, self.seasonal_order, self.series_id, len(train_series),
        )
        model = SARIMAX(
            self._train_series,
            order=self.order,
            seasonal_order=self.seasonal_order,
            enforce_stationarity=self.enforce_stationarity,
            enforce_invertibility=self.enforce_invertibility,
            trend="n",
        )
        self._fitted_model = model.fit(disp=False)
        logger.info(
            "SARIMA fit complete — AIC=%.2f", self._fitted_model.aic
        )
        return self

    # ── Predict ─────────────────────────────────────────────────────────────────

    def predict(
        self,
        steps: int = 13,
        alpha: float = 0.05,
    ) -> pd.DataFrame:
        """
        Generate point forecasts and analytical confidence intervals.

        Parameters
        ----------
        steps : forecast horizon in weeks
        alpha : significance level for CI (default 0.05 → 95% CI)

        Returns
        -------
        pd.DataFrame with columns:
            predicted_value, lower_bound, upper_bound
        """
        self._check_fitted()
        forecast = self._fitted_model.get_forecast(steps=steps)
        summary = forecast.summary_frame(alpha=alpha)

        result = pd.DataFrame({
            "predicted_value": summary["mean"].values,
            "lower_bound": summary["mean_ci_lower"].values,
            "upper_bound": summary["mean_ci_upper"].values,
        })
        # Clip to non-negative (sales can't be negative)
        result = result.clip(lower=0.0)
        return result

    # ── Diagnostics ─────────────────────────────────────────────────────────────

    def diagnostics(self, lags: int = 10) -> Dict:
        """
        Ljung-Box test on model residuals.

        Returns
        -------
        dict with keys:
            lb_stat    : float (Ljung-Box test statistic)
            lb_pvalue  : float
            passed     : bool  (True = residuals are white noise → good fit)
            aic        : float
            bic        : float
            residuals  : pd.Series
        """
        self._check_fitted()
        resid = self._fitted_model.resid.dropna()
        lb = acorr_ljungbox(resid, lags=[lags], return_df=True)
        lb_stat = float(lb["lb_stat"].iloc[0])
        lb_pvalue = float(lb["lb_pvalue"].iloc[0])
        passed = lb_pvalue > 0.05

        return {
            "lb_stat": round(lb_stat, 4),
            "lb_pvalue": round(lb_pvalue, 6),
            "passed": passed,
            "aic": round(self._fitted_model.aic, 2),
            "bic": round(self._fitted_model.bic, 2),
            "residuals": resid,
        }

    # ── Save / Load ─────────────────────────────────────────────────────────────

    def save(self, save_dir: str, version: str = "v1") -> str:
        """
        Persists the fitted model via joblib.

        Returns
        -------
        str : path where model was saved
        """
        self._check_fitted()
        path = Path(save_dir) / f"sarima_{self.series_id}_{version}.pkl"
        path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(
            {
                "fitted_model": self._fitted_model,
                "order": self.order,
                "seasonal_order": self.seasonal_order,
                "series_id": self.series_id,
            },
            path,
        )
        logger.info("Saved SARIMA model → %s", path)
        return str(path)

    @classmethod
    def load(cls, path: str) -> "SARIMAForecaster":
        """Loads a previously saved SARIMA model from disk."""
        data = joblib.load(path)
        obj = cls(
            series_id=data["series_id"],
            order=data["order"],
            seasonal_order=data["seasonal_order"],
        )
        obj._fitted_model = data["fitted_model"]
        logger.info("Loaded SARIMA model from %s", path)
        return obj

    def log_to_mlflow(
        self,
        val_mase: float,
        artifact_path: str = "sarima_model",
        run_id: Optional[str] = None,
    ) -> str:
        """
        Logs model params, metrics, and artifact to an MLflow run.

        Returns
        -------
        str : MLflow run_id
        """
        with mlflow.start_run(run_id=run_id) as run:
            mlflow.log_params({
                "series_id": self.series_id,
                "order": str(self.order),
                "seasonal_order": str(self.seasonal_order),
                "model_type": "SARIMA",
            })
            mlflow.log_metrics({"val_mase": val_mase})
            diag = self.diagnostics()
            mlflow.log_metrics({
                "aic": diag["aic"],
                "bic": diag["bic"],
                "ljungbox_pvalue": diag["lb_pvalue"],
            })
            return run.info.run_id

    # ── Private helpers ─────────────────────────────────────────────────────────

    def _check_fitted(self):
        if self._fitted_model is None:
            raise RuntimeError(
                f"Model '{self.series_id}' has not been fitted yet. Call .fit() first."
            )
