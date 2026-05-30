"""
drift_detection.py — Production Drift & Alert System

Implements three drift monitors:
    1. Rolling MASE alert   : 4-week MASE > 1.15 → Slack alert
    2. Forecast bias alert  : |ME| > 10% of mean demand for 3 consecutive weeks
    3. PSI shift detection  : PSI > 0.2 on input features → trigger retrain

All alerts send a structured payload to Slack (via webhook) or log if
SLACK_WEBHOOK_URL is not set.

Usage:
    from src.production.drift_detection import DriftDetector

    detector = DriftDetector()
    detector.check_rolling_mase(series_id, mase_history)
    detector.check_forecast_bias(series_id, error_history, mean_demand)
    psi = detector.compute_psi(reference_dist, current_dist)
"""

import logging
import os
from typing import List, Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

SLACK_WEBHOOK_URL = os.getenv("SLACK_WEBHOOK_URL", "")
DASHBOARD_BASE_URL = os.getenv(
    "DASHBOARD_URL", "http://localhost:8501"
)

# Alert thresholds (from spec)
MASE_ALERT_THRESHOLD = 1.15       # 4-week rolling MASE > 1.15
BIAS_PCT_THRESHOLD = 0.10          # |ME| > 10% of mean demand
BIAS_CONSECUTIVE_WEEKS = 3         # must persist for 3 consecutive weeks
PSI_RETRAIN_THRESHOLD = 0.20       # PSI > 0.2 → trigger retrain


class DriftDetector:
    """
    Monitors live forecast quality and input feature distributions.
    Sends structured Slack alerts or logs warnings if webhook not configured.
    """

    def __init__(
        self,
        slack_webhook_url: Optional[str] = None,
        dashboard_url: Optional[str] = None,
    ):
        self.slack_url = slack_webhook_url or SLACK_WEBHOOK_URL
        self.dashboard_url = dashboard_url or DASHBOARD_BASE_URL

    # ── Rolling MASE Alert ────────────────────────────────────────────────────

    def check_rolling_mase(
        self,
        series_id: str,
        mase_history: List[float],
        window: int = 4,
    ) -> bool:
        """
        Checks if 4-week rolling MASE exceeds threshold.

        Parameters
        ----------
        series_id    : e.g. "CA_1__FOODS"
        mase_history : list of recent MASE values (most-recent last)
        window       : rolling window size (default 4)

        Returns
        -------
        bool : True if alert was triggered
        """
        if len(mase_history) < window:
            return False

        rolling_mase = float(np.mean(mase_history[-window:]))

        if rolling_mase > MASE_ALERT_THRESHOLD:
            msg = (
                f"🚨 *MASE Alert* | Series: `{series_id}`\n"
                f"Rolling {window}-week MASE = *{rolling_mase:.3f}* "
                f"(threshold: {MASE_ALERT_THRESHOLD})\n"
                f"📊 <{self.dashboard_url}|Open Dashboard>"
            )
            self._send_alert(msg, alert_type="mase_alert", series_id=series_id,
                             metric_value=rolling_mase)
            return True

        logger.debug(
            "[%s] Rolling MASE=%.3f — OK (threshold %.2f)",
            series_id, rolling_mase, MASE_ALERT_THRESHOLD,
        )
        return False

    # ── Forecast Bias Alert ───────────────────────────────────────────────────

    def check_forecast_bias(
        self,
        series_id: str,
        error_history: List[float],
        mean_demand: float,
        consecutive_weeks: int = BIAS_CONSECUTIVE_WEEKS,
    ) -> bool:
        """
        Checks for systematic forecast bias: |ME| > 10% of mean demand
        for `consecutive_weeks` in a row.

        Mean Error (ME) = mean(forecast - actual); positive = over-forecast.

        Parameters
        ----------
        series_id        : series identifier
        error_history    : list of per-week signed errors (forecast - actual)
        mean_demand      : mean of training demand (for threshold scaling)
        consecutive_weeks: number of consecutive weeks bias must persist

        Returns
        -------
        bool : True if bias alert triggered
        """
        if len(error_history) < consecutive_weeks:
            return False

        recent = error_history[-consecutive_weeks:]
        bias_threshold = BIAS_PCT_THRESHOLD * mean_demand

        # Check if ALL recent weeks exceed the absolute bias threshold
        all_biased = all(abs(e) > bias_threshold for e in recent)
        if not all_biased:
            return False

        # Determine direction
        avg_me = float(np.mean(recent))
        direction = "over-forecasting" if avg_me > 0 else "under-forecasting"

        msg = (
            f"⚠️ *Bias Alert* | Series: `{series_id}`\n"
            f"Systematic *{direction}* detected for {consecutive_weeks} consecutive weeks\n"
            f"Avg ME = *{avg_me:.1f}* (threshold: ±{bias_threshold:.1f} = "
            f"{BIAS_PCT_THRESHOLD*100:.0f}% of mean demand {mean_demand:.1f})\n"
            f"📊 <{self.dashboard_url}|Open Dashboard>"
        )
        self._send_alert(msg, alert_type="bias_alert", series_id=series_id,
                         metric_value=avg_me)
        return True

    # ── PSI Distribution Shift ────────────────────────────────────────────────

    def compute_psi(
        self,
        reference: np.ndarray,
        current: np.ndarray,
        n_bins: int = 10,
    ) -> float:
        """
        Population Stability Index (PSI) — measures distribution shift.

        PSI < 0.10 : No significant change
        PSI 0.10–0.20 : Minor change, monitor
        PSI > 0.20  : Significant shift → trigger retrain

        Parameters
        ----------
        reference : feature values from training distribution
        current   : feature values from recent production window (last 4 weeks)
        n_bins    : number of bins for histogram approximation

        Returns
        -------
        float : PSI score
        """
        reference = np.asarray(reference, dtype=float)
        current = np.asarray(current, dtype=float)

        # Build bins on reference distribution
        breakpoints = np.histogram_bin_edges(reference, bins=n_bins)
        breakpoints[0] = -np.inf
        breakpoints[-1] = np.inf

        ref_counts, _ = np.histogram(reference, bins=breakpoints)
        cur_counts, _ = np.histogram(current, bins=breakpoints)

        # Avoid zeros (replace with small epsilon)
        eps = 1e-6
        ref_pct = ref_counts / (len(reference) + eps)
        cur_pct = cur_counts / (len(current) + eps)
        ref_pct = np.where(ref_pct == 0, eps, ref_pct)
        cur_pct = np.where(cur_pct == 0, eps, cur_pct)

        psi = float(np.sum((cur_pct - ref_pct) * np.log(cur_pct / ref_pct)))
        return round(psi, 6)

    def check_psi_alert(
        self,
        series_id: str,
        reference: np.ndarray,
        current: np.ndarray,
        feature_name: str = "sales",
    ) -> bool:
        """
        Computes PSI and triggers a retrain alert if PSI > 0.2.

        Returns
        -------
        bool : True if PSI alert (retrain needed)
        """
        psi = self.compute_psi(reference, current)

        if psi > PSI_RETRAIN_THRESHOLD:
            msg = (
                f"🔄 *PSI Distribution Shift* | Series: `{series_id}`\n"
                f"Feature: `{feature_name}` | PSI = *{psi:.4f}* "
                f"(threshold: {PSI_RETRAIN_THRESHOLD})\n"
                f"→ Automatic retrain triggered\n"
                f"📊 <{self.dashboard_url}|Open Dashboard>"
            )
            self._send_alert(msg, alert_type="psi_alert", series_id=series_id,
                             metric_value=psi)
            return True

        logger.debug(
            "[%s] PSI=%.4f for '%s' — OK (threshold %.2f)",
            series_id, psi, feature_name, PSI_RETRAIN_THRESHOLD,
        )
        return False

    # ── Batch check (used by Airflow DAG) ────────────────────────────────────

    def run_all_checks(
        self,
        series_id: str,
        mase_history: List[float],
        error_history: List[float],
        mean_demand: float,
        reference_dist: Optional[np.ndarray] = None,
        current_dist: Optional[np.ndarray] = None,
    ) -> dict:
        """
        Runs all three drift checks for a series and returns a summary dict.

        Returns
        -------
        dict: {mase_alert: bool, bias_alert: bool, psi_alert: bool, psi_score: float}
        """
        mase_alert = self.check_rolling_mase(series_id, mase_history)
        bias_alert = self.check_forecast_bias(series_id, error_history, mean_demand)

        psi_alert = False
        psi_score = 0.0
        if reference_dist is not None and current_dist is not None:
            psi_score = self.compute_psi(reference_dist, current_dist)
            psi_alert = psi_score > PSI_RETRAIN_THRESHOLD

        return {
            "series_id": series_id,
            "mase_alert": mase_alert,
            "bias_alert": bias_alert,
            "psi_alert": psi_alert,
            "psi_score": psi_score,
        }

    # ── Internal alert dispatch ────────────────────────────────────────────────

    def _send_alert(
        self,
        message: str,
        alert_type: str,
        series_id: str,
        metric_value: float,
    ):
        """Sends a Slack message or logs if webhook not configured."""
        if self.slack_url:
            try:
                import urllib.request
                import json
                payload = json.dumps({"text": message}).encode("utf-8")
                req = urllib.request.Request(
                    self.slack_url,
                    data=payload,
                    headers={"Content-Type": "application/json"},
                )
                with urllib.request.urlopen(req, timeout=5) as resp:
                    if resp.status == 200:
                        logger.info(
                            "Slack alert sent: %s | series=%s | value=%.4f",
                            alert_type, series_id, metric_value,
                        )
            except Exception as e:
                logger.error("Failed to send Slack alert: %s", e)
        else:
            logger.warning(
                "DRIFT ALERT [%s] series=%s value=%.4f | "
                "Set SLACK_WEBHOOK_URL to enable Slack notifications.\n%s",
                alert_type, series_id, metric_value, message,
            )
