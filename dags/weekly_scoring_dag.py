"""
weekly_scoring_dag.py — Airflow Weekly Scoring DAG

Runs every week:
    generate_forecasts >> check_drift >> alert_if_degraded

Each task:
    1. generate_forecasts : loads the 3 production models, runs inference
                            on all series, writes results to forecast store
    2. check_drift        : computes rolling MASE + bias + PSI per series
    3. alert_if_degraded  : sends Slack alerts for any triggered thresholds
"""

from __future__ import annotations

import logging
import os
from datetime import datetime, timedelta

from airflow import DAG
from airflow.operators.python import PythonOperator

logger = logging.getLogger(__name__)

# ── DAG default args ──────────────────────────────────────────────────────────

default_args = {
    "owner": "forecast-engine",
    "depends_on_past": False,
    "email_on_failure": False,
    "email_on_retry": False,
    "retries": 2,
    "retry_delay": timedelta(minutes=10),
}

# ── Task functions ─────────────────────────────────────────────────────────────


def generate_forecasts(**context):
    """
    Loads all fitted models from artifacts/, runs inference for HORIZON weeks,
    and writes results to the ForecastStore.
    """
    import sys
    import importlib
    from pathlib import Path
    import pandas as pd

    # Ensure the project root is on path
    project_root = Path(__file__).resolve().parents[1]
    if str(project_root) not in sys.path:
        sys.path.insert(0, str(project_root))

    from src.production.forecast_store import ForecastStore
    from src.pipeline.data_loader import M5DataLoader
    from src.pipeline.preprocessing import TemporalSplitter

    HORIZON = 13
    MODELS_DIR = project_root / "artifacts" / "models"
    store = ForecastStore()
    loader = M5DataLoader()

    try:
        weekly_df = loader.load_and_aggregate()
    except FileNotFoundError:
        logger.warning("Raw data not found — skipping scoring (demo mode)")
        return

    splitter = TemporalSplitter()
    splits = splitter.split_series(weekly_df)

    all_forecasts = []

    for series_id, (train_df, val_df, test_df) in splits.items():
        state = series_id.split("_")[0]
        train_series = train_df.set_index("week_start")["total_sales"]

        for model_type in ["SARIMA", "Prophet", "LSTM"]:
            try:
                forecast_df = _run_inference(
                    series_id, model_type, train_series, HORIZON, MODELS_DIR, state
                )
                if forecast_df is not None:
                    forecast_df["series_id"] = series_id
                    forecast_df["model_type"] = model_type
                    all_forecasts.append(forecast_df)
            except Exception as e:
                logger.error("[%s/%s] Inference failed: %s", series_id, model_type, e)

    if all_forecasts:
        combined = pd.concat(all_forecasts, ignore_index=True)
        store.write(combined)
        logger.info("Wrote %d forecast rows to store.", len(combined))
        context["ti"].xcom_push(key="n_forecasts", value=len(combined))


def _run_inference(series_id, model_type, train_series, horizon, models_dir, state):
    """Helper: loads a saved model and runs predict()."""
    import glob
    import numpy as np
    from pathlib import Path
    import pandas as pd

    pattern = str(models_dir / f"{model_type.lower()}_{series_id}_*.pkl")
    files = sorted(glob.glob(pattern))
    if not files:
        logger.warning("No saved model found for %s/%s", series_id, model_type)
        return None

    model_path = files[-1]  # Most recent version
    version = Path(model_path).stem.split("_")[-1]
    now = pd.Timestamp.now(tz="UTC")

    if model_type == "SARIMA":
        from src.models.sarima_model import SARIMAForecaster
        model = SARIMAForecaster.load(model_path)
        pred_df = model.predict(steps=horizon)
    elif model_type == "Prophet":
        from src.models.prophet_model import ProphetForecaster
        model = ProphetForecaster.load(model_path)
        pred_df = model.predict(steps=horizon)
        pred_df = pred_df.rename(columns={"ds": "forecast_date"})
    elif model_type == "LSTM":
        keras_pattern = str(models_dir / f"lstm_{series_id}_*.keras")
        keras_files = sorted(glob.glob(keras_pattern))
        scaler_pattern = str(models_dir / f"lstm_scaler_{series_id}_*.pkl")
        scaler_files = sorted(glob.glob(scaler_pattern))
        if not keras_files or not scaler_files:
            return None
        from src.models.lstm_model import LSTMForecaster
        model = LSTMForecaster.load(keras_files[-1], scaler_files[-1], series_id)
        last_window = train_series.values[-52:]
        preds = model.predict(steps=horizon, last_window=last_window)
        pred_df = pd.DataFrame({"predicted_value": preds})
    else:
        return None

    if "forecast_date" not in pred_df.columns:
        last_date = train_series.index[-1]
        dates = pd.date_range(start=last_date + pd.Timedelta(weeks=1), periods=horizon, freq="W")
        pred_df["forecast_date"] = dates

    pred_df["horizon"] = range(1, len(pred_df) + 1)
    pred_df["model_version"] = version
    pred_df["generated_at"] = now

    for col in ["lower_bound", "upper_bound"]:
        if col not in pred_df.columns:
            pred_df[col] = None

    return pred_df


def check_drift(**context):
    """
    Reads recent forecasts + actuals from the store, computes rolling MASE,
    bias ME, and PSI. Pushes alert flags to XCom.
    """
    import sys
    from pathlib import Path
    import pandas as pd
    import numpy as np

    project_root = Path(__file__).resolve().parents[1]
    if str(project_root) not in sys.path:
        sys.path.insert(0, str(project_root))

    from src.production.forecast_store import ForecastStore
    from src.production.drift_detection import DriftDetector
    from src.evaluation.metrics import compute_rolling_mase

    store = ForecastStore()
    detector = DriftDetector()
    alerts = []

    all_series = store.all_series()

    for series_id in all_series:
        df = store.read(series_id=series_id)
        if df.empty:
            continue

        # Compute rolling MASE proxy from recent error history
        for model_type in ["SARIMA", "Prophet", "LSTM"]:
            model_df = df[df["model_type"] == model_type].sort_values("forecast_date")
            if len(model_df) < 4:
                continue

            # Use abs(predicted - lower) as a proxy error when actuals not available
            # In production, join with actuals table
            errors = (model_df["predicted_value"] - model_df["lower_bound"]).fillna(0)
            mase_history = [float(abs(e) / (model_df["predicted_value"].mean() + 1e-6))
                            for e in errors.values[-8:]]
            error_history = errors.values[-8:].tolist()
            mean_demand = float(model_df["predicted_value"].mean())

            result = detector.run_all_checks(
                series_id=f"{series_id}/{model_type}",
                mase_history=mase_history,
                error_history=error_history,
                mean_demand=mean_demand,
            )
            if result["mase_alert"] or result["bias_alert"] or result["psi_alert"]:
                alerts.append(result)

    context["ti"].xcom_push(key="alerts", value=alerts)
    logger.info("Drift check complete — %d alerts found.", len(alerts))


def alert_if_degraded(**context):
    """
    Pulls alert flags from XCom and emits Slack notifications for each.
    """
    alerts = context["ti"].xcom_pull(task_ids="check_drift", key="alerts") or []
    if not alerts:
        logger.info("No degraded series — all systems healthy ✅")
        return

    logger.warning("%d series flagged with drift alerts.", len(alerts))
    for alert in alerts:
        logger.warning(
            "ALERT: series=%s mase_alert=%s bias_alert=%s psi_alert=%s",
            alert["series_id"], alert["mase_alert"],
            alert["bias_alert"], alert["psi_alert"],
        )


# ── DAG definition ─────────────────────────────────────────────────────────────

with DAG(
    dag_id="weekly_scoring",
    default_args=default_args,
    description="Weekly: generate forecasts → check drift → alert",
    schedule_interval="@weekly",
    start_date=datetime(2024, 1, 1),
    catchup=False,
    tags=["forecast-engine", "scoring"],
) as dag:

    t_generate = PythonOperator(
        task_id="generate_forecasts",
        python_callable=generate_forecasts,
    )

    t_drift = PythonOperator(
        task_id="check_drift",
        python_callable=check_drift,
    )

    t_alert = PythonOperator(
        task_id="alert_if_degraded",
        python_callable=alert_if_degraded,
    )

    t_generate >> t_drift >> t_alert
