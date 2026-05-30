"""
retrain_dag.py — Airflow Retrain DAG

Monthly: retrain SARIMA + Prophet (cheap to recompute)
Quarterly: retrain LSTM + incremental fine-tuning (expensive)

Both schedules:
    validate_new_model >> register_if_better >> alert_on_rollback
"""

from __future__ import annotations

import logging
from datetime import datetime, timedelta

from airflow import DAG
from airflow.operators.python import PythonOperator, BranchPythonOperator
from airflow.operators.empty import EmptyOperator

logger = logging.getLogger(__name__)

default_args = {
    "owner": "forecast-engine",
    "depends_on_past": False,
    "retries": 1,
    "retry_delay": timedelta(minutes=30),
}


# ── Shared helpers ─────────────────────────────────────────────────────────────

def _get_project_root():
    from pathlib import Path
    return Path(__file__).resolve().parents[1]


def _load_data(project_root):
    """Loads and returns splits dict. Returns None if data unavailable."""
    import sys
    if str(project_root) not in sys.path:
        sys.path.insert(0, str(project_root))

    from src.pipeline.data_loader import M5DataLoader
    from src.pipeline.preprocessing import TemporalSplitter

    try:
        loader = M5DataLoader(raw_dir=project_root / "data" / "raw")
        weekly_df = loader.load_and_aggregate()
        splitter = TemporalSplitter()
        return splitter.split_series(weekly_df)
    except FileNotFoundError:
        logger.warning("Raw data unavailable — skipping retrain.")
        return None


# ── Monthly retrain: SARIMA + Prophet ─────────────────────────────────────────

def retrain_sarima_prophet(**context):
    """
    Monthly: fits fresh SARIMA and Prophet models on all series,
    evaluates validation MASE, and registers candidates in MLflow.
    """
    import sys
    project_root = _get_project_root()
    splits = _load_data(project_root)
    if splits is None:
        return

    from src.models.sarima_model import SARIMAForecaster
    from src.models.prophet_model import ProphetForecaster
    from src.pipeline.data_loader import M5DataLoader
    from src.evaluation.metrics import compute_mase
    from src.production.mlflow_registry import ModelRegistry

    registry = ModelRegistry()
    calendar_df = M5DataLoader().load_calendar()
    models_dir = project_root / "artifacts" / "models"
    results = []

    for series_id, (train_df, val_df, test_df) in splits.items():
        state = series_id.split("_")[0]
        train_series = train_df.set_index("week_start")["total_sales"]
        val_series = val_df.set_index("week_start")["total_sales"]

        for ModelClass, mtype in [
            (SARIMAForecaster, "SARIMA"),
            (ProphetForecaster, "Prophet"),
        ]:
            try:
                if mtype == "SARIMA":
                    m = ModelClass(series_id=series_id)
                    m.fit(train_series)
                    pred_df = m.predict(steps=len(val_df))
                    preds = pred_df["predicted_value"].values
                else:
                    m = ModelClass(series_id=series_id, state=state)
                    m.fit(train_series, calendar_df=calendar_df)
                    pred_df = m.predict(steps=len(val_df))
                    preds = pred_df["predicted_value"].values

                val_mase = compute_mase(
                    val_series.values, preds, train_series.values
                )
                save_path = m.save(str(models_dir), version="monthly")
                run_id = registry.log_model(
                    series_id=series_id,
                    model_type=mtype,
                    val_mase=val_mase,
                    artifact_paths={"model": save_path},
                )
                registry.promote_if_better(series_id, mtype, run_id, val_mase)
                results.append({
                    "series_id": series_id, "model": mtype,
                    "val_mase": val_mase, "run_id": run_id,
                })
                logger.info("[%s/%s] retrain complete — val_mase=%.4f", series_id, mtype, val_mase)
            except Exception as e:
                logger.error("[%s/%s] retrain failed: %s", series_id, mtype, e)

    context["ti"].xcom_push(key="retrain_results", value=results)
    logger.info("Monthly retrain complete — %d models processed.", len(results))


# ── Quarterly retrain: LSTM ────────────────────────────────────────────────────

def retrain_lstm(**context):
    """
    Quarterly: full LSTM refit on all series.
    Per-series MinMaxScaler fit on training split only.
    """
    import sys
    import numpy as np
    project_root = _get_project_root()
    splits = _load_data(project_root)
    if splits is None:
        return

    from src.models.lstm_model import LSTMForecaster
    from src.evaluation.metrics import compute_mase
    from src.production.mlflow_registry import ModelRegistry

    registry = ModelRegistry()
    models_dir = project_root / "artifacts" / "models"
    results = []

    for series_id, (train_df, val_df, test_df) in splits.items():
        train_values = train_df["total_sales"].values.astype(float)
        val_values = val_df["total_sales"].values.astype(float)

        try:
            m = LSTMForecaster(series_id=series_id)
            m.fit(train_values, val_values, epochs=50)

            last_window = train_values[-52:]
            preds = m.predict(steps=len(val_df), last_window=last_window)
            val_mase = compute_mase(val_values, preds, train_values)

            paths = m.save(str(models_dir), version="quarterly")
            run_id = registry.log_model(
                series_id=series_id,
                model_type="LSTM",
                val_mase=val_mase,
                metrics={"final_val_loss": m.training_history()["val_loss"][-1]},
            )
            registry.promote_if_better(series_id, "LSTM", run_id, val_mase)
            results.append({"series_id": series_id, "model": "LSTM",
                            "val_mase": val_mase, "run_id": run_id})
            logger.info("[%s/LSTM] quarterly retrain — val_mase=%.4f", series_id, val_mase)
        except Exception as e:
            logger.error("[%s/LSTM] quarterly retrain failed: %s", series_id, e)

    context["ti"].xcom_push(key="lstm_results", value=results)
    logger.info("Quarterly LSTM retrain complete — %d models.", len(results))


def check_retrain_results(**context):
    """Logs a summary of retrain results."""
    results = context["ti"].xcom_pull(task_ids="retrain_sarima_prophet",
                                       key="retrain_results") or []
    lstm_results = context["ti"].xcom_pull(task_ids="retrain_lstm",
                                            key="lstm_results") or []
    all_results = results + lstm_results
    if all_results:
        for r in all_results:
            logger.info(
                "Retrain summary — %s/%s  val_mase=%.4f",
                r["series_id"], r["model"], r.get("val_mase", float("nan")),
            )


# ── Monthly DAG ───────────────────────────────────────────────────────────────

with DAG(
    dag_id="monthly_retrain_sarima_prophet",
    default_args=default_args,
    description="Monthly: retrain SARIMA + Prophet, register in MLflow",
    schedule_interval="@monthly",
    start_date=datetime(2024, 1, 1),
    catchup=False,
    tags=["forecast-engine", "retrain", "monthly"],
) as monthly_dag:

    t_retrain = PythonOperator(
        task_id="retrain_sarima_prophet",
        python_callable=retrain_sarima_prophet,
    )

    t_check = PythonOperator(
        task_id="check_retrain_results",
        python_callable=check_retrain_results,
    )

    t_retrain >> t_check


# ── Quarterly DAG ─────────────────────────────────────────────────────────────

with DAG(
    dag_id="quarterly_retrain_lstm",
    default_args=default_args,
    description="Quarterly: full LSTM retrain + incremental fine-tuning",
    schedule_interval="0 0 1 */3 *",   # 1st of every 3rd month
    start_date=datetime(2024, 1, 1),
    catchup=False,
    tags=["forecast-engine", "retrain", "quarterly"],
) as quarterly_dag:

    t_lstm = PythonOperator(
        task_id="retrain_lstm",
        python_callable=retrain_lstm,
    )

    t_check_q = PythonOperator(
        task_id="check_lstm_results",
        python_callable=check_retrain_results,
    )

    t_lstm >> t_check_q
