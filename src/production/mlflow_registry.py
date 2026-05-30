"""
mlflow_registry.py — MLflow Model Registry & Rollback

Every model version is logged to MLflow before being used in production.
If a new model's validation MASE degrades > 5% vs the current production
model, it is automatically rolled back and a human review alert is sent.

Usage:
    from src.production.mlflow_registry import ModelRegistry

    registry = ModelRegistry()
    registry.log_model(model_obj, series_id, model_type, val_mase, artifact_path)
    registry.promote_if_better(series_id, model_type, new_val_mase)
"""

import logging
import os
from dataclasses import dataclass
from typing import Dict, Optional

import mlflow
from mlflow.tracking import MlflowClient

logger = logging.getLogger(__name__)

MLFLOW_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI", "artifacts/mlruns")
ROLLBACK_THRESHOLD = 0.05   # 5% MASE degradation triggers rollback
EXPERIMENT_NAME = "ForecastEngine"


@dataclass
class ModelVersion:
    series_id: str
    model_type: str
    run_id: str
    val_mase: float
    version_tag: str


class ModelRegistry:
    """
    Manages model versioning, promotion, and rollback in MLflow.

    Workflow:
        1. log_model()        → registers new run in MLflow
        2. promote_if_better()→ compares vs current production;
                                 promotes or rolls back + alerts
        3. get_production()   → retrieves current production model info
    """

    def __init__(self, tracking_uri: Optional[str] = None):
        self.tracking_uri = tracking_uri or MLFLOW_TRACKING_URI
        mlflow.set_tracking_uri(self.tracking_uri)
        self._client = MlflowClient(tracking_uri=self.tracking_uri)
        self._ensure_experiment()

    # ── Log ──────────────────────────────────────────────────────────────────────

    def log_model(
        self,
        series_id: str,
        model_type: str,
        val_mase: float,
        params: Optional[Dict] = None,
        metrics: Optional[Dict] = None,
        artifact_paths: Optional[Dict[str, str]] = None,
    ) -> str:
        """
        Creates a new MLflow run and logs model metadata.

        Parameters
        ----------
        series_id       : e.g. "CA_1__FOODS"
        model_type      : "SARIMA" | "Prophet" | "LSTM"
        val_mase        : validation MASE for this model version
        params          : additional model hyperparameters to log
        metrics         : additional metrics to log
        artifact_paths  : dict mapping name → local file path to log as artifacts

        Returns
        -------
        str : MLflow run_id
        """
        with mlflow.start_run(experiment_id=self._experiment_id) as run:
            run_id = run.info.run_id
            mlflow.set_tags({
                "series_id": series_id,
                "model_type": model_type,
                "stage": "candidate",
            })
            mlflow.log_params({
                "series_id": series_id,
                "model_type": model_type,
                **(params or {}),
            })
            mlflow.log_metrics({
                "val_mase": val_mase,
                **(metrics or {}),
            })
            if artifact_paths:
                for name, path in artifact_paths.items():
                    mlflow.log_artifact(path, artifact_path=name)

            logger.info(
                "Logged %s/%s to MLflow — run_id=%s, val_mase=%.4f",
                series_id, model_type, run_id, val_mase,
            )
        return run_id

    # ── Promote ──────────────────────────────────────────────────────────────────

    def promote_if_better(
        self,
        series_id: str,
        model_type: str,
        new_run_id: str,
        new_val_mase: float,
    ) -> bool:
        """
        Promotes new model to production if it doesn't degrade MASE by > 5%.

        Parameters
        ----------
        series_id    : series identifier
        model_type   : model type string
        new_run_id   : MLflow run_id of the new candidate model
        new_val_mase : validation MASE of the new model

        Returns
        -------
        bool : True if promoted, False if rolled back
        """
        current = self.get_production(series_id, model_type)

        if current is None:
            # No existing production model → always promote
            self._set_production(series_id, model_type, new_run_id, new_val_mase)
            logger.info(
                "No existing production model for %s/%s — promoting new model (MASE=%.4f)",
                series_id, model_type, new_val_mase,
            )
            return True

        current_mase = current["val_mase"]
        degradation = (new_val_mase - current_mase) / (current_mase + 1e-9)

        if degradation > ROLLBACK_THRESHOLD:
            # New model is worse → rollback
            self._mark_run_tag(new_run_id, "stage", "rolled_back")
            msg = (
                f"🔴 AUTO-ROLLBACK | {series_id}/{model_type}\n"
                f"New MASE={new_val_mase:.4f} vs Production MASE={current_mase:.4f} "
                f"(degradation={degradation*100:.1f}% > {ROLLBACK_THRESHOLD*100:.0f}%)\n"
                f"→ Keeping previous run_id={current['run_id']} | Human review required."
            )
            logger.warning(msg)
            self._human_review_alert(msg)
            return False

        # Promote new model
        self._set_production(series_id, model_type, new_run_id, new_val_mase)
        self._mark_run_tag(current["run_id"], "stage", "archived")
        logger.info(
            "Promoted %s/%s: MASE %.4f → %.4f (Δ=%.1f%%)",
            series_id, model_type,
            current_mase, new_val_mase,
            degradation * 100,
        )
        return True

    # ── Query ────────────────────────────────────────────────────────────────────

    def get_production(
        self, series_id: str, model_type: str
    ) -> Optional[Dict]:
        """
        Returns metadata for the current production model, or None.

        Returns
        -------
        dict: {run_id, val_mase, series_id, model_type} or None
        """
        runs = self._client.search_runs(
            experiment_ids=[self._experiment_id],
            filter_string=(
                f"tags.series_id = '{series_id}' AND "
                f"tags.model_type = '{model_type}' AND "
                f"tags.stage = 'production'"
            ),
            order_by=["start_time DESC"],
            max_results=1,
        )
        if not runs:
            return None
        run = runs[0]
        return {
            "run_id": run.info.run_id,
            "val_mase": run.data.metrics.get("val_mase"),
            "series_id": series_id,
            "model_type": model_type,
        }

    def list_all_production(self) -> list:
        """Returns list of all current production model records."""
        runs = self._client.search_runs(
            experiment_ids=[self._experiment_id],
            filter_string="tags.stage = 'production'",
            order_by=["tags.series_id ASC"],
        )
        return [
            {
                "run_id": r.info.run_id,
                "series_id": r.data.tags.get("series_id"),
                "model_type": r.data.tags.get("model_type"),
                "val_mase": r.data.metrics.get("val_mase"),
            }
            for r in runs
        ]

    # ── Private helpers ──────────────────────────────────────────────────────────

    def _ensure_experiment(self):
        """Creates the MLflow experiment if it doesn't exist."""
        experiment = self._client.get_experiment_by_name(EXPERIMENT_NAME)
        if experiment is None:
            self._experiment_id = self._client.create_experiment(EXPERIMENT_NAME)
            logger.info("Created MLflow experiment '%s'", EXPERIMENT_NAME)
        else:
            self._experiment_id = experiment.experiment_id

    def _set_production(
        self, series_id: str, model_type: str, run_id: str, val_mase: float
    ):
        self._mark_run_tag(run_id, "stage", "production")
        logger.info(
            "Set production: %s/%s run_id=%s val_mase=%.4f",
            series_id, model_type, run_id, val_mase,
        )

    def _mark_run_tag(self, run_id: str, key: str, value: str):
        try:
            self._client.set_tag(run_id, key, value)
        except Exception as e:
            logger.warning("Could not set tag on run %s: %s", run_id, e)

    @staticmethod
    def _human_review_alert(message: str):
        """Log alert for human review. Can be extended to email/PagerDuty."""
        logger.critical("HUMAN REVIEW REQUIRED: %s", message)
