"""
lstm_model.py — LSTM Forecaster (Univariate)

Architecture: LSTM(64 units) → Dropout(0.2) → Dense(1)
Sliding window input: 52 weeks
Strict temporal train/val/test split — no shuffling.
Per-series MinMaxScaler fit on training window ONLY.

Usage:
    from src.models.lstm_model import LSTMForecaster
    from src.pipeline.preprocessing import TemporalSplitter, SeriesScaler

    model = LSTMForecaster(series_id="CA_1__FOODS")
    model.fit(train_values, val_values, epochs=50)
    forecast = model.predict(steps=13, last_window=last_52_values)
    model.save("artifacts/models/")
"""

import logging
import os
from pathlib import Path
from typing import Dict, Optional, Tuple

import joblib
import mlflow
import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

WINDOW_SIZE = 52   # 52 weeks = 1 year of context
LSTM_UNITS = 64
DROPOUT_RATE = 0.2
DEFAULT_EPOCHS = 50
DEFAULT_BATCH_SIZE = 16


class LSTMForecaster:
    """
    Univariate LSTM forecaster for one (store_id × cat_id) series.

    Follows strict temporal integrity:
        - No data shuffling
        - Scaler fit on training data only
        - Sliding window constructed from scaled training data

    Produces:
        - Point forecasts (inverse-transformed to original scale)
        - Training loss history
    """

    def __init__(
        self,
        series_id: str,
        window_size: int = WINDOW_SIZE,
        lstm_units: int = LSTM_UNITS,
        dropout_rate: float = DROPOUT_RATE,
    ):
        self.series_id = series_id
        self.window_size = window_size
        self.lstm_units = lstm_units
        self.dropout_rate = dropout_rate
        self._model = None
        self._scaler = None
        self._history = None

    # ── Build ────────────────────────────────────────────────────────────────────

    def _build_model(self):
        """Builds the Keras model: LSTM(64) → Dropout(0.2) → Dense(1)."""
        try:
            import tensorflow as tf
            from tensorflow import keras
        except ImportError:
            raise ImportError("TensorFlow not installed. Run: pip install tensorflow")

        tf.random.set_seed(42)
        model = keras.Sequential([
            keras.layers.Input(shape=(self.window_size, 1)),
            keras.layers.LSTM(self.lstm_units, return_sequences=False),
            keras.layers.Dropout(self.dropout_rate),
            keras.layers.Dense(1),
        ])
        model.compile(optimizer="adam", loss="mse")
        return model

    # ── Fit ──────────────────────────────────────────────────────────────────────

    def fit(
        self,
        train_values: np.ndarray,
        val_values: np.ndarray,
        epochs: int = DEFAULT_EPOCHS,
        batch_size: int = DEFAULT_BATCH_SIZE,
        verbose: int = 0,
    ) -> "LSTMForecaster":
        """
        Fits the LSTM model. The scaler is fit on train_values ONLY.

        Parameters
        ----------
        train_values : raw (unscaled) training sales values (1-D)
        val_values   : raw validation values (1-D)
        epochs       : number of training epochs
        batch_size   : mini-batch size
        verbose      : 0=silent, 1=progress bar, 2=one line/epoch
        """
        from sklearn.preprocessing import MinMaxScaler
        from src.pipeline.preprocessing import make_lstm_windows

        # ── Scale — fit on TRAIN only ──────────────────────────────────────────
        self._scaler = MinMaxScaler(feature_range=(0, 1))
        train_scaled = self._scaler.fit_transform(
            train_values.reshape(-1, 1)
        ).flatten()
        val_scaled = self._scaler.transform(
            val_values.reshape(-1, 1)
        ).flatten()

        # ── Sliding windows ────────────────────────────────────────────────────
        X_train, y_train = make_lstm_windows(train_scaled, self.window_size)

        # For validation: prepend last window_size from train
        val_full = np.concatenate([train_scaled[-self.window_size:], val_scaled])
        X_val, y_val = make_lstm_windows(val_full, self.window_size)

        logger.info(
            "Training LSTM on %s — X_train=%s, X_val=%s, epochs=%d",
            self.series_id, X_train.shape, X_val.shape, epochs,
        )

        self._model = self._build_model()

        try:
            from tensorflow import keras
            callbacks = [
                keras.callbacks.EarlyStopping(
                    monitor="val_loss", patience=10, restore_best_weights=True
                )
            ]
            history = self._model.fit(
                X_train, y_train,
                validation_data=(X_val, y_val),
                epochs=epochs,
                batch_size=batch_size,
                callbacks=callbacks,
                shuffle=False,  # Never shuffle time series data
                verbose=verbose,
            )
            self._history = {
                "loss": history.history["loss"],
                "val_loss": history.history["val_loss"],
            }
            logger.info(
                "LSTM training complete — final val_loss=%.6f",
                self._history["val_loss"][-1],
            )
        except Exception as e:
            logger.error("LSTM training failed: %s", e)
            raise

        return self

    # ── Predict ──────────────────────────────────────────────────────────────────

    def predict(
        self,
        steps: int = 13,
        last_window: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        """
        Iterative multi-step forecast using last available window.

        Parameters
        ----------
        steps       : forecast horizon in weeks
        last_window : last `window_size` raw (unscaled) values before test set.
                      If None, uses the last window seen during training.

        Returns
        -------
        np.ndarray : point forecasts in original (unscaled) units
        """
        self._check_fitted()

        if last_window is None:
            raise ValueError("last_window must be provided: the last 52 raw sales values.")

        if len(last_window) < self.window_size:
            raise ValueError(
                f"last_window length {len(last_window)} < window_size {self.window_size}"
            )

        # Scale the seed window using training scaler
        window_scaled = self._scaler.transform(
            last_window[-self.window_size:].reshape(-1, 1)
        ).flatten()

        predictions_scaled = []
        current_window = window_scaled.copy()

        for _ in range(steps):
            x = current_window[-self.window_size:].reshape(1, self.window_size, 1)
            pred_scaled = self._model.predict(x, verbose=0)[0, 0]
            predictions_scaled.append(pred_scaled)
            # Slide window: drop oldest, append new prediction
            current_window = np.append(current_window[1:], pred_scaled)

        preds = np.array(predictions_scaled).reshape(-1, 1)
        return self._scaler.inverse_transform(preds).flatten().clip(min=0)

    def predict_on_test(
        self,
        test_values: np.ndarray,
        last_train_window: np.ndarray,
    ) -> np.ndarray:
        """
        Evaluates model on the test set using teacher-forcing for the seed,
        then iterative prediction for all test steps.

        Parameters
        ----------
        test_values       : actual test values (raw, for length reference)
        last_train_window : last window_size raw values from training set

        Returns
        -------
        np.ndarray : predictions aligned with test_values
        """
        return self.predict(steps=len(test_values), last_window=last_train_window)

    def training_history(self) -> Dict:
        """Returns training loss history dict: {loss: [...], val_loss: [...]}."""
        if self._history is None:
            raise RuntimeError("Model not yet trained.")
        return self._history

    # ── Save / Load ──────────────────────────────────────────────────────────────

    def save(self, save_dir: str, version: str = "v1") -> Dict[str, str]:
        """
        Saves model weights (.h5 / .keras) and scaler (.pkl).

        Returns
        -------
        dict: {model_path: ..., scaler_path: ...}
        """
        self._check_fitted()
        save_dir = Path(save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)

        model_path = save_dir / f"lstm_{self.series_id}_{version}.keras"
        scaler_path = save_dir / f"lstm_scaler_{self.series_id}_{version}.pkl"

        self._model.save(str(model_path))
        joblib.dump(self._scaler, str(scaler_path))

        logger.info("Saved LSTM model → %s", model_path)
        logger.info("Saved LSTM scaler → %s", scaler_path)
        return {"model_path": str(model_path), "scaler_path": str(scaler_path)}

    @classmethod
    def load(cls, model_path: str, scaler_path: str, series_id: str) -> "LSTMForecaster":
        """Loads a saved LSTM model and scaler from disk."""
        try:
            import tensorflow as tf
        except ImportError:
            raise ImportError("TensorFlow not installed.")

        obj = cls(series_id=series_id)
        obj._model = tf.keras.models.load_model(model_path)
        obj._scaler = joblib.load(scaler_path)
        logger.info("Loaded LSTM model from %s", model_path)
        return obj

    def log_to_mlflow(
        self,
        val_mase: float,
        run_id: Optional[str] = None,
    ) -> str:
        """Logs LSTM params and metrics to MLflow."""
        with mlflow.start_run(run_id=run_id) as run:
            mlflow.log_params({
                "series_id": self.series_id,
                "model_type": "LSTM",
                "lstm_units": self.lstm_units,
                "dropout_rate": self.dropout_rate,
                "window_size": self.window_size,
            })
            mlflow.log_metrics({"val_mase": val_mase})
            if self._history:
                mlflow.log_metrics({
                    "final_train_loss": self._history["loss"][-1],
                    "final_val_loss": self._history["val_loss"][-1],
                })
            return run.info.run_id

    # ── Private ──────────────────────────────────────────────────────────────────

    def _check_fitted(self):
        if self._model is None or self._scaler is None:
            raise RuntimeError(
                f"LSTM model '{self.series_id}' not fitted. Call .fit() first."
            )
