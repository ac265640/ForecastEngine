"""
run_pipeline.py — End-to-End Pipeline Runner

Runs the full ForecastEngine pipeline:
    1. Load & aggregate M5 data
    2. Feature engineering
    3. EDA (ADF, STL, anomaly detection) per series
    4. Train SARIMA, Prophet, LSTM per series
    5. Evaluate all models (MASE, RMSE, MAE) on the held-out test set
    6. Write forecasts to the ForecastStore
    7. Log all model versions to MLflow

Usage:
    python scripts/run_pipeline.py
    python scripts/run_pipeline.py --series CA_1__FOODS CA_2__FOODS
    python scripts/run_pipeline.py --dry-run   (uses synthetic data, no M5 files needed)
    python scripts/run_pipeline.py --skip-lstm (skip LSTM — faster iteration)
    python scripts/run_pipeline.py --epochs 20
"""

import argparse
import logging
import sys
import time
from pathlib import Path

# ── Ensure project root is on sys.path ─────────────────────────────────────────
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("run_pipeline")


# ── Argument parser ────────────────────────────────────────────────────────────

def parse_args():
    parser = argparse.ArgumentParser(
        description="ForecastEngine — End-to-End Pipeline Runner"
    )
    parser.add_argument(
        "--series", nargs="*",
        help="Specific series IDs to process (default: all)",
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Run with synthetic data — no M5 files required",
    )
    parser.add_argument(
        "--skip-lstm", action="store_true",
        help="Skip LSTM training (faster iteration)",
    )
    parser.add_argument(
        "--skip-eda", action="store_true",
        help="Skip EDA step",
    )
    parser.add_argument(
        "--epochs", type=int, default=50,
        help="LSTM training epochs (default: 50)",
    )
    parser.add_argument(
        "--models-dir", type=str, default="artifacts/models",
        help="Directory to save trained models",
    )
    parser.add_argument(
        "--max-series", type=int, default=None,
        help="Maximum number of series to process (for quick testing)",
    )
    return parser.parse_args()


# ── Dry-run synthetic data ────────────────────────────────────────────────────

def generate_synthetic_data(n_series: int = 3, n_weeks: int = 250) -> pd.DataFrame:
    """
    Generates synthetic weekly sales data that mimics M5 structure.
    Used when --dry-run flag is set (no M5 CSV files needed).
    """
    logger.info("Generating synthetic data (%d series × %d weeks) …", n_series, n_weeks)
    rng = np.random.default_rng(42)

    stores = ["CA_1", "TX_1", "WI_1"]
    cats = ["FOODS", "HOBBIES", "HOUSEHOLD"]
    dates = pd.date_range(start="2011-01-29", periods=n_weeks, freq="W")

    rows = []
    for i in range(n_series):
        store = stores[i % len(stores)]
        cat = cats[i % len(cats)]
        series_id = f"{store}__{cat}"
        state = store.split("_")[0]

        # Trend + seasonality + noise
        trend = np.linspace(500, 650, n_weeks)
        seasonal = 80 * np.sin(2 * np.pi * np.arange(n_weeks) / 52)
        noise = rng.normal(0, 40, n_weeks)
        sales = np.clip(trend + seasonal + noise, 0, None)

        # Calendar features
        snap = rng.binomial(1, 0.3, n_weeks).astype(np.int8)
        is_holiday = np.zeros(n_weeks, dtype=np.int8)
        is_holiday[np.array([47, 48]) % n_weeks] = 1   # Thanksgiving / Black Friday

        for j, date in enumerate(dates):
            rows.append({
                "series_id": series_id,
                "store_id": store,
                "cat_id": cat,
                "week_id": date.strftime("%G-W%V"),
                "week_start": date,
                "total_sales": float(sales[j]),
                "snap_CA": snap[j] if state == "CA" else 0,
                "snap_TX": snap[j] if state == "TX" else 0,
                "snap_WI": snap[j] if state == "WI" else 0,
                "is_holiday": is_holiday[j],
                "is_thanksgiving": int(j % 52 == 47),
                "is_black_friday": int(j % 52 == 48),
            })

    return pd.DataFrame(rows)


# ── Main pipeline ─────────────────────────────────────────────────────────────

def main():
    args = parse_args()
    t_start = time.time()

    logger.info("=" * 60)
    logger.info("  ForecastEngine Pipeline  |  dry-run=%s", args.dry_run)
    logger.info("=" * 60)

    # ── Step 1: Load data ──────────────────────────────────────────────────────
    logger.info("STEP 1 — Loading data …")
    calendar_df = None

    if args.dry_run:
        weekly_df = generate_synthetic_data(n_series=3)
    else:
        try:
            from src.pipeline.data_loader import M5DataLoader
            loader = M5DataLoader()
            weekly_df = loader.load_and_aggregate()
            calendar_df = loader.load_calendar()
        except FileNotFoundError as e:
            logger.error(str(e))
            logger.info("Tip: use --dry-run to run without M5 data files.")
            sys.exit(1)

    series_ids = sorted(weekly_df["series_id"].unique())

    # Filter / limit series
    if args.series:
        series_ids = [s for s in series_ids if s in args.series]
        logger.info("Running on %d requested series.", len(series_ids))
    if args.max_series:
        series_ids = series_ids[:args.max_series]
        logger.info("Capped to %d series (--max-series).", len(series_ids))

    logger.info("Total series to process: %d", len(series_ids))

    # ── Step 2: Feature engineering ───────────────────────────────────────────
    logger.info("STEP 2 — Feature engineering …")
    try:
        from src.pipeline.feature_engineering import FeatureEngineer
        fe = FeatureEngineer(calendar_df=calendar_df)
        weekly_df = fe.transform(weekly_df)
    except Exception as e:
        logger.warning("Feature engineering failed (non-fatal): %s", e)

    # ── Step 3: Temporal splits ────────────────────────────────────────────────
    logger.info("STEP 3 — Computing temporal splits …")
    from src.pipeline.preprocessing import TemporalSplitter
    splitter = TemporalSplitter()
    splits = splitter.split_series(weekly_df)

    # ── Step 4: EDA ────────────────────────────────────────────────────────────
    if not args.skip_eda:
        logger.info("STEP 4 — Running EDA …")
        eda_dir = ROOT / "artifacts" / "eda"
        eda_dir.mkdir(parents=True, exist_ok=True)
        try:
            from src.pipeline.eda import run_full_eda
            for sid in series_ids[:5]:   # EDA on first 5 series for speed
                series_data = weekly_df[weekly_df["series_id"] == sid]
                sales = series_data.set_index("week_start")["total_sales"]
                report = run_full_eda(
                    sales, series_id=sid,
                    calendar_df=calendar_df,
                    save_dir=eda_dir,
                )
                adf = report["adf"]
                anomalies = report["anomalies"]
                n_anom = anomalies["is_anomaly"].sum() if not anomalies.empty else 0
                logger.info(
                    "[EDA] %s → %s (p=%.4f), %d anomalies",
                    sid, adf["status"], adf["p_value"], n_anom,
                )
        except Exception as e:
            logger.warning("EDA failed (non-fatal): %s", e)
    else:
        logger.info("STEP 4 — EDA skipped.")

    # ── Step 5: Train & evaluate models ───────────────────────────────────────
    logger.info("STEP 5 — Training models …")

    models_dir = Path(args.models_dir)
    models_dir.mkdir(parents=True, exist_ok=True)

    from src.evaluation.metrics import evaluate_all_models, build_results_table
    from src.production.forecast_store import ForecastStore
    from src.production.mlflow_registry import ModelRegistry

    store = ForecastStore()
    registry = ModelRegistry()

    all_results = []
    all_forecasts = []
    TEST_HORIZON = 13

    for sid in series_ids:
        train_df, val_df, test_df = splits[sid]
        train_series = train_df.set_index("week_start")["total_sales"]
        val_series   = val_df.set_index("week_start")["total_sales"]
        test_series  = test_df.set_index("week_start")["total_sales"]

        train_vals = train_series.values.astype(float)
        val_vals   = val_series.values.astype(float)
        test_vals  = test_series.values.astype(float)
        state      = sid.split("_")[0]

        forecasts_for_eval = {}
        test_forecast_rows = []

        # ── SARIMA ────────────────────────────────────────────────────────────
        try:
            from src.models.sarima_model import SARIMAForecaster
            logger.info("  [%s] Fitting SARIMA …", sid)
            sarima = SARIMAForecaster(series_id=sid)
            sarima.fit(train_series)
            sarima_preds_df = sarima.predict(steps=TEST_HORIZON)
            sarima_preds = sarima_preds_df["predicted_value"].values
            forecasts_for_eval["SARIMA"] = sarima_preds

            sarima_path = sarima.save(str(models_dir))
            val_preds_df = sarima.predict(steps=len(val_df))
            val_mase_sarima = evaluate_all_models(
                sid, val_vals, train_vals, {"SARIMA": val_preds_df["predicted_value"].values}
            )[0]["MASE"]
            run_id = registry.log_model(
                sid, "SARIMA", val_mase_sarima,
                artifact_paths={"model": sarima_path},
            )
            registry.promote_if_better(sid, "SARIMA", run_id, val_mase_sarima)

            _append_forecast_rows(
                test_forecast_rows, sid, test_series,
                sarima_preds_df, "SARIMA", "v1"
            )
        except Exception as e:
            logger.error("  [%s] SARIMA failed: %s", sid, e)

        # ── Prophet ───────────────────────────────────────────────────────────
        try:
            from src.models.prophet_model import ProphetForecaster
            logger.info("  [%s] Fitting Prophet …", sid)
            prophet = ProphetForecaster(series_id=sid, state=state)
            prophet.fit(train_series, calendar_df=calendar_df)
            prophet_preds_df = prophet.predict(steps=TEST_HORIZON)
            prophet_preds = prophet_preds_df["predicted_value"].values
            forecasts_for_eval["Prophet"] = prophet_preds

            prophet_path = prophet.save(str(models_dir))
            val_preds_df_p = prophet.predict(steps=len(val_df))
            val_mase_prophet = evaluate_all_models(
                sid, val_vals, train_vals, {"Prophet": val_preds_df_p["predicted_value"].values}
            )[0]["MASE"]
            run_id = registry.log_model(
                sid, "Prophet", val_mase_prophet,
                artifact_paths={"model": prophet_path},
            )
            registry.promote_if_better(sid, "Prophet", run_id, val_mase_prophet)

            prophet_preds_df["forecast_date"] = prophet_preds_df["ds"] \
                if "ds" in prophet_preds_df.columns else test_series.index
            _append_forecast_rows(
                test_forecast_rows, sid, test_series,
                prophet_preds_df, "Prophet", "v1"
            )
        except Exception as e:
            logger.error("  [%s] Prophet failed: %s", sid, e)

        # ── LSTM ──────────────────────────────────────────────────────────────
        if not args.skip_lstm:
            try:
                from src.models.lstm_model import LSTMForecaster
                logger.info("  [%s] Fitting LSTM (epochs=%d) …", sid, args.epochs)
                lstm = LSTMForecaster(series_id=sid)
                lstm.fit(train_vals, val_vals, epochs=args.epochs)

                last_window = train_vals[-52:]
                lstm_preds = lstm.predict(steps=TEST_HORIZON, last_window=last_window)
                forecasts_for_eval["LSTM"] = lstm_preds

                lstm_paths = lstm.save(str(models_dir))
                val_preds_lstm = lstm.predict(steps=len(val_df), last_window=last_window)
                val_mase_lstm = evaluate_all_models(
                    sid, val_vals, train_vals, {"LSTM": val_preds_lstm}
                )[0]["MASE"]
                run_id = registry.log_model(
                    sid, "LSTM", val_mase_lstm,
                    metrics={"final_val_loss": lstm.training_history()["val_loss"][-1]},
                )
                registry.promote_if_better(sid, "LSTM", run_id, val_mase_lstm)

                lstm_df = pd.DataFrame({
                    "predicted_value": lstm_preds,
                    "lower_bound": lstm_preds * 0.88,
                    "upper_bound": lstm_preds * 1.12,
                })
                _append_forecast_rows(test_forecast_rows, sid, test_series, lstm_df, "LSTM", "v1")
            except Exception as e:
                logger.error("  [%s] LSTM failed: %s", sid, e)
        else:
            logger.info("  [%s] LSTM skipped.", sid)

        # ── Evaluate on test set ───────────────────────────────────────────────
        if forecasts_for_eval:
            results = evaluate_all_models(sid, test_vals, train_vals, forecasts_for_eval)
            all_results.extend(results)

        if test_forecast_rows:
            all_forecasts.extend(test_forecast_rows)

    # ── Step 6: Save results & forecasts ──────────────────────────────────────
    logger.info("STEP 6 — Saving results …")

    if all_results:
        results_df = build_results_table(all_results)
        results_path = ROOT / "artifacts" / "evaluation_results.csv"
        results_df.to_csv(results_path, index=False)
        logger.info("Saved evaluation results → %s", results_path)
        logger.info("\n%s", results_df.to_string(index=False))

    if all_forecasts:
        forecasts_df = pd.DataFrame(all_forecasts)
        n_written = store.write(forecasts_df)
        logger.info("Wrote %d forecast rows to ForecastStore.", n_written)

    elapsed = time.time() - t_start
    logger.info("=" * 60)
    logger.info("  Pipeline complete in %.1f seconds", elapsed)
    logger.info("  MLflow UI: mlflow ui --backend-store-uri artifacts/mlruns")
    logger.info("  Dashboard: streamlit run src/dashboard/app.py")
    logger.info("=" * 60)


def _append_forecast_rows(rows, sid, test_series, pred_df, model_type, version):
    """Builds forecast store rows from a prediction DataFrame."""
    dates = test_series.index
    preds = pred_df["predicted_value"].values[:len(dates)]
    lower = pred_df.get("lower_bound", pd.Series([None] * len(preds))).values[:len(dates)]
    upper = pred_df.get("upper_bound", pd.Series([None] * len(preds))).values[:len(dates)]

    import pandas as pd as _pd
    now = _pd.Timestamp.now(tz="UTC")

    for i, date in enumerate(dates):
        rows.append({
            "series_id": sid,
            "forecast_date": date,
            "horizon": i + 1,
            "predicted_value": float(preds[i]) if i < len(preds) else None,
            "lower_bound": float(lower[i]) if lower is not None and i < len(lower) else None,
            "upper_bound": float(upper[i]) if upper is not None and i < len(upper) else None,
            "model_type": model_type,
            "model_version": version,
            "generated_at": now,
        })


if __name__ == "__main__":
    main()
