# ForecastEngine

**Production-grade retail demand forecasting system** built on the M5 Walmart dataset.

[![Live Demo](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://ac265640-forecastengine.streamlit.app)
&nbsp;
![Python](https://img.shields.io/badge/Python-3.9%2B-blue?logo=python&logoColor=white)
![Streamlit](https://img.shields.io/badge/Streamlit-1.50-red?logo=streamlit&logoColor=white)
![MLflow](https://img.shields.io/badge/MLflow-tracked-orange?logo=mlflow&logoColor=white)
![Airflow](https://img.shields.io/badge/Airflow-DAGs-lightblue?logo=apacheairflow&logoColor=white)
![License](https://img.shields.io/badge/License-MIT-green)

---

## Overview

ForecastEngine is a complete end-to-end forecasting pipeline for retail demand at the store × category level. It trains three model families (SARIMA, Prophet, LSTM), evaluates them on a strictly held-out test set, writes forecasts to a production store, and surfaces everything through an interactive Streamlit dashboard.

The system is designed to run without any dataset — it ships with a synthetic data generator that mimics the M5 Walmart dataset so you can explore the full dashboard immediately.

---

## Live Dashboard

> **[Open Dashboard](https://ac265640-forecastengine.streamlit.app)**

The dashboard runs  in **Interactive Demo Mode** by default (no dataset required). Load the M5 data locally and run the pipeline to switch to production mode with real model forecasts.

---

## Architecture

```
forecasting/
├── data/
│   ├── raw/                      # Place M5 CSVs here (optional)
│   └── processed/                # Parquet outputs — auto-generated
├── src/
│   ├── pipeline/
│   │   ├── data_loader.py        # M5 ingestion + weekly aggregation
│   │   ├── eda.py                # ADF test, STL, ACF/PACF, anomaly detection
│   │   ├── preprocessing.py      # Temporal splits, MinMaxScaler (no leakage)
│   │   └── feature_engineering.py
│   ├── models/
│   │   ├── sarima_model.py       # SARIMA(1,1,1)(1,1,1)[52]
│   │   ├── prophet_model.py      # Prophet + holiday effects
│   │   └── lstm_model.py         # LSTM(64) → Dropout → Dense
│   ├── evaluation/
│   │   └── metrics.py            # MASE, RMSE, MAE
│   ├── production/
│   │   ├── forecast_store.py     # SQLite forecast store
│   │   ├── drift_detection.py    # Rolling MASE + PSI alerts
│   │   └── mlflow_registry.py    # Model versioning + rollback
│   └── dashboard/
│       ├── app.py                # Main Streamlit entry point
│       ├── tab_eda.py            # Tab 1 — EDA Explorer
│       ├── tab_forecast.py       # Tab 2 — Forecast Comparison
│       ├── tab_deep_dive.py      # Tab 3 — Model Deep Dive
│       ├── tab_business.py       # Tab 4 — Business Impact
│       └── tab_monitoring.py     # Tab 5 — Monitoring View
├── dags/
│   ├── weekly_scoring_dag.py     # Airflow: weekly inference
│   └── retrain_dag.py            # Airflow: monthly/quarterly retrain
├── tests/                        # Pytest suite
├── scripts/
│   └── run_pipeline.py           # End-to-end runner
├── requirements.txt
└── environment.yml
```

---

## Dashboard — 5 Tabs

| Tab | What it shows |
|---|---|
| **EDA Explorer** | Series selector, STL decomposition overlay, anomaly markers with calendar event labels, ADF stationarity badge |
| **Forecast Comparison** | All 3 model forecasts vs actuals, horizon slider (4/8/13/26 weeks), live MASE/RMSE/MAE table |
| **Model Deep Dive** | SARIMA residual ACF + Ljung-Box p-value; Prophet component decomposition; LSTM training loss curve + prediction scatter |
| **Business Impact** | Newsvendor calculator — optimal order quantity and expected weekly cost per model based on forecast accuracy |
| **Monitoring View** | Rolling 4-week MASE chart with alert threshold, forecast bias (ME) chart, GREEN/AMBER/RED health table per series |

---

## Models

| Model | Key design choices | Best horizon |
|---|---|---|
| **SARIMA(1,1,1)(1,1,1)[52]** | Orders from ACF/PACF, Ljung-Box residual diagnostic | 4–8 weeks |
| **Prophet** | Additive model, Fourier seasonality, SNAP days + Thanksgiving + Black Friday holiday regressors | 8–26 weeks |
| **LSTM(64)** | 52-week sliding window, strict temporal 80/10/10 split, per-series MinMaxScaler fit on train only | 13–26 weeks |

### Evaluation: MASE

- Primary metric: **MASE** — a value below 1.0 means the model beats the naïve seasonal baseline (last year's same week)
- Secondary: RMSE, MAE
- Test set: final **13 weeks**, held out completely, never shuffled

---

## Setup

### 1. Clone

```bash
git clone https://github.com/ac265640/ForecastEngine.git
cd ForecastEngine
```

### 2. Install dependencies

**Option A — pip (recommended):**
```bash
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

**Option B — conda:**
```bash
conda env create -f environment.yml
conda activate forecastengine
```

### 3. Run the dashboard (no dataset needed)

```bash
streamlit run src/dashboard/app.py
```

The app starts in **Interactive Demo Mode** with synthetic data — all 5 tabs are fully functional.

### 4. Run the full pipeline (requires M5 dataset)

Download the M5 dataset from [Kaggle](https://www.kaggle.com/competitions/m5-forecasting-accuracy/data) and place these files in `data/raw/`:

```
data/raw/
├── sales_train_validation.csv
├── calendar.csv
└── sell_prices.csv
```

Then run:

```bash
python scripts/run_pipeline.py
```

This will:
1. Load and aggregate M5 data to weekly level → `data/processed/`
2. Run EDA (ADF tests, STL decomposition, anomaly detection)
3. Train SARIMA, Prophet, and LSTM per store × category series
4. Evaluate all models on the held-out test set
5. Write forecasts to the forecast store (`artifacts/forecast_store.db`)
6. Log model versions to MLflow

### 5. MLflow tracking UI

```bash
mlflow ui --backend-store-uri artifacts/mlruns
```

Open `http://localhost:5000` to view all experiment runs, metrics, and model versions.

### 6. Airflow (optional)

```bash
export AIRFLOW_HOME=$(pwd)/airflow_home
airflow db init
airflow webserver --port 8080 &
airflow scheduler &
```

Copy the DAGs from `dags/` into your `AIRFLOW_HOME/dags/` folder. Two DAGs are provided:

- `weekly_scoring_dag` — runs inference on all 3 models and writes to the forecast store
- `retrain_dag` — monthly SARIMA + Prophet retrain; quarterly LSTM fine-tune

---

## Production Monitoring

| Alert | Trigger | Action |
|---|---|---|
| Rolling MASE | 4-week MASE > 1.15 on any series | Slack alert with series name + dashboard link |
| Forecast bias | Rolling ME > 10% of mean demand for 3 consecutive weeks | Slack alert (systematic bias, not noise) |
| Distribution shift | PSI > 0.2 on input features vs training distribution | Automatic retrain triggered immediately |
| Model degradation | New version MASE degrades > 5% vs production | Auto-rollback to previous version + human review alert |

---

## Environment Variables

| Variable | Description | Default |
|---|---|---|
| `SLACK_WEBHOOK_URL` | Slack webhook for drift and bias alerts | log-only mode |
| `FORECAST_STORE_PATH` | Path to SQLite forecast store | `artifacts/forecast_store.db` |
| `MLFLOW_TRACKING_URI` | MLflow tracking server URI | `artifacts/mlruns` |
| `DATA_RAW_PATH` | Path to raw M5 CSVs | `data/raw/` |
| `DATA_PROCESSED_PATH` | Path for processed Parquet outputs | `data/processed/` |

---

## Running Tests

```bash
pytest tests/ -v --cov=src
```

---

## Stack

`pandas` · `statsmodels` · `prophet` · `tensorflow/keras` · `scikit-learn` · `joblib` · `apache-airflow` · `mlflow` · `streamlit` · `plotly` · `sqlalchemy` · `scipy`

---

## License

MIT
