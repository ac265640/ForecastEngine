"""
forecast_store.py — Forecast Store (SQLite / Parquet)

All model forecasts are written here. Downstream systems query this store;
they never call models directly.

Schema:
    series_id, forecast_date, horizon, predicted_value,
    lower_bound, upper_bound, model_version, model_type, generated_at

Usage:
    from src.production.forecast_store import ForecastStore

    store = ForecastStore()
    store.write(forecasts_df)
    df = store.read(series_id="CA_1__FOODS", model_type="SARIMA")
"""

import logging
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import pandas as pd
from sqlalchemy import (
    Column, DateTime, Float, Integer, String, Text,
    create_engine, text
)
from sqlalchemy.orm import declarative_base, sessionmaker

logger = logging.getLogger(__name__)

STORE_PATH = os.getenv(
    "FORECAST_STORE_PATH", "artifacts/forecast_store.db"
)

Base = declarative_base()


class ForecastRecord(Base):
    __tablename__ = "forecasts"

    id               = Column(Integer, primary_key=True, autoincrement=True)
    series_id        = Column(String(128), nullable=False, index=True)
    forecast_date    = Column(DateTime, nullable=False, index=True)
    horizon          = Column(Integer, nullable=False)          # weeks ahead
    predicted_value  = Column(Float, nullable=False)
    lower_bound      = Column(Float, nullable=True)
    upper_bound      = Column(Float, nullable=True)
    model_type       = Column(String(32), nullable=False)       # SARIMA | Prophet | LSTM
    model_version    = Column(String(64), nullable=False)
    generated_at     = Column(DateTime, nullable=False)


class ForecastStore:
    """
    Thin wrapper around a SQLite database for reading/writing forecasts.
    Swappable to Parquet on S3 by setting FORECAST_STORE_PATH to a
    Parquet file path and calling write_parquet() / read_parquet().
    """

    def __init__(self, store_path: Optional[str] = None):
        self.store_path = Path(store_path or STORE_PATH)
        self.store_path.parent.mkdir(parents=True, exist_ok=True)
        self._engine = create_engine(
            f"sqlite:///{self.store_path}",
            connect_args={"check_same_thread": False},
        )
        Base.metadata.create_all(self._engine)
        self._Session = sessionmaker(bind=self._engine)
        logger.info("ForecastStore initialised at %s", self.store_path)

    # ── Write ────────────────────────────────────────────────────────────────────

    def write(self, df: pd.DataFrame) -> int:
        """
        Writes a batch of forecasts to the store.

        Expected columns:
            series_id, forecast_date, horizon, predicted_value,
            lower_bound, upper_bound, model_type, model_version
        Optional (auto-filled if missing):
            generated_at  ← UTC now

        Returns
        -------
        int : number of rows written
        """
        required = {
            "series_id", "forecast_date", "horizon",
            "predicted_value", "model_type", "model_version",
        }
        missing = required - set(df.columns)
        if missing:
            raise ValueError(f"ForecastStore.write() missing columns: {missing}")

        df = df.copy()
        if "generated_at" not in df.columns:
            df["generated_at"] = datetime.now(timezone.utc)
        if "lower_bound" not in df.columns:
            df["lower_bound"] = None
        if "upper_bound" not in df.columns:
            df["upper_bound"] = None

        df["forecast_date"] = pd.to_datetime(df["forecast_date"])
        df["generated_at"] = pd.to_datetime(df["generated_at"])

        records = df[
            [
                "series_id", "forecast_date", "horizon", "predicted_value",
                "lower_bound", "upper_bound", "model_type", "model_version",
                "generated_at",
            ]
        ].to_dict(orient="records")

        session = self._Session()
        try:
            session.bulk_insert_mappings(ForecastRecord, records)
            session.commit()
            logger.info("Wrote %d forecast rows to store.", len(records))
            return len(records)
        except Exception as e:
            session.rollback()
            raise e
        finally:
            session.close()

    # ── Read ─────────────────────────────────────────────────────────────────────

    def read(
        self,
        series_id: Optional[str] = None,
        model_type: Optional[str] = None,
        model_version: Optional[str] = None,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
    ) -> pd.DataFrame:
        """
        Queries the forecast store with optional filters.

        Returns
        -------
        pd.DataFrame sorted by (series_id, model_type, forecast_date)
        """
        query = "SELECT * FROM forecasts WHERE 1=1"
        params = {}

        if series_id:
            query += " AND series_id = :series_id"
            params["series_id"] = series_id
        if model_type:
            query += " AND model_type = :model_type"
            params["model_type"] = model_type
        if model_version:
            query += " AND model_version = :model_version"
            params["model_version"] = model_version
        if start_date:
            query += " AND forecast_date >= :start_date"
            params["start_date"] = start_date
        if end_date:
            query += " AND forecast_date <= :end_date"
            params["end_date"] = end_date

        query += " ORDER BY series_id, model_type, forecast_date"

        with self._engine.connect() as conn:
            df = pd.read_sql(text(query), conn, params=params)

        df["forecast_date"] = pd.to_datetime(df["forecast_date"])
        df["generated_at"] = pd.to_datetime(df["generated_at"])
        return df

    def latest_version(self, series_id: str, model_type: str) -> Optional[str]:
        """Returns the most recently generated model_version for a series+model."""
        query = text("""
            SELECT model_version FROM forecasts
            WHERE series_id = :sid AND model_type = :mt
            ORDER BY generated_at DESC LIMIT 1
        """)
        with self._engine.connect() as conn:
            result = conn.execute(query, {"sid": series_id, "mt": model_type}).fetchone()
        return result[0] if result else None

    def all_series(self) -> list:
        """Returns list of all unique series_ids in the store."""
        with self._engine.connect() as conn:
            result = conn.execute(text("SELECT DISTINCT series_id FROM forecasts"))
            return [row[0] for row in result.fetchall()]

    # ── Parquet export / import ───────────────────────────────────────────────

    def export_parquet(self, path: str) -> str:
        """Dumps entire forecast store to a Parquet file (for S3 upload etc)."""
        df = self.read()
        df.to_parquet(path, index=False)
        logger.info("Exported %d rows to Parquet → %s", len(df), path)
        return path

    @classmethod
    def from_parquet(cls, parquet_path: str, store_path: Optional[str] = None) -> "ForecastStore":
        """Loads forecasts from a Parquet file into the store."""
        store = cls(store_path=store_path)
        df = pd.read_parquet(parquet_path)
        store.write(df)
        return store
