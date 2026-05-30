"""
data_loader.py — M5 Walmart Dataset Loader

Loads the three M5 CSV files, melts sales from wide→long format,
aggregates daily sales to weekly level per (store_id × category_id),
and saves the result as Parquet.

Usage:
    from src.pipeline.data_loader import M5DataLoader
    loader = M5DataLoader()
    df = loader.load_and_aggregate()
"""

import os
import logging
from pathlib import Path
from typing import Optional

import pandas as pd
import numpy as np

logger = logging.getLogger(__name__)

# ── Default paths (override via env vars) ──────────────────────────────────────
RAW_DIR = Path(os.getenv("DATA_RAW_PATH", "data/raw"))
PROCESSED_DIR = Path(os.getenv("DATA_PROCESSED_PATH", "data/processed"))

SALES_FILE = RAW_DIR / "sales_train_validation.csv"
CALENDAR_FILE = RAW_DIR / "calendar.csv"
PRICES_FILE = RAW_DIR / "sell_prices.csv"

PROCESSED_SALES_FILE = PROCESSED_DIR / "weekly_sales.parquet"
PROCESSED_CALENDAR_FILE = PROCESSED_DIR / "calendar_features.parquet"


class M5DataLoader:
    """
    Loads the M5 Walmart Forecasting dataset and produces a clean weekly
    aggregated DataFrame keyed by (store_id, cat_id, week_id).
    """

    def __init__(
        self,
        raw_dir: Optional[Path] = None,
        processed_dir: Optional[Path] = None,
        force_reload: bool = False,
    ):
        self.raw_dir = Path(raw_dir) if raw_dir else RAW_DIR
        self.processed_dir = Path(processed_dir) if processed_dir else PROCESSED_DIR
        self.force_reload = force_reload
        self.processed_dir.mkdir(parents=True, exist_ok=True)

    # ── Public API ──────────────────────────────────────────────────────────────

    def load_and_aggregate(self) -> pd.DataFrame:
        """
        Main entry point. Returns a weekly aggregated DataFrame with columns:
            series_id, store_id, cat_id, week_id, week_start, total_sales
        Saves result to Parquet for fast reloads.
        """
        out_path = self.processed_dir / "weekly_sales.parquet"
        if out_path.exists() and not self.force_reload:
            logger.info("Loading aggregated weekly sales from cache: %s", out_path)
            return pd.read_parquet(out_path)

        logger.info("Loading raw M5 files …")
        self._check_files_exist()

        sales_long = self._load_sales_long()
        calendar_df = self._load_calendar()
        sales_long = self._join_calendar(sales_long, calendar_df)
        weekly = self._aggregate_to_weekly(sales_long)

        logger.info("Saving weekly sales Parquet → %s", out_path)
        weekly.to_parquet(out_path, index=False)
        return weekly

    def load_calendar(self) -> pd.DataFrame:
        """
        Returns the processed calendar DataFrame (one row per date) with all
        event and SNAP flags encoded. Saves to Parquet.
        """
        out_path = self.processed_dir / "calendar_features.parquet"
        if out_path.exists() and not self.force_reload:
            return pd.read_parquet(out_path)

        self._check_files_exist()
        cal = self._load_calendar()
        cal.to_parquet(out_path, index=False)
        return cal

    def load_sell_prices(self) -> pd.DataFrame:
        """Returns the sell prices DataFrame (raw, for reference)."""
        self._check_files_exist()
        return pd.read_csv(self.raw_dir / "sell_prices.csv")

    # ── Private helpers ─────────────────────────────────────────────────────────

    def _check_files_exist(self):
        """Raises FileNotFoundError with a helpful message if any M5 file is missing."""
        missing = []
        for f in [SALES_FILE, CALENDAR_FILE, PRICES_FILE]:
            path = self.raw_dir / f.name
            if not path.exists():
                missing.append(str(path))
        if missing:
            raise FileNotFoundError(
                "Missing M5 dataset files. Download from:\n"
                "  https://www.kaggle.com/competitions/m5-forecasting-accuracy/data\n"
                "and place them in: {}\n\nMissing:\n  {}".format(
                    self.raw_dir, "\n  ".join(missing)
                )
            )

    def _load_sales_long(self) -> pd.DataFrame:
        """
        Reads sales_train_validation.csv, melts from wide (one col per day)
        to long format (one row per item-day). Keeps: item_id, store_id,
        cat_id, dept_id, state_id, d (day_id), sales.
        """
        logger.info("Reading sales CSV (this may take a moment) …")
        sales = pd.read_csv(self.raw_dir / "sales_train_validation.csv")

        id_cols = ["id", "item_id", "dept_id", "cat_id", "store_id", "state_id"]
        day_cols = [c for c in sales.columns if c.startswith("d_")]

        logger.info(
            "Melting %d rows × %d day columns …", len(sales), len(day_cols)
        )
        sales_long = sales[id_cols + day_cols].melt(
            id_vars=id_cols, value_vars=day_cols, var_name="d", value_name="sales"
        )
        # sales is always non-negative integer
        sales_long["sales"] = sales_long["sales"].astype(np.int32)
        return sales_long

    def _load_calendar(self) -> pd.DataFrame:
        """
        Reads calendar.csv and engineers binary event flags:
            snap_CA, snap_TX, snap_WI, is_thanksgiving, is_black_friday,
            is_holiday, event_name_1, event_type_1.
        """
        logger.info("Reading calendar CSV …")
        cal = pd.read_csv(self.raw_dir / "calendar.csv")
        cal["date"] = pd.to_datetime(cal["date"])

        # SNAP flags per state
        cal["snap_CA"] = cal["snap_CA"].astype(np.int8)
        cal["snap_TX"] = cal["snap_TX"].astype(np.int8)
        cal["snap_WI"] = cal["snap_WI"].astype(np.int8)

        # Binary: any event on this day
        cal["is_holiday"] = (cal["event_name_1"].notna()).astype(np.int8)

        # Specific holidays
        cal["is_thanksgiving"] = (
            cal["event_name_1"].str.lower().str.contains("thanksgiving", na=False)
        ).astype(np.int8)

        cal["is_black_friday"] = (
            cal["event_name_1"].str.lower().str.contains("blackfriday", na=False)
        ).astype(np.int8)

        # Week identifier (ISO year-week string, e.g., "2011-W01")
        cal["week_id"] = cal["date"].dt.strftime("%G-W%V")
        cal["week_start"] = cal["date"] - pd.to_timedelta(cal["date"].dt.dayofweek, unit="D")

        return cal

    def _join_calendar(self, sales_long: pd.DataFrame, cal: pd.DataFrame) -> pd.DataFrame:
        """Merges the calendar into the long-format sales DataFrame on 'd' column."""
        logger.info("Joining calendar …")
        cal_subset = cal[
            [
                "d",
                "date",
                "week_id",
                "week_start",
                "snap_CA",
                "snap_TX",
                "snap_WI",
                "is_holiday",
                "is_thanksgiving",
                "is_black_friday",
            ]
        ]
        return sales_long.merge(cal_subset, on="d", how="left")

    def _aggregate_to_weekly(self, sales_long: pd.DataFrame) -> pd.DataFrame:
        """
        Aggregates daily sales to weekly (store_id × cat_id × week_id).

        For SNAP flags: takes max (1 if any day in week was a SNAP day).
        For holiday flags: takes max within week.
        Returns columns:
            series_id, store_id, cat_id, week_id, week_start, total_sales,
            snap_CA, snap_TX, snap_WI, is_holiday, is_thanksgiving, is_black_friday
        """
        logger.info("Aggregating to weekly …")

        # Map state to SNAP column
        state_snap_map = {"CA": "snap_CA", "TX": "snap_TX", "WI": "snap_WI"}

        # Determine which SNAP column to use per row based on state_id
        snap_cols = ["snap_CA", "snap_TX", "snap_WI"]

        agg = (
            sales_long.groupby(["store_id", "cat_id", "week_id", "week_start"])
            .agg(
                total_sales=("sales", "sum"),
                snap_CA=("snap_CA", "max"),
                snap_TX=("snap_TX", "max"),
                snap_WI=("snap_WI", "max"),
                is_holiday=("is_holiday", "max"),
                is_thanksgiving=("is_thanksgiving", "max"),
                is_black_friday=("is_black_friday", "max"),
            )
            .reset_index()
        )

        # Create a unified series_id
        agg["series_id"] = agg["store_id"] + "__" + agg["cat_id"]

        # Sort chronologically
        agg = agg.sort_values(["series_id", "week_start"]).reset_index(drop=True)

        # Cast types
        agg["week_start"] = pd.to_datetime(agg["week_start"])
        for col in snap_cols + ["is_holiday", "is_thanksgiving", "is_black_friday"]:
            agg[col] = agg[col].astype(np.int8)

        logger.info(
            "Weekly aggregation complete: %d rows, %d unique series",
            len(agg),
            agg["series_id"].nunique(),
        )
        return agg


# ── Convenience function ────────────────────────────────────────────────────────

def load_weekly_sales(force_reload: bool = False) -> pd.DataFrame:
    """Shortcut: returns aggregated weekly sales DataFrame."""
    return M5DataLoader(force_reload=force_reload).load_and_aggregate()


def load_calendar(force_reload: bool = False) -> pd.DataFrame:
    """Shortcut: returns processed calendar DataFrame."""
    return M5DataLoader(force_reload=force_reload).load_calendar()
