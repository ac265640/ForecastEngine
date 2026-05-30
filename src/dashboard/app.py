"""
app.py — ForecastEngine Streamlit Dashboard

Main entry point for the interactive dashboard. Organises the application
into 5 tabs, manages sidebar configurations, and coordinates demo/production
data loading.
"""

import os
import sys
from pathlib import Path
import streamlit as st
import pandas as pd
import numpy as np

# Ensure project root is in python path
project_root = Path(__file__).resolve().parent.parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from src.dashboard import (
    tab_eda,
    tab_forecast,
    tab_deep_dive,
    tab_business,
    tab_monitoring,
)
from src.production.forecast_store import ForecastStore

# ── PAGE CONFIGURATION ────────────────────────────────────────────────────────
st.set_page_config(
    page_title="ForecastEngine | Retail Demand Intelligence",
    page_icon="chart_with_upwards_trend",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── CUSTOM CSS FOR AESTHETICS (Dark Mode, Glassmorphism, Premium Styling) ─────
st.markdown(
    """
    <style>
        /* General background and typography adjustments */
        @import url('https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;600;700&display=swap');
        
        html, body, [class*="css"] {
            font-family: 'Outfit', -apple-system, BlinkMacSystemFont, sans-serif;
        }
        
        .main {
            background-color: #0b0f19;
            color: #e2e8f0;
        }
        
        /* Sidebar styling */
        section[data-testid="stSidebar"] {
            background-color: #0f172a !important;
            border-right: 1px solid #1e293b;
        }
        
        /* Premium custom metric cards styling */
        div.css-1r6g72t, div.stMetric {
            background: rgba(30, 41, 59, 0.45);
            backdrop-filter: blur(8px);
            border-radius: 12px;
            border: 1px solid rgba(255, 255, 255, 0.05);
            padding: 15px;
            box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1), 0 2px 4px -1px rgba(0, 0, 0, 0.06);
            transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
        }
        
        div.stMetric:hover {
            transform: translateY(-2px);
            border-color: rgba(96, 165, 250, 0.3);
            box-shadow: 0 10px 15px -3px rgba(0, 0, 0, 0.3), 0 4px 6px -2px rgba(0, 0, 0, 0.05);
        }
        
        /* Header custom styling */
        h1 {
            color: #f8fafc;
            font-weight: 700 !important;
            letter-spacing: -0.025em;
            background: linear-gradient(135deg, #60a5fa 0%, #a78bfa 100%);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            margin-bottom: 0.8rem !important;
        }
        
        h2, h3 {
            color: #f1f5f9;
            font-weight: 600 !important;
        }
        
        /* Beautiful Divider */
        hr {
            border-color: rgba(255, 255, 255, 0.05) !important;
        }
        
        /* Custom status labels */
        .badge-stationary {
            background: #22c55e;
            color: white;
            padding: 4px 12px;
            border-radius: 12px;
            font-weight: 600;
            font-size: 13px;
        }
        
        .badge-nonstationary {
            background: #ef4444;
            color: white;
            padding: 4px 12px;
            border-radius: 12px;
            font-weight: 600;
            font-size: 13px;
        }
    </style>
    """,
    unsafe_allow_html=True,
)

# ── HELPER: SYNTHETIC DATA GENERATORS (DEMO FALLBACK) ─────────────────────────
@st.cache_data(show_spinner=False)
def generate_synthetic_data():
    """Generates synthetic weekly sales and calendar data mimicking M5."""
    # Date range: 150 weeks ending last Sunday
    end_date = pd.Timestamp.now() - pd.Timedelta(days=pd.Timestamp.now().dayofweek + 1)
    dates = pd.date_range(end=end_date, periods=150, freq="W")
    
    # Series combinations
    stores = ["CA_1", "CA_2", "TX_1", "TX_2", "WI_1"]
    categories = ["FOODS", "HOBBIES", "HOUSEHOLD"]
    series_ids = [f"{s}__{c}" for s in stores for c in categories]
    
    weekly_records = []
    
    # Calendar features
    calendar_records = []
    rng = np.random.default_rng(42)
    
    for i, dt in enumerate(dates):
        # 1-indexed ISO year-week
        week_id = dt.strftime("%G-W%V")
        # Event flag setup
        is_holiday = 0
        event_name = None
        
        # Inject some holidays for anomalies
        if dt.month == 11 and dt.day >= 20 and dt.day <= 27:
            is_holiday = 1
            event_name = "Thanksgiving"
        elif dt.month == 12 and dt.day >= 22 and dt.day <= 28:
            is_holiday = 1
            event_name = "Christmas"
        elif dt.month == 7 and dt.day >= 1 and dt.day <= 7:
            is_holiday = 1
            event_name = "IndependenceDay"
        elif rng.random() < 0.04:
            is_holiday = 1
            event_name = rng.choice(["SuperBowl", "LaborDay", "Easter"])
            
        calendar_records.append({
            "d": f"d_{i*7 + 1}",
            "date": dt,
            "week_id": week_id,
            "week_start": dt,
            "snap_CA": 1 if rng.random() < 0.3 else 0,
            "snap_TX": 1 if rng.random() < 0.3 else 0,
            "snap_WI": 1 if rng.random() < 0.3 else 0,
            "is_holiday": is_holiday,
            "is_thanksgiving": 1 if event_name == "Thanksgiving" else 0,
            "is_black_friday": 1 if event_name == "Christmas" else 0, # approximation
            "event_name_1": event_name,
        })
        
    calendar_df = pd.DataFrame(calendar_records)
    
    # Generate sales series
    for series in series_ids:
        state = series.split("_")[0]
        # Base demand and weekly seasonality
        base_demand = {"CA": 600, "TX": 500, "WI": 400}[state]
        cat_multiplier = {"FOODS": 1.5, "HOBBIES": 0.6, "HOUSEHOLD": 0.9}[series.split("__")[1]]
        
        level = base_demand * cat_multiplier
        t = np.arange(len(dates))
        
        # Deterministic components: trend + seasonality
        trend = 0.1 * t * level / 100
        seasonality = level * 0.15 * np.sin(2 * np.pi * t / 52)
        
        # Noise
        noise = rng.normal(0, level * 0.08, len(dates))
        
        sales = np.clip(level + trend + seasonality + noise, 10, None)
        
        # Inject anomalies on holidays
        for idx, cal_rec in enumerate(calendar_records):
            if cal_rec["is_holiday"] == 1:
                sales[idx] += level * rng.choice([0.25, -0.2, 0.35])
            # Unexplained anomaly sometimes
            elif idx == 42 or idx == 115:
                sales[idx] += level * 0.45
                
        for idx, dt in enumerate(dates):
            weekly_records.append({
                "series_id": series,
                "store_id": series.split("__")[0],
                "cat_id": series.split("__")[1],
                "week_id": calendar_records[idx]["week_id"],
                "week_start": dt,
                "total_sales": round(sales[idx], 2),
                "snap_CA": calendar_records[idx]["snap_CA"],
                "snap_TX": calendar_records[idx]["snap_TX"],
                "snap_WI": calendar_records[idx]["snap_WI"],
                "is_holiday": calendar_records[idx]["is_holiday"],
                "is_thanksgiving": calendar_records[idx]["is_thanksgiving"],
                "is_black_friday": calendar_records[idx]["is_black_friday"],
            })
            
    weekly_df = pd.DataFrame(weekly_records)
    return weekly_df, calendar_df


# ── DATA LOADING ROUTINE ──────────────────────────────────────────────────────
def load_dashboard_data():
    """Tries to load production dataset or falls back to synthetic demo data."""
    try:
        from src.pipeline.data_loader import M5DataLoader
        # Check cache parquet files
        processed_weekly = Path("data/processed/weekly_sales.parquet")
        processed_calendar = Path("data/processed/calendar_features.parquet")
        
        if processed_weekly.exists() and processed_calendar.exists():
            weekly_df = pd.read_parquet(processed_weekly)
            calendar_df = pd.read_parquet(processed_calendar)
            
            # Ensure proper datetime formats
            weekly_df["week_start"] = pd.to_datetime(weekly_df["week_start"])
            calendar_df["date"] = pd.to_datetime(calendar_df["date"])
            calendar_df["week_start"] = pd.to_datetime(calendar_df["week_start"])
            
            return weekly_df, calendar_df, False
    except Exception:
        pass
    
    # Fallback to demo mode
    weekly_df, calendar_df = generate_synthetic_data()
    return weekly_df, calendar_df, True


# ── MAIN LAYOUT ───────────────────────────────────────────────────────────────
def main():
    # Sidebar header
    st.sidebar.markdown(
        """
        <div style='text-align: center; padding-bottom: 20px;'>
            <h2 style='color: #60a5fa; margin: 0; font-size: 24px;'>ForecastEngine</h2>
            <p style='color: #94a3b8; font-size: 12px; margin: 5px 0 0 0;'>Retail Demand Intelligence Suite</p>
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.sidebar.divider()

    # Load data
    weekly_df, calendar_df, is_demo_fallback = load_dashboard_data()

    # Mode Selector in Sidebar
    if is_demo_fallback:
        st.sidebar.info("M5 dataset not loaded. Running in **Interactive Demo Mode** with synthetic data.")
        mode = "Interactive Demo Mode"
    else:
        mode = st.sidebar.selectbox(
            "System Data Source",
            ["Production Database Mode", "Interactive Demo Mode"],
            help="Toggle between real pipeline database and synthetic demo simulator."
        )

    st.sidebar.subheader("Quick Controls")
    st.sidebar.info(
        "Use the tabs on the main screen to explore data, "
        "inspect model predictions, perform cost calculators, "
        "and monitor drift alerts."
    )

    # Initialize store if production
    store = None
    results_df = None
    if mode == "Production Database Mode":
        try:
            store = ForecastStore()
            # If we have results from pipeline evaluation, let's load them
            # Usually saved in artifacts or sqlite, let's check sqlite table 'forecasts'
            # Or construct a dummy/calculated metrics table if DB is empty
            db_df = store.read()
            if not db_df.empty:
                # Group and estimate MASE/RMSE/MAE based on actuals vs predictions
                # Let's extract actual test results from store
                pass
        except Exception as e:
            st.sidebar.error(f"Error accessing Forecast Store: {e}")

    # Header section
    st.title("ForecastEngine demand forecast dashboard")
    
    # Render Tabs
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "EDA Explorer",
        "Forecast Comparison",
        "Model Deep Dive",
        "Business Impact",
        "Monitoring View"
    ])

    with tab1:
        tab_eda.render(weekly_df, calendar_df)

    with tab2:
        tab_forecast.render(weekly_df, forecast_store=store, results_df=results_df)

    with tab3:
        tab_deep_dive.render(weekly_df, calendar_df)

    with tab4:
        tab_business.render(results_df=results_df, weekly_df=weekly_df)

    with tab5:
        tab_monitoring.render(weekly_df, forecast_store=store, results_df=results_df)

    # Footer
    st.markdown("---")
    st.markdown(
        """
        <div style='text-align: center; color: #64748b; font-size: 11px; padding: 10px 0;'>
            ForecastEngine v1.0.0 • Developed for Production retail operations.
        </div>
        """,
        unsafe_allow_html=True,
    )

if __name__ == "__main__":
    main()
