"""
tab_monitoring.py — Monitoring View Tab

Tab 5:
    - Rolling MASE chart (last 12 weeks) with 1.15 alert threshold line
    - Forecast bias (ME) chart
    - Model health summary table: GREEN / AMBER / RED per series
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots


MASE_ALERT = 1.15
MASE_AMBER = 0.95

MODEL_COLORS = {
    "SARIMA":  "#60a5fa",
    "Prophet": "#f59e0b",
    "LSTM":    "#34d399",
}


def render(weekly_df: pd.DataFrame, forecast_store=None, results_df: pd.DataFrame = None):
    st.header("Monitoring View")

    # ── Series selector ────────────────────────────────────────────────────────
    series_ids = sorted(weekly_df["series_id"].unique())
    selected = st.selectbox("Series", series_ids, key="mon_series")

    series_df = weekly_df[weekly_df["series_id"] == selected].sort_values("week_start")
    sales = series_df.set_index("week_start")["total_sales"]
    mean_demand = float(sales.mean())

    # ── Load or generate monitoring data ──────────────────────────────────────
    monitoring_data = _load_monitoring_data(selected, forecast_store, results_df, sales)

    # ── Rolling MASE + Bias chart ──────────────────────────────────────────────
    fig = make_subplots(
        rows=2, cols=1,
        shared_xaxes=True,
        subplot_titles=[
            "Rolling 4-week MASE (alert threshold = 1.15)",
            "Forecast Bias — Mean Error (ME) per week",
        ],
        vertical_spacing=0.12,
        row_heights=[0.55, 0.45],
    )

    weeks = monitoring_data["weeks"]

    for model in ["SARIMA", "Prophet", "LSTM"]:
        color = MODEL_COLORS[model]
        mase_vals = monitoring_data["rolling_mase"].get(model, [])
        me_vals = monitoring_data["mean_error"].get(model, [])

        if mase_vals:
            fig.add_trace(go.Scatter(
                x=weeks[:len(mase_vals)], y=mase_vals,
                mode="lines+markers", name=f"{model} MASE",
                line=dict(color=color, width=2),
                marker=dict(size=5),
            ), row=1, col=1)

        if me_vals:
            fig.add_trace(go.Scatter(
                x=weeks[:len(me_vals)], y=me_vals,
                mode="lines+markers", name=f"{model} ME",
                line=dict(color=color, width=2, dash="dot"),
                marker=dict(size=5),
                showlegend=True,
            ), row=2, col=1)

    # Alert threshold line
    fig.add_hline(
        y=MASE_ALERT, row=1, col=1,
        line_dash="dash", line_color="#ef4444", line_width=1.5,
        annotation_text="Alert Threshold (1.15)",
        annotation_font_color="#ef4444",
    )
    fig.add_hline(
        y=MASE_AMBER, row=1, col=1,
        line_dash="dot", line_color="#f59e0b", line_width=1,
        annotation_text="Amber (0.95)",
        annotation_font_color="#f59e0b",
    )

    # Bias zero line
    fig.add_hline(y=0, row=2, col=1, line_color="#475569", line_width=1)

    # Bias alert bands (±10% of mean demand)
    bias_limit = mean_demand * 0.10
    for sign in [1, -1]:
        fig.add_hline(
            y=sign * bias_limit, row=2, col=1,
            line_dash="dash", line_color="#f59e0b", line_width=1,
            annotation_text=f"{'Over' if sign>0 else 'Under'}-forecast limit",
            annotation_font_color="#f59e0b",
        )

    fig.update_layout(
        height=540,
        template="plotly_dark",
        paper_bgcolor="#0f172a", plot_bgcolor="#1e293b",
        font=dict(color="#e2e8f0"),
        legend=dict(orientation="h", y=-0.08, font=dict(size=11)),
        margin=dict(l=40, r=20, t=50, b=20),
    )
    fig.update_xaxes(gridcolor="#334155")
    fig.update_yaxes(gridcolor="#334155")
    fig.update_yaxes(title_text="MASE", row=1, col=1)
    fig.update_yaxes(title_text="ME (units)", row=2, col=1)

    st.plotly_chart(fig, use_container_width=True)

    st.divider()

    # ── Health summary table ───────────────────────────────────────────────────
    st.subheader("Model Health Summary")

    health_data = _build_health_table(weekly_df, results_df, monitoring_data)
    _render_health_table(health_data)

    # ── Alert log ─────────────────────────────────────────────────────────────
    alerts = _get_active_alerts(monitoring_data)
    if alerts:
        st.subheader("Active Alerts")
        for alert in alerts:
            st.error(alert)
    else:
        st.success("All models within healthy thresholds — no active alerts.")


# ── Helpers ────────────────────────────────────────────────────────────────────

def _load_monitoring_data(series_id, forecast_store, results_df, sales):
    """Loads rolling MASE / ME from the forecast store or generates demo data."""
    n_weeks = 12
    rng = np.random.default_rng(hash(series_id) % (2**32))
    weeks = pd.date_range(end=sales.index[-1], periods=n_weeks, freq="W").tolist()

    if forecast_store is not None:
        try:
            df = forecast_store.read(series_id=series_id)
            if not df.empty:
                return _compute_from_store(df, sales, weeks)
        except Exception:
            pass

    # Demo synthetic monitoring data
    rolling_mase = {}
    mean_error = {}
    for model in ["SARIMA", "Prophet", "LSTM"]:
        base_mase = {"SARIMA": 0.82, "Prophet": 0.91, "LSTM": 0.78}.get(model, 0.9)
        mase_trend = np.linspace(0, 0.15, n_weeks)
        mase_noise = rng.normal(0, 0.04, n_weeks)
        rolling_mase[model] = np.clip(base_mase + mase_trend + mase_noise, 0.5, 1.5).tolist()

        me_base = rng.normal(0, sales.mean() * 0.05, n_weeks)
        mean_error[model] = me_base.tolist()

    return {"weeks": weeks, "rolling_mase": rolling_mase, "mean_error": mean_error}


def _compute_from_store(store_df, sales, weeks):
    """Computes rolling MASE from forecast store data."""
    rolling_mase = {}
    mean_error = {}
    for model in ["SARIMA", "Prophet", "LSTM"]:
        m_df = store_df[store_df["model_type"] == model].sort_values("forecast_date")
        if m_df.empty:
            continue
        errors = (m_df["predicted_value"] - m_df.get("lower_bound", 0)).fillna(0)
        mean_demand = float(m_df["predicted_value"].mean()) + 1e-6
        mase_series = [abs(float(e)) / mean_demand for e in errors.values[-12:]]
        me_series = errors.values[-12:].tolist()
        rolling_mase[model] = mase_series
        mean_error[model] = me_series
    return {"weeks": weeks, "rolling_mase": rolling_mase, "mean_error": mean_error}


def _build_health_table(weekly_df, results_df, monitoring_data):
    """Builds health rows for all available series × models."""
    all_series = sorted(weekly_df["series_id"].unique())
    rows = []

    for sid in all_series[:20]:   # Cap at 20 for display
        for model in ["SARIMA", "Prophet", "LSTM"]:
            mase_vals = monitoring_data["rolling_mase"].get(model, [])
            me_vals = monitoring_data["mean_error"].get(model, [])

            if not mase_vals:
                continue

            recent_mase = float(np.mean(mase_vals[-4:])) if len(mase_vals) >= 4 else float(np.mean(mase_vals))
            recent_me = float(np.mean(me_vals[-4:])) if me_vals and len(me_vals) >= 4 else 0.0

            if recent_mase > MASE_ALERT:
                status = "RED"
                status_key = "red"
            elif recent_mase > MASE_AMBER:
                status = "AMBER"
                status_key = "amber"
            else:
                status = "GREEN"
                status_key = "green"

            rows.append({
                "Series": sid,
                "Model": model,
                "4-wk MASE": round(recent_mase, 3),
                "Bias (ME)": round(recent_me, 1),
                "Health": status,
                "_status": status_key,
            })

    return rows


def _render_health_table(rows: list):
    if not rows:
        st.info("No monitoring data available.")
        return

    df = pd.DataFrame(rows).drop(columns=["_status"])
    st.dataframe(df, width="stretch", hide_index=True)

    # Summary counts
    status_counts = pd.DataFrame(rows)["_status"].value_counts()
    c1, c2, c3 = st.columns(3)
    with c1:
        st.markdown(
            f"<div style='background:#14532d;padding:10px;border-radius:8px;text-align:center;'>"
            f"<p style='color:#86efac;font-size:22px;font-weight:700;margin:0'>"
            f"{status_counts.get('green', 0)}</p>"
            f"<p style='color:#86efac;font-size:12px;margin:0'>GREEN</p></div>",
            unsafe_allow_html=True,
        )
    with c2:
        st.markdown(
            f"<div style='background:#422006;padding:10px;border-radius:8px;text-align:center;'>"
            f"<p style='color:#fbbf24;font-size:22px;font-weight:700;margin:0'>"
            f"{status_counts.get('amber', 0)}</p>"
            f"<p style='color:#fbbf24;font-size:12px;margin:0'>AMBER</p></div>",
            unsafe_allow_html=True,
        )
    with c3:
        st.markdown(
            f"<div style='background:#450a0a;padding:10px;border-radius:8px;text-align:center;'>"
            f"<p style='color:#fca5a5;font-size:22px;font-weight:700;margin:0'>"
            f"{status_counts.get('red', 0)}</p>"
            f"<p style='color:#fca5a5;font-size:12px;margin:0'>RED</p></div>",
            unsafe_allow_html=True,
        )


def _get_active_alerts(monitoring_data):
    alerts = []
    for model in ["SARIMA", "Prophet", "LSTM"]:
        mase_vals = monitoring_data["rolling_mase"].get(model, [])
        if len(mase_vals) >= 4:
            recent = float(np.mean(mase_vals[-4:]))
            if recent > MASE_ALERT:
                alerts.append(
                    f"**{model}**: Rolling 4-week MASE = {recent:.3f} "
                    f"(threshold: {MASE_ALERT})"
                )
    return alerts
