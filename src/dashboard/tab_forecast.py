"""
tab_forecast.py — Forecast Comparison Tab

Tab 2: All 3 model forecasts plotted against actuals.
Horizon slider: 4 / 8 / 13 / 26 weeks.
Live-updating MASE / RMSE / MAE table per model as horizon changes.
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go


HORIZON_OPTIONS = [4, 8, 13, 26]
MODEL_COLORS = {
    "SARIMA":  "#60a5fa",
    "Prophet": "#f59e0b",
    "LSTM":    "#34d399",
}


def _hex_to_rgba(hex_color: str, alpha: float = 0.1) -> str:
    """Convert a CSS hex color string to an rgba() string suitable for Plotly."""
    h = hex_color.lstrip("#")
    r, g, b = int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16)
    return f"rgba({r},{g},{b},{alpha})"


def render(
    weekly_df: pd.DataFrame,
    forecast_store=None,
    results_df: pd.DataFrame = None,
):
    st.header("Forecast Comparison")

    series_ids = sorted(weekly_df["series_id"].unique())
    col1, col2 = st.columns([3, 1])

    with col1:
        selected = st.selectbox("Series", series_ids, key="fc_series")
    with col2:
        horizon = st.select_slider(
            "Horizon (weeks)", options=HORIZON_OPTIONS, value=13, key="fc_horizon"
        )

    series_df = weekly_df[weekly_df["series_id"] == selected].sort_values("week_start")
    sales = series_df.set_index("week_start")["total_sales"]

    # ── Forecast data ─────────────────────────────────────────────────────────
    all_forecasts = {}

    if forecast_store is not None:
        # Load from the store
        for model_type in ["SARIMA", "Prophet", "LSTM"]:
            try:
                df = forecast_store.read(series_id=selected, model_type=model_type)
                if not df.empty:
                    all_forecasts[model_type] = df.sort_values("forecast_date").head(horizon)
            except Exception:
                pass

    if not all_forecasts:
        # Demo: generate synthetic forecasts around the last known actuals
        all_forecasts = _generate_demo_forecasts(sales, horizon)
        st.info(
            "**Demo mode** — showing synthetic forecasts. "
            "Run the pipeline to populate real model forecasts."
        )

    # ── Main forecast chart ────────────────────────────────────────────────────
    fig = go.Figure()

    # Actuals (show last 52 weeks for context)
    recent_sales = sales.iloc[-52:]
    fig.add_trace(go.Scatter(
        x=recent_sales.index, y=recent_sales.values,
        mode="lines", name="Actuals",
        line=dict(color="#e2e8f0", width=2),
    ))

    for model_name, forecast_data in all_forecasts.items():
        color = MODEL_COLORS.get(model_name, "#94a3b8")

        if isinstance(forecast_data, pd.DataFrame):
            dates = pd.to_datetime(forecast_data.get(
                "forecast_date",
                forecast_data.get("ds", pd.Series(dtype="datetime64[ns]"))
            ))
            preds = forecast_data.get("predicted_value", forecast_data.get("yhat")).values
            lower = forecast_data.get("lower_bound", forecast_data.get("yhat_lower"))
            upper = forecast_data.get("upper_bound", forecast_data.get("yhat_upper"))
        else:
            # numpy array fallback
            last_date = sales.index[-1]
            dates = pd.date_range(last_date + pd.Timedelta(weeks=1), periods=len(forecast_data), freq="W")
            preds = np.asarray(forecast_data)
            lower = upper = None

        fig.add_trace(go.Scatter(
            x=dates, y=preds,
            mode="lines+markers", name=model_name,
            line=dict(color=color, width=2, dash="dash"),
            marker=dict(size=5),
        ))

        if lower is not None and upper is not None:
            lower_vals = lower.values if hasattr(lower, "values") else np.asarray(lower)
            upper_vals = upper.values if hasattr(upper, "values") else np.asarray(upper)
            fig.add_trace(go.Scatter(
                x=list(dates) + list(dates[::-1]),
                y=list(upper_vals) + list(lower_vals[::-1]),
                fill="toself",
                fillcolor=_hex_to_rgba(color, alpha=0.1),
                line=dict(color="rgba(0,0,0,0)"),
                showlegend=False,
                name=f"{model_name} CI",
            ))

    # Vertical line at forecast start
    # Plotly 6.x requires datetime x-values as milliseconds-since-epoch for add_vline
    forecast_start_ms = int(sales.index[-1].value // 10 ** 6)
    fig.add_vline(
        x=forecast_start_ms,
        line_dash="dot", line_color="#475569",
        annotation_text="Forecast Start", annotation_position="top right",
    )

    fig.update_layout(
        template="plotly_dark",
        paper_bgcolor="#0f172a",
        plot_bgcolor="#1e293b",
        font=dict(color="#e2e8f0"),
        height=440,
        title=dict(text=f"<b>{selected}</b> — {horizon}-week Forecast", font=dict(size=15)),
        legend=dict(orientation="h", y=-0.12),
        margin=dict(l=40, r=20, t=50, b=40),
        xaxis=dict(gridcolor="#334155"),
        yaxis=dict(gridcolor="#334155", title="Weekly Sales"),
    )
    st.plotly_chart(fig, use_container_width=True)

    # ── Metrics table ─────────────────────────────────────────────────────────
    st.subheader(f"Metrics at {horizon}-week Horizon")

    if results_df is not None and not results_df.empty:
        series_results = results_df[results_df["series_id"] == selected]
        if not series_results.empty:
            _render_metrics_table(series_results)
        else:
            _render_demo_metrics_table()
    else:
        _render_demo_metrics_table()

    # ── Key insight callout ───────────────────────────────────────────────────
    st.info(
        "**Key Insight**: SARIMA typically wins at short horizons (4–8 weeks) "
        "due to its precise autocorrelation modelling. "
        "LSTM tends to outperform at longer horizons (13–26 weeks) "
        "by capturing non-linear patterns."
    )


def _render_metrics_table(df: pd.DataFrame):
    def color_mase(val):
        if val < 0.8:
            return "background-color: #14532d; color: #86efac"
        elif val < 1.0:
            return "background-color: #1a3a2a; color: #4ade80"
        elif val < 1.15:
            return "background-color: #422006; color: #fbbf24"
        else:
            return "background-color: #450a0a; color: #fca5a5"

    display = df[["model", "MASE", "RMSE", "MAE"]].copy()
    # Use .map() — applymap() is deprecated in pandas 2.x
    styled = display.style.map(color_mase, subset=["MASE"])
    st.dataframe(styled, width="stretch", hide_index=True)


def _render_demo_metrics_table():
    """Shows placeholder metrics when pipeline hasn't run yet."""
    demo = pd.DataFrame({
        "Model": ["SARIMA", "Prophet", "LSTM"],
        "MASE":  ["—", "—", "—"],
        "RMSE":  ["—", "—", "—"],
        "MAE":   ["—", "—", "—"],
        "Status": ["Run pipeline to evaluate", "Run pipeline to evaluate", "Run pipeline to evaluate"],
    })
    st.dataframe(demo, width="stretch", hide_index=True)


def _generate_demo_forecasts(sales: pd.Series, horizon: int) -> dict:
    """Generates synthetic forecasts for demo mode."""
    last_date = sales.index[-1]
    future_dates = pd.date_range(
        last_date + pd.Timedelta(weeks=1), periods=horizon, freq="W"
    )
    last_val = float(sales.iloc[-1])
    trend = (sales.iloc[-1] - sales.iloc[-8]) / 8 if len(sales) >= 8 else 0
    base = np.array([last_val + trend * i for i in range(1, horizon + 1)])

    rng = np.random.default_rng(42)
    forecasts = {}
    for model, noise_scale, bias in [
        ("SARIMA", 0.05, 0.02),
        ("Prophet", 0.08, -0.01),
        ("LSTM", 0.06, 0.03),
    ]:
        noise = rng.normal(0, last_val * noise_scale, size=horizon)
        preds = np.clip(base + noise + last_val * bias, 0, None)
        ci_width = last_val * 0.12
        forecasts[model] = pd.DataFrame({
            "forecast_date": future_dates,
            "predicted_value": preds,
            "lower_bound": np.clip(preds - ci_width, 0, None),
            "upper_bound": preds + ci_width,
        })
    return forecasts
