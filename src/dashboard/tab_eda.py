"""
tab_eda.py — EDA Explorer Tab

Tab 1 of the ForecastEngine Streamlit dashboard.

Features:
    - Series selector (store × category dropdown)
    - Plotly line chart with toggleable STL overlay
    - Anomaly markers annotated with calendar event labels
    - ADF test result badge: STATIONARY / NON-STATIONARY
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots


def render(weekly_df: pd.DataFrame, calendar_df: pd.DataFrame):
    st.header("EDA Explorer")

    series_ids = sorted(weekly_df["series_id"].unique())
    col1, col2 = st.columns([3, 1])
    with col1:
        selected = st.selectbox("Select Series (Store × Category)", series_ids,
                                key="eda_series")
    with col2:
        show_stl = st.toggle("Show STL Decomposition", value=True)

    series_df = weekly_df[weekly_df["series_id"] == selected].sort_values("week_start")
    sales = series_df.set_index("week_start")["total_sales"]

    # ── ADF test ──────────────────────────────────────────────────────────────
    try:
        from src.pipeline.eda import run_adf_test, run_stl_decomposition, detect_anomalies
        adf = run_adf_test(sales)
        badge_color = "#22c55e" if adf["status"] == "STATIONARY" else "#ef4444"
        st.markdown(
            f'<span style="background:{badge_color};color:white;padding:4px 12px;'
            f'border-radius:12px;font-weight:600;font-size:14px;">'
            f'{adf["status"]}</span> &nbsp;'
            f'<span style="color:#94a3b8;font-size:13px;">ADF p-value = {adf["p_value"]}</span>',
            unsafe_allow_html=True,
        )
    except Exception as e:
        st.warning(f"ADF test unavailable: {e}")
        adf = None

    st.divider()

    # ── Main line chart ───────────────────────────────────────────────────────
    try:
        from src.pipeline.eda import run_stl_decomposition, detect_anomalies, rolling_stats

        stl = run_stl_decomposition(sales)
        anomaly_df = detect_anomalies(
            sales.reset_index(drop=True),
            stl["residual"],
            calendar_df=calendar_df,
            index=pd.Series(sales.index),
        )

        # Rolling stats
        roll = rolling_stats(sales)
        roll.index = sales.index

        # Build figure
        if show_stl:
            fig = make_subplots(
                rows=4, cols=1,
                shared_xaxes=True,
                row_heights=[0.45, 0.2, 0.2, 0.15],
                subplot_titles=["Sales + Rolling Mean", "Trend", "Seasonality", "Residual"],
                vertical_spacing=0.05,
            )
        else:
            fig = make_subplots(rows=1, cols=1)

        dates = sales.index

        # Row 1: Actuals + rolling mean + anomalies
        fig.add_trace(go.Scatter(
            x=dates, y=sales.values,
            mode="lines", name="Weekly Sales",
            line=dict(color="#60a5fa", width=1.5),
        ), row=1, col=1)

        fig.add_trace(go.Scatter(
            x=dates, y=roll["rolling_mean"].values,
            mode="lines", name="4-week Rolling Mean",
            line=dict(color="#f59e0b", width=2, dash="dash"),
        ), row=1, col=1)

        # Anomaly markers
        expl_mask = anomaly_df["anomaly_class"] == "explainable"
        unexpl_mask = anomaly_df["anomaly_class"] == "unexplained"

        if expl_mask.any():
            expl_idx = anomaly_df[expl_mask].index
            expl_dates = [dates[i] for i in expl_idx if i < len(dates)]
            expl_sales = [sales.values[i] for i in expl_idx if i < len(dates)]
            expl_labels = anomaly_df.loc[expl_mask, "event_label"].values
            fig.add_trace(go.Scatter(
                x=expl_dates, y=expl_sales,
                mode="markers+text",
                marker=dict(color="#f59e0b", size=10, symbol="star"),
                text=expl_labels, textposition="top center",
                name="Explainable Anomaly",
            ), row=1, col=1)

        if unexpl_mask.any():
            unexpl_idx = anomaly_df[unexpl_mask].index
            unexpl_dates = [dates[i] for i in unexpl_idx if i < len(dates)]
            unexpl_sales = [sales.values[i] for i in unexpl_idx if i < len(dates)]
            fig.add_trace(go.Scatter(
                x=unexpl_dates, y=unexpl_sales,
                mode="markers",
                marker=dict(color="#ef4444", size=10, symbol="x"),
                name="Unexplained Anomaly",
            ), row=1, col=1)

        if show_stl:
            # Row 2: Trend
            fig.add_trace(go.Scatter(
                x=dates, y=stl["trend"].values,
                mode="lines", name="Trend",
                line=dict(color="#a78bfa", width=2),
            ), row=2, col=1)

            # Row 3: Seasonality
            fig.add_trace(go.Scatter(
                x=dates, y=stl["seasonal"].values,
                mode="lines", name="Seasonality",
                line=dict(color="#34d399", width=1.5),
                fill="tozeroy", fillcolor="rgba(52,211,153,0.1)",
            ), row=3, col=1)

            # Row 4: Residual + 3σ bands
            resid = stl["residual"].values
            sigma = resid.std()
            fig.add_trace(go.Scatter(
                x=dates, y=resid,
                mode="lines", name="Residual",
                line=dict(color="#94a3b8", width=1),
            ), row=4, col=1)
            for sign, label in [(1, "+3σ"), (-1, "−3σ")]:
                fig.add_hline(
                    y=sign * 3 * sigma, line_dash="dot",
                    line_color="#ef4444", row=4, col=1,
                    annotation_text=label,
                )

        fig.update_layout(
            height=700 if show_stl else 380,
            template="plotly_dark",
            paper_bgcolor="#0f172a",
            plot_bgcolor="#1e293b",
            font=dict(color="#e2e8f0"),
            legend=dict(orientation="h", y=-0.05),
            margin=dict(l=40, r=20, t=40, b=20),
            title=dict(text=f"<b>{selected}</b>", font=dict(size=16, color="#e2e8f0")),
        )
        fig.update_xaxes(gridcolor="#334155", showgrid=True)
        fig.update_yaxes(gridcolor="#334155", showgrid=True)

        st.plotly_chart(fig, use_container_width=True)

        # ── Anomaly summary table ─────────────────────────────────────────────
        anomalies_only = anomaly_df[anomaly_df["is_anomaly"]]
        if not anomalies_only.empty:
            with st.expander(f"Anomaly Summary ({len(anomalies_only)} detected)"):
                display = anomalies_only[["week_index", "sales", "residual",
                                          "anomaly_class", "event_label"]].copy()
                if "week_index" in display.columns:
                    display["week_index"] = display["week_index"].astype(str)
                st.dataframe(display, width="stretch")

    except Exception as e:
        st.error(f"EDA chart failed: {e}")
        _fallback_chart(sales)


def _fallback_chart(sales: pd.Series):
    """Renders a simple sales chart if EDA modules are unavailable."""
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=sales.index, y=sales.values,
        mode="lines", name="Weekly Sales",
        line=dict(color="#60a5fa", width=2),
    ))
    fig.update_layout(
        template="plotly_dark",
        paper_bgcolor="#0f172a",
        plot_bgcolor="#1e293b",
        height=380,
        title="Weekly Sales",
    )
    st.plotly_chart(fig, use_container_width=True)
