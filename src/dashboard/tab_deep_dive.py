"""
tab_deep_dive.py — Model Deep Dive Tab

Tab 3:
    - SARIMA: residual ACF plot + Ljung-Box p-value card
    - Prophet: component decomposition (trend, weekly, yearly, holidays)
    - LSTM: training loss curve + prediction vs actual scatter
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots


MODEL_COLORS = {
    "SARIMA":  "#60a5fa",
    "Prophet": "#f59e0b",
    "LSTM":    "#34d399",
}


def _hex_to_rgba(hex_color: str, alpha: float = 0.08) -> str:
    """Convert a CSS hex color string to an rgba() string suitable for Plotly."""
    h = hex_color.lstrip("#")
    r, g, b = int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16)
    return f"rgba({r},{g},{b},{alpha})"


def render(
    weekly_df: pd.DataFrame,
    calendar_df: pd.DataFrame,
    sarima_model=None,
    prophet_model=None,
    lstm_model=None,
):
    st.header("Model Deep Dive")

    model_tab = st.radio(
        "Select Model",
        ["SARIMA", "Prophet", "LSTM"],
        horizontal=True,
        key="deepdive_model",
    )

    series_ids = sorted(weekly_df["series_id"].unique())
    selected = st.selectbox("Series", series_ids, key="dd_series")

    series_df = weekly_df[weekly_df["series_id"] == selected].sort_values("week_start")
    sales = series_df.set_index("week_start")["total_sales"]

    st.divider()

    if model_tab == "SARIMA":
        _render_sarima(sales, selected, sarima_model)
    elif model_tab == "Prophet":
        _render_prophet(sales, selected, calendar_df, prophet_model)
    elif model_tab == "LSTM":
        _render_lstm(sales, selected, lstm_model)


# ── SARIMA ─────────────────────────────────────────────────────────────────────

def _render_sarima(sales: pd.Series, series_id: str, sarima_model=None):
    st.subheader("SARIMA — Residual Analysis")

    if sarima_model is None:
        # Demo: fit a lightweight model on the available series
        try:
            from src.pipeline.preprocessing import TemporalSplitter
            from src.models.sarima_model import SARIMAForecaster

            splitter = TemporalSplitter()
            train, val, test = splitter.split(sales.reset_index())
            train_series = train.set_index("week_start")["total_sales"]

            with st.spinner("Fitting SARIMA for diagnostics…"):
                model = SARIMAForecaster(series_id=series_id)
                model.fit(train_series)
                diag = model.diagnostics()
        except Exception as e:
            st.warning(f"Could not fit SARIMA model: {e}")
            diag = _demo_sarima_diag()
    else:
        diag = sarima_model.diagnostics()

    # ── Ljung-Box metric cards ─────────────────────────────────────────────
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        _metric_card("Ljung-Box p-value", f"{diag['lb_pvalue']:.4f}")
    with c2:
        status = "PASS" if diag["passed"] else "FAIL"
        color = "#22c55e" if diag["passed"] else "#ef4444"
        st.markdown(
            f"<div style='background:#1e293b;padding:12px;border-radius:8px;"
            f"border-left:4px solid {color};'>"
            f"<p style='color:#94a3b8;font-size:12px;margin:0'>Residual Test</p>"
            f"<p style='color:{color};font-size:20px;font-weight:700;margin:0'>{status}</p>"
            f"</div>",
            unsafe_allow_html=True,
        )
    with c3:
        _metric_card("AIC", f"{diag['aic']:.1f}")
    with c4:
        _metric_card("BIC", f"{diag['bic']:.1f}")

    st.caption(
        "Ljung-Box tests whether residuals are white noise. "
        "p > 0.05 = residuals are uncorrelated (model captures all structure). "
        "PASS = good fit."
    )

    # ── Residual ACF plot ──────────────────────────────────────────────────
    resid = diag["residuals"]
    try:
        from src.pipeline.eda import compute_acf_pacf
        acf_data = compute_acf_pacf(resid, nlags=40)
        lags = acf_data["lags"]
        acf_vals = acf_data["acf_values"]
        acf_ci = acf_data["acf_confint"]

        fig = go.Figure()
        fig.add_bar(
            x=lags, y=acf_vals,
            marker_color="#60a5fa", opacity=0.8, name="ACF",
        )
        upper_ci = acf_ci[:, 1] - acf_vals
        lower_ci = acf_ci[:, 0] - acf_vals
        fig.add_scatter(
            x=lags, y=upper_ci, mode="lines",
            line=dict(color="#ef4444", dash="dot", width=1),
            name="+95% CI", showlegend=True,
        )
        fig.add_scatter(
            x=lags, y=lower_ci, mode="lines",
            line=dict(color="#ef4444", dash="dot", width=1),
            name="−95% CI", showlegend=True,
        )
        fig.add_hline(y=0, line_color="#e2e8f0", line_width=0.8)
        fig.update_layout(
            title="Residual ACF (should lie within CI bands → white noise)",
            template="plotly_dark",
            paper_bgcolor="#0f172a", plot_bgcolor="#1e293b",
            font=dict(color="#e2e8f0"),
            height=320, margin=dict(l=40, r=20, t=50, b=40),
            xaxis=dict(title="Lag", gridcolor="#334155"),
            yaxis=dict(title="ACF", gridcolor="#334155"),
        )
        st.plotly_chart(fig, use_container_width=True)
    except Exception as e:
        st.warning(f"ACF plot unavailable: {e}")


# ── Prophet ────────────────────────────────────────────────────────────────────

def _render_prophet(
    sales: pd.Series,
    series_id: str,
    calendar_df: pd.DataFrame,
    prophet_model=None,
):
    st.subheader("Prophet — Component Decomposition")

    if prophet_model is None:
        try:
            from src.pipeline.preprocessing import TemporalSplitter
            from src.models.prophet_model import ProphetForecaster

            splitter = TemporalSplitter()
            train, val, test = splitter.split(sales.reset_index())
            train_series = train.set_index("week_start")["total_sales"]
            state = series_id.split("_")[0]

            with st.spinner("Fitting Prophet for component decomposition…"):
                model = ProphetForecaster(series_id=series_id, state=state)
                model.fit(train_series, calendar_df=calendar_df)
                _ = model.predict(steps=13)
                components = model.get_components()
        except Exception as e:
            st.warning(f"Could not fit Prophet model: {e}")
            components = _demo_prophet_components(sales)
    else:
        _ = prophet_model.predict(steps=13)
        components = prophet_model.get_components()

    comp_names = list(components.keys())
    if not comp_names:
        st.info("No components available.")
        return

    rows = len(comp_names)
    fig = make_subplots(
        rows=rows, cols=1,
        shared_xaxes=True,
        subplot_titles=[c.capitalize() for c in comp_names],
        vertical_spacing=0.06,
    )
    comp_colors = ["#a78bfa", "#60a5fa", "#34d399", "#f59e0b", "#f87171"]

    for i, (comp_name, comp_vals) in enumerate(components.items(), start=1):
        color = comp_colors[(i - 1) % len(comp_colors)]
        x = list(range(len(comp_vals)))
        fig.add_trace(
            go.Scatter(
                x=x, y=comp_vals.values if hasattr(comp_vals, "values") else comp_vals,
                mode="lines", name=comp_name.capitalize(),
                line=dict(color=color, width=2),
                fill="tozeroy" if comp_name != "trend" else None,
                fillcolor=_hex_to_rgba(color, alpha=0.08) if comp_name != "trend" else None,
            ),
            row=i, col=1,
        )

    fig.update_layout(
        height=180 * rows,
        template="plotly_dark",
        paper_bgcolor="#0f172a", plot_bgcolor="#1e293b",
        font=dict(color="#e2e8f0"),
        title=dict(text="Prophet Component Decomposition", font=dict(size=15)),
        margin=dict(l=40, r=20, t=50, b=20),
        showlegend=False,
    )
    fig.update_xaxes(gridcolor="#334155")
    fig.update_yaxes(gridcolor="#334155")
    st.plotly_chart(fig, use_container_width=True)


# ── LSTM ───────────────────────────────────────────────────────────────────────

def _render_lstm(sales: pd.Series, series_id: str, lstm_model=None):
    st.subheader("LSTM — Training Curves & Predictions")

    if lstm_model is None:
        history = _demo_lstm_history()
        test_actual, test_pred = _demo_lstm_predictions(sales)
        st.info("Demo mode — showing synthetic training curves. Run the pipeline for real results.")
    else:
        try:
            history = lstm_model.training_history()
            from src.pipeline.preprocessing import TemporalSplitter
            splitter = TemporalSplitter()
            train, val, test = splitter.split(sales.reset_index())
            train_vals = train["total_sales"].values
            test_actual = test["total_sales"].values
            last_window = train_vals[-52:]
            test_pred = lstm_model.predict(steps=len(test_actual), last_window=last_window)
        except Exception as e:
            st.warning(f"Could not load LSTM results: {e}")
            history = _demo_lstm_history()
            test_actual, test_pred = _demo_lstm_predictions(sales)

    col1, col2 = st.columns(2)

    # Loss curve
    with col1:
        fig_loss = go.Figure()
        epochs = list(range(1, len(history["loss"]) + 1))
        fig_loss.add_trace(go.Scatter(
            x=epochs, y=history["loss"],
            mode="lines", name="Train Loss",
            line=dict(color="#60a5fa", width=2),
        ))
        fig_loss.add_trace(go.Scatter(
            x=epochs, y=history["val_loss"],
            mode="lines", name="Val Loss",
            line=dict(color="#f59e0b", width=2, dash="dash"),
        ))
        fig_loss.update_layout(
            title="Training Loss Curve",
            template="plotly_dark",
            paper_bgcolor="#0f172a", plot_bgcolor="#1e293b",
            font=dict(color="#e2e8f0"),
            height=320, margin=dict(l=40, r=20, t=50, b=40),
            xaxis=dict(title="Epoch", gridcolor="#334155"),
            yaxis=dict(title="MSE Loss", gridcolor="#334155"),
            legend=dict(orientation="h", y=-0.15),
        )
        st.plotly_chart(fig_loss, use_container_width=True)

    # Predicted vs Actual scatter
    with col2:
        fig_scatter = go.Figure()
        min_val = min(test_actual.min(), test_pred.min())
        max_val = max(test_actual.max(), test_pred.max())

        # Perfect prediction line
        fig_scatter.add_trace(go.Scatter(
            x=[min_val, max_val], y=[min_val, max_val],
            mode="lines", name="Perfect Fit",
            line=dict(color="#475569", dash="dot", width=1.5),
        ))
        fig_scatter.add_trace(go.Scatter(
            x=test_actual, y=test_pred,
            mode="markers", name="Test Predictions",
            marker=dict(color="#34d399", size=8, opacity=0.8,
                        line=dict(color="#1e293b", width=1)),
        ))
        fig_scatter.update_layout(
            title="Predicted vs Actual (Test Set)",
            template="plotly_dark",
            paper_bgcolor="#0f172a", plot_bgcolor="#1e293b",
            font=dict(color="#e2e8f0"),
            height=320, margin=dict(l=40, r=20, t=50, b=40),
            xaxis=dict(title="Actual", gridcolor="#334155"),
            yaxis=dict(title="Predicted", gridcolor="#334155"),
        )
        st.plotly_chart(fig_scatter, use_container_width=True)


# ── Helpers ────────────────────────────────────────────────────────────────────

def _metric_card(label: str, value: str):
    st.markdown(
        f"<div style='background:#1e293b;padding:12px;border-radius:8px;'>"
        f"<p style='color:#94a3b8;font-size:12px;margin:0'>{label}</p>"
        f"<p style='color:#e2e8f0;font-size:20px;font-weight:700;margin:0'>{value}</p>"
        f"</div>",
        unsafe_allow_html=True,
    )


def _demo_sarima_diag():
    rng = np.random.default_rng(0)
    return {
        "lb_stat": 8.23,
        "lb_pvalue": 0.412,
        "passed": True,
        "aic": -184.3,
        "bic": -167.1,
        "residuals": pd.Series(rng.normal(0, 1, 200)),
    }


def _demo_prophet_components(sales: pd.Series):
    n = len(sales)
    t = np.arange(n)
    return {
        "trend":    pd.Series(sales.mean() + t * 2.5, name="trend"),
        "weekly":   pd.Series(30 * np.sin(2 * np.pi * t / 7), name="weekly"),
        "yearly":   pd.Series(80 * np.sin(2 * np.pi * t / 52), name="yearly"),
        "holidays": pd.Series(np.where(t % 52 == 47, 120, 0), name="holidays"),
    }


def _demo_lstm_history():
    epochs = 40
    rng = np.random.default_rng(1)
    train_loss = np.exp(-np.linspace(0, 3, epochs)) + rng.normal(0, 0.01, epochs)
    val_loss = np.exp(-np.linspace(0, 2.5, epochs)) + rng.normal(0, 0.015, epochs)
    return {
        "loss": train_loss.clip(0).tolist(),
        "val_loss": val_loss.clip(0).tolist(),
    }


def _demo_lstm_predictions(sales: pd.Series):
    rng = np.random.default_rng(2)
    actual = sales.values[-13:]
    predicted = actual * (1 + rng.normal(0, 0.08, size=len(actual)))
    return actual, predicted.clip(0)
