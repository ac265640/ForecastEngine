"""
tab_business.py — Business Impact Tab

Tab 4:
    - Newsvendor calculator: holding cost/unit + stockout cost/unit inputs
    - Optimal order quantity + expected weekly cost per model's forecast accuracy
    - Bar chart: cost differential between models
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from scipy.stats import norm


MODEL_COLORS = {
    "SARIMA":  "#60a5fa",
    "Prophet": "#f59e0b",
    "LSTM":    "#34d399",
}


def render(results_df: pd.DataFrame = None, weekly_df: pd.DataFrame = None):
    st.header("Business Impact")
    st.markdown(
        "Use the **Newsvendor model** to translate forecast accuracy into "
        "expected weekly inventory costs. Better forecasts → lower costs."
    )

    # ── Inputs ─────────────────────────────────────────────────────────────────
    st.subheader("Cost Parameters")
    col1, col2, col3 = st.columns(3)
    with col1:
        holding_cost = st.number_input(
            "Holding cost / unit / week ($)",
            min_value=0.01, max_value=100.0, value=2.0, step=0.5,
            help="Cost of holding one unsold unit for one week.",
        )
    with col2:
        stockout_cost = st.number_input(
            "Stockout cost / unit / week ($)",
            min_value=0.01, max_value=500.0, value=8.0, step=1.0,
            help="Cost of one unit of unmet demand (lost margin + goodwill).",
        )
    with col3:
        mean_demand = st.number_input(
            "Mean weekly demand (units)",
            min_value=1.0, max_value=100000.0, value=500.0, step=50.0,
            help="Average weekly demand for this series.",
        )

    st.divider()

    # ── Load or generate model metrics ────────────────────────────────────────
    if results_df is not None and not results_df.empty:
        model_metrics = _extract_model_metrics(results_df, mean_demand)
    else:
        model_metrics = _demo_model_metrics(mean_demand)
        st.info(
            "**Demo mode** — using illustrative MASE/MAE values. "
            "Run the pipeline for actual model metrics."
        )

    # ── Newsvendor calculation ─────────────────────────────────────────────────
    st.subheader("Optimal Order Quantities")
    critical_ratio = stockout_cost / (stockout_cost + holding_cost)

    results = []
    for model_name, metrics in model_metrics.items():
        mae = metrics["MAE"]
        # Approximate forecast std from MAE (assume normal errors: σ ≈ MAE * √(π/2))
        sigma = mae * np.sqrt(np.pi / 2)
        # Optimal order quantity (Newsvendor formula)
        z = norm.ppf(critical_ratio)
        q_star = mean_demand + z * sigma
        q_star = max(0, q_star)

        # Expected weekly cost
        # E[holding] = holding_cost × E[max(Q-D, 0)]
        # E[stockout] = stockout_cost × E[max(D-Q, 0)]
        expected_holding = holding_cost * sigma * norm.pdf(z)
        expected_stockout = stockout_cost * sigma * (1 - critical_ratio) * norm.pdf(z)
        total_cost = expected_holding + expected_stockout

        results.append({
            "model": model_name,
            "mae": mae,
            "sigma": sigma,
            "q_star": q_star,
            "expected_cost": total_cost,
        })

    results_df_calc = pd.DataFrame(results)
    results_df_calc = results_df_calc.sort_values("expected_cost").reset_index(drop=True)

    # ── KPI cards ──────────────────────────────────────────────────────────────
    cols = st.columns(len(results))
    for col_idx, (_, row) in enumerate(results_df_calc.iterrows()):
        with cols[col_idx]:
            color = MODEL_COLORS.get(row["model"], "#94a3b8")
            is_best = col_idx == 0  # first row after sort is the cheapest model
            border = f"border: 2px solid {color};" if is_best else ""
            badge = "Best" if is_best else ""
            st.markdown(
                f"<div style='background:#1e293b;padding:16px;border-radius:12px;"
                f"{border}text-align:center;'>"
                f"<p style='color:{color};font-size:15px;font-weight:700;margin:0'>"
                f"{row['model']} {badge}</p>"
                f"<p style='color:#94a3b8;font-size:12px;margin:6px 0 2px'>Optimal Order</p>"
                f"<p style='color:#e2e8f0;font-size:22px;font-weight:700;margin:0'>"
                f"{row['q_star']:.0f} units</p>"
                f"<p style='color:#94a3b8;font-size:12px;margin:6px 0 2px'>Expected Cost/Week</p>"
                f"<p style='color:{color};font-size:18px;font-weight:600;margin:0'>"
                f"${row['expected_cost']:.2f}</p>"
                f"<p style='color:#64748b;font-size:11px;margin:4px 0 0'>MAE = {row['mae']:.1f}</p>"
                f"</div>",
                unsafe_allow_html=True,
            )

    st.divider()

    # ── Cost differential bar chart ────────────────────────────────────────────
    st.subheader("Weekly Cost by Model")

    best_cost = results_df_calc["expected_cost"].min()
    results_df_calc["savings_vs_worst"] = (
        results_df_calc["expected_cost"].max() - results_df_calc["expected_cost"]
    )

    fig = go.Figure()
    for _, row in results_df_calc.iterrows():
        color = MODEL_COLORS.get(row["model"], "#94a3b8")
        fig.add_trace(go.Bar(
            name=row["model"],
            x=[row["model"]],
            y=[row["expected_cost"]],
            marker_color=color,
            marker_line_color=color,
            marker_line_width=1.5,
            opacity=0.85,
            text=f"${row['expected_cost']:.2f}",
            textposition="outside",
            textfont=dict(color="#e2e8f0", size=14),
        ))

    # Savings annotation
    worst_cost = results_df_calc["expected_cost"].max()
    savings = worst_cost - best_cost
    best_model = results_df_calc.iloc[0]["model"]
    worst_model = results_df_calc.iloc[-1]["model"]

    fig.update_layout(
        title=dict(
            text=f"Using <b>{best_model}</b> vs <b>{worst_model}</b> saves "
                 f"<b>${savings:.2f}/week</b> in inventory costs",
            font=dict(size=14, color="#e2e8f0"),
        ),
        template="plotly_dark",
        paper_bgcolor="#0f172a", plot_bgcolor="#1e293b",
        font=dict(color="#e2e8f0"),
        height=380,
        showlegend=False,
        barmode="group",
        margin=dict(l=40, r=20, t=80, b=40),
        yaxis=dict(title="Expected Weekly Cost ($)", gridcolor="#334155"),
        xaxis=dict(title="Model"),
    )
    st.plotly_chart(fig, use_container_width=True)

    # ── Formula explanation ────────────────────────────────────────────────────
    with st.expander("Newsvendor Formula Details"):
        st.markdown(f"""
**Critical Ratio** = Stockout Cost / (Stockout + Holding) = 
`{stockout_cost:.2f} / ({stockout_cost:.2f} + {holding_cost:.2f}) = {critical_ratio:.3f}`

**Optimal Order Quantity** Q* = μ + z × σ  
where z = `norm.ppf({critical_ratio:.3f}) = {norm.ppf(critical_ratio):.3f}`  
and σ is approximated from model MAE (σ ≈ MAE × √(π/2))

A **MASE < 1.0** means the model beats a naïve seasonal baseline —  
lower MAE directly translates to tighter confidence intervals and lower expected inventory costs.
        """)


# ── Helpers ────────────────────────────────────────────────────────────────────

def _extract_model_metrics(results_df: pd.DataFrame, mean_demand: float) -> dict:
    metrics = {}
    for model in ["SARIMA", "Prophet", "LSTM"]:
        row = results_df[results_df["model"] == model]
        if not row.empty:
            metrics[model] = {
                "MASE": float(row["MASE"].mean()),
                "RMSE": float(row["RMSE"].mean()),
                "MAE": float(row["MAE"].mean()),
            }
    return metrics


def _demo_model_metrics(mean_demand: float) -> dict:
    """Realistic illustrative metrics for demo mode."""
    base_mae = mean_demand * 0.15
    return {
        "SARIMA":  {"MASE": 0.82, "RMSE": base_mae * 1.3, "MAE": base_mae * 0.95},
        "Prophet": {"MASE": 0.91, "RMSE": base_mae * 1.4, "MAE": base_mae * 1.05},
        "LSTM":    {"MASE": 0.78, "RMSE": base_mae * 1.2, "MAE": base_mae * 0.88},
    }
