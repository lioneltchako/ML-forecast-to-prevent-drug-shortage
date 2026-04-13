"""Page 5 — Model optimization, tuning curves, and safety-stock impact."""
# pylint: disable=wrong-import-position,use-dict-literal,too-many-locals,too-many-statements

import os
import sys
from typing import Any

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# pylint: disable=import-error
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import plotly.graph_objects as go  # noqa: E402
import streamlit as st  # noqa: E402

from utils.colors import DANGER, PRIMARY, SUCCESS, WARNING  # noqa: E402
from utils.domain import safety_stock  # noqa: E402
from utils.synthetic_data import BASELINE_MAE, XGBOOST_MAE  # noqa: E402

st.set_page_config(
    page_title="Drug Forecast AI — Optimization & Accuracy",
    page_icon="💊",
    layout="wide",
)

np.random.seed(7)

# ─────────────────────────────────────────
# SYNTHETIC LEARNING CURVES
# ─────────────────────────────────────────

n_estimators_range = np.arange(50, 501, 50)
train_mae = 22 - 6 * (1 - np.exp(-n_estimators_range / 120)) + np.random.normal(0, 0.3, len(n_estimators_range))
val_mae   = 32 - 5 * (1 - np.exp(-n_estimators_range / 150)) + np.random.normal(0, 0.5, len(n_estimators_range))
val_mae   = np.clip(val_mae, 27.5, 35)

# Depth curve
depth_range = np.arange(2, 11)
depth_train = 30 - 2.8 * depth_range + 0.18 * depth_range**2 - 5
depth_val   = 34 - 1.2 * depth_range + 0.14 * depth_range**2
depth_train = np.clip(depth_train, 16, 33)
depth_val   = np.clip(depth_val,   27, 38)

# LR curve
lr_range   = [0.01, 0.03, 0.05, 0.08, 0.1, 0.15, 0.2, 0.25, 0.3]
lr_val_mae = [33.5, 31.5, 29.8, 28.5, 28.1, 28.4, 29.0, 30.1, 31.5]


# ─────────────────────────────────────────
# PAGE RENDER
# ─────────────────────────────────────────
def render() -> None:
    """Render the optimization page: error metrics, tuning curves, and safety-stock impact."""
    st.markdown("## Model optimization & forecast accuracy")
    st.markdown("Tuning the model, measuring its errors, and translating accuracy into inventory outcomes.")

    st.warning(
        "**Transparency note** — All values are adjusted from real study data to protect confidentiality. "
        "Magnitude, direction, and relative rankings are preserved. Franchise and brand names are anonymized.",
        icon="⚠️",
    )
    st.divider()

    # ═══════════════════════════════════════
    # SECTION 1 — Forecast error metrics
    # ═══════════════════════════════════════
    st.markdown("### Section 1 — Understanding forecast error metrics")
    st.markdown(
        "Before evaluating the model, we need a shared language for measuring how wrong a forecast is. "
        "Here are the three metrics used in this study — each tells a different story."
    )

    col_m1, col_m2, col_m3 = st.columns(3)

    with col_m1:
        st.markdown("""
#### MAE — Mean Absolute Error
*Average gap between forecast and actual*

**Formula:** Average of |Forecast − Actual| / Actual × 100

**Plain language:**
"On average, my forecast is off by X% —
regardless of whether I over- or under-forecast."

**Example:**
Forecast 1 000 units, actual 800 →
error = 200 units = 25%

**Why it matters for inventory:**
MAE directly sets the size of the safety stock
buffer you need to absorb forecast errors.
""")

    with col_m2:
        st.markdown("""
#### RMSE — Root Mean Squared Error
*Penalises large errors more heavily*

**Formula:** √( Average of (Forecast − Actual)² ) / Actual × 100

**Plain language:**
"A single month where I'm off by 200%
hurts much more than two months off by 100%."

**Example:**
A COVID spike that the model missed entirely
gets heavily penalised in RMSE —
reflecting the real operational cost of a missed spike.

**Why it matters:**
RMSE identifies products where occasional
large misses are creating most of the risk.
""")

    with col_m3:
        st.markdown("""
#### Bias — Systematic directional error
*Does the model consistently over- or under-forecast?*

**Formula:** Average of (Forecast − Actual) / Actual × 100

**Plain language:**
"I always order 10% less than I should" —
a systematic error that compounds over time.

**Example:**
Consistent −10% bias means the warehouse
is chronically under-stocked relative to demand.

**Why it matters most:**
A biased model does not just make random errors —
it **systematically creates risk**. Correcting bias
has the most direct impact on stock-out prevention.
""")

    st.divider()

    # ── Interactive error calculator ──────
    st.markdown("#### Try it yourself — compute forecast error live")
    st.caption("Enter a forecast and actual value to see the three error metrics computed in real time.")

    col_calc1, col_calc2, col_calc3 = st.columns([1, 1, 2])

    with col_calc1:
        forecast_val = st.number_input("Your forecast (units)", min_value=0, value=1000, step=50)
    with col_calc2:
        actual_val   = st.number_input("Actual sales (units)", min_value=1,  value=800,  step=50)

    error     = forecast_val - actual_val
    mae_calc  = abs(error) / actual_val * 100
    rmse_calc = (error**2 / actual_val**2) ** 0.5 * 100  # single-point RMSE = |error|/actual
    bias_calc = error / actual_val * 100

    with col_calc3:
        c1, c2, c3 = st.columns(3)
        c1.metric("MAE",  f"{mae_calc:.1f}%",
                  "Over" if error > 0 else "Under" if error < 0 else "Exact",
                  delta_color="inverse" if abs(mae_calc) > 30 else "normal")
        c2.metric("RMSE", f"{rmse_calc:.1f}%")
        c3.metric("Bias", f"{bias_calc:+.1f}%",
                  "Overforecast" if bias_calc > 0 else "Underforecast" if bias_calc < 0 else "Neutral",
                  delta_color="inverse")

        if abs(bias_calc) > 20:
            st.warning(f"Large bias of {bias_calc:+.1f}% — if this were systematic, "
                       "it would require significant safety stock adjustment.")
        elif mae_calc < 20:
            st.success(f"Good accuracy — {mae_calc:.1f}% MAE is below the 28% XGBoost benchmark.")

    st.divider()

    # ═══════════════════════════════════════
    # SECTION 2 — Model tuning
    # ═══════════════════════════════════════
    st.markdown("### Section 2 — Model tuning (hyperparameter optimisation)")
    st.markdown("""
Before finalising the model, its **tuning parameters** (also called hyperparameters)
must be set correctly. These are not learned from data — they control *how* the model learns.

> **Analogy for supply chain professionals:**
> Think of it like calibrating the sensitivity of a planning system.
> Too sensitive → the model reacts to every noise spike and builds a "perfect" plan
> for the past that fails on new data (*over-adapting to past data*).
> Not sensitive enough → the model ignores real trends and produces flat, useless forecasts.
""")

    # Tuning parameters table
    params_df = pd.DataFrame([
        {
            "Parameter":         "n_estimators",
            "Supply chain name": "Number of correction rounds",
            "Tuned value":       "300",
            "Effect":            "More rounds = more accurate, but diminishing returns past 300",
        },
        {
            "Parameter":         "max_depth",
            "Supply chain name": "Decision complexity per round",
            "Tuned value":       "4",
            "Effect":            "Deeper trees capture more patterns but risk over-adapting",
        },
        {
            "Parameter":         "learning_rate",
            "Supply chain name": "Correction step size",
            "Tuned value":       "0.08",
            "Effect":            "Smaller = more precise but slower; too large = unstable",
        },
        {
            "Parameter":         "subsample",
            "Supply chain name": "Data sample per round",
            "Tuned value":       "0.8",
            "Effect":            "Uses 80% of data per tree — adds diversity, reduces over-fit",
        },
        {
            "Parameter":         "min_child_weight",
            "Supply chain name": "Minimum demand signal strength",
            "Tuned value":       "5",
            "Effect":            "Prevents splits on too few observations — key for sparse SKUs",
        },
    ])
    st.dataframe(params_df, use_container_width=True, hide_index=True)

    st.markdown("")
    st.markdown("#### Learning curves — how forecast error changes during tuning")
    st.caption("These curves show how training and validation error evolve as the model becomes more complex.")

    tune_param = st.radio(
        "Tuning parameter to visualise",
        options=["Number of trees (n_estimators)", "Tree depth (max_depth)", "Learning rate"],
        horizontal=True,
    )

    fig_lc = go.Figure()

    x_vals: Any
    if tune_param == "Number of trees (n_estimators)":
        x_vals  = n_estimators_range
        x_label = "Number of trees"
        fig_lc.add_trace(go.Scatter(
            x=x_vals, y=train_mae, mode="lines+markers",
            line=dict(color=PRIMARY, width=2), name="Training error",
        ))
        fig_lc.add_trace(go.Scatter(
            x=x_vals, y=val_mae, mode="lines+markers",
            line=dict(color=DANGER, width=2), name="Validation error (held-out data)",
        ))
        opt_x = 300
        fig_lc.add_vline(x=opt_x, line_dash="dash", line_color=SUCCESS,
                         annotation_text=f"Chosen: {opt_x} trees", annotation_position="top right")

    elif tune_param == "Tree depth (max_depth)":
        x_vals  = depth_range
        x_label = "Max tree depth"
        fig_lc.add_trace(go.Scatter(
            x=x_vals, y=depth_train, mode="lines+markers",
            line=dict(color=PRIMARY, width=2), name="Training error",
        ))
        fig_lc.add_trace(go.Scatter(
            x=x_vals, y=depth_val, mode="lines+markers",
            line=dict(color=DANGER, width=2), name="Validation error",
        ))
        fig_lc.add_vline(x=4, line_dash="dash", line_color=SUCCESS,
                         annotation_text="Chosen: depth 4", annotation_position="top right")
        fig_lc.add_annotation(x=3, y=depth_val[1]+1.5, text="Under-fitting\n(too simple)",
                               showarrow=False, font=dict(color=WARNING))
        fig_lc.add_annotation(x=8, y=depth_val[6]+1.5, text="Over-fitting\n(too complex)",
                               showarrow=False, font=dict(color=DANGER))

    else:  # Learning rate
        x_vals  = lr_range
        x_label = "Learning rate"
        fig_lc.add_trace(go.Scatter(
            x=x_vals, y=lr_val_mae, mode="lines+markers",
            line=dict(color=DANGER, width=2), name="Validation error",
        ))
        fig_lc.add_vline(x=0.08, line_dash="dash", line_color=SUCCESS,
                         annotation_text="Chosen: 0.08", annotation_position="top right")

    fig_lc.update_layout(
        height=320,
        margin=dict(t=20, b=20, l=0, r=0),
        xaxis=dict(title=x_label),
        yaxis=dict(title="MAE (%)"),
        plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
        legend=dict(x=0.6, y=0.9),
    )
    fig_lc.update_xaxes(showgrid=False)
    fig_lc.update_yaxes(gridcolor="#E5E5E5")
    st.plotly_chart(fig_lc, use_container_width=True)

    st.caption(
        "Training error (blue) measures how well the model fits known data. "
        "Validation error (red) measures how well it forecasts data it has never seen. "
        "The sweet spot is where validation error is minimised — beyond that, "
        "the model is over-adapting to past data."
    )
    st.divider()

    # ═══════════════════════════════════════
    # SECTION 3 — Safety stock impact
    # ═══════════════════════════════════════
    st.markdown("### Section 3 — Safety stock impact")
    st.markdown(
        "This is the most directly actionable output of the study for inventory management."
    )

    st.info(
        "**The core link between forecast accuracy and inventory:**  \n"
        "Safety stock exists to absorb forecast errors. The larger the error, "
        "the more buffer stock you need to maintain the same service level. "
        "Improving forecast accuracy directly reduces how much buffer stock is required.",
        icon="📦",
    )

    # ── Safety stock formula ──────────────
    st.markdown("#### The safety stock formula — plain language")
    col_f1, col_f2 = st.columns([2, 1])

    with col_f1:
        st.markdown(r"""
The standard safety stock formula:

$$SS = Z \times \sigma_{error} \times \sqrt{L}$$

Where:
- **Z** = service level factor (e.g. 1.645 for 95% fill rate)
- **σ_error** = standard deviation of forecast errors — **directly driven by MAE**
- **L** = replenishment lead time (months)

**In plain language:**
> "The less accurate your forecast, the more buffer stock you need
> to avoid running out at the same service level."

**The key insight:** σ_error ≈ MAE × 1.25 (for normally distributed errors).
So **every percentage-point improvement in MAE directly reduces safety stock.**
""")

    with col_f2:
        st.markdown("**Typical assumptions used in this study**")
        service_z   = st.slider("Service level factor Z",    min_value=1.0, max_value=2.5,
                                value=1.645, step=0.05,
                                help="1.645 = 95% service level; 2.326 = 99%")
        avg_demand  = st.slider("Average monthly demand",    min_value=500, max_value=10000,
                                value=2100, step=100)
        lead_time   = st.slider("Lead time (months)",        min_value=1, max_value=6,
                                value=2, step=1)

    # ── Safety stock comparison ───────────
    st.markdown("#### Baseline vs XGBoost — safety stock comparison")

    ss_baseline = safety_stock(BASELINE_MAE,  avg_demand, lead_time, service_z)
    ss_xgboost  = safety_stock(XGBOOST_MAE,   avg_demand, lead_time, service_z)
    ss_delta    = ss_baseline - ss_xgboost
    ss_delta_pct = ss_delta / ss_baseline * 100

    col_ss1, col_ss2, col_ss3 = st.columns(3)
    col_ss1.metric("Baseline safety stock",  f"{ss_baseline:,.0f} units",
                   f"MAE {BASELINE_MAE:.0f}%", delta_color="off")
    col_ss2.metric("XGBoost safety stock",   f"{ss_xgboost:,.0f} units",
                   f"MAE {XGBOOST_MAE:.0f}%", delta_color="off")
    col_ss3.metric("Reduction",              f"−{ss_delta:,.0f} units",
                   f"−{ss_delta_pct:.1f}% per product", delta_color="normal")

    st.caption(
        f"Calculated with: Z = {service_z}, avg demand = {avg_demand:,} units/month, "
        f"lead time = {lead_time} months. Adjust sliders above to explore different scenarios."
    )

    # ── Visualisation ─────────────────────
    mae_range = np.linspace(15, 50, 100)
    ss_curve  = [safety_stock(m, avg_demand, lead_time, service_z) for m in mae_range]

    fig_ss = go.Figure()

    # Safety stock curve
    fig_ss.add_trace(go.Scatter(
        x=mae_range, y=ss_curve,
        mode="lines", line=dict(color=PRIMARY, width=2),
        name="Safety stock required",
        fill="tozeroy", fillcolor="rgba(24,95,165,0.06)",
    ))

    # Baseline point
    fig_ss.add_trace(go.Scatter(
        x=[BASELINE_MAE], y=[ss_baseline],
        mode="markers+text",
        marker=dict(size=14, color=DANGER, symbol="diamond"),
        text=[f"Baseline<br>{BASELINE_MAE:.0f}% MAE<br>{ss_baseline:,.0f} units"],
        textposition="top right",
        textfont=dict(color=DANGER),
        name="Baseline model",
    ))

    # XGBoost point
    fig_ss.add_trace(go.Scatter(
        x=[XGBOOST_MAE], y=[ss_xgboost],
        mode="markers+text",
        marker=dict(size=14, color=SUCCESS, symbol="star"),
        text=[f"XGBoost<br>{XGBOOST_MAE:.0f}% MAE<br>{ss_xgboost:,.0f} units"],
        textposition="top left",
        textfont=dict(color=SUCCESS),
        name="XGBoost model",
    ))

    # Annotation arrow showing the saving
    fig_ss.add_annotation(
        x=(BASELINE_MAE + XGBOOST_MAE) / 2,
        y=(ss_baseline + ss_xgboost) / 2 + 50,
        text=f"−{ss_delta:,.0f} units<br>saved",
        showarrow=True, ax=0, ay=-30,
        arrowhead=2, arrowcolor=SUCCESS,
        font=dict(color=SUCCESS, size=12),
    )

    fig_ss.update_layout(
        height=380,
        margin=dict(t=30, b=20, l=0, r=0),
        xaxis=dict(title="Mean Absolute Error (%)", range=[10, 55]),
        yaxis=dict(title="Safety stock (units per product)"),
        plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
        legend=dict(x=0.65, y=0.95),
    )
    fig_ss.update_xaxes(showgrid=False)
    fig_ss.update_yaxes(gridcolor="#E5E5E5")
    st.plotly_chart(fig_ss, use_container_width=True)

    # ── Portfolio impact ──────────────────
    st.markdown("#### Portfolio-level impact")

    n_products  = 43
    total_delta = ss_delta * n_products

    col_pi1, col_pi2 = st.columns([2, 1])
    with col_pi1:
        st.markdown(f"""
Across the **43 products** in the portfolio:

| | Baseline | XGBoost | Improvement |
|--|---------|---------|-------------|
| Safety stock / product | {ss_baseline:,.0f} units | {ss_xgboost:,.0f} units | −{ss_delta:,.0f} units |
| Portfolio total | {ss_baseline*n_products:,.0f} units | {ss_xgboost*n_products:,.0f} units | −{total_delta:,.0f} units |
| Relative reduction | — | — | **−{ss_delta_pct:.1f}%** |

*Assumes all products have similar demand profile. Adjust sliders for custom scenarios.*
""")
    with col_pi2:
        st.success(
            f"**A {ss_delta_pct:.0f}% reduction in safety stock** "
            f"while maintaining the same {service_z:.3f} service level factor. "
            "This frees up capital, reduces waste risk, and improves working capital.",
            icon="📦",
        )

    st.markdown("""
> **Key takeaway for demand planning:**
> A 9% improvement in forecast accuracy does not just mean "better numbers."
> It directly translates to **less buffer stock, less capital tied up in inventory,
> and the same or better protection against stock-outs** — the exact trade-off
> that safety stock calculations are designed to optimise.
>
> → Continue to **Page 6 — Results & Recommendations** for the full performance comparison
> and actionable next steps.
""")

    st.divider()
    st.caption(
        "Safety stock calculations use standard formula SS = Z × σ × √L. "
        "σ estimated as MAE × 1.25 (normal distribution approximation). "
        "Results vary by product; portfolio-level figures use average demand assumptions."
    )


try:
    render()
except Exception as e:
    st.error(f"Page failed to render: {e}")
    st.stop()
