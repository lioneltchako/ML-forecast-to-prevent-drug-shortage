import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
import pandas as pd
import numpy as np

from utils.colors import PRIMARY, DANGER, SUCCESS, WARNING, NEUTRAL, FRANCHISE_COLORS
from utils.synthetic_data import (
    BASELINE_MAE, BASELINE_RMSE, BASELINE_BIAS,
    XGBOOST_MAE,  XGBOOST_RMSE,  XGBOOST_BIAS,
    FRANCHISE_RESULTS, BRAND_RESULTS,
)

st.set_page_config(
    page_title="Drug Forecast AI — Results & Recommendations",
    page_icon="💊",
    layout="wide",
)


# ─────────────────────────────────────────
# HEATMAP HELPER
# ─────────────────────────────────────────

def build_heatmap(results_df: pd.DataFrame, group_col: str) -> go.Figure:
    """
    Build a side-by-side heatmap showing Baseline vs XGBoost
    for MAE / RMSE / Bias across groups (franchise or brand).
    """
    groups   = results_df[group_col].unique()
    metrics  = ["MAE", "RMSE", "Bias"]
    models   = ["Baseline", "XGBoost"]

    # Build z-matrix: rows = group × model, cols = metric
    row_labels = []
    z_vals     = []
    hover_text = []

    for grp in groups:
        for mdl in models:
            subset = results_df[(results_df[group_col] == grp) & (results_df["model"] == mdl)]
            if subset.empty:
                continue
            row = subset.iloc[0]
            row_labels.append(f"{grp} — {mdl}")
            vals = [row["MAE"], row["RMSE"], abs(row["Bias"])]
            z_vals.append(vals)
            hover_text.append([
                f"<b>{grp} | {mdl}</b><br>MAE: ~{row['MAE']}%",
                f"<b>{grp} | {mdl}</b><br>RMSE: ~{row['RMSE']}%",
                f"<b>{grp} | {mdl}</b><br>Bias: {row['Bias']:+d}%",
            ])

    z_arr   = np.array(z_vals, dtype=float)
    # Normalise each metric column to [0,1] for consistent coloring
    z_norm  = np.zeros_like(z_arr)
    for j in range(z_arr.shape[1]):
        col = z_arr[:, j]
        z_norm[:, j] = (col - col.min()) / (col.max() - col.min() + 1e-9)

    fig = go.Figure(go.Heatmap(
        z=z_norm,
        x=metrics,
        y=row_labels,
        colorscale=[[0, SUCCESS], [0.5, WARNING], [1, DANGER]],
        showscale=True,
        colorbar=dict(
            title="Relative error<br>(green=low, red=high)",
            tickvals=[0, 0.5, 1],
            ticktext=["Low", "Medium", "High"],
        ),
        text=[[f"~{int(z_arr[i, j])}%" for j in range(3)] for i in range(len(row_labels))],
        texttemplate="%{text}",
        hovertext=hover_text,
        hovertemplate="%{hovertext}<extra></extra>",
    ))

    fig.update_layout(
        height=max(300, len(row_labels) * 38),
        margin=dict(t=20, b=20, l=0, r=0),
        xaxis=dict(side="top"),
        plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
    )
    return fig


# ─────────────────────────────────────────
# PAGE RENDER
# ─────────────────────────────────────────
def render():

    st.markdown("## Results & recommendations")
    st.markdown(
        "Did the ML model deliver — and what should the business do next?"
    )

    st.warning(
        "**Transparency note** — All KPIs are approximated from real study values "
        "to protect confidential data. The magnitude, direction, and relative "
        "ranking of results are fully preserved. "
        "Franchise and brand names are anonymized. Approximated values are prefixed with ~.",
        icon="⚠️",
    )
    st.divider()

    # ═══════════════════════════════════════
    # SECTION 1 — Global performance
    # ═══════════════════════════════════════
    st.markdown("### Section 1 — Global performance")
    st.markdown("Comparing the existing baseline model to XGBoost across all 43 products.")

    # KPI cards
    col_h1, col_h2, col_h3 = st.columns(3)
    col_h1.metric("Baseline MAE",  f"~{BASELINE_MAE:.0f}%",  "Existing model",    delta_color="off")
    col_h2.metric("XGBoost MAE",   f"~{XGBOOST_MAE:.0f}%",   "ML model",         delta_color="off")
    col_h3.metric("Improvement",   f"~{(BASELINE_MAE-XGBOOST_MAE)/BASELINE_MAE*100:.0f}%",
                  "Relative MAE reduction", delta_color="normal")

    st.markdown("")

    # Comparison table
    global_df = pd.DataFrame([
        {
            "Model":     "Baseline",
            "MAE":       f"~{BASELINE_MAE:.0f}%",
            "RMSE":      f"~{BASELINE_RMSE:.0f}%",
            "Bias":      f"~{BASELINE_BIAS:+.0f}%",
            "MAE_num":   BASELINE_MAE,
            "RMSE_num":  BASELINE_RMSE,
            "Bias_num":  abs(BASELINE_BIAS),
        },
        {
            "Model":     "XGBoost",
            "MAE":       f"~{XGBOOST_MAE:.0f}%",
            "RMSE":      f"~{XGBOOST_RMSE:.0f}%",
            "Bias":      f"~{XGBOOST_BIAS:+.0f}%",
            "MAE_num":   XGBOOST_MAE,
            "RMSE_num":  XGBOOST_RMSE,
            "Bias_num":  abs(XGBOOST_BIAS),
        },
        {
            "Model":     "Δ Improvement",
            "MAE":       f"~{BASELINE_MAE-XGBOOST_MAE:.0f}pp",
            "RMSE":      f"~{BASELINE_RMSE-XGBOOST_RMSE:.0f}pp",
            "Bias":      f"~{abs(BASELINE_BIAS)-abs(XGBOOST_BIAS):.0f}pp closer to zero",
            "MAE_num":   BASELINE_MAE - XGBOOST_MAE,
            "RMSE_num":  BASELINE_RMSE - XGBOOST_RMSE,
            "Bias_num":  abs(BASELINE_BIAS) - abs(XGBOOST_BIAS),
        },
    ])

    st.dataframe(
        global_df[["Model", "MAE", "RMSE", "Bias"]],
        use_container_width=True, hide_index=True,
    )

    col_interp1, col_interp2, col_interp3 = st.columns(3)
    with col_interp1:
        st.info(
            f"**MAE: {BASELINE_MAE:.0f}% → {XGBOOST_MAE:.0f}%**  \n"
            "The average forecast is now {:.0f} percentage points closer to actual demand. "
            "This directly reduces the safety stock buffer needed.".format(BASELINE_MAE - XGBOOST_MAE),
            icon="📊",
        )
    with col_interp2:
        st.info(
            f"**RMSE: {BASELINE_RMSE:.0f}% → {XGBOOST_RMSE:.0f}%**  \n"
            "Fewer large misses — the model handles volatility spikes "
            "more robustly than the baseline.",
            icon="📉",
        )
    with col_interp3:
        st.success(
            f"**Bias: {BASELINE_BIAS:+.0f}% → {XGBOOST_BIAS:+.0f}%**  \n"
            "The systematic underestimation is almost entirely eliminated. "
            "This is the most impactful change for stock-out prevention.",
            icon="⚖️",
        )

    # Visual comparison bar chart
    metrics_chart = pd.DataFrame({
        "Metric":   ["MAE", "RMSE", "|Bias|"] * 2,
        "Value":    [BASELINE_MAE, BASELINE_RMSE, abs(BASELINE_BIAS),
                     XGBOOST_MAE,  XGBOOST_RMSE,  abs(XGBOOST_BIAS)],
        "Model":    ["Baseline"] * 3 + ["XGBoost"] * 3,
    })

    fig_global = px.grouped_bar = go.Figure()
    for model, color in [("Baseline", DANGER), ("XGBoost", SUCCESS)]:
        subset = metrics_chart[metrics_chart["Model"] == model]
        fig_global.add_trace(go.Bar(
            x=subset["Metric"], y=subset["Value"],
            name=model, marker_color=color,
            text=[f"~{v:.0f}%" for v in subset["Value"]],
            textposition="outside",
        ))

    fig_global.update_layout(
        barmode="group",
        height=320,
        margin=dict(t=20, b=20, l=0, r=0),
        yaxis=dict(title="Error (%)"),
        plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
        legend=dict(x=0.75, y=0.9),
    )
    fig_global.update_xaxes(showgrid=False)
    fig_global.update_yaxes(gridcolor="#E5E5E5")
    st.plotly_chart(fig_global, use_container_width=True)
    st.divider()

    # ═══════════════════════════════════════
    # SECTION 2 — Performance by franchise
    # ═══════════════════════════════════════
    st.markdown("### Section 2 — Performance by therapeutic franchise")
    st.markdown(
        "XGBoost does not improve equally across all franchises. "
        "Understanding where it excels and where it struggles is critical for deployment planning."
    )

    st.caption(
        "Heatmap: green = low error (better), red = high error (worse). "
        "Each franchise shows two rows: Baseline and XGBoost."
    )

    fig_franchise = build_heatmap(FRANCHISE_RESULTS, "franchise")
    st.plotly_chart(fig_franchise, use_container_width=True)

    col_fw1, col_fw2 = st.columns(2)
    with col_fw1:
        st.success(
            "**Where XGBoost wins most:**  \n"
            "Cardiology and Immunology — stable, high-volume franchises with clear seasonal patterns. "
            "The model's demand signals have the most signal to learn from.",
            icon="✅",
        )
    with col_fw2:
        st.warning(
            "**Where improvement is more modest:**  \n"
            "Solid Tumors and Neuroscience — high-volatility franchises driven by "
            "exceptional events (clinical decisions, compassionate use, individual patient prescriptions) "
            "that are inherently difficult to forecast from historical data alone.",
            icon="⚠️",
        )

    st.divider()

    # ═══════════════════════════════════════
    # SECTION 3 — Performance by brand
    # ═══════════════════════════════════════
    st.markdown("### Section 3 — Performance by brand")
    st.markdown(
        "Brand-level results show greater variance than franchise-level averages. "
        "Some brands benefit dramatically; others show modest improvement."
    )
    st.caption("Brands anonymized as Brand-A through Brand-J. Green = lower error.")

    # Brand filter
    franchise_filter = st.multiselect(
        "Filter by franchise",
        options=BRAND_RESULTS["franchise"].unique().tolist(),
        default=BRAND_RESULTS["franchise"].unique().tolist(),
    )

    filtered_brands = BRAND_RESULTS[BRAND_RESULTS["franchise"].isin(franchise_filter)]

    if filtered_brands.empty:
        st.warning("Select at least one franchise.")
    else:
        fig_brand = build_heatmap(filtered_brands, "brand")
        st.plotly_chart(fig_brand, use_container_width=True)

    col_bw1, col_bw2 = st.columns(2)
    with col_bw1:
        st.success(
            "**Brand-A (Cardiology):** MAE improves from ~24% to ~21% — "
            "a high-volume, stable product where the lag features are extremely predictive.",
            icon="🏆",
        )
    with col_bw2:
        st.warning(
            "**Brand-J (Neuroscience):** MAE moves from ~45% to ~43% — "
            "very high volatility driven by specialist prescribing patterns. "
            "ML helps at the margin, but a safety stock buffer remains essential.",
            icon="⚡",
        )

    st.divider()

    # ═══════════════════════════════════════
    # SECTION 4 — Recommendations
    # ═══════════════════════════════════════
    st.markdown("### Section 4 — Three actionable recommendations")
    st.markdown(
        "These recommendations are written for a supply chain director "
        "deciding how to deploy ML forecasting in a pharmaceutical portfolio."
    )

    st.markdown("---")

    # Recommendation 1
    col_r1a, col_r1b = st.columns([1, 8])
    col_r1a.markdown("## 1")
    with col_r1b:
        st.markdown("#### Deploy ML forecasting on high-volume, stable products first")
        st.success(
            "**Lowest risk. Fastest inventory impact.**  \n\n"
            "Stable products (CV < 50%) represent ~33 out of 43 SKUs and the majority of revenue. "
            "These are exactly where XGBoost performs best — MAE improvements of 3–5 percentage points "
            "are reliable, and the bias correction (~−10% → ~−1%) immediately reduces systematic under-ordering.  \n\n"
            "**Action:** Run XGBoost forecasts in parallel with the existing model for 2 planning cycles. "
            "Compare against actuals. Replace the existing model where XGBoost consistently wins.",
            icon="🚀",
        )

    st.markdown("---")

    # Recommendation 2
    col_r2a, col_r2b = st.columns([1, 8])
    col_r2a.markdown("## 2")
    with col_r2b:
        st.markdown("#### For high-volatility products, combine ML forecast with a safety stock buffer")
        st.warning(
            "**The model improves accuracy — but cannot fully predict exceptional events.**  \n\n"
            "High-volatility products (Neuroscience, Solid Tumors) have demand driven by factors "
            "outside any historical pattern: single prescriber decisions, compassionate use, "
            "or sudden therapeutic guideline changes.  \n\n"
            "XGBoost still reduces bias and improves average accuracy, but the residual error "
            "remains high. For these products, the right strategy is:  \n"
            "- Use XGBoost as the base forecast  \n"
            "- Apply a **higher safety stock multiplier** (Z = 2.0–2.3 vs 1.645 for stable products)  \n"
            "- Review these SKUs manually each planning cycle  \n\n"
            "**Action:** Segment products into two tiers — ML-automated and ML-assisted — "
            "based on CV threshold.",
            icon="🛡️",
        )

    st.markdown("---")

    # Recommendation 3
    col_r3a, col_r3b = st.columns([1, 8])
    col_r3a.markdown("## 3")
    with col_r3b:
        st.markdown("#### Integrate exceptional event flags into the planning process")
        st.info(
            "**They have the largest single impact on forecast error.**  \n\n"
            "The COVID flag, patent expiry events, and market disruption flags are among "
            "the top 5 most important demand signals in the model. Without them, "
            "the model interprets a pandemic or a competitor withdrawal as noise — "
            "and produces systematically wrong forecasts for months afterward.  \n\n"
            "In practice, this means:  \n"
            "- Maintain a **live event calendar** linked to the forecasting system  \n"
            "- Flag known upcoming events: patent expiries, regulatory changes, "
            "competitor launches, demand campaigns  \n"
            "- Retrain the model each quarter to incorporate the most recent event patterns  \n\n"
            "**Action:** Assign ownership of the event calendar to the demand planning team. "
            "Treat event flags as first-class inputs — not as post-hoc adjustments.",
            icon="📅",
        )

    st.markdown("---")
    st.divider()

    # ═══════════════════════════════════════
    # SUMMARY CARD
    # ═══════════════════════════════════════
    st.markdown("### Summary — what this study delivers")

    col_sum1, col_sum2 = st.columns([2, 1])

    with col_sum1:
        st.markdown(f"""
| What we measured | Baseline model | XGBoost model | Improvement |
|-----------------|---------------|--------------|-------------|
| Mean Absolute Error | ~{BASELINE_MAE:.0f}% | ~{XGBOOST_MAE:.0f}% | **~{(BASELINE_MAE-XGBOOST_MAE)/BASELINE_MAE*100:.0f}% relative reduction** |
| Root Mean Squared Error | ~{BASELINE_RMSE:.0f}% | ~{XGBOOST_RMSE:.0f}% | **~{(BASELINE_RMSE-XGBOOST_RMSE)/BASELINE_RMSE*100:.0f}% relative reduction** |
| Systematic bias | ~{BASELINE_BIAS:.0f}% | ~{XGBOOST_BIAS:.0f}% | **Nearly eliminated** |
| Safety stock impact | Higher buffer needed | Lower buffer needed | **Direct inventory cost reduction** |

*All KPIs are approximated from real study values to protect confidential data.*
""")

    with col_sum2:
        st.success(
            "**Overall:**  \nML forecasting reduces forecast error by **~9%** "
            "while systematically correcting the underestimation bias of the existing model.  \n\n"
            "For a portfolio of 43 products, this translates to a **measurable reduction in "
            "safety stock requirements** at the same service level — and a lower risk of "
            "the stock-outs that triggered this study.",
            icon="🎯",
        )

    st.markdown("""
> **For demand planners and supply chain directors:**
> The value of ML forecasting is not in replacing human judgement —
> it is in giving demand planners a more accurate, unbiased starting point,
> so that human adjustments can focus where they matter most:
> on the exceptional events that no model can fully predict.
""")

    st.divider()
    st.caption(
        "⚠️ All KPIs are approximated from real study values to protect confidential data. "
        "Magnitude, direction, and relative rankings are fully preserved. "
        "Franchise and brand names are anonymized. "
        "Sources: internal study data (anonymized) · ANSM · French Senate inquiry 2023"
    )


render()
