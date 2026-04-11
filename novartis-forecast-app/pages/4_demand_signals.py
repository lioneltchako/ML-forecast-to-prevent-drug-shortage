import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import streamlit as st
import plotly.graph_objects as go
import pandas as pd
import numpy as np

from utils.colors import PRIMARY, DANGER, SUCCESS, WARNING, NEUTRAL
from utils.synthetic_data import FEATURE_IMPORTANCE

st.set_page_config(
    page_title="Drug Forecast AI — Demand Signals",
    page_icon="💊",
    layout="wide",
)

# ─────────────────────────────────────────
# SIGNAL CATEGORY METADATA
# ─────────────────────────────────────────

CATEGORY_COLORS = {
    "Historical":  PRIMARY,
    "Seasonality": WARNING,
    "Events":      DANGER,
    "Product":     NEUTRAL,
}

CATEGORY_META = {
    "Historical": {
        "icon":    "📈",
        "title":   "Historical consumption patterns",
        "tagline": "What sold last month, last quarter, last year — the strongest predictors",
        "body": """
These are the **lag variables** — past sales values fed directly into the model as inputs.

| Signal | Technical name | What it tells the model |
|--------|---------------|------------------------|
| Last month's sales | `sales_lag_1` | The single strongest predictor — demand is autocorrelated |
| Sales 2 months ago | `sales_lag_2` | Short-term momentum signal |
| Sales 3 months ago | `sales_lag_3` | Bridges short and medium-term trends |
| Sales 6 months ago | `sales_lag_6` | Half-year context — detects mid-cycle patterns |
| Same month last year | `sales_lag_12` | Captures year-on-year seasonality exactly |
| 3-month moving average | `rolling_mean_3m` | Smooths noise; identifies recent trend direction |
| 6-month moving average | `rolling_mean_6m` | Medium-term demand level signal |
| Short-term trend (EMA) | `ema_3m` | Exponential moving average — weights recent months more |

**Why these dominate:** Pharmaceutical demand is highly autocorrelated.
Knowing what sold last month is by far the best predictor of what will sell next month.
Combined, historical consumption patterns explain ~70% of total feature importance.
""",
    },
    "Seasonality": {
        "icon":    "📅",
        "title":   "Seasonality signals",
        "tagline": "Recurring annual patterns in demand — encoded mathematically",
        "body": """
Pharmaceutical demand follows consistent annual cycles:
respiratory products peak in winter, allergy products in spring, etc.

The model captures this using **trigonometric encoding**:

| Signal | Technical name | What it tells the model |
|--------|---------------|------------------------|
| Seasonal cycle — sine | `sin_month` | Encodes month position on an annual cycle |
| Seasonal cycle — cosine | `cos_month` | Second component needed for full cycle coverage |
| Quarter of year | `quarter` | Coarser seasonal signal (Q1–Q4) |

**Why sin/cos instead of raw month number?**
See the visual explanation below.
""",
    },
    "Events": {
        "icon":    "⚡",
        "title":   "Exceptional demand events",
        "tagline": "Events that break normal patterns — the model needs to know about them",
        "body": """
Certain events cause demand to deviate sharply from its normal pattern.
Without explicitly flagging these, the model would either over-fit to them
or produce very large errors when they recur.

| Signal | Technical name | What it tells the model |
|--------|---------------|------------------------|
| COVID-19 period | `covid_flag` | Binary: 1 during Jan 2020–Jul 2021, 0 otherwise |
| Historical stock-out | `stockout_flag` | Flags periods where recorded sales were constrained by supply |
| Patent expiry event | `patent_loss` | Demand typically drops sharply when generics enter the market |
| Market disruption | `market_event` | Other significant market events (competitor withdrawal, etc.) |

**Key insight:** The COVID flag alone is one of the top 5 most important features.
Without it, the model would interpret the 2020 demand dip as a real structural change
and permanently under-forecast those products.

**For demand planners:** This is the equivalent of including event notes in your
planning horizon. The model "knows" something unusual happened and adjusts accordingly.
""",
    },
    "Product": {
        "icon":    "🏷️",
        "title":   "Product characteristics",
        "tagline": "Product-level context that shifts baseline demand",
        "body": """
These features provide the model with product-level context
that influences the baseline level of demand.

| Signal | Technical name | What it tells the model |
|--------|---------------|------------------------|
| Product category | `MITM` | Medication intended for temporary market (impacts demand profile) |
| Distribution channel | `CDM` | Channel mix affects volume and ordering patterns |

**Why these matter:** Two products in the same franchise with different distribution
channels have systematically different demand profiles.
Without this context, the model would be forced to treat them identically.
""",
    },
}


# ─────────────────────────────────────────
# PAGE RENDER
# ─────────────────────────────────────────
def render():

    st.markdown("## Demand signals")
    st.markdown(
        "What information drives forecast accuracy — "
        "and how the model learns to use it."
    )

    st.warning(
        "**Transparency note** — Feature names and importance values on this page "
        "are synthetic and generated to match the direction and relative magnitude "
        "of the real study findings. No actual model outputs are reproduced.",
        icon="⚠️",
    )
    st.divider()

    # ── Introduction ──────────────────────
    st.markdown("### How the model learns from signals in the data")
    col_intro1, col_intro2 = st.columns([2, 1])

    with col_intro1:
        st.markdown("""
A machine learning model does not "understand" demand the way a planner does.
Instead, it learns mathematical relationships between **input signals** and **future sales**.

The quality of those input signals — which ones you include, how you engineer them —
directly determines how accurate the forecasts will be.

In this study, **17 demand signals** were engineered across 4 categories:
""")
        col_c1, col_c2, col_c3, col_c4 = st.columns(4)
        col_c1.metric("Historical patterns", "8 signals", "~70% of importance")
        col_c2.metric("Seasonality",          "3 signals", "~11% of importance")
        col_c3.metric("Event flags",          "4 signals", "~11% of importance")
        col_c4.metric("Product context",      "2 signals", "~4% of importance")

    with col_intro2:
        st.info(
            "The terminology used in machine learning is **feature engineering** — "
            "creating informative inputs from raw data. In supply chain terms, "
            "this is simply asking: *what information would a good demand planner "
            "look at before making a forecast?*",
            icon="💡",
        )
    st.divider()

    # ── Signal categories ─────────────────
    st.markdown("### Signal categories — explore each group")

    tabs = st.tabs([
        f"{meta['icon']} {name}"
        for name, meta in CATEGORY_META.items()
    ])

    for tab, (cat_name, meta) in zip(tabs, CATEGORY_META.items()):
        with tab:
            st.markdown(f"**{meta['tagline']}**")
            st.markdown(meta["body"])

            # Mini importance chart for this category
            cat_features = FEATURE_IMPORTANCE[FEATURE_IMPORTANCE["category"] == cat_name].copy()
            cat_features = cat_features.sort_values("importance", ascending=True)

            fig_cat = go.Figure(go.Bar(
                x=cat_features["importance"],
                y=cat_features["label"],
                orientation="h",
                marker_color=CATEGORY_COLORS[cat_name],
                text=[f"{v:.1%}" for v in cat_features["importance"]],
                textposition="outside",
            ))
            fig_cat.update_layout(
                height=max(150, len(cat_features) * 42),
                margin=dict(t=10, b=10, l=0, r=60),
                xaxis=dict(title="Relative importance", tickformat=".0%"),
                plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
                showlegend=False,
            )
            fig_cat.update_xaxes(showgrid=False)
            fig_cat.update_yaxes(showgrid=False)
            st.plotly_chart(fig_cat, use_container_width=True)

    st.divider()

    # ── Sin/cos encoding visual ───────────
    st.markdown("### Why sin/cos encoding works better than a month number")
    col_sc1, col_sc2 = st.columns([2, 3])

    with col_sc1:
        st.markdown("""
**The problem with raw month numbers:**

If you feed the model "month = 12" (December) and "month = 1" (January),
the model sees them as 11 apart numerically — but they are only **1 month apart**
in reality. December and January have similar demand profiles.

**The solution — trigonometric encoding:**

By encoding month as:
- `sin_month = sin(2π × month / 12)`
- `cos_month = cos(2π × month / 12)`

...month 12 and month 1 become adjacent points on a circle.
The model "feels" the circular continuity of the year.

This is the standard approach for any cyclical variable:
hour of day, day of week, or month of year.
""")

    with col_sc2:
        months   = np.arange(1, 13)
        sin_vals = np.sin(2 * np.pi * months / 12)
        cos_vals = np.cos(2 * np.pi * months / 12)
        month_names = ["Jan", "Feb", "Mar", "Apr", "May", "Jun",
                       "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]

        fig_sc = go.Figure()

        # Circle of months
        theta = np.linspace(0, 2 * np.pi, 200)
        fig_sc.add_trace(go.Scatter(
            x=np.cos(theta), y=np.sin(theta),
            mode="lines", line=dict(color="#DDDDDD", width=1),
            showlegend=False, hoverinfo="skip",
        ))

        # Month points on the circle
        fig_sc.add_trace(go.Scatter(
            x=cos_vals, y=sin_vals,
            mode="markers+text",
            text=month_names,
            textposition="top center",
            textfont=dict(size=11),
            marker=dict(
                size=12,
                color=[WARNING if m in [12, 1, 6, 7] else PRIMARY for m in months],
            ),
            showlegend=False,
        ))

        # Highlight Dec-Jan adjacency
        fig_sc.add_trace(go.Scatter(
            x=[cos_vals[11], cos_vals[0]],
            y=[sin_vals[11], sin_vals[0]],
            mode="lines",
            line=dict(color=DANGER, width=2, dash="dot"),
            showlegend=True, name="Dec–Jan (adjacent)",
        ))

        fig_sc.update_layout(
            height=340,
            margin=dict(t=20, b=20, l=20, r=20),
            xaxis=dict(range=[-1.5, 1.5], showgrid=False, zeroline=False, visible=False),
            yaxis=dict(range=[-1.5, 1.5], showgrid=False, zeroline=False, visible=False,
                       scaleanchor="x"),
            plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
            legend=dict(x=0.02, y=0.02),
        )
        st.plotly_chart(fig_sc, use_container_width=True)
        st.caption(
            "Months plotted on the unit circle using sin/cos encoding. "
            "December and January appear adjacent — as they should be for a demand model."
        )

    st.divider()

    # ── Full feature importance chart ─────
    st.markdown("### Complete feature importance ranking")
    st.caption(
        "Sorted by relative importance (contribution to forecast accuracy). "
        "Each bar is colored by signal category."
    )

    fi_sorted = FEATURE_IMPORTANCE.sort_values("importance", ascending=True)
    colors    = [CATEGORY_COLORS[c] for c in fi_sorted["category"]]

    fig_fi = go.Figure(go.Bar(
        x=fi_sorted["importance"],
        y=fi_sorted["label"],
        orientation="h",
        marker_color=colors,
        text=[f"{v:.1%}" for v in fi_sorted["importance"]],
        textposition="outside",
        customdata=fi_sorted[["feature", "category"]].values,
        hovertemplate=(
            "<b>%{y}</b><br>"
            "Technical name: %{customdata[0]}<br>"
            "Category: %{customdata[1]}<br>"
            "Importance: %{x:.1%}<extra></extra>"
        ),
    ))
    fig_fi.update_layout(
        height=580,
        margin=dict(t=20, b=20, l=0, r=80),
        xaxis=dict(title="Relative importance", tickformat=".0%"),
        plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
        showlegend=False,
    )
    fig_fi.update_xaxes(showgrid=False)
    fig_fi.update_yaxes(showgrid=False)
    st.plotly_chart(fig_fi, use_container_width=True)

    # Category legend
    col_leg1, col_leg2, col_leg3, col_leg4 = st.columns(4)
    col_leg1.markdown(f"<span style='color:{PRIMARY}'>■</span> **Historical patterns**",     unsafe_allow_html=True)
    col_leg2.markdown(f"<span style='color:{WARNING}'>■</span> **Seasonality signals**",      unsafe_allow_html=True)
    col_leg3.markdown(f"<span style='color:{DANGER}'>■</span> **Exceptional events**",        unsafe_allow_html=True)
    col_leg4.markdown(f"<span style='color:{NEUTRAL}'>■</span> **Product characteristics**",  unsafe_allow_html=True)

    st.divider()

    # ── Top 5 signals ─────────────────────
    st.markdown("### The 5 most important signals — and what they tell us")

    top5 = FEATURE_IMPORTANCE.head(5)
    interpretations = {
        "sales_lag_1":     "Demand is **autocorrelated** — last month's sales is by far the best predictor of next month's. "
                           "The model leans on this heavily for stable products.",
        "sales_lag_2":     "Short-term **momentum** — two consecutive months of growth signal a trend "
                           "the model should continue; two months of decline suggest a correction.",
        "rolling_mean_3m": "The recent **demand level** — a smoothed view of where demand is right now, "
                           "filtering out month-to-month noise.",
        "sales_lag_12":    "**Year-on-year seasonality** — the same month last year provides the cleanest "
                           "seasonal benchmark. Especially powerful for products with strong annual cycles.",
        "ema_3m":          "**Trend direction** — the exponential moving average weights recent months more "
                           "heavily, making it sensitive to demand shifts without over-reacting to spikes.",
    }

    for i, (_, row) in enumerate(top5.iterrows(), 1):
        with st.container():
            col_rank, col_content = st.columns([1, 10])
            col_rank.markdown(f"### {i}")
            col_content.markdown(
                f"**{row['label']}** `{row['feature']}` — importance: **{row['importance']:.1%}**"
            )
            col_content.markdown(interpretations.get(row["feature"], ""))
        if i < 5:
            st.markdown("---")

    st.markdown("""
> **Key takeaway for demand planning:**
> The model is essentially asking *"what did we sell recently, and does the current
> month follow the usual seasonal pattern?"* — the same questions a good demand planner
> would ask, but applied consistently across 43 products simultaneously.
>
> → Continue to **Page 5 — Optimization & Accuracy** to see how the model was tuned
> and what its errors mean for inventory.
""")

    st.divider()
    st.caption(
        "⚠️ Feature importance values are synthetic and generated to match the direction "
        "and relative magnitude of the real study findings. "
        "Importance is measured as the mean decrease in forecast error contribution (gain)."
    )


try:
    render()
except Exception as e:
    st.error(f"Page failed to render: {e}")
    st.stop()
