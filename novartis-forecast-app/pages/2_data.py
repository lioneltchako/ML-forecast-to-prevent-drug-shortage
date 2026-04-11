import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
import pandas as pd

from utils.colors import PRIMARY, DANGER, SUCCESS, WARNING, FRANCHISE_COLORS_LIST
from utils.synthetic_data import build_dataset, FRANCHISES, COVID_START, COVID_END

st.set_page_config(
    page_title="Drug Forecast AI — The Data",
    page_icon="💊",
    layout="wide",
)


# ─────────────────────────────────────────
# DERIVED DATA (cached — runs once per session)
# ─────────────────────────────────────────

@st.cache_data(show_spinner=False)
def get_page_data():
    df = build_dataset()

    monthly_agg = (
        df.groupby("date")["sales"].sum().reset_index()
        .rename(columns={"sales": "total_sales"})
    )

    monthly_avg = (
        df.assign(month=df["date"].dt.month)
        .groupby("month")["sales"].mean().reset_index()
        .rename(columns={"sales": "avg_sales"})
    )
    monthly_avg["month_name"] = (
        pd.to_datetime(monthly_avg["month"], format="%m").dt.strftime("%b")
    )

    cv_dist = (
        df[["product_id", "cv"]].drop_duplicates()
        .assign(segment=lambda x: pd.cut(
            x["cv"],
            bins=[0, 50, 100, 200],
            labels=["Stable (CV < 50%)", "Variable (50–100%)", "High volatility (CV > 100%)"],
        ))
    )

    franchise_dist = pd.DataFrame([
        {"franchise": f, "share": m["share"], "products": m["products"]}
        for f, m in FRANCHISES.items()
    ])

    return df, monthly_agg, monthly_avg, cv_dist, franchise_dist


# ─────────────────────────────────────────
# PAGE RENDER
# ─────────────────────────────────────────
def render():

    st.markdown("## The data")
    st.markdown("Exploring what pharmaceutical sales data looks like before building any model.")

    st.warning(
        "**Transparency note** — All data on this page is fully synthetic. "
        "It was generated to match the statistical properties of the real dataset "
        "(same number of products, franchise distribution, seasonality pattern, "
        "COVID-19 impact shape, and order-of-magnitude sales volumes). "
        "No real figures or identifiers from the original study are used.",
        icon="⚠️",
    )
    st.divider()

    try:
        df, monthly_agg, monthly_avg, cv_dist, franchise_dist = get_page_data()
    except Exception as e:
        st.error(f"Failed to load data: {e}")
        st.stop()

    # ── Dataset overview ──────────────────
    st.markdown("### Dataset overview")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Products",       "43",           "unique SKUs")
    c2.metric("Time period",    "63 months",    "Jan 2018 → Mar 2023")
    c3.metric("Franchises",     "5",            "therapeutic areas")
    c4.metric("Monthly median", "~2 100 units", "per product")
    st.divider()

    # ── Franchise distribution ────────────
    st.markdown("### Franchise distribution")
    col_f1, col_f2 = st.columns([1, 1])

    with col_f1:
        fig_f = px.pie(
            franchise_dist, names="franchise", values="share",
            color_discrete_sequence=FRANCHISE_COLORS_LIST, hole=0.45,
        )
        fig_f.update_traces(textposition="outside", textinfo="percent+label")
        fig_f.update_layout(height=320, margin=dict(t=10, b=10, l=0, r=0),
                            showlegend=False, paper_bgcolor="rgba(0,0,0,0)")
        st.plotly_chart(fig_f, use_container_width=True)

    with col_f2:
        st.markdown("**Products per franchise**")
        for _, row in franchise_dist.iterrows():
            st.markdown(f"**{row['franchise']}** — {row['products']} products · {row['share']}% of total sales")
            st.progress(int(row["share"]) / 100)
    st.divider()

    # ── Aggregated sales trend ────────────
    st.markdown("### Aggregated monthly sales trend")
    st.caption("Total synthetic sales across all 43 products. The COVID-19 period is highlighted.")

    fig_trend = go.Figure()
    fig_trend.add_vrect(
        x0=COVID_START, x1=COVID_END, fillcolor="#FAECE7", opacity=0.4,
        layer="below", line_width=0,
        annotation_text="COVID-19 period", annotation_position="top left",
        annotation_font_size=11,
    )
    fig_trend.add_trace(go.Scatter(
        x=monthly_agg["date"], y=monthly_agg["total_sales"],
        mode="lines", line=dict(color=PRIMARY, width=2),
        fill="tozeroy", fillcolor="rgba(24,95,165,0.08)", name="Total sales",
    ))
    fig_trend.update_layout(
        height=320, margin=dict(t=20, b=20, l=0, r=0),
        yaxis=dict(title="Total units sold"),
        plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)", showlegend=False,
    )
    fig_trend.update_xaxes(showgrid=False)
    fig_trend.update_yaxes(gridcolor="#E5E5E5")
    st.plotly_chart(fig_trend, use_container_width=True)

    col_t1, col_t2 = st.columns(2)
    with col_t1:
        st.info("**Upward trend** — Overall sales grow steadily, reflecting portfolio expansion and market growth.", icon="📈")
    with col_t2:
        st.warning("**COVID-19 dip** — A significant drop in 2020–2021 due to supply chain disruptions and demand shifts.", icon="🦠")
    st.divider()

    # ── Seasonality ───────────────────────
    st.markdown("### Seasonality — average sales by month")
    st.caption("A recurring annual pattern is visible: sales peak mid-year and dip in early months.")

    colors = [
        WARNING if row["avg_sales"] == monthly_avg["avg_sales"].max() else PRIMARY
        for _, row in monthly_avg.iterrows()
    ]
    fig_seas = go.Figure(go.Bar(
        x=monthly_avg["month_name"], y=monthly_avg["avg_sales"].round(0),
        marker_color=colors,
        text=monthly_avg["avg_sales"].round(0).astype(int), textposition="outside",
    ))
    fig_seas.update_layout(
        height=300, margin=dict(t=20, b=20, l=0, r=0),
        yaxis=dict(title="Average units sold"),
        plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)", showlegend=False,
    )
    fig_seas.update_xaxes(showgrid=False)
    fig_seas.update_yaxes(gridcolor="#E5E5E5")
    st.plotly_chart(fig_seas, use_container_width=True)

    st.markdown(
        "> This seasonality pattern is a key signal that the model must learn. "
        "It is captured using **trigonometric features** (sin/cos of month) in the feature engineering step."
    )
    st.divider()

    # ── Volatility explorer ───────────────
    st.markdown("### Sales volatility across products")
    st.caption("The coefficient of variation (CV) measures how unpredictable each product's sales are.")

    col_cv1, col_cv2 = st.columns([2, 1])
    with col_cv1:
        segment_counts = cv_dist["segment"].value_counts().reset_index()
        segment_counts.columns = ["segment", "count"]
        segment_counts = segment_counts.sort_values("segment")

        fig_cv = px.bar(
            segment_counts, x="segment", y="count", color="segment",
            color_discrete_map={
                "Stable (CV < 50%)":          SUCCESS,
                "Variable (50–100%)":         WARNING,
                "High volatility (CV > 100%)": DANGER,
            },
            text="count",
        )
        fig_cv.update_traces(textposition="outside")
        fig_cv.update_layout(
            height=300, margin=dict(t=20, b=20, l=0, r=0),
            yaxis=dict(title="Number of products"), xaxis=dict(title=""),
            plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)", showlegend=False,
        )
        fig_cv.update_xaxes(showgrid=False)
        fig_cv.update_yaxes(gridcolor="#E5E5E5")
        st.plotly_chart(fig_cv, use_container_width=True)

    with col_cv2:
        st.markdown("**What CV tells us**")
        st.success("**Stable products (CV < 50%)** — Predictable demand. ~33 products fall here.")
        st.warning("**Variable products (50–100%)** — Moderate unpredictability. ~3 products.")
        st.error("**High volatility (CV > 100%)** — Very hard to forecast. ~7 products. These drive most of the model error.")
    st.divider()

    # ── Product-level explorer ────────────
    st.markdown("### Product-level sales explorer")
    st.caption("Select a franchise and product to inspect individual sales patterns.")

    col_sel1, col_sel2 = st.columns(2)
    with col_sel1:
        selected_franchise = st.selectbox("Franchise", options=list(FRANCHISES.keys()))
    with col_sel2:
        products_in_franchise = df[df["franchise"] == selected_franchise]["product_id"].unique()
        selected_product = st.selectbox("Product", options=sorted(products_in_franchise))

    product_df = df[df["product_id"] == selected_product].copy()
    product_cv = product_df["cv"].iloc[0]
    if product_cv < 50:
        vlabel, vcolor = "Stable",          SUCCESS
    elif product_cv < 100:
        vlabel, vcolor = "Variable",        WARNING
    else:
        vlabel, vcolor = "High volatility", DANGER

    col_pm1, col_pm2, col_pm3 = st.columns(3)
    col_pm1.metric("Franchise",         selected_franchise)
    col_pm2.metric("CV",                f"{product_cv}%", vlabel)
    col_pm3.metric("Avg monthly sales", f"{int(product_df['sales'].mean()):,} units")

    franchise_list = list(FRANCHISES.keys())
    fig_prod = go.Figure()
    fig_prod.add_vrect(
        x0=COVID_START, x1=COVID_END, fillcolor="#FAECE7", opacity=0.4,
        layer="below", line_width=0,
        annotation_text="COVID-19", annotation_position="top left", annotation_font_size=10,
    )
    fig_prod.add_trace(go.Scatter(
        x=product_df["date"], y=product_df["sales"],
        mode="lines+markers", marker=dict(size=4),
        line=dict(color=FRANCHISE_COLORS_LIST[franchise_list.index(selected_franchise)], width=2),
        name=selected_product,
    ))
    fig_prod.update_layout(
        height=300, margin=dict(t=20, b=20, l=0, r=0),
        yaxis=dict(title="Units sold"),
        plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)", showlegend=False,
    )
    fig_prod.update_xaxes(showgrid=False)
    fig_prod.update_yaxes(gridcolor="#E5E5E5")
    st.plotly_chart(fig_prod, use_container_width=True)
    st.divider()

    # ── Outlier handling ──────────────────
    st.markdown("### How exceptional demand events were handled")
    col_o1, col_o2 = st.columns(2)

    with col_o1:
        st.markdown("""
**Detection — IQR method**

For each product, exceptional values were identified
using the interquartile range:

- Lower bound = Q1 − 1.5 × IQR
- Upper bound = Q3 + 1.5 × IQR

Values outside these bounds were flagged as
**exceptional demand events**.
""")

    with col_o2:
        st.markdown("""
**Imputation — KNN (k = 5)**

Rather than deleting flagged values, they were
replaced using **K-Nearest Neighbors imputation**
(k = 5 neighbors).

This preserves dataset size and uses the
multidimensional structure of the data to
estimate the most plausible replacement value.
""")

    with st.expander("Why not simply delete exceptional values?"):
        st.markdown("""
Deleting values would reduce dataset size, which is already
limited for some products (minimum 39 months of history).

KNN imputation is preferred because:
- It preserves the temporal structure of the series
- It leverages information from similar products
- It avoids introducing artificial constants (e.g. mean imputation)
  which would reduce variance and distort the signal
""")
    st.divider()

    # ── Transition ────────────────────────
    st.markdown("### What this data tells us about modelling needs")
    st.markdown("Three properties of this dataset directly shape the modelling decisions made next:")

    col_m1, col_m2, col_m3 = st.columns(3)
    with col_m1:
        st.info("**Seasonality is real** — Monthly patterns are consistent enough to be learned. → Trigonometric features needed.", icon="📅")
    with col_m2:
        st.warning("**COVID-19 is an anomaly** — It must be explicitly flagged, not learned as a trend. → Binary covid_period feature needed.", icon="🦠")
    with col_m3:
        st.error("**High volatility products exist** — A single model will struggle with them. → Per-product training strategy needed.", icon="⚠️")

    st.markdown("""
> → Continue to **Page 3 — Model Selection** to see how the forecasting algorithm was chosen.
""")

    st.caption(
        "⚠️ All data on this page is synthetic and generated solely for illustration. "
        "Statistical properties match the real dataset; no actual figures are reproduced."
    )


render()
