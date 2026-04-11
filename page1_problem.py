import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
import pandas as pd


# ─────────────────────────────────────────
# ANONYMIZATION NOTE
# ─────────────────────────────────────────
# This app is based on a real industry study conducted at a major
# French pharmaceutical company (anonymized for confidentiality).
#
# The following data has been slightly modified to protect
# confidential information while preserving the magnitude
# and direction of the original findings:
#   - Baseline model KPIs (MAE, RMSE, Bias) — rounded/approximated
#   - All company and product names removed
#
# All other statistics come from fully public sources:
# ANSM, French Senate inquiry 2023, WHO, Novartis Annual Report 2022.
# ─────────────────────────────────────────


# ─────────────────────────────────────────
# DATA  (public sources)
# ─────────────────────────────────────────

shortage_data = pd.DataFrame({
    "Year":    [2016, 2017, 2018, 2019, 2020, 2021, 2022, 2023],
    "Signals": [400,  530,  868,  1200, 1504, 2160, 3500, 4900],
})

causes_data = pd.DataFrame({
    "Cause": [
        "Unexpected demand increase",
        "Manufacturing incidents",
        "Raw material shortage",
        "Regulatory updates",
        "Logistics disruptions",
        "Strategic decisions",
    ],
    "Percentage": [32, 24, 18, 10, 9, 7],
})

categories_data = pd.DataFrame({
    "Category": [
        "Cardiology", "Neurology", "Oncology",
        "Infectiology", "Immunology", "Other",
    ],
    "Shortages_pct": [22, 18, 15, 14, 12, 19],
})

ranking_data = pd.DataFrame({
    "Year":    [1995, 2000, 2005, 2010, 2015, 2022],
    "Ranking": [1,    2,    3,    4,    5,    5],
})

# ─────────────────────────────────────────
# BASELINE KPIs
# Slightly modified from real study values
# to protect confidential data.
# Magnitude and direction fully preserved.
# ─────────────────────────────────────────
BASELINE_MAE  = "~30%"
BASELINE_RMSE = "~70%"
BASELINE_BIAS = "~−10%"


# ─────────────────────────────────────────
# COLOUR PALETTE
# ─────────────────────────────────────────
PRIMARY = "#185FA5"
DANGER  = "#E24B4A"
SUCCESS = "#1D9E75"
WARNING = "#BA7517"
NEUTRAL = "#888780"


# ─────────────────────────────────────────
# PAGE RENDER
# ─────────────────────────────────────────
def render():

    # ── Header ────────────────────────────
    st.markdown("## The drug shortage crisis in France")
    st.markdown(
        "Understanding why accurate demand forecasting "
        "has become a public-health priority."
    )

    # ── Anonymization disclosure ──────────
    st.warning(
        "**Transparency note** — This study was conducted at a major French "
        "pharmaceutical company whose name has been anonymized for confidentiality. "
        "Baseline model KPIs shown at the bottom of this page have been slightly "
        "modified (rounded/approximated) to protect proprietary data. "
        "The magnitude and direction of all findings are fully preserved. "
        "All other statistics on this page come from public sources.",
        icon="⚠️",
    )
    st.divider()


    # ── KPI summary row ───────────────────
    st.markdown("### At a glance")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric(
        label="Stock-out signals (2022)",
        value="3 500",
        delta="+62% vs 2021",
        delta_color="inverse",
    )
    c2.metric(
        label="Population affected",
        value="37%",
        delta="of French patients",
        delta_color="off",
    )
    c3.metric(
        label="Signals doubled",
        value="2×",
        delta="Jan → Aug 2023",
        delta_color="inverse",
    )
    c4.metric(
        label="Drugs at risk of discontinuation",
        value="~700",
        delta="manufacturers considering exit",
        delta_color="inverse",
    )
    st.divider()


    # ── Shortage trend ────────────────────
    st.markdown("### Shortage signals reported to ANSM (2016 – 2023)")
    col_chart, col_text = st.columns([2, 1])

    with col_chart:
        fig = go.Figure()
        fig.add_trace(go.Bar(
            x=shortage_data["Year"],
            y=shortage_data["Signals"],
            marker_color=[
                DANGER if y >= 2022 else PRIMARY
                for y in shortage_data["Year"]
            ],
            text=shortage_data["Signals"],
            textposition="outside",
        ))
        fig.update_layout(
            height=320,
            margin=dict(t=20, b=20, l=0, r=0),
            xaxis=dict(tickmode="linear", dtick=1),
            yaxis=dict(title="Number of signals"),
            plot_bgcolor="rgba(0,0,0,0)",
            paper_bgcolor="rgba(0,0,0,0)",
            showlegend=False,
        )
        fig.update_xaxes(showgrid=False)
        fig.update_yaxes(gridcolor="#E5E5E5")
        st.plotly_chart(fig, use_container_width=True)

    with col_text:
        st.markdown("""
**What this chart shows**

Shortage signals sent to ANSM have grown
**nearly 12× in 7 years**.

The sharp acceleration from 2020 reflects:
- The COVID-19 pandemic
- The winter 2022–23 triple epidemic
  (COVID + flu + bronchiolitis)
- Growing dependency on Asian
  active ingredient suppliers
""")
    st.divider()


    # ── Root causes ───────────────────────
    st.markdown("### What causes shortages?")
    col_pie, col_causes = st.columns([1, 1])

    with col_pie:
        fig2 = px.pie(
            causes_data,
            names="Cause",
            values="Percentage",
            color_discrete_sequence=[
                PRIMARY, DANGER, WARNING, SUCCESS, NEUTRAL, "#AFA9EC"
            ],
            hole=0.45,
        )
        fig2.update_traces(textposition="outside", textinfo="percent+label")
        fig2.update_layout(
            height=340,
            margin=dict(t=10, b=10, l=0, r=0),
            showlegend=False,
            paper_bgcolor="rgba(0,0,0,0)",
        )
        st.plotly_chart(fig2, use_container_width=True)

    with col_causes:
        st.markdown("**Key insight per cause**")
        insights = {
            "Unexpected demand increase":
                "32% of cases — the main driver. "
                "Traditional models cannot anticipate epidemic spikes.",
            "Manufacturing incidents":
                "24% — quality control failures at a single "
                "global production site can trigger worldwide shortages.",
            "Raw material shortage":
                "18% — 80% of active ingredients are produced "
                "outside Europe, mostly in Asia.",
            "Regulatory updates":
                "10% — legal changes force reformulations "
                "or temporary market withdrawals.",
            "Logistics disruptions":
                "9% — geopolitical tensions impact "
                "packaging material supply.",
            "Strategic decisions":
                "7% — manufacturers deprioritise low-margin "
                "older drugs in favour of innovative products.",
        }
        for cause, text in insights.items():
            with st.expander(cause):
                st.write(text)
    st.divider()


    # ── Interactive category explorer ─────
    st.markdown("### Explore the impact by therapeutic category")
    selected = st.multiselect(
        "Select categories to display",
        options=categories_data["Category"].tolist(),
        default=categories_data["Category"].tolist(),
    )
    filtered = categories_data[
        categories_data["Category"].isin(selected)
    ].sort_values("Shortages_pct", ascending=True)

    if filtered.empty:
        st.warning("Please select at least one category.")
    else:
        fig3 = go.Figure(go.Bar(
            x=filtered["Shortages_pct"],
            y=filtered["Category"],
            orientation="h",
            marker_color=PRIMARY,
            text=[f"{v}%" for v in filtered["Shortages_pct"]],
            textposition="outside",
        ))
        fig3.update_layout(
            height=max(200, len(filtered) * 55),
            margin=dict(t=10, b=20, l=0, r=40),
            xaxis=dict(title="Share of total shortages (%)"),
            plot_bgcolor="rgba(0,0,0,0)",
            paper_bgcolor="rgba(0,0,0,0)",
        )
        fig3.update_xaxes(showgrid=False)
        fig3.update_yaxes(showgrid=False)
        st.plotly_chart(fig3, use_container_width=True)
    st.divider()


    # ── Production decline ────────────────
    st.markdown("### France's declining position as a drug producer")
    col_rank, col_rank_text = st.columns([2, 1])

    with col_rank:
        fig4 = go.Figure()
        fig4.add_trace(go.Scatter(
            x=ranking_data["Year"],
            y=ranking_data["Ranking"],
            mode="lines+markers+text",
            text=ranking_data["Ranking"].apply(lambda r: f"#{r}"),
            textposition="top center",
            line=dict(color=DANGER, width=2),
            marker=dict(size=10, color=DANGER),
        ))
        fig4.update_layout(
            height=280,
            margin=dict(t=20, b=20, l=0, r=0),
            yaxis=dict(
                title="European ranking",
                autorange="reversed",
                tickvals=[1, 2, 3, 4, 5],
                ticktext=["1st", "2nd", "3rd", "4th", "5th"],
            ),
            xaxis=dict(tickmode="linear", dtick=5),
            plot_bgcolor="rgba(0,0,0,0)",
            paper_bgcolor="rgba(0,0,0,0)",
            showlegend=False,
        )
        fig4.update_yaxes(gridcolor="#E5E5E5")
        fig4.update_xaxes(showgrid=False)
        st.plotly_chart(fig4, use_container_width=True)

    with col_rank_text:
        st.markdown("""
**From 1st to 5th in 25 years**

France was Europe's leading drug producer
in the 1990s. Four decades of offshoring
have created a critical dependency:

- Only **1 in 3** drugs consumed in France
  are produced domestically
- France no longer attracts production
  of innovative, high-value drugs
- ~700 molecules risk being abandoned
  by manufacturers
""")
    st.divider()


    # ── Transition to next page ───────────
    st.markdown("### Where does our case study company fit in?")
    st.markdown("""
The company studied operates across **8 therapeutic franchises**
and **60 brands** in France. Its forecasting accuracy directly
determines whether essential medicines reach patients on time.

During the study period, the existing forecasting model showed:
""")

    col_n1, col_n2, col_n3 = st.columns(3)
    col_n1.metric(
        label="Baseline MAE",
        value=BASELINE_MAE,
        delta="Mean absolute error",
        delta_color="off",
    )
    col_n2.metric(
        label="Baseline RMSE",
        value=BASELINE_RMSE,
        delta="Root mean squared error",
        delta_color="off",
    )
    col_n3.metric(
        label="Baseline Bias",
        value=BASELINE_BIAS,
        delta="Systematic underestimation",
        delta_color="inverse",
    )

    st.caption(
        "⚠️ These KPIs are approximated from real study values "
        "to protect confidential data. "
        "Magnitude and direction are fully preserved."
    )

    st.markdown("""
> The baseline model was **systematically underestimating demand**,
> increasing the risk of stock-outs for essential medicines.
> **The question becomes: can Machine Learning do better?**
""")

    st.divider()
    st.caption(
        "Public sources: ANSM annual reports 2016–2023 · "
        "French Senate inquiry on drug shortages (2023) · "
        "WHO Global medicine shortage report · "
        "Novartis Annual Report 2022"
    )


# ─────────────────────────────────────────
# ENTRY POINT
# ─────────────────────────────────────────
if __name__ == "__main__":
    st.set_page_config(
        page_title="Drug Forecast AI — The Problem",
        page_icon="💊",
        layout="wide",
    )
    render()
