import streamlit as st

st.set_page_config(
    page_title="Drug Forecast AI",
    page_icon="💊",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─────────────────────────────────────────
# LANDING PAGE
# ─────────────────────────────────────────

st.markdown("## Drug Demand Forecasting")
st.markdown("## Preventing Medicine Shortages with Machine Learning")
st.divider()

st.markdown(
    "**Master's thesis** — CentraleSupélec × [anonymized company] &nbsp;|&nbsp; "
    "**Lionel Davy Kouemeni Tchako** — 2023"
)
st.markdown("")

# ── Summary cards ─────────────────────────
c1, c2, c3, c4 = st.columns(4)
with c1:
    st.info(
        "**The Problem**  \nDrug shortages in France — nearly 12× increase "
        "in shortage signals over 7 years.",
        icon="🏥",
    )
with c2:
    st.info(
        "**The Data**  \n43 products · 5 therapeutic franchises · "
        "63 months of pharmaceutical sales history.",
        icon="📊",
    )
with c3:
    st.info(
        "**The Approach**  \nXGBoost trained on engineered demand signals — "
        "lags, seasonality, and event flags.",
        icon="🤖",
    )
with c4:
    st.success(
        "**The Result**  \nApprox. 9% improvement in forecast accuracy · "
        "directly reducing safety stock needs and stock-out risk.",
        icon="🎯",
    )

st.divider()

# ── Key result ────────────────────────────
col_r, col_n = st.columns([2, 1])
with col_r:
    st.markdown("""
> *Built as an interactive alternative to a static thesis —
> combining supply chain thinking with data science methods.*

This dashboard walks through the full study end-to-end:

| Page | What it covers |
|------|---------------|
| **1 — The Problem** | Why drug shortages matter and why forecasting accuracy is critical |
| **2 — The Data** | What the dataset looks like — 43 products, 5 years, COVID impact |
| **3 — Model Selection** | How XGBoost was chosen over simpler forecasting approaches |
| **4 — Demand Signals** | Which input signals drive forecast accuracy and why |
| **5 — Optimization & Accuracy** | How the model was tuned and what the errors mean in practice |
| **6 — Results & Recommendations** | Full performance comparison and three actionable recommendations |
""")

with col_n:
    st.metric(label="Forecast error improvement", value="approx. 9%", delta="MAE reduction vs baseline",
              delta_color="normal")
    st.metric(label="Bias correction", value="approx. −10% → approx. −1%", delta="Systematic underestimation fixed",
              delta_color="normal")
    st.metric(label="Products covered", value="43", delta="across 5 franchises",
              delta_color="off")

st.divider()
st.markdown("**Navigate using the sidebar to explore each section.**")
st.caption(
    "Sources: ANSM annual reports · French Senate inquiry 2023 · WHO · "
    "CentraleSupélec research framework · Anonymized company data"
)
