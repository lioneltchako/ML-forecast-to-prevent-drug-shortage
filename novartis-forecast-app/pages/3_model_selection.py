"""Page 3 — Algorithm comparison and XGBoost selection rationale."""
# pylint: disable=wrong-import-position,use-dict-literal,too-many-locals,too-many-statements

import os
import sys
from typing import TypedDict

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# pylint: disable=import-error
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import plotly.graph_objects as go  # noqa: E402
import streamlit as st  # noqa: E402

from utils.colors import NEUTRAL, SUCCESS, WARNING  # noqa: E402


class _AlgoMeta(TypedDict):
    mae: int
    training_speed: int
    interpretability: int
    seasonality: int
    outlier_handling: int
    color: str
    description: str
    icon: str


# ─────────────────────────────────────────
# TABLE STYLING HELPERS  (no matplotlib)
# ─────────────────────────────────────────
# Approach chosen: Styler.apply() with pure-CSS rgb() interpolation.
# Alternatives considered:
#   st.column_config ProgressColumn — no colour gradient, only bar width; poor for
#     qualitative 1-5 scores where green/red context matters.
#   st.html() manual HTML table — full control but verbose and fragile to maintain.
#   Styler.background_gradient() — requires matplotlib; ruled out by constraints.
# Pure-CSS interpolation matches the original RdYlGn look, adds zero dependencies,
# and works across pandas 2.x / 3.x.

def _col_gradient(series: pd.Series, higher_is_better: bool = True, use_abs: bool = False) -> list[str]:
    """Return CSS background strings for a Series using a red-yellow-green gradient."""
    vals = series.abs() if use_abs else series
    col_min = float(vals.min())
    col_max = float(vals.max())
    styles: list[str] = []
    for val in series:
        v = abs(float(val)) if use_abs else float(val)
        ratio = 0.0 if col_max == col_min else (v - col_min) / (col_max - col_min)
        if not higher_is_better:
            ratio = 1.0 - ratio   # flip: smaller value → greener
        # Interpolate: red (226,75,74) → yellow (232,197,71) → green (29,158,117)
        if ratio < 0.5:
            t = ratio * 2.0
            r, g, b = int(226 + 6 * t), int(75 + 122 * t), int(74 - 3 * t)
        else:
            t = (ratio - 0.5) * 2.0
            r, g, b = int(232 - 203 * t), int(197 - 39 * t), int(71 + 46 * t)
        styles.append(f"background-color: rgb({r},{g},{b}); color: #1a1a1a;")
    return styles


def _highlight_winner(row: pd.Series) -> list[str]:
    """Bold text + green left border on the winner row (any row whose Algorithm contains ✅)."""
    if "✅" in str(row.get("Algorithm", row.name)):
        return ["font-weight: bold; border-left: 3px solid #1D9E75;" for _ in row]
    return ["" for _ in row]

st.set_page_config(
    page_title="Drug Forecast AI — Model Selection",
    page_icon="💊",
    layout="wide",
)

# ─────────────────────────────────────────
# ALGORITHM DATA
# ─────────────────────────────────────────

ALGORITHMS: dict[str, _AlgoMeta] = {
    "Decision Tree": {
        "mae":              42,
        "training_speed":   5,
        "interpretability": 5,
        "seasonality":      2,
        "outlier_handling": 2,
        "color": NEUTRAL,
        "description": """
**Decision Tree** — A single tree that splits data on the most informative rule at each step.

- **How it works in plain language:** Like a flowchart of yes/no questions —
  "Was last month's sales > 5 000 units? If yes, predict high; if no, predict low."
- **Why it can struggle with demand forecasting:** A single tree memorises the training data too closely
  (*over-adapts to past data*). It is highly sensitive to exceptional demand events
  and tends to produce jagged, unstable forecasts.
- **Verdict:** Useful for understanding the data, but too fragile for reliable monthly forecasting.
""",
        "icon": "🌿",
    },
    "Random Forest": {
        "mae":              35,
        "training_speed":   3,
        "interpretability": 3,
        "seasonality":      3,
        "outlier_handling": 3,
        "color": "#6EA8D8",
        "description": """
**Random Forest** — Hundreds of Decision Trees trained on different random subsets of the data,
each giving a vote; the final forecast is the average.

- **How it works in plain language:** Like asking 200 demand planners to independently forecast
  next month's sales, each looking at a different slice of history, then averaging their answers.
- **Why it is better:** Averaging cancels out individual errors and reduces over-adaptation.
  Handles moderate volatility well.
- **Why it falls short here:** Still weaker than XGBoost on structured tabular data;
  slower to train and less able to correct systematic errors (like the negative bias we observed).
- **Verdict:** A strong baseline — but not the best available option.
""",
        "icon": "🌲",
    },
    "ExtraTrees": {
        "mae":              34,
        "training_speed":   4,
        "interpretability": 3,
        "seasonality":      3,
        "outlier_handling": 3,
        "color": "#5DB37E",
        "description": """
**ExtraTrees (Extremely Randomised Trees)** — Similar to Random Forest but uses random thresholds
for splits instead of searching for the optimal split.

- **How it works in plain language:** Like Random Forest, but each planner is also told to make
  their splits more randomly — this introduces even more diversity across trees.
- **Why it is fast:** No split-search step means much faster training than Random Forest.
- **Why it falls slightly short:** The extra randomness reduces bias but can hurt precision
  on datasets with clear structured patterns — like our seasonal pharmaceutical data.
- **Verdict:** Competitive with Random Forest; neither matches XGBoost on this dataset.
""",
        "icon": "🌳",
    },
    "AdaBoost": {
        "mae":              38,
        "training_speed":   2,
        "interpretability": 2,
        "seasonality":      2,
        "outlier_handling": 2,
        "color": WARNING,
        "description": """
**AdaBoost (Adaptive Boosting)** — Builds trees sequentially, each one focusing harder
on the examples the previous tree got wrong.

- **How it works in plain language:** Like a team where each new analyst is specifically
  assigned to fix the mistakes made by the previous one.
- **Why it struggles here:** AdaBoost is very sensitive to exceptional demand events.
  When a COVID spike or patent-loss event creates a large error, the next tree
  over-weights it — amplifying rather than smoothing those errors.
- **Verdict:** Works well for classification problems; poorly suited to our regression task
  with volatile pharmaceutical demand.
""",
        "icon": "🔁",
    },
    "XGBoost": {
        "mae":              28,
        "training_speed":   4,
        "interpretability": 3,
        "seasonality":      5,
        "outlier_handling": 5,
        "color": SUCCESS,
        "description": """
**XGBoost (Extreme Gradient Boosting)** — Like AdaBoost, builds trees sequentially to
correct prior errors, but uses a more sophisticated mathematical framework (gradient descent)
to do so more efficiently and robustly.

- **How it works in plain language:** An ensemble of trees where each new tree is precisely
  trained to fill the gaps left by all previous trees — using the gradient of the error
  to know exactly where to focus. Think of it as a team where each new analyst knows not
  just *what* went wrong, but *how much* and *in which direction* to correct.
- **Why it excels here:**
  - Natively handles the structured, tabular format of our engineered demand signals
  - Built-in regularisation (L1/L2) prevents over-adapting to past data
  - Directly corrects the negative bias (systematic underestimation) through its boosting mechanism
  - Handles seasonal patterns when combined with sin/cos features
- **Verdict:** Best overall performance across all metrics that matter for demand planning.
""",
        "icon": "🚀",
    },
}

# ─────────────────────────────────────────
# COMPARISON TABLE DATA
# ─────────────────────────────────────────

comparison_df = pd.DataFrame([
    {
        "Algorithm":       name,
        "MAE (%)":         d["mae"],
        "Speed":           d["training_speed"],
        "Interpretability": d["interpretability"],
        "Handles Seasonality": d["seasonality"],
        "Handles Outliers":    d["outlier_handling"],
    }
    for name, d in ALGORITHMS.items()
])

# ─────────────────────────────────────────
# BENCHMARK DATA (synthetic)
# ─────────────────────────────────────────

np.random.seed(99)
benchmark = pd.DataFrame({
    "Algorithm": list(ALGORITHMS.keys()),
    "MAE":       [d["mae"] + np.random.uniform(-1, 1) for d in ALGORITHMS.values()],
    "RMSE":      [d["mae"] * 2.3 + np.random.uniform(-2, 2) for d in ALGORITHMS.values()],
    "Color":     [d["color"] for d in ALGORITHMS.values()],
})
# Fix XGBoost to match known KPIs
benchmark.loc[benchmark["Algorithm"] == "XGBoost", "MAE"]  = 28.0
benchmark.loc[benchmark["Algorithm"] == "XGBoost", "RMSE"] = 63.0

# ─────────────────────────────────────────
# KPI BENCHMARK DATA (synthetic — page transparency note applies)
# ─────────────────────────────────────────
kpi_benchmark = pd.DataFrame([
    {"Algorithm": "Random Forest",     "Bias (%)": -4.8, "MAE": 142, "RMSE": 187},
    {"Algorithm": "Gradient Boosting", "Bias (%)": -3.1, "MAE": 118, "RMSE": 161},
    {"Algorithm": "XGBoost ✅",         "Bias (%)": -1.0, "MAE":  91, "RMSE": 134},
    {"Algorithm": "LightGBM",          "Bias (%)": -2.4, "MAE": 107, "RMSE": 152},
    {"Algorithm": "CatBoost",          "Bias (%)": -3.6, "MAE": 115, "RMSE": 163},
])


# ─────────────────────────────────────────
# PAGE RENDER
# ─────────────────────────────────────────
def render() -> None:
    """Render the model-selection page: algorithm comparison, radar charts, and benchmark."""
    st.markdown("## Model selection")
    st.markdown("How we chose the right forecasting algorithm — and why it matters for demand planning.")

    st.warning(
        "**Transparency note** — Benchmark results on this page are synthetic and generated "
        "to illustrate relative algorithm performance. They reflect the direction and magnitude "
        "of the real study findings without reproducing actual figures. "
        "Approximated KPIs are prefixed with ~.",
        icon="⚠️",
    )
    st.divider()

    # ── Why not Excel / moving averages? ──
    st.markdown("### Why not just use Excel or a moving average?")
    col_why1, col_why2 = st.columns([3, 2])

    with col_why1:
        st.markdown("""
Simple forecasting tools — spreadsheets, moving averages, exponential smoothing —
work well when demand is **stable and predictable**. For roughly a third of our products, they
hold up reasonably. But for the rest, three challenges make them inadequate:

1. **They cannot capture non-linear interactions.**
   A COVID wave + end-of-year seasonality + a product nearing patent expiry
   all interact in complex ways that a weighted average cannot model.

2. **They have no memory of exceptional events.**
   A 3-month moving average treats a pandemic as just "another data point"
   and normalises it away — then repeats the same error the next time.

3. **They cannot correct systematic bias.**
   If a model consistently underestimates demand, a moving average will keep
   underestimating. A machine learning model learns from its own errors and corrects.
""")

    with col_why2:
        st.info(
            "The existing model at the study company had a **approx. −10% systematic bias** — "
            "it consistently under-ordered, leading directly to stock-out risk. "
            "Simple models cannot self-correct this kind of structural error.",
            icon="📉",
        )
        st.markdown("")
        st.success(
            "Machine learning approaches, and XGBoost in particular, reduced that bias "
            "from **approx. −10% to approx. −1%** — a near-complete elimination of systematic "
            "underestimation.",
            icon="✅",
        )
    st.divider()

    # ── Algorithm comparison table ────────
    st.markdown("### Algorithms evaluated")
    st.caption(
        "Five tree-based ensemble algorithms were tested. "
        "Scores for Speed, Interpretability, Seasonality, and Outlier Handling "
        "are on a 1–5 scale (5 = best)."
    )

    # Styled table — pure-CSS gradient, no matplotlib dependency
    styled = (
        comparison_df.style
        .apply(_col_gradient, higher_is_better=False, subset=["MAE (%)"])
        .apply(_col_gradient, higher_is_better=True,  subset=["Speed"])
        .apply(_col_gradient, higher_is_better=True,  subset=["Interpretability"])
        .apply(_col_gradient, higher_is_better=True,  subset=["Handles Seasonality"])
        .apply(_col_gradient, higher_is_better=True,  subset=["Handles Outliers"])
        .format({"MAE (%)": "{:.0f}%"})
    )
    st.dataframe(styled, use_container_width=True, hide_index=True)

    st.caption(
        "MAE = Mean Absolute Error (forecast error as % of actual sales). "
        "Lower is better. Synthetic benchmark values."
    )
    st.divider()

    # ── KPI benchmark ─────────────────────
    st.markdown("### 📊 Benchmark results on dataset")
    st.caption(
        "Each model was trained and evaluated on the same historical demand dataset. "
        "KPIs are averaged across all SKUs and forecast horizons (4–12 weeks). "
        "All values are synthetic and prefixed with ~ per the transparency note above."
    )

    # Metric cards: bias and MAE improvement vs legacy
    col_bk1, col_bk2, col_bk3 = st.columns(3)
    col_bk1.metric("XGBoost Bias",  "approx. −1.0%", "−9 pp vs legacy (approx. −10%)", delta_color="normal")
    col_bk2.metric("XGBoost MAE",   "approx. 91",    "−51 units vs Random Forest (approx. 142)", delta_color="normal")
    col_bk3.metric("XGBoost RMSE",  "approx. 134",   "−53 vs Random Forest (approx. 187)", delta_color="normal")

    # Styled KPI table
    kpi_styled = (
        kpi_benchmark.style
        .apply(_col_gradient, higher_is_better=False, use_abs=True, subset=["Bias (%)"])
        .apply(_col_gradient, higher_is_better=False, subset=["MAE"])
        .apply(_col_gradient, higher_is_better=False, subset=["RMSE"])
        .apply(_highlight_winner, axis=1)
        .format({"Bias (%)": "{:+.1f}%", "MAE": "{:.0f}", "RMSE": "{:.0f}"})
    )
    st.dataframe(kpi_styled, use_container_width=True, hide_index=True)

    # Bar chart: MAE comparison across models
    kpi_colors = [
        SUCCESS if "✅" in str(row["Algorithm"]) else NEUTRAL
        for _, row in kpi_benchmark.iterrows()
    ]
    fig_kpi = go.Figure(go.Bar(
        x=kpi_benchmark["Algorithm"],
        y=kpi_benchmark["MAE"],
        marker_color=kpi_colors,
        text=[f"approx. {v:.0f}" for v in kpi_benchmark["MAE"]],
        textposition="outside",
    ))
    fig_kpi.add_hline(
        y=91, line_dash="dash", line_color=SUCCESS,
        annotation_text="XGBoost MAE: approx. 91", annotation_position="top right",
    )
    fig_kpi.update_layout(
        height=300,
        margin=dict(t=30, b=20, l=0, r=0),
        yaxis=dict(title="~MAE (units)", range=[0, 210]),
        plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
        showlegend=False,
    )
    fig_kpi.update_xaxes(showgrid=False)
    fig_kpi.update_yaxes(gridcolor="#E5E5E5")
    st.plotly_chart(fig_kpi, use_container_width=True)

    col_kb1, col_kb2 = st.columns(2)
    with col_kb1:
        st.success(
            "**XGBoost achieved the lowest bias (approx. −1%)** — down from approx. −10% with the legacy model. "
            "This nearly eliminates systematic under-ordering.",
            icon="⚖️",
        )
    with col_kb2:
        st.success(
            "**XGBoost led on both error metrics** — approx. 91 MAE and approx. 134 RMSE "
            "vs 107–142 and 152–187 for other models.",
            icon="🏆",
        )
    st.divider()

    # ── Interactive algorithm explorer ────
    st.markdown("### Explore each algorithm")
    st.caption("Select an algorithm to read how it works and why it does or doesn't suit demand forecasting.")

    selected_algo = st.radio(
        "Algorithm",
        options=list(ALGORITHMS.keys()),
        horizontal=True,
    )

    algo_data = ALGORITHMS[selected_algo]
    col_desc, col_radar = st.columns([3, 2])

    with col_desc:
        st.markdown(f"#### {algo_data['icon']} {selected_algo}")
        st.markdown(algo_data["description"])

    with col_radar:
        categories   = ["MAE Accuracy", "Speed", "Interpretability", "Seasonality", "Outlier Handling"]
        # Invert MAE score so higher = better for radar
        mae_score    = max(1, 6 - round((algo_data["mae"] - 25) / 4))
        values       = [
            mae_score,
            algo_data["training_speed"],
            algo_data["interpretability"],
            algo_data["seasonality"],
            algo_data["outlier_handling"],
        ]
        values_closed = values + [values[0]]
        cats_closed   = categories + [categories[0]]

        fig_radar = go.Figure()
        fig_radar.add_trace(go.Scatterpolar(
            r=values_closed, theta=cats_closed,
            fill="toself",
            fillcolor=f"rgba({int(algo_data['color'][1:3], 16)}, "
                      f"{int(algo_data['color'][3:5], 16)}, "
                      f"{int(algo_data['color'][5:7], 16)}, 0.25)",
            line=dict(color=algo_data["color"], width=2),
            name=selected_algo,
        ))
        fig_radar.update_layout(
            polar=dict(
                radialaxis=dict(visible=True, range=[0, 5], tickvals=[1, 2, 3, 4, 5]),
                bgcolor="rgba(0,0,0,0)",
            ),
            showlegend=False,
            height=300,
            margin=dict(t=30, b=30, l=30, r=30),
            paper_bgcolor="rgba(0,0,0,0)",
        )
        st.plotly_chart(fig_radar, use_container_width=True)
        st.caption("Radar chart: all axes scaled 1–5, higher = better. MAE Accuracy is inverted (lower error → higher score).")

    st.divider()

    # ── Benchmark results ─────────────────
    st.markdown("### Benchmark results — out-of-sample testing")
    st.caption(
        "Each algorithm was tested on held-out data it had never seen during training "
        "(out-of-sample testing). Results below are synthetic and illustrative."
    )

    col_b1, col_b2 = st.columns([3, 2])

    with col_b1:
        fig_bench = go.Figure()
        fig_bench.add_trace(go.Bar(
            x=benchmark["Algorithm"],
            y=benchmark["MAE"],
            marker_color=benchmark["Color"].tolist(),
            text=[f"approx. {v:.0f}%" for v in benchmark["MAE"]],
            textposition="outside",
            name="MAE",
        ))
        fig_bench.add_hline(
            y=28, line_dash="dash", line_color=SUCCESS,
            annotation_text="XGBoost target (approx. 28%)",
            annotation_position="top right",
        )
        fig_bench.update_layout(
            height=320,
            margin=dict(t=30, b=20, l=0, r=0),
            yaxis=dict(title="Mean Absolute Error (%)", range=[0, 55]),
            plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
            showlegend=False,
        )
        fig_bench.update_xaxes(showgrid=False)
        fig_bench.update_yaxes(gridcolor="#E5E5E5")
        st.plotly_chart(fig_bench, use_container_width=True)

    with col_b2:
        st.markdown("**What these results mean in practice**")
        for _, row in benchmark.iterrows():
            delta_vs_xgb = row["MAE"] - 28.0
            if row["Algorithm"] == "XGBoost":
                st.success(f"**{row['Algorithm']}** — ~{row['MAE']:.0f}% MAE — **Winner**", icon="🏆")
            elif delta_vs_xgb <= 8:
                st.info(f"**{row['Algorithm']}** — approx. {row['MAE']:.0f}% MAE — +{delta_vs_xgb:.0f}pp vs XGBoost")
            else:
                st.error(f"**{row['Algorithm']}** — approx. {row['MAE']:.0f}% MAE — +{delta_vs_xgb:.0f}pp vs XGBoost")

    st.divider()

    # ── XGBoost justification ─────────────
    st.markdown("### Why XGBoost — in supply chain terms")

    col_j1, col_j2, col_j3 = st.columns(3)
    with col_j1:
        st.success(
            "**Best forecast accuracy**  \n"
            "approx. 28% MAE vs approx. 34–42% for alternatives — "
            "directly translates to lower safety stock requirements.",
            icon="🎯",
        )
    with col_j2:
        st.success(
            "**Corrects systematic bias**  \n"
            "Reduces the underestimation bias from approx. −10% to approx. −1% — "
            "the single most impactful improvement for stock-out prevention.",
            icon="⚖️",
        )
    with col_j3:
        st.success(
            "**Handles the full complexity of demand**  \n"
            "Seasonality, COVID disruptions, patent expiries, and product mix "
            "are all captured through engineered demand signals.",
            icon="🔧",
        )

    st.markdown("""
> **Bottom line for demand planning:** XGBoost gives the best forecast accuracy on both
> stable and volatile products, corrects the existing model's systematic underestimation,
> and produces a forecast that is ready to feed directly into safety stock calculations.
>
> → Continue to **Page 4 — Demand Signals** to see which input signals drive that accuracy.
""")

    st.divider()
    st.caption(
        "⚠️ Benchmark results are synthetic and illustrative. "
        "Algorithm descriptions reflect established machine learning literature. "
        "Out-of-sample testing used a walk-forward validation scheme over 6-month horizons."
    )


try:
    render()
except Exception as e:
    st.error(f"Page failed to render: {e}")
    st.stop()
