## Drug Demand Forecasting
## Preventing Medicine Shortages with Machine Learning

Master's thesis — CentraleSupélec × [anonymized company]
Lionel Davy Kouemeni Tchako — 2023

This interactive dashboard presents an end-to-end demand forecasting study:
- The business problem: drug shortages in France
- The data: 43 products, 5 years of sales history
- The approach: XGBoost with demand signal engineering
- The result: approx. 9% MAE improvement over the baseline model,
  directly reducing stock-out risk and safety stock needs

> Built as an interactive alternative to a static thesis —
> combining supply chain thinking with data science methods.

---

## Business context

Drug shortages in France have increased nearly 12x in 7 years.
A Fortune 500 pharmaceutical manufacturer needed to improve its
demand forecasting accuracy to prevent stock-outs of essential
medicines across a portfolio of 43 products spanning multiple
therapeutic franchises.

This study evaluated multiple ML algorithms and found that XGBoost,
combined with engineered demand signals (lag features, seasonality
encoding, and event flags), reduced forecast error by approx. 9%
and nearly eliminated systematic underestimation bias
(from approx. -10% to approx. -1%).

---

## Features

- 6-page interactive Streamlit dashboard
- Fully synthetic dataset matching real statistical properties
- Algorithm comparison with radar charts and styled tables
- Feature importance analysis with sin/cos encoding visualization
- Interactive hyperparameter tuning curves
- Safety stock impact calculator with live sliders
- Franchise and brand-level performance heatmaps
- Three actionable deployment recommendations

---

## How to run locally

```bash
# 1. Install dependencies
pip install streamlit plotly pandas numpy

# 2. Move into the app directory
cd pharma-forecast-app

# 3. Launch the app
streamlit run app.py
```

The app will open at `http://localhost:8501`.

---

## How to deploy on Streamlit Cloud

1. Push this repository to GitHub (the `pharma-forecast-app/` folder must be at the root or specify it as the app directory)
2. Go to [share.streamlit.io](https://share.streamlit.io), connect your repo, and set the main file to `app.py`

---

## App structure

```
pharma-forecast-app/
├── app.py                   ← Home / landing page
├── pages/
│   ├── 1_problem.py         ← The drug shortage crisis in France
│   ├── 2_data.py            ← Dataset exploration (synthetic data)
│   ├── 3_model_selection.py ← Algorithm comparison and XGBoost selection
│   ├── 4_demand_signals.py  ← Feature engineering and importance
│   ├── 5_optimization.py    ← Tuning, error metrics, safety stock impact
│   └── 6_results.py         ← Results, heatmaps, and recommendations
├── utils/
│   ├── colors.py            ← Shared colour palette
│   ├── synthetic_data.py    ← Shared data generators and KPI constants
│   ├── domain.py            ← Domain logic (safety stock formula)
│   └── disclaimer.py        ← Shared anonymization banners
├── tests/
│   └── test_core.py         ← Smoke tests for data and domain logic
├── requirements.txt
└── README.md
```

---

## Methodology

- Data: 43 products, 63 months (Jan 2018 – Mar 2023), 5 franchises (A–E)
- Preprocessing: IQR-based outlier detection, KNN imputation (k=5)
- Feature engineering: 17 demand signals across 4 categories
  (historical lags, seasonality encoding, event flags, product context)
- Models evaluated: Decision Tree, Random Forest, ExtraTrees, AdaBoost, XGBoost
- Validation: walk-forward cross-validation over 6-month horizons
- Metrics: MAE, RMSE, Bias

---

## Anonymization statement

This dashboard was built on a real industry study conducted at a major French
pharmaceutical company. To protect confidential information:

- The company name has been removed
- All product names and identifiers have been removed
- Therapeutic franchises are labeled Franchise A through Franchise E
- All data shown is **fully synthetic**, generated to match the statistical
  properties of the real dataset (same product count, franchise distribution,
  seasonality shape, COVID impact, and order-of-magnitude volumes)
- Baseline model KPIs (MAE, RMSE, Bias) have been slightly rounded/approximated;
  the magnitude and direction of all findings are fully preserved
- Approximated KPIs are labeled with "approx." throughout the app

---

## Sources

- ANSM (Agence nationale de sécurité du médicament) — annual shortage reports 2016–2023
- French Senate inquiry on drug shortages (2023)
- WHO Global medicine shortage report
- Company Annual Report 2022
- CentraleSupélec research framework
- Original study: internal data from anonymized pharmaceutical company
