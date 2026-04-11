"""
Shared synthetic data generators used across multiple pages.

All data is fully synthetic. Statistical properties (franchise distribution,
seasonality shape, COVID impact, coefficient-of-variation distribution) mirror
the real dataset without reproducing any actual figures.
"""

import numpy as np
import pandas as pd

np.random.seed(42)

# ─────────────────────────────────────────
# CONSTANTS
# ─────────────────────────────────────────

FRANCHISES = {
    "Cardiology":   {"products": 19, "color": "#185FA5", "share": 44.88},
    "Hematology":   {"products":  7, "color": "#1D9E75", "share": 15.63},
    "Solid Tumors": {"products":  7, "color": "#E24B4A", "share": 15.63},
    "Immunology":   {"products":  6, "color": "#BA7517", "share": 13.58},
    "Neuroscience": {"products":  4, "color": "#AFA9EC", "share": 10.28},
}

MONTHS = pd.date_range(start="2018-01-01", end="2023-03-01", freq="MS")
N_MONTHS = len(MONTHS)

COVID_START = "2020-01-01"
COVID_END   = "2021-07-01"

# Approximated KPIs — slightly modified from real values
BASELINE_MAE  = 30.0   # %
BASELINE_RMSE = 70.0   # %
BASELINE_BIAS = -10.0  # %

XGBOOST_MAE   = 28.0   # %  (~9% relative improvement on the ensemble metric)
XGBOOST_RMSE  = 63.0   # %
XGBOOST_BIAS  = -1.0   # %


# ─────────────────────────────────────────
# PRODUCT SALES GENERATOR
# ─────────────────────────────────────────

def generate_product_sales(
    base_volume: int,
    cv: float,
    franchise: str,
    has_covid_dip: bool = True,
    seed_offset: int = 0,
) -> np.ndarray:
    rng = np.random.default_rng(42 + seed_offset)
    t = np.arange(N_MONTHS)

    trend      = base_volume * (1 + 0.004 * t)
    phase      = {"Immunology": np.pi / 6, "Neuroscience": np.pi / 3}.get(franchise, 0)
    seasonality = 1 + 0.12 * np.sin(2 * np.pi * t / 12 + phase)

    covid_mask = np.ones(N_MONTHS)
    if has_covid_dip:
        covid_mask[24:43] = np.linspace(1.0, 0.72, 19)
        covid_mask[43:50] = np.linspace(0.72, 1.0, 7)

    noise = rng.normal(1, cv * 0.3, N_MONTHS)
    noise = np.clip(noise, 0.3, 2.5)

    sales = trend * seasonality * covid_mask * noise
    return np.maximum(sales, 0).astype(int)


# ─────────────────────────────────────────
# FULL DATASET
# ─────────────────────────────────────────

def build_dataset() -> pd.DataFrame:
    rng = np.random.default_rng(42)
    records = []
    product_id = 1
    for franchise, meta in FRANCHISES.items():
        for i in range(meta["products"]):
            base = int(rng.integers(500, 18_000))
            cv_choice = rng.choice(
                [rng.uniform(0.05, 0.45),
                 rng.uniform(0.50, 0.95),
                 rng.uniform(1.00, 1.60)],
                p=[0.77, 0.07, 0.16],
            )
            sales = generate_product_sales(base, cv_choice, franchise, seed_offset=product_id)
            for j, date in enumerate(MONTHS):
                records.append({
                    "product_id": f"P{product_id:03d}",
                    "franchise":  franchise,
                    "date":       date,
                    "sales":      sales[j],
                    "cv":         round(float(cv_choice) * 100, 1),
                })
            product_id += 1
    return pd.DataFrame(records)


# ─────────────────────────────────────────
# FEATURE IMPORTANCE (synthetic, page 4)
# ─────────────────────────────────────────

FEATURE_IMPORTANCE = pd.DataFrame([
    # Historical consumption patterns
    {"feature": "sales_lag_1",       "importance": 0.255, "category": "Historical", "label": "Last month's sales"},
    {"feature": "sales_lag_2",       "importance": 0.110, "category": "Historical", "label": "Sales 2 months ago"},
    {"feature": "sales_lag_3",       "importance": 0.055, "category": "Historical", "label": "Sales 3 months ago"},
    {"feature": "sales_lag_6",       "importance": 0.050, "category": "Historical", "label": "Sales 6 months ago"},
    {"feature": "sales_lag_12",      "importance": 0.075, "category": "Historical", "label": "Same month last year"},
    {"feature": "rolling_mean_3m",   "importance": 0.085, "category": "Historical", "label": "3-month moving average"},
    {"feature": "rolling_mean_6m",   "importance": 0.065, "category": "Historical", "label": "6-month moving average"},
    {"feature": "ema_3m",            "importance": 0.070, "category": "Historical", "label": "Short-term trend (EMA 3m)"},
    # Seasonality signals
    {"feature": "sin_month",         "importance": 0.050, "category": "Seasonality", "label": "Seasonal cycle — sine"},
    {"feature": "cos_month",         "importance": 0.038, "category": "Seasonality", "label": "Seasonal cycle — cosine"},
    {"feature": "quarter",           "importance": 0.020, "category": "Seasonality", "label": "Quarter of year"},
    # Exceptional demand events
    {"feature": "covid_flag",        "importance": 0.050, "category": "Events",     "label": "COVID-19 period flag"},
    {"feature": "stockout_flag",     "importance": 0.030, "category": "Events",     "label": "Historical stock-out"},
    {"feature": "patent_loss",       "importance": 0.018, "category": "Events",     "label": "Patent expiry event"},
    {"feature": "market_event",      "importance": 0.012, "category": "Events",     "label": "Market disruption"},
    # Product characteristics
    {"feature": "MITM",              "importance": 0.025, "category": "Product",    "label": "Product category (MITM)"},
    {"feature": "CDM",               "importance": 0.017, "category": "Product",    "label": "Distribution channel"},
]).sort_values("importance", ascending=False).reset_index(drop=True)

# Normalize to sum exactly to 1
FEATURE_IMPORTANCE["importance"] = (
    FEATURE_IMPORTANCE["importance"] / FEATURE_IMPORTANCE["importance"].sum()
)


# ─────────────────────────────────────────
# FRANCHISE-LEVEL RESULTS (page 6)
# ─────────────────────────────────────────

FRANCHISE_RESULTS = pd.DataFrame([
    {"franchise": "Cardiology",   "model": "Baseline", "MAE": 28, "RMSE": 65, "Bias": -8},
    {"franchise": "Cardiology",   "model": "XGBoost",  "MAE": 25, "RMSE": 58, "Bias": -1},
    {"franchise": "Hematology",   "model": "Baseline", "MAE": 32, "RMSE": 72, "Bias": -11},
    {"franchise": "Hematology",   "model": "XGBoost",  "MAE": 29, "RMSE": 65, "Bias": -2},
    {"franchise": "Solid Tumors", "model": "Baseline", "MAE": 38, "RMSE": 80, "Bias": -13},
    {"franchise": "Solid Tumors", "model": "XGBoost",  "MAE": 35, "RMSE": 76, "Bias": -4},
    {"franchise": "Immunology",   "model": "Baseline", "MAE": 30, "RMSE": 68, "Bias": -9},
    {"franchise": "Immunology",   "model": "XGBoost",  "MAE": 27, "RMSE": 62, "Bias": -2},
    {"franchise": "Neuroscience", "model": "Baseline", "MAE": 42, "RMSE": 88, "Bias": -15},
    {"franchise": "Neuroscience", "model": "XGBoost",  "MAE": 39, "RMSE": 83, "Bias": -4},
])

# ─────────────────────────────────────────
# BRAND-LEVEL RESULTS (page 6)
# ─────────────────────────────────────────

BRAND_RESULTS = pd.DataFrame([
    {"brand": "Brand-A", "franchise": "Cardiology",   "model": "Baseline", "MAE": 24, "RMSE": 60, "Bias": -6},
    {"brand": "Brand-A", "franchise": "Cardiology",   "model": "XGBoost",  "MAE": 21, "RMSE": 55, "Bias": -1},
    {"brand": "Brand-B", "franchise": "Cardiology",   "model": "Baseline", "MAE": 30, "RMSE": 68, "Bias": -9},
    {"brand": "Brand-B", "franchise": "Cardiology",   "model": "XGBoost",  "MAE": 27, "RMSE": 61, "Bias": -2},
    {"brand": "Brand-C", "franchise": "Hematology",   "model": "Baseline", "MAE": 35, "RMSE": 75, "Bias": -12},
    {"brand": "Brand-C", "franchise": "Hematology",   "model": "XGBoost",  "MAE": 31, "RMSE": 68, "Bias": -3},
    {"brand": "Brand-D", "franchise": "Hematology",   "model": "Baseline", "MAE": 29, "RMSE": 70, "Bias": -10},
    {"brand": "Brand-D", "franchise": "Hematology",   "model": "XGBoost",  "MAE": 26, "RMSE": 63, "Bias": -2},
    {"brand": "Brand-E", "franchise": "Solid Tumors", "model": "Baseline", "MAE": 40, "RMSE": 82, "Bias": -14},
    {"brand": "Brand-E", "franchise": "Solid Tumors", "model": "XGBoost",  "MAE": 36, "RMSE": 78, "Bias": -5},
    {"brand": "Brand-F", "franchise": "Solid Tumors", "model": "Baseline", "MAE": 36, "RMSE": 78, "Bias": -12},
    {"brand": "Brand-F", "franchise": "Solid Tumors", "model": "XGBoost",  "MAE": 34, "RMSE": 74, "Bias": -3},
    {"brand": "Brand-G", "franchise": "Immunology",   "model": "Baseline", "MAE": 28, "RMSE": 65, "Bias": -8},
    {"brand": "Brand-G", "franchise": "Immunology",   "model": "XGBoost",  "MAE": 25, "RMSE": 59, "Bias": -2},
    {"brand": "Brand-H", "franchise": "Immunology",   "model": "Baseline", "MAE": 33, "RMSE": 71, "Bias": -11},
    {"brand": "Brand-H", "franchise": "Immunology",   "model": "XGBoost",  "MAE": 30, "RMSE": 65, "Bias": -3},
    {"brand": "Brand-I", "franchise": "Neuroscience", "model": "Baseline", "MAE": 38, "RMSE": 82, "Bias": -13},
    {"brand": "Brand-I", "franchise": "Neuroscience", "model": "XGBoost",  "MAE": 35, "RMSE": 79, "Bias": -3},
    {"brand": "Brand-J", "franchise": "Neuroscience", "model": "Baseline", "MAE": 45, "RMSE": 93, "Bias": -17},
    {"brand": "Brand-J", "franchise": "Neuroscience", "model": "XGBoost",  "MAE": 43, "RMSE": 87, "Bias": -5},
])
