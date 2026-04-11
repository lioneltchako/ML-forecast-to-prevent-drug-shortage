"""
Domain logic shared across pages.
"""

import numpy as np


def safety_stock(mae_pct: float, avg_demand: float, lead_time: float, service_z: float) -> float:
    """
    Safety stock formula: SS = Z × σ_error × √(lead_time)
    σ_error ≈ MAE × 1.25  (normal distribution approximation)
    Ref: Hopp & Spearman, Factory Physics, 2001
    """
    sigma = (mae_pct / 100) * avg_demand * 1.25
    return service_z * sigma * np.sqrt(lead_time)
