"""Smoke tests for core data and domain logic.

These tests verify invariants of the synthetic dataset and the safety-stock
formula without requiring the Streamlit runtime.
"""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

# pylint: disable=wrong-import-position
from utils.domain import safety_stock  # noqa: E402
from utils.synthetic_data import FEATURE_IMPORTANCE, build_dataset  # noqa: E402


def test_dataset_shape() -> None:
    """Dataset must have 43 products x 63 months and required columns."""
    df = build_dataset()
    assert len(df) == 43 * 63, f"Expected {43 * 63} rows, got {len(df)}"
    assert set(df.columns) >= {"product_id", "franchise", "date", "sales", "cv"}


def test_no_negative_sales() -> None:
    """Synthetic sales should never be negative."""
    df = build_dataset()
    assert df["sales"].ge(0).all(), "Found negative sales values"


def test_feature_importance_sums_to_one() -> None:
    """Feature importance values must be normalized to sum to exactly 1."""
    total = FEATURE_IMPORTANCE["importance"].sum()
    assert abs(total - 1.0) < 1e-9, f"Feature importance sums to {total}, expected 1.0"


def test_safety_stock_positive() -> None:
    """Safety stock must be positive for realistic inputs."""
    result = safety_stock(28.0, 2100, 2, 1.645)
    assert result > 0, f"Safety stock should be positive, got {result}"


def test_safety_stock_decreases_with_lower_mae() -> None:
    """Lower MAE must yield lower safety stock at the same service level."""
    ss_high = safety_stock(30.0, 2100, 2, 1.645)
    ss_low = safety_stock(28.0, 2100, 2, 1.645)
    assert ss_low < ss_high, "Lower MAE should produce lower safety stock"
