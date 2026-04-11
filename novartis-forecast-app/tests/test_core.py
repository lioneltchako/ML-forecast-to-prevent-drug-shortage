import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))


def test_dataset_shape():
    from utils.synthetic_data import build_dataset
    df = build_dataset()
    assert len(df) == 43 * 63, f"Expected {43 * 63} rows, got {len(df)}"
    assert set(df.columns) >= {"product_id", "franchise", "date", "sales", "cv"}


def test_no_negative_sales():
    from utils.synthetic_data import build_dataset
    df = build_dataset()
    assert df["sales"].ge(0).all(), "Found negative sales values"


def test_feature_importance_sums_to_one():
    from utils.synthetic_data import FEATURE_IMPORTANCE
    total = FEATURE_IMPORTANCE["importance"].sum()
    assert abs(total - 1.0) < 1e-9, f"Feature importance sums to {total}, expected 1.0"


def test_safety_stock_positive():
    from utils.domain import safety_stock
    result = safety_stock(28.0, 2100, 2, 1.645)
    assert result > 0, f"Safety stock should be positive, got {result}"


def test_safety_stock_decreases_with_lower_mae():
    from utils.domain import safety_stock
    ss_high = safety_stock(30.0, 2100, 2, 1.645)
    ss_low  = safety_stock(28.0, 2100, 2, 1.645)
    assert ss_low < ss_high, "Lower MAE should produce lower safety stock"
