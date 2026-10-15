import numpy as np
import pandas as pd

from churn.data import check_data
from churn.data import load_and_prepare

def test_check_data_flags_a_category_that_always_churn():
    rng = np.random.default_rng(0)
    n = 1000
    contract = rng.choice(["Monthly", "Annual"], size = n)
    y = pd.Series(np.where(contract == "Monthly", 1, rng.integers(0, 2, size = n)))
    X = pd.DataFrame({"Contract": contract})
    warnings_found = check_data(X, y)
    assert any("perfect rule" in w for w in warnings_found)

def test_check_data_is_quiet_on_clean_random_data():
    rng = np.random.default_rng(1)
    n = 1000
    X = pd.DataFrame({"usage": rng.normal(size = n), "plan": rng.choice(["A", "B", "C"], size = n)})
    y = pd.Series(rng.integers(0, 2, size = n))
    warnings_found = check_data(X, y)
    assert not any("possible leakage" in w for w in warnings_found)

def test_load_and_prepare_applies_the_config_rules(tmp_path):
    csv = tmp_path / "data" / "tiny.csv"
    csv.parent.mkdir()
    pd.DataFrame({
        "id": [1, 2, 3, 4],
        "total": ["10.5", " ", "30", "40"],
        "age": [0, 30, 40, 50],
        "label": ["Yes", "No", "No", "Yes"],
    }).to_csv(csv, index = False)

    cfg = {"_root": str(tmp_path),
           "data_path": "data/tiny.csv",
           "target": "label",
           "positive_label": "Yes",
           "id_column": "id",
           "drop_columns": [],
           "numeric_from_text": ["total"],
           "zero_means_missing": ["age"],
           "sample_rows": None}
    X, y, ids = load_and_prepare(cfg)

    assert list(y) == [1, 0, 0, 1]
    assert "id" not in X.columns and "label" not in X.columns
    assert X["total"].dtype == "float64" and X["total"].isnull().sum() == 1
    assert X["age"].isnull().sum() == 1
    assert list(ids) == [1, 2, 3, 4]