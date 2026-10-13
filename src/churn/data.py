from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split


def load_and_prepare(cfg):
    """Read the raw CSV and apply the rules written in the config.
    Returns X (features), y (0/1 target) and ids (customer IDs, or None)."""
    df = pd.read_csv(Path(cfg["_root"]) / cfg["data_path"])
    target = cfg["target"]

    # rows without a target value cannot be used for learning
    n_missing_target = int(df[target].isnull().sum())
    if n_missing_target > 0:
        print("Dropped", n_missing_target, "row(s) with a missing target")
        df = df[df[target].notnull()].copy()

    y = (df[target] == cfg["positive_label"]).astype(int)

    ids = None
    if cfg["id_column"] is not None:
        ids = df[cfg["id_column"]]

    columns_to_drop = [target] + cfg["drop_columns"]
    if cfg["id_column"] is not None:
        columns_to_drop.append(cfg["id_column"])
    X = df.drop(columns=columns_to_drop)

    # columns stored as text that are really numbers (blank text becomes a missing value)
    for col in cfg["numeric_from_text"]:
        X[col] = pd.to_numeric(X[col], errors="coerce")

    # columns where 0 is impossible and really means "unknown"
    for col in cfg["zero_means_missing"]:
        X[col] = X[col].replace(0, np.nan)

    return X, y, ids


def check_data(X, y, ids=None):
    """Print a data-quality report and return a list of warnings."""
    found = []
    print("Rows:", X.shape[0], "| feature columns:", X.shape[1])
    print("Churn rate:", round(y.mean(), 4))

    if ids is not None and not ids.is_unique:
        found.append("The ID column has repeated values")

    print("Duplicate feature rows:", int(X.duplicated().sum()))

    # missing values
    missing = X.isnull().mean()
    missing = missing[missing > 0].sort_values(ascending=False)
    print("Columns with missing values:", len(missing))
    for col, share in missing.head(3).items():
        print("   ", col, "->", round(share * 100, 1), "% missing")
    for col, share in missing.items():
        if share > 0.4:
            found.append(f"Column '{col}' is {share * 100:.0f}% missing")

    num_cols = X.select_dtypes(include="number").columns
    cat_cols = X.select_dtypes(exclude="number").columns

    # many categories
    for col in cat_cols:
        if X[col].nunique() > 50:
            print("Column", col, "has", X[col].nunique(), "categories (rare ones will be grouped)")

    # leak detector 1: can one numeric column alone almost perfectly predict the target?
    best_name, best_auc = None, 0.5
    for col in num_cols:
        x = X[col].fillna(X[col].median())
        if x.nunique() > 1:
            auc = roc_auc_score(y, x)
            auc = max(auc, 1 - auc)
            if auc > best_auc:
                best_name, best_auc = col, auc
            if auc > 0.9:
                found.append(f"Column '{col}' alone gives AUC {auc:.3f}: possible leakage or synthetic data")
    print("Strongest single numeric column:", best_name, "(AUC", round(best_auc, 3), ")")

    # leak detector 2: does one category always (or never) churn?
    for col in cat_cols:
        stats = y.groupby(X[col]).agg(["mean", "size"])
        for value, row in stats.iterrows():
            big_enough = row["size"] >= 0.01 * len(y)
            if big_enough and (row["mean"] >= 0.999 or row["mean"] <= 0.001):
                found.append(f"{col} = {value!r} has a churn rate of {row['mean'] * 100:.1f}% "
                             f"on {int(row['size'])} customers: a perfect rule, possible leakage or synthetic data")

    if found:
        print("WARNINGS:")
        for message in found:
            print("   -", message)
    else:
        print("No warnings.")
    return found


def make_split(X, y, cfg):
    """Stratified train/test split using the settings in the config."""
    return train_test_split(X, y, test_size=cfg["test_size"], stratify=y,
                            random_state=cfg["random_state"])