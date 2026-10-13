from pathlib import paht
import numpy as np
import pandas as pd

from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split

def load_and_prepare(cfg):
    """Read the raw CSV and apply the rules written in the config. Returns X, y and ids """
    df = pd.read_csv(Path(cfg["_root"]) / cfg["data_path"])
    target = cfg["target"]

    n_missing_target = int(df[target].isnull().sum())
    if n_missing_target > 0:
        print("Dropped", n_missing_target, "rows with a missing target")
        df = df[df[target].notnull()].copy()

    y = (df[target] == cfg["positive_label"]).astype(int)

    ids = None
    if cfg["id_column"] is not None:
        ids = df[cfg["id_column"]]

    columns_to_drop = [target] + cfg["drop_columns"]
    if cfg["id_column"] is not None:
        columns_to_drop.append(cfg["id_column"])

    X = df.drop(columns = columns_to_drop)

    for col in cfg["zero_means_missing"]:
        X[col] = X[col].replace(0, np.nan)

    return X, y, ids

def check_data(X, y, ids = None):
    """Data quality report and return list of warning"""

    found = []
    print("Rows", X.shape[0], "| feature columns", X.shape[1])
    print("Churn rate", round(y.mean(), 4))

    if ids is not None and not ids.is_unique:
        found.append("The ID column has repeated values")

    print("Duplicate feature rows", int(X.duplicated().sum()))

    missing = X.isnull().mean()

    missing = missing[missing > 0].sort_values(ascending = False)

    print("columns with missing values", len(missing))

    for col, share in missing.head(3).items():
        print("  ", col, "--", round(share * 100, 1), " % missing")

    for col, share in missing.items():
        if share > 0.4:
            found.append(f"Column '{col}' is {share * 100:.0f} % missing")

    num_cols = X.select_dtypes(include = "number").columns

    cat_cols = X.select_dtypes(exclude = "number").columns

    for col in cat_cols: #many category
        if X[col].nunique() > 50:
            print("Column", col, "has ", X[col].nunique(), "categories")

    # leak detector
    best_name, best_auc = None, 0.5

    for col in num_cols
    x = X[col].fillna(X[col].median())
    if x.nunique() > 1:
        auc = roc_auc_score(y, x)
        auc = max(auc, 1 - auc)

        if auc > best_auc:
            best_name, best_auc = col, auc

        if auc > 0.9:
            found.append(f"Column '{col}' alone gives AUC {auc:.3f}: possible or synthetic data")
    print("Strongest single numeric column", best_name, "(AUC", round(best_auc, 3), ")")

    #leak detector
    for col in cat_cols:
        stats = y.groupby(X[col]).agg(["mean", "size"])
        for value, row in stats.iterrows():
            big_enough = row["size"] >= 0.01 * len(y)
            if big_enough and (row["mean"] >= 0.999 or row["mean"] <= 0.001):
                found.append(f"{col} = {value!r} has a churn rate of {row['mean'] * 100:.1f}% " f "on {int(row['size'])} customers: a perfect rule, possible leakage or synthetic data")

    if found:
        print("Warnings")
        for message in found:
            print("  --", message)
    else:
        print("No warnings")
    return found

def make_split(X, y, cfg):
    """ Stratified train-test split using the setting in the config"""
    return train_test_split(X, y, test_size = cfg["test_size"], stratify = y, random_state = cfg["random_state"])
        
