import json
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import sklearn

from churn.config import load_config
from churn.data import load_and_prepare
from churn.data import make_split
from churn.model import build_models


def export(config_path):
    cfg = load_config(config_path)
    root = Path(cfg["_root"])

    # which model won
    results_file = root / "reports" / ("results_" + cfg["name"] + ".json")
    model_name, threshold = "Logistic Regression", None
    if results_file.exists():
        with open(results_file, "r", encoding="utf-8") as f:
            summary = json.load(f)
        model_name = summary["primary_model"]
        threshold = summary["threshold"]
    print("Exporting model:", model_name)

    X, y, ids = load_and_prepare(cfg)
    X_train, X_test, y_train, y_test = make_split(X, y, cfg)
    model = build_models(X_train, cfg["random_state"])[model_name]
    model.fit(X_train, y_train)

    # what the app needs to know
    numeric = {}
    for col in X_train.select_dtypes(include="number").columns:
        numeric[col] = {"min": float(X_train[col].min()),
                        "max": float(X_train[col].max()),
                        "default": float(X_train[col].median())}
    categorical = {}
    for col in X_train.select_dtypes(exclude="number").columns:
        counts = X_train[col].value_counts()
        categorical[col] = {"options": sorted(counts.index.tolist()),
                            "default": counts.index[0]}

    # average of every transformed feature: app explain a score "versus an average customer"
    feature_means = model[:-1].transform(X_train).mean(axis=0)

    meta = {
        "dataset": cfg["name"],
        "model_name": model_name,
        "explainable": bool(hasattr(model[-1], "coef_")),
        "trained_on_rows": int(len(X_train)),
        "baseline_churn_rate": float(y_train.mean()),
        "threshold": threshold,
        "business": cfg["business"],
        "feature_columns": X_train.columns.tolist(),
        "numeric": numeric,
        "categorical": categorical,
        "feature_means": [float(v) for v in feature_means],
        "sklearn_version": sklearn.__version__,
    }

    models_dir = root / "models"
    models_dir.mkdir(exist_ok=True)
    joblib.dump(model, models_dir / (cfg["name"] + "_model.joblib"), compress=3)
    with open(models_dir / (cfg["name"] + "_meta.json"), "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)

    app_dir = root / "app"
    app_dir.mkdir(exist_ok=True)
        lines = ["streamlit",
             "sqlalchemy>=2.0",
             "psycopg2-binary",
             "pandas==" + pd.__version__,
             "numpy==" + np.__version__,
             "scikit-learn==" + sklearn.__version__]
    (app_dir / "requirements.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")

    print("Saved model, metadata and app/requirements.txt")
    print("Python:", sys.version.split()[0], "| scikit-learn:", sklearn.__version__)

if __name__ == "__main__":
    export(sys.argv[1])