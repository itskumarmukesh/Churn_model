import numpy as np
import pandas as pd

from sklearn.metrics import average_precision_score, brier_score_loss, confusion_matrix, roc_auc_score
from sklearn.model_selection import StratifiedKFold, cross_val_predict, cross_validate

def make_cv(cfg):
    return StratifiedKFold(n_splits = cfg["cv_folds"], shuffle = True, random_state = cfg["random_state"])

def compare_models(models, X_train, y_train, cfg):
    """Cross-validate every candidate model on the training data only"""

    rows = []
    for name, model in models.items():
        s = cross_validate(model, X_train, y_train, cv = make_cv(cfg), scoring = ["roc_auc", "average_precision"])

        rows.append({
            "model": name, 
            "roc_auc": s["test_roc_auc"].mean(),
            "roc_auc_std": s["test_roc_auc"].std(),
            "pr_auc": s["test_average_precision"].mean()})

        print(" -done", name, flush = True)
    return pd.DataFrame(rows)

def pick_primary(results, margin = 0.005):
    """Models are listed from simplest to complex. Keep the simplest one, unless a more comples one beats the current best by atleast some margin"""
    best_name = results.loc[0, "model"]
    best_auc = results.loc[0, "roc_auc"]

    for i in range(1, len(results)):
        if results.loc[i, "roc_auc"] >= best_auc + margin:
            best_name = results.loc[i, "model"]
            best_auc = results.loc[i, "roc_auc"]
    return best_name

def out_of_fold_probs(model, X_train, y_train, cfg):
    """Every training customer is predicted by a model that never saw the customer"""
    return cross_val_predict(model, X_train, y_train, cv = make_cv(cfg), method = "predict_proba")[:, 1]

def profit_per_customer(y_true, flagged, business):
    """y_true: numpy array of 0/1. flagged: numpy array of True/False """
    hits = (flagged & (y_true == 1)).sum()
    gain = hits * business["success_rate"] * business["value_saved"]
    cost = flagged.sum() * business["cost_contact"]
    return (gain - cost) / len(y_true)

def choose_threshold(y_true, prob, business):
    """Try many threshold and keep the one wiht the highest profit. If not threshold make money , return 1.01"""
    rows = []
    for t in list(np.arange(0.05, 1.0, 0.05)) + [1.01]:
        flagged = prob >= t
        rows.append({
            "threshold": round(float(t), 2),
            "flagged_pct" : round(flagged.mean() * 100, 1),
            "profit_per_customer": round(profit_per_customer(y_true, flagged, business), 2)})

    table = pd.DataFrame(rows)
    best = table.loc[table["profit_per_customer"].idxmax()]
    if best["profit_per_customer"] <= 0:
        return 1.01, table
    return float(best["threshold"]), table

def evaluate_on_test(model, X_train, y_train, X_test, y_test, threshold, cfg):
    """The one and only look at the test set"""
    business = cfg["business"]
    model.fit(X_train, y_train)
    prob = model.predict_proba(X_test)[:, 1]
    y_arr = y_test.values
    n = len(y_arr)

    #boostrap
    rng = np.random.default_rng(cfg["random_state"])
    aucs = []

    for i in range(cfg["n_bootstrap"]):
        idx = rng.integers(0, n, n)
        aucs.append(roc_auc_score(y_arr[idx], prob[idx]))
    low, high = np.percentile(aucs, [2.5, 97.5])

    flagged = prob >= threshold
    tn, fp, fn, tp = confusion_matrix(y_arr, flagged.astype(int), labels = [0, 1]). ravel()
    precision = tp/ (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0

    #if we only contact the riskiest

    order = np.argsort(-prob)
    k = int(n * 0.2)
    caught = y_arr[order][:k].sum()

    return {
        "roc_auc": roc_auc_score(y_arr, prob),
        "roc_auc_low": low,
        "roc_auc_high": high,
        "pr_auc": average_precision_score(y_arr, prob),
        "brier": brier_score_loss(y_arr, prob),
        "threshold": threshold,
        "share_contacted_pct": flagged.mean() * 100,
        "precision": precision,
        "recall": recall,
        "tp": tp, "fp": fp, "fn": fn, "tn":tn,
        "profit_model": profit_per_customer(y_arr, flagged, business),
        "profit_everyone": profit_per_customer(y_arr, np.ones(n, dtype = bool), business),
        "top20_share_of_churners_pct": caught / y_arr.sum() * 100, 
        "top20_lift": (caught / k) / y_arr.mean(),
    }
    