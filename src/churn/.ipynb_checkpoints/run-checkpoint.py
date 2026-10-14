import json
import sys
from pathlib import Path

from churn.config import load_config
from churn.data import check_data, load_and_prepare, make_split
from churn.evaluate import (choose_threshold, compare_models, evaluate_on_test, out_of_fold_probs, pick_primary)
from churn.model import build_models

def run(config_path):

    cfg = load_config(config_path)
    print("Dataset", cfg["name"])

    print("1. Data Check")
    X, y, ids = load_and_prepare(cfg)
    problems = check_data(X, y, ids)
    X_train, X_test, y_train, y_test = make_split(X, y, cfg)
    print("Train", X_train.shape, "| Test", X_test.shape, "(test set stays locked until step 4)")

    print("2. Model Comparison (cross-validation, training data only)")
    models = build_models(X_train, cfg["random_state"])
    results = compare_models(models, X_train, y_train, cfg)
    print(results.round(4).to_string(index = False))
    primary_name = pick_primary(results)
    print("Primary model", primary_name)

    print("3. Threshold (chosen on training data only)")
    b = cfg["business"]
    print("Business assumpotion: value saved", b["value_saved"], "| offer cost", b["cost_contact"], "| success rate", b["success_rate"])

    oof = out_of_fold_probs(models[primary_name], X_train, y_train, cfg)
    threshold, table = choose_threshold(y_train.values, oof, b)

    if threshold > 1:
        print("No threshold make a profit under these assumptions: the campaign is not worth running")
    else:
        print("Chosen threshold", threshold)

    print("4. Test-Set (opened Once)")
    t = evaluate_on_test(models[primary_name], X_train, y_train, X_test, y_test, threshold, cfg)
    print(f"ROC-AUC {t['roc_auc']:.4f} (95% range {t['roc_auc_low']:.4f} to {t['roc_auc_high']:.4f})")
    print(f"PR-AUC {t['pr_auc']:.4f} (random guessing: {y_test.mean():.3f}) | Brier {t['brier']:.4f}")
    print(f"Rishkiest 20 % of customer: reaches {t['top20_share_of_churners_pct']:.1f}% of churners"f"(lift {t['top20_lift']:.2f})")

    if threshold <= 1:
        print(f"At threshold {threshold}: contacts {t['share_contacted_pct']:.1f}% of customers | precision {t['precision']:.3f} | recall {t['recall']:.3f}")
        print(f"Profit per customer: model-based {t['profit_model']:.1f} | contact everyone {t['profit_everyone']:.1f} | contact nobody 0")

    summary = {
        "dataset": cfg["name"],
        "rows": len(y), 
        "churn_rate": float(y.mean()),
        "data_warnings": problems,
        "model_comparison": results.to_dict("records"),
        "primary_model": primary_name,
        "threshold": threshold,
        "business_assuptions": b,
        "test": t
        }
    
    report_dir = Path(cfg["_root"]) / "reports"
    report_dir.mkdir(exist_ok = True)
    out_file = report_dir / ("results_" + cfg["name"] + ".json")

    with open(out_file, "w") as f:
        json.dump(summary, f, indent = 2, default = lambda o: o.item())
    print("Saved", out_file)
    return summary

if __name__ == "__main__":
    run(sys.argv[1])
        