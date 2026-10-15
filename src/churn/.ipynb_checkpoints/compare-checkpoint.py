import json
import sys
from pathlib import Path

def load_results(reports_dir):
    """Read every results_*.json file saved by run.py"""
    results = []

    for path in sorted(Path(reports_dir).glob("results_*.json")):
        with open(path, "r", encoding = "utf-8") as f:
            results.append(json.load(f))
    return results

def pr_skill(pr_auc, churn_rate):
        """How are PR_AUC is above random guessing: 0 -> random, 1 -> perfect. Unlike lift (capped by 1 / churn rate) it does not depend on how common churn is"""
        return (pr_auc - churn_rate) / (1 - churn_rate)

def build_rows(results):
        rows = []
        for r in results:
            t = r["test"]
            cv_auc = 0.0
            for m in r["model_comparison"]:
                if m["model"] == r["primary_model"]:
                    cv_auc = m["roc_auc"]
            
            suspicious = r.get("too_good_to_be_true", False)
            if r["threshold"] > 1:
                profit = "campaign not profitable"
            else:
                profit = f"{t['profit_model']:+.1f} vs {t['profit_everyone']:+.1f}"

            rows.append({
                "suspicious": suspicious,
                "cells": [
                    r["dataset"],
                    f"{r['rows']:,}",
                    f"{r['churn_rate'] * 100:.1f}%",
                    r["primary_model"],
                    f"{cv_auc:.3f}",
                    f"{t['roc_auc']:.3f} ({t['roc_auc_low']:.3f} to {t['roc_auc_high']:.3f})",
                    f"{pr_skill(t['pr_auc'], r['churn_rate']):.2f}",
                    f"{t['top20_share_of_churners_pct']:.1f}% (lift {t['top20_lift']:.2f})",
                    profit,
                    "Do Not Trust: too good to be true" if suspicious else "ok",
                ],
            })
        #true datasets first, then alphabetical
        rows.sort(key = lambda row: (row["suspicious"], row["cells"][0]))
        return rows

HEADER = ["Dataset", "Rows used", "Churn rate", "Primary model", "CV ROC-AUC", "Test ROC-AUC (95% range)", "PR-AUC above random", "Riskiest 20% reaches", "Profit per customer: model vs contact everyone", "Trust"]

def to_markdown(rows):
    lines = ["# Cross Dataset Comparison","", "| " + " | ".join(HEADER) + " |", "|" + " --|" * len(HEADER)]
    for row in rows:
        lines.append("| " + " | ".join(row["cells"]) + " | ")
    lines += ["", "How to read this table", "-- PR-AUC above random = ( PR-AUC - churn rate) / (1 - churn rate). 0 means random, 1 means perfect. It can be compared across datasets; raw AUC, and lift, cannot",
            " -- Datasets marked DO NOT TRUST look synthetic or leaky. They test the pipeline, not model quality",
            " - The business columns use the same illustratice assumptions for every dataset",
            ""]

    return "\n".join(lines)

def main(root):
    reports_dir = Path(root) / "reports"
    rows = build_rows(load_results(reports_dir))
    markdown = to_markdown(rows)
    print(markdown)
    out_file = reports_dir / "Cross_Dataset_Comparison.md"
    out_file.write_text(markdown, encoding = "utf-8")
    print(out_file, "Saving Done")

if __name__ == "__main__":
    main(sys.argv[1])