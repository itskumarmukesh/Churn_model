# Customer Churn Intelligence
A reusable, config-driven pipeline that predicts which telecom customers are likely to leave and turns the predictions into a retention decision. Built with strict evaluation discipline, a locked test set, no data leakage, honest probabilities, automatic data-quality gates and stated limitations.

## Results
Same code, three datasets (test set, opened once per run):

| Dataset | Churn rate | Model chosen automatically | Test ROC-AUC (95% range) | PR-AUC above random* | Status |
|---|---|---|---|---|---|
| IBM Telco (7,043 customers) | 26.5% | Logistic Regression | 0.842 (0.819 to 0.863) | 0.50 | ok |
| Cell2Cell (51,047 customers) | 28.8% | Boosting (deep) | 0.674 (0.663 to 0.685) | 0.23 | ok |
| Kaggle synthetic (50k sample) | 56.9% | Boosting (shallow) | 0.997 | 1.00 | stopped / flagged |

*(PR-AUC - churn rate) / (1 - churn rate): 0 is random, 1 is perfect. Comparable across datasets, unlike raw AUC or lift.

**Telco business view**, under illustrative assumptions (a saved customer is worth $800, an offer costs $100, 25% of contacted churners are retained):
- Threshold 0.55, chosen on training data only: contacts 18.2% of customers, precision 0.68, recall 0.46.
- The riskiest 20% of customers contain 50.0% of all churners (lift 2.51).
- Profit per customer: +$6.4 (model-targeted) vs -$46.9 (contact everyone).

Cell2Cell has much less signal in its columns: the same assumptions give only +$1.3 per customer, and the report says so rather than hiding it.

## What makes this project rigorous
- **Locked test set,** opened once per run. Model and threshold are chosen on training data first.
- **No leakage:** imputing, scaling and encoding are learned inside cross-validation folds.
- **Automatic model choice:** candidates are listed from simplest to most complex; a more complex model wins only if it beats the current best by at least 0.005 ROC-AUC.
- **Two safety nets:** (1) the pipeline stops if one column or category gives the answer away (e.g. a category that always churns); (2) it flags any test ROC-AUC above 0.97. Both fired on the synthetic Kaggle file, and removing the obvious shortcut did not make it realistic, which is why both nets exist.
- **Honest probabilities:** class weighting was dropped because it inflated predicted churn (average 0.41 vs real 0.27).
- **Evidence-based feature work:** seven engineered features and hyper-parameter tuning gave no meaningful gain on Telco; scores plateaued near 0.85 ROC-AUC across model families.
- **Failure analysis:** on Telco the model is nearly blind to churners on 1-2 year contracts (recall 2%) and weak after the first year (recall 32%).
- **Tests:** 9 automated tests cover the model-selection rule, the profit calculation and the data checks.

## Business findings (Telco)
Month-to-month contracts, the first year of service, fiber optic service and electronic-check payment carry the highest churn; long contracts, tech support and online security the lowest.
These are associations, not proven causes. See `reports/Business Recommendation.md`.

## Ouickstart
pip install -r requirements.txt
cd src
python -m churn.run ..\configs\telco.yaml # or ../configs/telco.yaml on macOS/Linux
python -m churn.run ..\configs\cell2cell.yaml
python -m churn.compare .. # builds reports/CROSS_DATASET_COMPARISON.md
cd ..
python -m pytest tests -q

## Repository layout
- `src/churn/`: `config.py`, `data.py` (loading and data checks), `model.py`, `evaluate.py`, `run.py` (end-to-end), `compare.py` (cross-dataset table)
- `configs/`: one YAML per dataset
- `tests/`: pytest tests
- `notebooks/`: the step-by-step analysis (cleaning, EDA, features, models, evaluation, Cell2Cell)
- `sql/`: churn analysis queries (MySQL), cross-checked against pandas
- `reports/`: dataset cards, cleaning logs, evaluation report, recommendations, results
- `data/`: not committed; see the links below


## Data
- IBM Telco Customer Churn: https://github.com/IBM/telco-customer-churn-on-icp4d (IBM sample data, license: Apache License 2.0)
- Cell2Cell (Teradata Center for CRM, Duke University), via Kaggle: https://www.kaggle.com/datasets/jpacse/datasets-for-churn-telecom (license: Unknown)
- Kaggle customer churn file (440k rows): behaves like synthetic data; used only as a stress test of the data checks.

## Limitations
- Telco and Cell2Cell are single snapshots with no dates, so there is no time-based validation.
- The Telco test set was scored twice (the Step 7 notebook and the refactored pipeline), no decision changed after seeing either result.
- Telco's test set has 374 churners, so scores carry about +/-0.02 AUC of uncertainty.
- Business figures are assumptions; the real value depends on offer cost and success rate.
- The models find who is likely to leave, not what would keep them; that needs an A/B test.
- Cell2Cell's retention-team columns are excluded by default because they may be caused by the customer already wanting to leave.
See `reports/Evaluation_Report.md`.