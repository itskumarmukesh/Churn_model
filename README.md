# Customer Churn Intelligence
Predict which telecom customers are likely to leave, and turn those predictions into a retention decision a business can act on. The project follows a strict evaluation discipline: a locked test set, no data leakage, honest probabilities and stated limitations.

## Results (IBM Telco, 7,043 customers, 26.5% churn)
Single test-set evaluation | Logistic Regression
ROC-AUC | 0.841 (95% range 0.820 to 0.862)
PR-AUC | 0.629 (random guessing: 0.265)
Brier score | 0.138 (always predicting the average: 0.195)

Business view, under illustrative assumptions (a saved customer is worth $800, an offer costs $100, 25% of contacted churners are retained):
- Contacting the riskiest 20% of customers reaches 48.7% of all churners (lift 2.44).
- The threshold (0.50, the break-even probability) was chosen on training data only. On the test set it contacts 21.6% of customers with precision 0.66 and recall 0.54.
- Profit per customer: +$6.8 (model-targeted) vs -$46.9 (contact everyone).

(reports/figures/step7_test_curves.png)

## What makes this project rigorous:
- **Locked test set**, opened once. Model choice and threshold were fixed on training data first.
- **No leakage:** scaling and encoding are learned inside cross-validation folds.
- **Honest comparisons:** models and features were compared fold by fold on identical folds. Seven engineered features and hyper-parameter tuning gave no meaningful gain over the raw columns; scores plateaued near 0.85 ROC-AUC across model families and with more rows.
- **Honest probabilities:** class weighting was dropped because it inflated the average predicted churn to 0.41 when the real rate is 0.27.
- **Failure analysis:** the model is nearly blind to churners on 1-2 year contracts (recall 2%), and weak after the first year (recall 32%).

## Business findings
Month-to-month contracts, the first year of service, fiber optic service and electronic-check payment carry the highest churn; long contracts, tech support and online security the lowest.
These are associations, not proven causes. See `reports/Business Recommendation.md`.

## Generalising to other datasets (in progress)
The pipeline is being turned into a config. driven package so the same code runs on different churn datasets.
- Cell2Cell (51,047 customers, 56 features): cross-validated ROC-AUC 0.676 with gradient boosting vs 0.616 with logistic regression. Unlike Telco, a non-linear model wins clearly.
- A synthetic Kaggle churn file (440k rows) is used as a stress test: a single column separates it perfectly, and the pipeline should warn about it.

## Repository layout
- `notebooks/`: 02 cleaning, 03 EDA, 05 features, 06 models, 07 evaluation, 09 Cell2Cell
- `sql/`: churn analysis queries (MySQL), cross-checked against pandas
- `reports/`: dataset cards, cleaning log, EDA findings, evaluation report, recommendations
- `src/churn/`, `configs/`: reusable pipeline (in progress)
- `data/`: not committed; see the download links below

## Data
- IBM Telco Customer Churn: https://github.com/IBM/telco-customer-churn-on-icp4d (IBM sample data, license: Apache License 2.0)
- Cell2Cell (Teradata Center for CRM, Duke University), via Kaggle: https://www.kaggle.com/datasets/jpacse/datasets-for-churn-telecom (license: Unknown)

## Limitations
One snapshot of sample data with no dates, so no time-based validation. The test set has 374 churners (about +/-0.02 AUC of uncertainty). Business figures are assumptions. The model predicts who is likely to leave, not what would keep them; that needs an A/B test.
See `reports/EVALUATION_REPORT.md`.
