# Evaluation Report: Customer Churn Intelligence (IBM Telco)

## Setup
- Data: TBM Telco sample dataset, 7043 customers, 26.5% churn
- Split: 80/20 stratified (5634 train / 1409 test; the test set holds 374 churners)
- The test set was used once, after all modelling decisions were made
- Validation: 5 fold stratified cross - validation on training data only
- scaling is fitted inside each fold to avoid leakage

## Model Selection (training data only)
- Logistic Regression 0.846
- Random FOrest 0.846
- Gradient Boosting 0.843
- Decision Tree 0.825

- Baseline ("nobody churns"): accuracy 0.735, recall 0.
- Tuning: Logistic Regression and Random Forest gained nothing. Gradient Boosting improved 0.842 to 0.848 (better on 10 of 10 folds).
- Tuned Gradient Boosting vs Logistic Regression: +0.002 AUC (better on 8 of 10 folds, gap std 0.0045), so statistically a tie.
- Decision rule fixed before the test set was opened: Logistic Regression is the primary model (simpler, explainable); tuned Gradient Boosting is the challenger.
- class_weight="balanced" was dropped: identical AUC, but it inflated probabilities (average prediction 0.413 vs real churn 0.265; Brier 0.165 vs 0.135 without it).
- Feature engineering: 4 engineered features and 7 further ideas gave no measurable gain.

## Test Results
- Logistic Regression (primary) | 0.841 (0.820 to 0.852) | 0.629 | 0.138
- Tuned Gradient Boosting (challenger) | 0.844 (0.822 to 0.865) | 0.656 | 0.136

- Random guessing gives PR_AUC of about 0.265
- ROC_AUC difference (challenger - primary): 95% range -0.004 to 0.009, which includes 0
- The challengers higher PR_AUC was observed after the test set was opened, so it did not change the pre committed model choice

# Business Decision
- Contact a customer when probability > cost/(success rate * value) = 0.50
- The threshold of 0.5 was chosen using out of fold predictions on training data
- On the test set at threshold 0.50: 21.6% of customers contacted, precision 0.658, recall 0.535 (TP 200, FP 104, FN 174, TN 931)
- Profit per customer: 6.8 dollar (model-targeted) vs -46.9 dollar (contact everyone) vs 0 dollar (contact nobody)
- Sensitivity: at a 15% success rate the best threshold is 0.80 and profit is about 0.2 dollar at 40% it is 0.35 and profit is 27.6 dollar

## Limitations
- Blind to long-contract churners: recall on 1-2 year contracts is 2.2% (1 of 45 caught). They churn at only 7.1%, so they are not economically worth contacting at threshold 0.5, but the model also has no signal to find them. Hypothesis: they leave at renewal, the data has no contract end dates.
- Weak after the first year: recall 32% (108 of the 174 missed churners). Early-life churn (first year) is caught well: recall 69%.
- One snapshot of sample data from a fictional company; no dates, so no time-based validation.
- The test set has 374 churners, so scores carry about +/-0.02 AUC of uncertainty. The challenger's higher test PR-AUC (+0.008 to +0.048 in a bootstrap) did not replicate in cross-validation on training data (0.003/0.004, no consistent winner), so it is treated as test-split luck and the pre-committed model was kept.
- Business figures are assumptions; real value depends on offer cost and success rate.
- The model finds who is likely to leave, not what would keep them; this needs an A/B test.
- Model weights on correlated columns (tenure, MonthlyCharges, TotalCharges) cannot be read individually.
- Gender has no effect on the score (ROC-AUC 0.8466 vs 0.8467 without it) and is excluded by default in the reusable pipeline.