# Cross Dataset Comparison

| Dataset | Rows used | Churn rate | Primary model | CV ROC-AUC | Test ROC-AUC (95% range) | PR-AUC above random | Riskiest 20% reaches | Profit per customer: model vs contact everyone | Trust |
| --| --| --| --| --| --| --| --| --| --|
| cell2cell | 51,047 | 28.8% | Boosting (deep) | 0.677 | 0.674 (0.663 to 0.685) | 0.23 | 33.3% (lift 1.67) | +1.3 vs -42.4 | ok | 
| telco | 7,043 | 26.5% | Logistic Regression | 0.846 | 0.842 (0.819 to 0.863) | 0.50 | 50.0% (lift 2.51) | +6.4 vs -46.9 | ok | 
| kaggle_forced | 50,000 | 56.9% | Boosting (shallow) | 0.997 | 0.997 (0.996 to 0.998) | 1.00 | 35.1% (lift 1.76) | +56.0 vs +13.8 | Do Not Trust: too good to be true | 
| kaggle_nocontract | 50,000 | 56.9% | Boosting (shallow) | 0.987 | 0.986 (0.983 to 0.988) | 0.98 | 35.1% (lift 1.76) | +54.8 vs +13.8 | Do Not Trust: too good to be true | 

How to read this table
-- PR-AUC above random = ( PR-AUC - churn rate) / (1 - churn rate). 0 means random, 1 means perfect. It can be compared across datasets; raw AUC, and lift, cannot
 -- Datasets marked DO NOT TRUST look synthetic or leaky. They test the pipeline, not model quality
 - The business columns use the same illustratice assumptions for every dataset
