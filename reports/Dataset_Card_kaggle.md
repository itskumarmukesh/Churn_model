
#Dataset : Telecom Customer Churn (Kaggle)
source: https://www.kaggle.com/datasets/jpacse/datasets-for-churn-telecom
Download date: 5 OCT 2026
Lincense : Unknown

#Data Overview
Rows : 50000, Columns : 10
Telecom customer churn dataset.

#Target Variable
'Churn'
Class balance: 56.7% churned in the full file.

#Key features
Contract Length, Support Calls, and other customer attributes included in the dataset.

#Drawbacks
Monthly contracts have a 100% churn rate for 10,001 customers in the inspected sample.
Initial model scores near 0.99 raise concerns about data leakage or synthetic patterns.
A blank row exists at the end of the source file.
Dataset identity and license require verification against the original Kaggle page

#Use Case:
Stress test only. Test data validation, leakage detection, and churn-modeling pipeline safeguards. Do not use as a reliable benchmark for real-world churn prediction.