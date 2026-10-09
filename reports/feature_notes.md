 Feature notes -- IBM Telco churn
# New feature - built only from input columns, never from churn
# is_first_year: 1 if tenure <= 12 months. EDA : 47.4% churn in the first year
# monthly_echeck : 1 if month to month contract AND electronic check 53.7 % churn
# lives_alone : 1 if no partner AND no dependents 34.2 % churn vs 19.8 %

## Encoding:
# gender, InternetService, Contract, PaymentMethod turned into 0/1 (one-hot, first category dropped).

## Split and Scaling:
# 80% train / 20% test, random_state 42, Test set stays locked until the final evaluation
# tenure, MonthlyCharges, TotalCharges, num_addons scaled with StandardScaler fitted on the training set only

## customerID and Churn are not in X
# No customer appears in both train and test
# The scaler learned only from training data
# Limitation : the effect is small for simple flags, but next time explore the training set only
