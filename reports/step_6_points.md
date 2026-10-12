

## Primary model : Logistic regression (plain, no class weights)
## Challenger : Tuned gradient boosting
## Rule : at least 0.005 AUC in cross then challenger replaces the primary. 

## In simple, The real churn rate is 0.265.
The plain models average prediction is 0.266, which is honest.
The balanced model says 0.413, which is inflated.
0.135 vs 0.1647 a model that always guesses the average scores 0.195, so balanced is worse here

## Tuned vs default boosting: 0.0066 is 2.4times its std and wins 10 of 10, so the tuning is real
## Boosting vs Logistic Regression: 0.0018 is only 0.4times its std, 

## Hence, we should use plain. 
