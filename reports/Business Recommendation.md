## Business Recommendations
- Method: findings are only listed where the EDA, the SQL analysis and the model all agree. These are associations, not proven causes, each action should be validated with an A/B test

## Protect the first year and move month-to-month customers to longer contracts
- Evidence: churn is 47.4% in months 0-12 vs 9.5% after year 4; month-to-month 42.7% vs 2.8% for two-year contracts. The model's strongest protective signals are contract length and tenure.
- Action: an onboarding and early-life retention programme for month-to-month customers, with an incentive to switch to a 1-year contract.
- Caveat: customers who choose long contracts may simply be more loyal to begin with.

## Investigate fiber optic service
- Evidence: fiber churn 41.9% (DSL 19.0%); the largest risk weight in the model (odds x2.1).
- Action: review fiber pricing, reliability and complaints. The data has no churn reasons, so this needs operational data to diagnose.

## Offer technical support and online security to at-risk customers
- Evidence: churn 31% without these services vs 15% with them; both lower the model's risk.
- Action: free trial for fiber and month-to-month customers flagged by the model.
- Caveat: customers who take these services may be more engaged to begin with.

## Move customers to automatic payment
- Evidence: electronic-check customers churn 45.3% vs 15-19% for other methods, automatic payment 16.0% vs 34.7% for manual.
- Action: auto-pay enrolment incentive.

## Target the riskiest 20% of customers and measure the real success rate
- Evidence (test set): contacting the top 20% reaches 48.7% of all churners (lift 2.44, hit rate 64.8%). Under illustrative assumptions, profit per customer is 6.8 dollar vs -46.9 dollar for contacting everyone.
- Key uncertainty: the offer success rate. Profit ranges from about 0.2 dollar (15% success) to $27.6 (40% success) per customer, so a small A/B test of the real success rate is the first thing to do.

## 5. Limitations and failure cases
- 