CREATE DATABASE IF NOT EXISTS chrun_db;
USE churn_db;

DROP TABLE IF EXISTS customers;

CREATE TABLE customers (
    customerID    VARCHAR(20) PRIMARY KEY,
    gender    VARCHAR(10),
    seniorCitizen    TINYINT,
    Partner    TINYINT,
    Dependents    TINYINT,
    tenure    INT,
    PhoneService    TINYINT,
    MultipleLines    TINYINT,
    InternetService    VARCHAR(20),
    OnlineSecurity    VARCHAR(20),
    OnlineBackup    TINYINT,
    DeviceProtection    TINYINT,
    TechSupport    TINYINT,
    StreamingTV    TINYINT,
    StreamingMovies    TINYINT,
    Contract    VARCHAR(20),
    PaperlessBilling    TINYINT,
    PaymentMethod    VARCHAR(30),
    MonthlyCharges    DECIMAL(6,2),
    TotalCharges    DECIMAL(10,2),
    Churn    TINYINT
);

SELECT COUNT(*) AS customers FROM customers;

SELECT ROUND(AVG(Churn) * 100, 1) AS churn_pct FROM customers;

SELECT Contract,
    COUNT(*) AS customer,
    ROUND(AVG(Churn) * 100, 2) AS churn_pct
FROM customers
GROUP BY Contract
ORDER BY churn_pct DESC;

SELECT CASE
    WHEN tenure <= 12 THEN '0-12 months'
    WHEN tenure <= 24 THEN '13-24 months'
    WHEN tenure <= 48 THEN '25-48 months'
    ELSE '49-72 months'
   END AS tenure_group,
   COUNT(*) AS customers,
   ROUND(AVG(Churn)*100, 1) AS churn_pct
FROM customers
GROUP BY tenure_group
ORDER BY tenure_group;

SELECT COUNT(*) AS churned_customers,
    ROUND(SUM(MonthlyCharges), 2) AS monthly_revenue_lost
FROM customers
WHERE Churn = 1;

SELECT Contract,
    InternetService,
    PaymentMethod,
    COUNT(*) AS customers,
    ROUND(AVG(Churn) * 100, 2) AS churn_pct
FROM customers GROUP BY Contract, InternetService, PaymentMethod HAVING COUNT(*) >= 100
ORDER BY churn_pct DESC
LIMIT 5;

SELECT Contract, ROUND(AVG(Churn)* 100,2) AS churn_pct
FROM customers
GROUP BY Contract;
