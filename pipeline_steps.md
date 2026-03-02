# PL_Project Pipeline: Step-by-Step Documentation

This document describes the sequence of steps executed in the main pipeline of PL_Project, as defined in `main.py`. Each step is briefly explained to provide clarity on its purpose and function within the overall workflow.

---

## 1. Data Preprocessing
**Function:** `preprocess_data()`
- Cleans, formats, and prepares the raw input data for analysis. This includes handling missing values, encoding categorical variables, and generating features required for downstream modeling.

## 2. Survival Model Training
**Function:** `random_survival_forest()`
- Trains a Random Survival Forest model to predict time-to-event outcomes (e.g., churn). This model estimates survival probabilities for each subscription/customer over time.

## 3. Average Treatment Effect (ATE) Estimation (Linear Model)
**Function:** `estimate_ate_linear()`
- Estimates the average effect of treatments (e.g., campaign_cohort) on the outcome using a linear regression approach. Provides a baseline measure of treatment impact.

## 4. Segment-Level ATE Estimation (Monthly Spend)
**Function:** `estimate_ate_linear_segment('monthly_spend_estimated')`
- Calculates ATE for different segments based on estimated monthly spend, allowing for more granular insights into how treatment effects vary across spending levels.

## 5. Segment-Level ATE Estimation (Churn Duration)
**Function:** `estimate_ate_linear_segment('churn_duration_months')`
- Calculates ATE for segments defined by churn duration, revealing how treatment effects differ for customers with varying churn histories.

## 6. ATE Estimation via T-Learner
**Function:** `ate_t_learner()`
- Uses the T-Learner approach to estimate heterogeneous treatment effects, modeling outcomes separately for treated and control groups to capture individual-level uplift.

## 7. Survival Model Prediction (Control Group)
**Function:** `random_survival_forest(full_dataset=False, train_test=False)`
- Applies the Random Survival Forest model specifically to the control group, generating survival predictions for use in lifetime value (LTV) estimation and policy simulation.

## 8. Lifetime Value Prediction (Control Group)
**Function:** `predict_ltv_based_on_control_rsf()`
- Predicts the 12-month (time_horizon_months_for_customer_lifetime_value) lifetime value for each subscription/customer in the control group using survival probabilities and historical spend data.

## 9. Policy Simulation (Customer-Level Constraint)
**Function:** `simulate_policy(constraint_level='customer')`
- Simulates an optimal treatment allocation policy, enforcing a constraint that only one subscription per customer can be treated. Selects the subscription with the highest expected value and applies a maximum treatable percentage at the customer level. Outputs key metrics such as expected churn prevented, gross/net value saved, and ROI.

## 10. Policy Simulation (Subscription-Level Constraint)
**Function:** `simulate_policy(constraint_level='subscription')`
- Simulates the policy with the constraint applied at the subscription level, allowing multiple subscriptions per customer to be treated. Outputs the same set of policy metrics for comparison.

---

This pipeline enables robust estimation of treatment effects, prediction of customer value, and simulation of targeted intervention policies under different operational constraints. Each step builds upon the previous, culminating in actionable insights for retention and value optimization.

