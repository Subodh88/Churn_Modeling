# Data Scientist Take-Home Assignment
## Retention & Uplift Optimiser

## 1 — Business Scenario

The Postcode Lottery Group raises funds for charities through lottery participation. 
Every 
retained participant directly contributes to more funds for environment, health, and 
community causes.

We have a subscription based lottery. Customers subscribe using their postcodes as 
their lottery numbers, and each month there is a draw with various prizes. Customers 
participate in these draws as long as they have an active subscription. Each year 
there are 2 extra draws on top of regular 12 draws.

Historically, our retention strategy was reactive: we predicted who was likely to 
churn and then manually decided what to do. We now want to move to a **proactive, 
test-and-learn approach**:

1.  We want to **predict churn and lifetime value**.
2.  We want to **understand which interventions actually work** (and for whom).
3.  We want to roll this out in a way that is **robust, reproducible, and ready for 
    production**.

### The Experiment
Recently, we ran a randomized controlled experiment with three groups:
* **Control:** No special treatment.
* **Variant A:** A standard discount offer.
* **Variant B:** A value-add offer (e.g., extra draws, bonus tickets, or 
  charity-related perks).

*Note: Each active subscription at the start of the experiment was randomly assigned 
to a cohort, but due to operational constraints, not everyone in a treatment cohort 
actually received the communication.*

### Your Goal
Build a production-grade data science pipeline that:
1.  **Cleans and standardizes** a messy CRM export.
2.  **Builds robust models** to predict churn probability and estimate lifetime value.
3.  **Quantifies incremental impact** of Variant A and Variant B vs Control.
4.  **OPTIONAL:** Produces a targeting policy:** Determines who should get which offer, 
    under 
    budget constraints, to maximize expected net contribution to charity.

---

## 2 — The Data

**File:** `data/raw/advanced_case_study_data.csv`

Data come from 2 different main sources, given below as legacy_system_id. Each row 
represents one lottery subscription at the start of the experiment. 
This file is a raw legacy export and is **not in an ingest-ready format**. You 
should assume:
* Mixed decimal separators (`.` and `,`).
* Mixed thousand separators.
* Mixed date formats (US vs EU, text months).
* Mixed encodings for booleans and categories.
* Embedded JSON-like strings in some columns.
* And other anomalies that can appear in a data export.

### Data Schema

|    | Column Name                 | Description                                           | Example Values                      |
|----|:----------------------------|:------------------------------------------------------|:------------------------------------|
| 1  | `customer_id`               | Unique customer identifier                            | `b7a782741f667201b5`                |
| 2  | `subscription_id`           | Unique subscription identifier                        | `7db88cdd3c295d2276`                |
| 3  | `legacy_system_id`          | Data Source system identifier                         | `SYS_OLD`, `SYS_NEW`                |
| 4  | `subscription_date`         | Original subscription datetime                        | `2022-03-04 00:00:00`, `July 2018`  |
| 5  | `participant_age`           | Age at experiment start                               | `69`, `32`, `Unknown`               |
| 6  | `marketing_channel`         | Acquisition channel                                   | `DM`, `Direct mail`, `Online paid`  |
| 7  | `country_code`              | Country of lottery                                    | `NL`, `DE`, `GB`, `N/A`             |
| 8  | `postcode_area`             | Coarse location of subscription owner                 | `1012`, `SE-123 45`                 |
| 9  | `extra_draws_per_year`      | Opt-in for extra "13th/14th" month draws              | `0` (None), `1`, `2` (Both)         |
| 10 | `Add-ons`                   | Number of add-ons                                     | `0`, `1`, `3`, `0,0`                |
| 11 | `payment_method`            | Primary payment method                                | `Direct debit`, `Credit card`       |
| 12 | `failed_payments_12m`       | Count of failed payments in last 12 months            | `0`, `1`, `NaN`                     |
| 13 | `monthly_spend_estimated`   | Estimated monthly net spend (pre-tax)                 | `€15,50`, `EUR 15.50`, `1,250`      |
| 14 | `donation_share_charity`    | Share of monthly spend that goes to charity           | `0.40`, `40`                        |
| 15 | `engagement_history`        | Recent engagement in semi-structured format           | `{"email_opens": "5"}`, `opens=3`   |
| 16 | `web_sessions_90d_raw`      | Web sessions in last 90 days                          | `0`, `5`, `10+`, `missing`          |
| 17 | `c_service_contacts_12m`    | Number of customer service contacts in last 12 months | `0`, `1`, `99`                      |
| 18 | `complaints_12m`            | Number of complaints in last 12 months                | `0`, `2`, `5`                       |
| 19 | `lifetime_wins`             | Total number of historical prizes won                 | `0`, `1`, `15`                      |
| 20 | `win_dates`                 | Dates of prizes won                                   | `2022-03-04 00:00:00`, `July 2018`  |
| 21 | `campaign_cohort`           | Randomized experiment group                           | `Control`, `Variant_A`, `Variant_B` |
| 22 | `treatment_sent_flag`       | Whether the campaign communication was actually sent  | `1`, `0`, `Y`, `N`                  |
| 23 | `offer_cost_eur`            | Marketing + incentive cost                            | `0`, `€20,00`, `20.0`               |
| 24 | `baseline_churn_risk_band`  | Legacy rule-based churn band at experiment start      | `Low`, `Medium`, `High`             |
| 25 | `historic_revenue_12m`      | Observed revenue last 12 months (before experiment)   | `€180,00`, `1.200,00`               |
| 26 | `churned`                   | Whether subscription churned in follow-up window      | `1`, `0`, `Y`, `N`                  |
| 27 | `churn_date`                | Date of churn (if any)                                | `2023-01-15`, `NA`                  |
| 28 | `observation_end_date`      | End date of follow-up observation                     | `2023-03-31`                        |
| 29 | `revenue_next_12m_observed` | Observed revenue (after exp) over 12-month follow-up  | `€600,00`, `0`, `NA`                |
### Assumptions
* The experiment assignment (`campaign_cohort`) is randomized at the subscription level.
* Some participants in Variant A/B did not actually receive the treatment due to 
  operational issues (`treatment_sent_flag`), which you should consider when 
  defining causal estimands.
* All monetary values are intended to be interpreted in **EUR** after cleaning and 
  conversion, even if the raw export shows $ or USD.



# 3 — Your Assignment: What You Must Build

**Timeline:** 1 Week.

**Deliverable:** A Git repository (GitHub/GitLab) with reproducible code and a `design_doc.md`.

You have **one week** to work on this assignment. We expect a professional, 
reproducible repository (GitHub/GitLab-style) with code, analysis, and a short 
description. You will be invited to review your work together with our data 
scientists if your solution meets our evaluation criteria.

**Tech Stack:** You are free to choose your stack (e.g., Python with Pandas / 
scikit-learn / statsmodels, etc.), but you should:
* Follow good MLOps practices.
* Make your work easy to rerun from scratch.

**Suggested Structure:**
* `src/` for Python modules.
* `notebooks/` for exploratory work.
* `data/` for input/output data (with `data/raw/` and `data/processed/`).
* `configs/` for configuration.
* `tests/` for unit tests.

```text
├── README.md
├── design_doc.md
├── configs/
│   └── default.yaml
├── data/
│   ├── raw/
│   │   └── advanced_case_study_data.csv
│   └── processed/
│       └── retention_model_data.parquet
├── notebooks/
│   └── eda_and_experiments.ipynb
├── src/
│   ├── preprocessing.py
│   ├── models.py
│   ├── causal.py
│   ├── policy.py
│   └── run_all.py
└── tests/
    └── test_preprocessing.py
```

## Part A — Data Engineering & Pipeline 
(`src/preprocessing.py`)

### 1. Robust Ingestion
Build a single entry point (e.g., CLI or main function) that:
* Reads `data/raw/advanced_case_study_data.csv`.
* **Handles robustness:** Manage mixed encodings, decimal separators, and date formats.
* **Normalizes:** Standardize booleans and categorical encodings.
* *Note:* A simple `pd.read_csv()` with default options is expected to produce 
  incorrect types; your code should make this robust and reproducible.

### 2. Cleaning & Standardization
* Standardise all columns properly

### 3. Feature Engineering
* **Output:** Create a clean training dataset and write it to 
  `data/processed/retention_model_data.parquet` (or `.csv`), including:
    * All features you decide to use.
    * Treatment assignment variables.
    * Target variables.

### 4. Reproducibility
* All cleaning and feature engineering steps should be executable by running a 
  single command/script.
* You must be able to re-create the processed dataset from only the raw input.

---

## Part B — Predictive Modeling 
(`src/models.py`)

### 1. Churn Propensity Model
* **Baseline:** Build at least one baseline model (e.g., regularized logistic 
  regression, GLM) to predict `churned`.
* **Advanced:** Build at least one more advanced model (e.g., Random Forest, 
  Gradient Boosting, XGBoost, LightGBM).
* **Documentation:** Clearly describe:
    * How you approach training.
    * Which features you include or exclude (and why).
* **Evaluation:** Compare models using appropriate metrics (e.g., ROC AUC, PR AUC, 
  log loss, calibration plots, lift curves, etc.) and motivate your choice/conclusions.

### 2. Customer Value / LTV Approximation
* Using `monthly_spend_estimated` and/or `revenue_next_12m_observed`, define an 
  approximate subscription or customer value measure (e.g., expected next 12-month 
  revenue or contribution).
* Explain your choices (e.g., how you handle missing etc).
* Provide a conceptual model design for a more robust lifetime value calculation.

---

## Part C — Causal & Uplift Analysis 
(`src/causal.py`)

### 1. A/B Test Analysis
* Treat `campaign_cohort` as the treatment assignment.
* Estimate the **average treatment effect** on churn for:
    * Variant A vs Control.
    * Variant B vs Control.
* **Checks:** Verify the results, should we be concerned about bias.
* **Output:** Provide point estimates and confidence intervals, and interpret them.

### 2. Treatment Effect Heterogeneity (Segments)
* Identify whether certain segments (e.g., high vs low spend, shorter vs longer 
  tenure, etc) react differently to Variant A or B.
* **Method:** Choose the appropriate method to conduct this analysis and motivate 
  your choice. 
* **Summary:** Define which segments are particularly well suited for Variant A vs 
  Variant B.

### 3. Causal Estimation Uplift Modeling
* Build an explicit uplift model (e.g., two-model approach, T-learner, U-learner) to 
  estimate individual treatment effects.
* Compare this approach qualitatively and/or quantitatively to your simpler A/B 
  analysis.

---

## Part D (OPTIONAL-BONUS) — Targeting Policy & ROI 
(`notebooks/analysis.ipynb` or `src/policy.py`)

### 1. Define a Targeting Rule
Combine churn propensity, customer value, and treatment effect estimates to propose 
a targeting policy.
* *Example:* "Target subscriptions / customers with/where ......"
* You may choose to use Variant A for some segments and Variant B for others, or not 
  treat very low-value/low-risk customers.

### 2. Budget Constraint
* **Constraint:** You can treat at most **20%** of the total customer base.
* **Cost A:** Average cost of Variant A is **€10** per targeted subscription.
* **Cost B:** Average cost of Variant B is **€20** per targeted subscription.
* **Value:** Come up with an approximate for the value of saving a subscription.

*Task:* Construct a strategy that selects at most 20% of customers for treatment (A 
or B), respecting budget and any other sensible constraints.

### 3. ROI Calculation
On your test set (or via cross-validation), estimate the incremental net value of 
your strategy vs:
* (a) No campaign.
* (b) A naive strategy (e.g., random 20% targeting, or “treat top 20% by churn 
  probability”).

**Explicitly compute:**
* Incremental customers saved.
* Incremental revenue / contribution.
* Total campaign cost.
* **Overall ROI**.

---

## Part E — Engineering & Reproducibility

### 1. Configuration Management (`configs/`)
* Add a simple YAML configuration file (e.g., `configs/default.yaml`) that contains:
    * Paths to input/output data.
    * Key modeling & policy hyperparameters (e.g., random seeds, budget constraint).
* Your main scripts should be driven via this configuration file.

### 2. Project Structure & Entry Points
* **Script/CLI:** Provide a script to run the full pipeline end-to-end:
    `python -m src.run_all --config configs/default.yaml`
    * Runs preprocessing.
    * Trains/evaluates models.
    * (OPTIONAL-BONUS) Produces a file with recommended actions (`data/outputs/targeting_policy.csv`).
* **README:** A clear `README.md` explaining how to set up the environment, run 
  preprocessing, train models, and generate recommendations.

### 3. (OPTIONAL-BONUS)  Unit Tests (`tests/`)
* Write at least **2–3 unit tests** for:
    * Currency parsing and conversion.
    * Date parsing.
    * Parsing of `engagement_history`.
    * and others...
* Tests should run with a single command (e.g., `pytest`).

### 4. Versioning & Environment
* Include a `requirements.txt` or `environment.yml`.
* Ensure versions are pinned enough that your code is reproducible.

---

## Part F — Documentation

### 1. Design Document (`design_doc.md`)
Briefly describe:
* **Pipeline Architecture:** Ingestion, cleaning, feature engineering.
* **Modeling Choices:** Rationale for algorithms, key features, regularization, and 
  leakage prevention.
* **Causal Approach:** How you interpret your results.
* **Targeting Policy:** Connection to business constraints.
* **Traceability:** How your design ensures a specific recommendation can be traced 
  back to the exact data, code, and model.

### What to expect at the review session:
You will go over your solution with our data scientists (2) where you will have the 
chance to explain your design choices and answer questions from the group. 
* Questions about overall architecture and pipeline.
* Clarifying questions on **key technical decisions** (e.g., choice of uplift model, 
  handling dirty data).
* Questions on model and analysis quality, risks and performance.

### Do you need to prepare presentation?:
If you can find time it would be great to prepare a slide deck of 6-7 slides which 
* Explains **1–2 key technical decisions** (e.g., choice of uplift model, handling 
  dirty data).
* Shows **1–2 key results**:
* Contains **1–2 slides for non-technical stakeholders** (e.g., Head of Retention) 
  focusing on actionable insights.
* Mentions future improvements (e.g., deep learning, real-time inference).