# PL_Project: Pipeline README

## Overview
This project provides a complete pipeline for retention modeling, causal inference, and policy simulation for subscription-based businesses. The pipeline is modular and configurable via the `configs/default.yaml` file.

---

## How to Run the Pipeline

1. **Install Dependencies**
   - Ensure you have Python 3.11 or above version installed.
   - Install required packages:
     ```powershell
     pip install -r requirements.txt
     ```

2. **Prepare Data**
   - Place your raw data file at the location specified by `paths.raw_data` in `configs/default.yaml`. Must be a csv file.
   - Specify the name of the processed data in `paths.processed_data`. File extension must be csv.
   - Adjust configuration values in `configs/default.yaml` as needed for your data and experiment.

3. **Run the Pipeline**
   - Execute the main script:
     ```powershell
     python main.py > log.txt
     ```
   - The pipeline will preprocess data, train models, estimate treatment effects, predict lifetime value, and simulate policy interventions.
   - Outputs will be saved to the directory specified by `paths.outputs_dir`.
   - Read the file `pipeline_steps.md` to undertsand the workings of full pipeline.

4. **Review Results**
   - Key outputs include processed data, model artifacts, treatment effect estimates, and policy simulation summaries.
   - Check the outputs directory for CSVs, model files, and summary reports.

5. **Output Files**
   - *Processed output file* : This is stored in `paths.outputs_dir`. It will have the name as provided in `paths.processed_data`.
   - LTV_and_Risk_Predictions_Full_Sample.csv : This contains the life time value and the risk score for each sample point based on Random Survival Forest
   - ATE_Full_Sample_TreatmentName_DependentVariableName.csv : This file constains individual `ate`. There will be as many files as the number of treatments and number of dependent variable specified in  
     `causal_inference.dp_variables_for_treatment` in `configs/default.yaml`.
   - Control_based_LTV.csv : This file contains the LTV based on Random Survival Forest built using only controls. I.e., exlcuding samples which are in treatment group.  
   - Policy_output_customer.csv & Policy_output_subscription.csv are the policy output file showing who should be given which treatment.
---

## Configuration: Explanation of Terms in `default.yaml`

### paths
- **raw_data**: Path to the raw input data file (CSV).
- **processed_data**: Path where processed/cleaned data will be saved and loaded from.
- **outputs_dir**: Directory for all output files (model artifacts, results, summaries).

### model
- **rsf_random_state**: Random seed for reproducibility in Random Survival Forest training.
- **traintest_split_random_state**: Random seed for train/test data splitting.
- **rsf_model_name**: Path to save/load the trained Random Survival Forest model.

### variables
- **duration_col**: Column name for churn duration (in months).
- **event_col**: Column name indicating churn event (1 = churned, 0 = active).
- **treatment_col**: Column name for treatment assignment (e.g., campaign cohort).
- **covariates**: List of feature columns used for building Random Survival Forest.

### causal_inference
- **dp_variables_for_treatment**: Outcome variables for treatment effect estimation (e.g., churned, revenue).
- **control_variables_for_treatment**: Covariates used to control for confounding in causal inference.

### lifetime_value_variables
- **annual_discount_rate**: Discount rate used to calculate present value of future revenue (for LTV prediction).

### treatment_costs
- **variant_a**: Cost of applying treatment variant A (e.g., €10).
- **variant_b**: Cost of applying treatment variant B (e.g., €20).
- **max_treatable_percentage**: Maximum percentage of subscriptions/customers eligible for treatment (e.g., 20%).
- **time_horizon_months_for_customer_lifetime_value**: Time horizon (in months) for calculating customer lifetime value (e.g., 12) used in policy simulation.

---


