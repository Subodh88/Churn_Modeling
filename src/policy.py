from src.utils import load_config, pprint_df, get_current_date_time
import os
import pandas as pd
import numpy as np

def simulate_policy(constraint_level='customer'):
    config = load_config('configs/default.yaml')
    output_path = config['paths']['outputs_dir']
    ltv_file = os.path.join(output_path, f"Control_based_LTV.csv")
    ltv_df = pd.read_csv(ltv_file)

    data_path = config['paths']['processed_data']
    data = pd.read_parquet(data_path) if data_path.endswith('.parquet') else pd.read_csv(data_path)

    campaign_col = config['variables']['treatment_col']
    data[campaign_col] = data[campaign_col].str.lower()
    treatments = data[campaign_col].unique().tolist()
    treatments = [t for t in treatments if t.lower() != 'control']

    # Step 1: Load ITEs and LTV, standardize treatment-effect direction
    treatment_ate = pd.DataFrame(index=data.index, columns=treatments)
    for treatment in treatments:
        treatment_file = os.path.join(output_path, f"ATE_Full_Sample_{treatment}_churned.csv")
        treatment_df = pd.read_csv(treatment_file)
        tau = treatment_df[f'ITE_{treatment}']
        # Standardize: churn prevented probability
        u = np.clip(np.maximum(0, -tau), 0, 1)
        treatment_ate[treatment] = u

    # Add LTV
    treatment_ate['ltv'] = ltv_df['predicted_lifetime_value']
    treatment_ate['sero'] = 0
    treatment_ate['customer_id'] = ltv_df['customer_id']
    treatment_ate['subscription_id'] = ltv_df['subscription_id']

    # Step 2: Compute expected gross value saved
    for treatment in treatments:
        treatment_ate[f'saved_value_{treatment}'] = treatment_ate[treatment] * treatment_ate['ltv']

    # Step 3: Subtract treatment costs to get expected net gain
    costs = config['treatment_costs']
    for treatment in treatments:
        cost = costs.get(treatment, 0)
        treatment_ate[f'net_gain_{treatment}'] = treatment_ate[f'saved_value_{treatment}'] - cost

    # Step 4: Choose best action per subscription
    net_gain_cols = [f'net_gain_{t}' for t in treatments]
    net_gain_cols = net_gain_cols + ['sero']
    treatment_ate['best_net_gain'] = treatment_ate[net_gain_cols].max(axis=1)
    treatment_ate['best_action'] = treatment_ate[net_gain_cols].idxmax(axis=1).str.replace('net_gain_', '')
    treatment_ate.loc[treatment_ate['best_net_gain'] <= 0, 'best_action'] = 'none'

    # Step 5: Enforce max treatable % constraint
    max_treatable_pct = costs['max_treatable_percentage']/100
    treatment_ate['final_action'] = 'none'

    if constraint_level == 'customer':
        N_customers = treatment_ate['customer_id'].nunique()
        treatable = treatment_ate[treatment_ate['best_net_gain'] > 0].copy()
        # For each customer, select the subscription with maximum saved value
        treatable = treatable.loc[treatable.groupby('customer_id')[['saved_value_' + t for t in treatments]].idxmax().max(axis=1)]
        treatable = treatable.sort_values('best_net_gain', ascending=False)
        N_treat = min(int(max_treatable_pct * N_customers), len(treatable))
        treatable = treatable.iloc[:N_treat]
        treatment_ate.loc[treatable.index, 'final_action'] = treatable['best_action']
    else:
        N = len(treatment_ate)
        treatable = treatment_ate[treatment_ate['best_net_gain'] > 0].copy()
        treatable = treatable.sort_values('best_net_gain', ascending=False)
        N_treat = min(int(max_treatable_pct * N), len(treatable))
        treatable = treatable.iloc[:N_treat]
        treatment_ate.loc[treatable.index, 'final_action'] = treatable['best_action']

    # Writing the policy output to a CSV file
    timestamp = get_current_date_time()
    output_file = os.path.join(output_path, f"Policy_output_{constraint_level}.csv")
    treatment_ate.to_csv(output_file, index=False)
    # Step 6: Policy outputs
    n_treated = (treatment_ate['final_action'] != 'none').sum()
    pct_treated = n_treated / (treatment_ate['customer_id'].nunique() if constraint_level=='customer' else len(treatment_ate))
    variant_count = {}
    total_cost = 0
    for treatment in treatments:
        variant_count[treatment] = (treatment_ate['final_action'] == treatment).sum()
        total_cost += variant_count[treatment] * costs.get(treatment, 0)

    treated = treatment_ate[treatment_ate['final_action'] != 'none']
    expected_gross_value = treated.apply(lambda row: row[f'saved_value_{row["final_action"]}'], axis=1).sum()
    expected_net_value = treated['best_net_gain'].sum()
    roi_gross = expected_gross_value / total_cost if total_cost > 0 else 0
    roi_net = expected_net_value / total_cost if total_cost > 0 else 0

    summary = {
        'pct_treated': pct_treated,
        'total_cost': total_cost,
        'expected_gross_value': expected_gross_value,
        'expected_net_value': expected_net_value,
        'roi_gross': roi_gross,
        'roi_net': roi_net,
    }

    for treatment in treatments:
        summary[f'{treatment}%'] = round((variant_count[treatment]/n_treated) * 100,2) if n_treated > 0 else 0

    if constraint_level == 'customer':
        print("\nPolicy Simulation Summary : Customer-level treatment")
    else:
        print("\nPolicy Simulation Summary : Subscription-level treatment")
    summary_df = pd.DataFrame([summary])
    pprint_df(summary_df)
