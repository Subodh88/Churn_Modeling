from src.preprocessing import preprocess_data
from src.models import random_survival_forest, predict_ltv_based_on_control_rsf
from src.causal import estimate_ate_linear, estimate_ate_linear_segment, ate_t_learner
from src.policy import simulate_policy




def main():    
    preprocess_data()
    random_survival_forest()
    estimate_ate_linear()
    estimate_ate_linear_segment('monthly_spend_estimated')
    estimate_ate_linear_segment('churn_duration_months')
    ate_t_learner()
    random_survival_forest(full_dataset=False,train_test=False)
    predict_ltv_based_on_control_rsf()
    simulate_policy(constraint_level='customer')
    simulate_policy(constraint_level='subscription')
 


if __name__ == '__main__':
    main()
