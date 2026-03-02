import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sksurv.ensemble import RandomSurvivalForest
from sksurv.preprocessing import OneHotEncoder
from src.utils import load_config, pprint_df, get_current_date_time
import joblib
import os
import matplotlib.pyplot as plt
from sklearn.inspection import permutation_importance


def random_survival_forest(full_dataset=True,train_test=True):
	# Load random_state from config
	config = load_config('configs/default.yaml')
	rsf_random_state = config.get('model', {}).get('rsf_random_state', 42)
	train_test_split_random_state = config.get('model', {}).get('traintest_split_random_state', 50)
	rsf_model_name = config.get('model', {}).get('rsf_model_name', 'rsf_model')

	if os.path.exists(rsf_model_name):
		print(f"[WARNING] RSF model file '{rsf_model_name}' already exists and will be overwritten.")
		# Optionally, you could add a prompt here to ask the user if they want to continue or not
		#response = input("Do you want to continue? (y/n): ")
		response = 'y'
		if response.lower() == 'n':
			print("Operation cancelled.")
			return

	event_col = config.get('variables', {}).get('event_col', '')
	duration_col = config.get('variables', {}).get('duration_col', '')
	covariates = config.get('variables', {}).get('covariates', [])
 
	# if event_col, duration_col, or covariates are not defined in the config, raise an error
	if not event_col or not duration_col or not covariates:
		raise ValueError("event_col, duration_col, and covariates must be defined in the config file under 'variables' section.")

	data_path = config['paths']['processed_data']
	data = pd.read_parquet(data_path) if data_path.endswith('.parquet') else pd.read_csv(data_path)

	campaign_col = config['variables']['treatment_col']
	if not full_dataset:
		data[campaign_col] = data[campaign_col].str.lower()
		data = data[data[campaign_col].isin(['control'])]
		data.reset_index(drop=True, inplace=True)

	# Check if the specified columns exist in the data
	missing_cols = [col for col in [event_col, duration_col] + covariates if col not in data.columns]
	if missing_cols:
		raise ValueError(f"The following columns are missing in the data: {', '.join(missing_cols)}")


	# Prepare the data for the model
	y = data[[event_col, duration_col]].apply(lambda row: (row[event_col], row[duration_col]), axis=1).to_numpy(dtype=[('event', bool), ('duration', int)])
	X = data[covariates]


	# Go through each column and check if it's categorical, if yes, apply one-hot encoding
	for col in X.columns:
		num_unique_values = X[col].nunique()
		# check if each value in the column is a string, if yes, apply one-hot encoding
		if X[col].dtype == 'object' or X[col].dtype.name == 'category':
			# Apply one-hot encoding to the column
			encoder = OneHotEncoder()
			encoded_cols = encoder.fit_transform(X[[col]])
			# Create a DataFrame with the encoded columns and concatenate it with the original DataFrame
			encoded_df = pd.DataFrame(encoded_cols, columns=encoder.get_feature_names_out([col]), index=X.index)
			X = pd.concat([X.drop(columns=[col]), encoded_df], axis=1)


	if train_test:
		X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=train_test_split_random_state)

		# count the number of True and False in the event column of y_train and y_test and print them nicely using pprint_df
		train_event_counts = pd.Series(y_train['event']).value_counts().sort_index()
		test_event_counts = pd.Series(y_test['event']).value_counts().sort_index()
		print("Training set event counts:")
		train_event_counts = train_event_counts.to_frame(name='count').reset_index().rename(columns={'index': 'Category'})
		train_event_counts['Category'] = train_event_counts['Category'].map({False: 'Not churned', True: 'Churned'})
		train_event_counts['Percentage'] = (train_event_counts['count'] / train_event_counts['count'].sum() * 100).round(2).astype(str) + '%'
		pprint_df(train_event_counts)

		print("\nTesting set event counts:")
		test_event_counts = test_event_counts.to_frame(name='count').reset_index().rename(columns={'index': 'Category'})
		test_event_counts['Category'] = test_event_counts['Category'].map({False: 'Not churned', True: 'Churned'})
		test_event_counts['Percentage'] = (test_event_counts['count'] / test_event_counts['count'].sum() * 100).round(2).astype(str) + '%'
		pprint_df(test_event_counts)


	rsf = RandomSurvivalForest(n_estimators=100, min_samples_split=10, min_samples_leaf=15, n_jobs=-1, random_state=rsf_random_state)
	if train_test:
		# Fit the Random Survival Forest model
		rsf.fit(X_train, y_train)
		# Save the model to disk
		joblib.dump(rsf, rsf_model_name)
	else:
		rsf.fit(X, y)
		# Save the model to disk
		rsf_model_name = rsf_model_name.replace(".joblib", "")  # Remove .joblib extension if it exists
		rsf_model_name = rsf_model_name + "_control_dataset"
		rsf_model_name = rsf_model_name if rsf_model_name.endswith(".joblib") else rsf_model_name + ".joblib"  # Ensure it ends with .joblib
		joblib.dump(rsf, rsf_model_name)

	# Load the model back from disk
	#rsf = joblib.load(rsf_model_name)

	if train_test:
		c_index_train = rsf.score(X_train, y_train)
		print(f"\nTraining C-index: {c_index_train:.5f}")
		c_index_test = rsf.score(X_test, y_test)
		print(f"Testing C-index: {c_index_test:.5f}")

		print(f"\nCalculating & Plottting survival probability for the test samples. This may take a few moments...")
		# Predict the survival function for the test set
		surv_funcs = rsf.predict_survival_function(X_test)
		# Plot survival functions
		plt.figure(figsize=(10, 6))
		for i, surv_func in enumerate(surv_funcs):
			plt.step(surv_func.x, surv_func.y, label=f"Sample {i}")
		plt.xlabel("Time")
		plt.ylabel("Survival Probability")
		plt.title("Predicted Survival Functions")
		plt.savefig("predicted_survival_functions.png", dpi=300)
		plt.close()

		print(f"\nCalculating & Plottting cumulative hazard function for the test samples. This may take a few moments...")
		# Plot cumulative hazard functions
		cum_hazard_funcs = rsf.predict_cumulative_hazard_function(X_test)
		plt.figure(figsize=(10, 6))
		for i, cum_hazard_func in enumerate(cum_hazard_funcs):
			plt.step(cum_hazard_func.x, cum_hazard_func.y, label=f"Sample {i}")
		plt.xlabel("Time")
		plt.ylabel("Cumulative Hazard")
		plt.title("Predicted Cumulative Hazard Functions")
		plt.savefig("predicted_cumulative_hazard_functions.png", dpi=300)
		plt.close()

		# Get the survival probabilities at the last point of the survival function for each test sample
		surv_probs_at_last_time = np.array([surv_func.y[-1] for surv_func in surv_funcs])
		surv_hazard_at_last_time = np.array([cum_hazard_func.y[-1] for cum_hazard_func in cum_hazard_funcs])
		# Create a DataFrame to display the survival probabilities also with the actual event and duration for each test sample
		surv_probs_df = pd.DataFrame({
			"Survival Probability": surv_probs_at_last_time,
			"Cumulative Hazard": surv_hazard_at_last_time,
			"Actual Event": y_test["event"],
			"Actual Duration": y_test["duration"]
		})

		# Get the average survival probability for the churned and not churned samples in the test set
		avg_surv_prob_churned = surv_probs_df[surv_probs_df["Actual Event"] == True]["Survival Probability"].mean()
		avg_surv_prob_not_churned = surv_probs_df[surv_probs_df["Actual Event"] == False]["Survival Probability"].mean()

		# Check if the maximum survival probability for the churned samples is less than the minimum survival probability for the not churned samples, if yes, then the model is able to perfectly separate the two classes based on the predicted survival probabilities
		max_surv_prob_churned = surv_probs_df[surv_probs_df["Actual Event"] == True]["Survival Probability"].max()
		min_surv_prob_not_churned = surv_probs_df[surv_probs_df["Actual Event"] == False]["Survival Probability"].min()
		print(f"\nAverage survival probability for churned samples  : {avg_surv_prob_churned:.3f}")
		print(f"Average survival probability for not churned samples: {avg_surv_prob_not_churned:.3f}")
		print(f"Maximum survival probability for churned samples    : {max_surv_prob_churned:.3f}")
		print(f"Minimum survival probability for not churned samples: {min_surv_prob_not_churned:.3f}")
		if max_surv_prob_churned < min_surv_prob_not_churned:
			print("The model is able to perfectly separate the churned and not churned samples based on the predicted survival probabilities.")
		else:
			plt.figure(figsize=(10, 6))
			surv_probs_df[surv_probs_df["Actual Event"] == True]["Survival Probability"].hist(alpha=0.5, label="Churned")
			surv_probs_df[surv_probs_df["Actual Event"] == False]["Survival Probability"].hist(alpha=0.5, label="Not Churned")
			plt.xlabel("Predicted Survival Probability at Last Time Point")
			plt.ylabel("Frequency")
			plt.title("Distribution of Predicted Survival Probabilities for Churned and Not Churned Samples")
			plt.legend()
			plt.savefig("survival_probability_distribution.png", dpi=300)
			plt.close()

		churn_cutoff_prob = 0.3
		# Create a confusion matrix based on the predicted survival probabilities at the last time point using a cutoff of 0.3, if the predicted survival probability is less than 0.3, then we predict that the sample will churn, otherwise we predict that the sample will not churn
		surv_probs_df["Predicted Churn"] = surv_probs_df["Survival Probability"].apply(lambda x: True if x <= churn_cutoff_prob else False)
		confusion_matrix = pd.crosstab(surv_probs_df["Actual Event"], surv_probs_df["Predicted Churn"], rownames=["Actual"], colnames=["Predicted"])
		print(f"\nConfusion Matrix based on predicted survival probabilities at the last time point with a cutoff of {churn_cutoff_prob}:")
		pprint_df(confusion_matrix, showindex=True)
		# Calculate the accuracy, precision, recall, and F1-score based on the confusion matrix
		tp = confusion_matrix.loc[True, True]
		fp = confusion_matrix.loc[False, True]
		tn = confusion_matrix.loc[False, False]
		fn = confusion_matrix.loc[True, False]
		accuracy = (tp + tn) / (tp + fp + tn + fn) if (tp + fp + tn + fn) > 0 else 0
		precision = tp / (tp + fp) if (tp + fp) > 0 else 0
		recall = tp / (tp + fn) if (tp + fn) > 0 else 0
		f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
		print(f"Accuracy : {accuracy:.2f}")
		print(f"Precision: {precision:.2f}")
		print(f"Recall   : {recall:.2f}")
		print(f"F1-score : {f1_score:.2f}")

		# Risk Score
		risk = pd.Series(rsf.predict(X_test))
		surv_probs_df["Predicted Risk"] = risk
		surv_probs_df = surv_probs_df.sort_values(by="Predicted Risk", ascending=False)
		number_churned_test = surv_probs_df["Actual Event"].sum()
		# create a list of steps in increment of 100 until the number of churned samples in the test set
		Sample_steps = list(range(100, int(number_churned_test) + 100, 100))
		# find the number of churned and not churned samples in the top 100, 200, 300, 400, and 500 samples with the highest predicted risk and print them in a nice table
		risk_cutoff_results = []
		for cutoff in Sample_steps:
			top_samples = surv_probs_df.head(cutoff)
			churned_count = top_samples["Actual Event"].sum()
			not_churned_count = len(top_samples) - churned_count
			risk_cutoff_results.append({
				"Top": cutoff,
				"Churned Count": churned_count,
				"Not Churned Count": not_churned_count,
				"Churn Rate": round((churned_count / cutoff * 100),0) if cutoff > 0 else 0
			})

		risk_cutoff_df = pd.DataFrame(risk_cutoff_results)
		print("\nChurn split in top (# of samples) based on predicted risk:")
		pprint_df(risk_cutoff_df)

		print(f"\n Calculating & Plotting feature importance based on permutation importance. This may take a few moments...")
		result_importance = permutation_importance(rsf, X_test, y_test, n_repeats=15, random_state=train_test_split_random_state)
		result_importance_df = pd.DataFrame({k: result_importance[k] for k in ("importances_mean","importances_std")},index=X_test.columns).sort_values(by="importances_mean", ascending=False)
		# Plot feature importance
		plt.figure(figsize=(12, 8))
		plt.barh(result_importance_df.index, result_importance_df['importances_mean'], xerr=result_importance_df['importances_std'])
		plt.xlabel("Mean Permutation Importance")
		plt.title("Feature Importance based on Permutation Importance")
		plt.gca().invert_yaxis()  # Invert y-axis to have the most important feature at the top
		plt.savefig("feature_importance.png", dpi=300)
		plt.close()


		# Lifetime value calculation and risk calculation for the complete dataset
		min_churn_duration_churned = data[duration_col].min()
		monthly_discount_factor = 1/ ((1 + config['lifetime_value_variables']['annual_discount_rate']) ** (1/12))
		monthly_spend = data['monthly_spend_estimated'].values
		All_LVT = []

		# calculate survival probability at each time point for the complete dataset
		surv_probs_complete = rsf.predict_survival_function(X)
		for i, surv_func in enumerate(surv_probs_complete):
				step = surv_func.x
				prob = surv_func.y
				curr_monthly_spend = monthly_spend[i]
				t_query = []
				all_prob = []
				if min_churn_duration_churned != 1:
					t_query = [i for i in range(1,min_churn_duration_churned)]
					all_prob = []
					for t in t_query:
						prob_at_t1 = np.interp(t, surv_func.x, surv_func.y)
						all_prob.append(prob_at_t1)

				step = t_query + step.tolist()
				prob = all_prob + prob.tolist()
				lvt = 0
				for t, p in zip(step, prob):
					# calculate the discounted monthly spend at each time point and sum it up to get the lifetime value for each sample
					lvt += (curr_monthly_spend * (monthly_discount_factor ** t) * p)
				All_LVT.append(lvt)


		data['predicted_lifetime_value'] = All_LVT

		# Risk Score
		risk = pd.Series(rsf.predict(X))
		data["predicted_risk"] = risk
		current_date_time = get_current_date_time()

		Final_data = data[['customer_id', 'subscription_id', 'subscription_date', 'churn_date', 'churned']+[event_col, duration_col] + covariates + ['predicted_lifetime_value', 'predicted_risk']]
		output_path = config['paths']['outputs_dir']
		# check if this folder exists, if not, create it
		if not os.path.exists(output_path):
			os.makedirs(output_path)
		output_file = os.path.join(output_path, f"LTV_and_Risk_Predictions_Full_Sample.csv")
		Final_data.to_csv(output_file, index=False)
		print(f"\nLifetime value and risk predictions for the complete dataset have been saved to {output_file}.")

def predict_ltv_based_on_control_rsf():
	config = load_config('configs/default.yaml')
	rsf_model_name = config.get('model', {}).get('rsf_model_name', 'rsf_model')
	rsf_model_name = rsf_model_name.replace(".joblib", "")  # Remove .joblib extension if it exists
	rsf_model_name = rsf_model_name + "_control_dataset"
	rsf_model_name = rsf_model_name if rsf_model_name.endswith(".joblib") else rsf_model_name + ".joblib"  # Ensure it ends with .joblib

	if not os.path.exists(rsf_model_name):
		raise FileNotFoundError(f"RSF model file '{rsf_model_name}' not found. Please run the random_survival_forest function with full_dataset=True and train_test=False to create the model first.")

	rsf = joblib.load(rsf_model_name)

	event_col = config.get('variables', {}).get('event_col', '')
	duration_col = config.get('variables', {}).get('duration_col', '')
	covariates = config.get('variables', {}).get('covariates', [])

	# if event_col, duration_col, or covariates are not defined in the config, raise an error
	if not event_col or not duration_col or not covariates:
		raise ValueError(
			"event_col, duration_col, and covariates must be defined in the config file under 'variables' section.")

	data_path = config['paths']['processed_data']
	data = pd.read_parquet(data_path) if data_path.endswith('.parquet') else pd.read_csv(data_path)

	# Check if the specified columns exist in the data
	missing_cols = [col for col in [event_col, duration_col] + covariates if col not in data.columns]
	if missing_cols:
		raise ValueError(f"The following columns are missing in the data: {', '.join(missing_cols)}")

	X = data[covariates]

	# Go through each column and check if it's categorical, if yes, apply one-hot encoding
	for col in X.columns:
		num_unique_values = X[col].nunique()
		# check if each value in the column is a string, if yes, apply one-hot encoding
		if X[col].dtype == 'object' or X[col].dtype.name == 'category':
			# Apply one-hot encoding to the column
			encoder = OneHotEncoder()
			encoded_cols = encoder.fit_transform(X[[col]])
			# Create a DataFrame with the encoded columns and concatenate it with the original DataFrame
			encoded_df = pd.DataFrame(encoded_cols, columns=encoder.get_feature_names_out([col]), index=X.index)
			X = pd.concat([X.drop(columns=[col]), encoded_df], axis=1)

	# Lifetime value calculation and risk calculation for the complete dataset
	min_churn_duration_churned = data[duration_col].min()
	monthly_discount_factor = 1 / ((1 + config['lifetime_value_variables']['annual_discount_rate']) ** (1 / 12))
	monthly_spend = data['monthly_spend_estimated'].values
	All_LVT = []

	# calculate survival probability at each time point for the complete dataset
	surv_probs_complete = rsf.predict_survival_function(X)
	for i, surv_func in enumerate(surv_probs_complete):
		step = surv_func.x
		prob = surv_func.y
		curr_monthly_spend = monthly_spend[i]


		step = step.tolist()
		prob = prob.tolist()
		# reverse the order of step and prob so that the time points are in ascending order
		step = step[::-1]
		prob = prob[::-1]

		timeframe = config['treatment_costs']['time_horizon_months_for_customer_lifetime_value']
		prob = prob[0:timeframe]
		prob = prob[::-1]  # reverse the order of prob again to have the time points in ascending order
		lvt = 0
		for t, p in enumerate(prob):
			# calculate the discounted monthly spend at each time point and sum it up to get the lifetime value for each sample
			lvt += (curr_monthly_spend * (monthly_discount_factor ** t+1) * p)
		All_LVT.append(lvt)

	data['predicted_lifetime_value'] = All_LVT
	Final_data = data[['customer_id', 'subscription_id', 'subscription_date', 'churn_date', 'churned'] + [event_col,duration_col] + covariates + ['predicted_lifetime_value']]
	output_path = config['paths']['outputs_dir']
	# check if this folder exists, if not, create it
	if not os.path.exists(output_path):
		os.makedirs(output_path)
	output_file = os.path.join(output_path, f"Control_based_LTV.csv")
	Final_data.to_csv(output_file, index=False)

