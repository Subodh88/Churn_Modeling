from causalml.match import create_table_one
import statsmodels.api as sm
import statsmodels.formula.api as smf
import pandas as pd
from sksurv.preprocessing import OneHotEncoder
from src.utils import load_config
from statsmodels.sandbox.regression.gmm import IV2SLS
from copy import deepcopy
import xgboost as xgb
import numpy as np

def estimate_ate_linear():
	config = load_config('configs/default.yaml')
	data_path = config['paths']['processed_data']
	df = pd.read_parquet(data_path) if data_path.endswith('.parquet') else pd.read_csv(data_path)

	# Controls
	controls = config['causal_inference']['control_variables_for_treatment']
	campaign_col = config['variables']['treatment_col']

	columns_to_drop = []
	columns_to_add = []
	

	for col in controls:
		if col in df.columns:
			if df[col].dtype == 'object' or df[col].dtype.name == 'category':
				# replace space with underscore in column values to avoid issues with formula parsing
				df[col] = df[col].str.replace(' ', '_')
				# Apply one-hot encoding to the column
				encoder = OneHotEncoder()
				encoded_cols = encoder.fit_transform(df[[col]])
				# Create a DataFrame with the encoded columns and concatenate it with the original DataFrame
				encoded_df = pd.DataFrame(encoded_cols, columns=encoder.get_feature_names_out([col]), index=df.index)
				columns=encoder.get_feature_names_out([col])
				# replace = with space in column names to avoid issues with formula parsing
				columns = [c.replace('=', '_') for c in columns]
				columns = [c.replace('-', '_') for c in columns]
				# replace space with underscore in column names to avoid issues with formula parsing
				columns = [c.replace(' ', '_') for c in columns]
				encoded_df.columns = columns

				df = pd.concat([df.drop(columns=[col]), encoded_df], axis=1)
				columns_to_drop.append(col)
				columns_to_add.extend(encoded_df.columns.tolist())
		else:
			print(f"[WARNING] Control variable '{col}' not found in the DataFrame. Skipping this variable.")
			columns_to_drop.append(col)

	controls = [c for c in controls if c not in columns_to_drop] + columns_to_add
	# lower case all values in campaign_cohort to avoid case sensitivity issues
	df[campaign_col] = df[campaign_col].str.lower()

	treatments = df[campaign_col].unique().tolist()
	treatments = [t for t in treatments if t.lower() != 'control']

	# create a mapping from treatments value to instrument value (e.g. variant_a to treatment_a) to avoid issues with formula parsing
	treatment_map = {t:f"treatment_{i+1}" for i,t in enumerate(treatments)}

	orignal_to_instrument_map = {treatment_map[t]: t for t in treatments}

	treatments_renamed = list(treatment_map.values())

	df[campaign_col] = df[campaign_col].map(treatment_map).fillna(df[campaign_col])  # map to treatment_x or keep as is if not in treatments

	for col in treatments_renamed:
		df[col] = (df[campaign_col] == col).astype(int)
	
	for instrument in treatments_renamed:
		df_curr = deepcopy(df)
		# only keep the rows which are control and the current treatment group
		df_curr = df_curr[df_curr[campaign_col].isin(['control', instrument])]
		table_one = create_table_one(data=df_curr, treatment_col=instrument, features=controls)
		print(f"\nCovariate balance for {orignal_to_instrument_map[instrument]}:")
		print(table_one)
	

	causal_inference_config = config['causal_inference']
	dp_variables_for_treatment = causal_inference_config['dp_variables_for_treatment']
	controls_formula = ' + '.join(controls)

	for curr_dp_variable in dp_variables_for_treatment:
		formula = f'{curr_dp_variable} ~ ' + ' + '.join([f'treatment_{i+1}' for i in range(len(treatments))]) + f' + {controls_formula}'
		# Fit model with clustered standard errors at customer_id level
		model = smf.ols(formula, data=df).fit(cov_type='cluster', cov_kwds={'groups': df['customer_id']})
		#print(f"'n ATE estimates with clustered SEs:")
		#print(model.summary().as_text())
		# Extract ATE estimates and clustered SEs for each treatment
		temp = {}
		for instrument in treatments_renamed:
			ate_A = model.params[instrument]	
			se_A = model.bse[instrument]
			temp[f"ATE_{orignal_to_instrument_map[instrument]}"] = (ate_A, se_A)
			
		results = {}
		results['linear_model'] = temp

		'''
		2SLS estimation for TOT:
		First stage: treatment_sent_flag ~ instrument (variant) + controls
		Second stage: churned/revenue_next_12m_observed ~ predicted_treatment_sent_flag + controls
		
		'''
		for instrument in treatments_renamed:
			df_curr = deepcopy(df)
			# only keep the rows which are control and the current treatment group
			df_curr =  df_curr[df_curr['campaign_cohort'].isin(['control', instrument])]
			# Prepare exog and instrument matrices
			exog = ['treatment_sent_flag'] + controls
			instr = [instrument] + controls
			# 2SLS estimation
			endog = df_curr[curr_dp_variable]
			exog_df = sm.add_constant(df_curr[exog])
			instr_df = sm.add_constant(df_curr[instr])
			model = IV2SLS(endog, exog_df, instr_df)
			model_fit = model.fit()  # 'cov_type' not supported for IV2SLS.fit()
			#print(f"\n2SLS results for {instrument.replace('treatment_', 'Variant_')}:")
			#print(model_fit.summary().as_text())
			# Extract TOT estimate and SE
			tot = model_fit.params['treatment_sent_flag']
			se_tot = model_fit.bse['treatment_sent_flag']
			results_dict = {
				'TOT': tot,
				'SE_TOT': se_tot,
			}
			results[orignal_to_instrument_map[instrument]] = results_dict

		
		# Nicely print the ITT and TOT results
		print(f"\nFinal ATE and TOT estimates for dependent variable: '{curr_dp_variable}':")
		for variant in treatments:
			ate = results['linear_model'][f'ATE_{variant}']
			tot = results[variant]['TOT']
			se_ate = results['linear_model'][f'ATE_{variant}'][1]
			se_tot = results[variant]['SE_TOT']
			print(f"{variant}: ATE = {ate[0]:.4f} (SE: {se_ate:.4f}), TOT = {tot:.4f} (SE: {se_tot:.4f})")


def estimate_ate_linear_segment(segment):
	config = load_config('configs/default.yaml')
	data_path = config['paths']['processed_data']
	df = pd.read_parquet(data_path) if data_path.endswith('.parquet') else pd.read_csv(data_path)
	# Controls
	controls = config['causal_inference']['control_variables_for_treatment']
	campaign_col = config['variables']['treatment_col']

	columns_to_drop = []
	columns_to_add = []
	

	for col in controls:
		if col in df.columns:
			if df[col].dtype == 'object' or df[col].dtype.name == 'category':
				# replace space with underscore in column values to avoid issues with formula parsing
				df[col] = df[col].str.replace(' ', '_')
				# Apply one-hot encoding to the column
				encoder = OneHotEncoder()
				encoded_cols = encoder.fit_transform(df[[col]])
				# Create a DataFrame with the encoded columns and concatenate it with the original DataFrame
				encoded_df = pd.DataFrame(encoded_cols, columns=encoder.get_feature_names_out([col]), index=df.index)
				columns=encoder.get_feature_names_out([col])
				# replace = with space in column names to avoid issues with formula parsing
				columns = [c.replace('=', '_') for c in columns]
				columns = [c.replace('-', '_') for c in columns]
				# replace space with underscore in column names to avoid issues with formula parsing
				columns = [c.replace(' ', '_') for c in columns]
				encoded_df.columns = columns

				df = pd.concat([df.drop(columns=[col]), encoded_df], axis=1)
				columns_to_drop.append(col)
				columns_to_add.extend(encoded_df.columns.tolist())
		else:
			print(f"[WARNING] Control variable '{col}' not found in the DataFrame. Skipping this variable.")
			columns_to_drop.append(col)

	controls = [c for c in controls if c not in columns_to_drop] + columns_to_add
	# lower case all values in campaign_cohort to avoid case sensitivity issues
	df[campaign_col] = df[campaign_col].str.lower()

	treatments = df[campaign_col].unique().tolist()
	treatments = [t for t in treatments if t.lower() != 'control']

	# create a mapping from treatments value to instrument value (e.g. variant_a to treatment_a) to avoid issues with formula parsing
	treatment_map = {t: f"treatment_{i + 1}" for i, t in enumerate(treatments)}

	orignal_to_instrument_map = {treatment_map[t]: t for t in treatments}

	treatments_renamed = list(treatment_map.values())

	df[campaign_col] = df[campaign_col].map(treatment_map).fillna(df[campaign_col])  # map to treatment_x or keep as is if not in treatments

	for col in treatments_renamed:
		df[col] = (df[campaign_col] == col).astype(int)

	Curr_Segment = segment
	# Check if the segment column exists and is a numeric column
	if Curr_Segment not in df.columns:
		raise ValueError(f"Segment column '{Curr_Segment}' not found in the DataFrame.")
	if df[Curr_Segment].dtype == 'object' or df[Curr_Segment].dtype.name == 'category':
		segment_values = df[Curr_Segment].unique().tolist()
		segment_names = []
		for index, value in enumerate(segment_values):
			df[f"Segment_{index+1}"] = (df[Curr_Segment] == value).astype(int)
			segment_names.append(f"Segment_{index+1}")
		print(f"Segment '{Curr_Segment}' is categorical with {len(segment_values)} unique values: {segment_values}")

	else:
		segment_values = df[Curr_Segment].unique().tolist()
		if len(segment_values) > 10:
			bins = 3
			quantile_points = [0] + [i*(1/bins) for i in range(1, bins)] + [1]
			# get the quantile values for the specified quantiles
			quantile_values = df[Curr_Segment].quantile(quantile_points).values
			bin_ranges = []
			seg_number = 1
			segment_names = []
			for ll,ul in zip(quantile_values[:-1], quantile_values[1:]):
				# for the last segment, include the upper bound to capture all values
				if ul == quantile_values[-1]:
					df[f"Segment_{seg_number}"] = ((df[Curr_Segment] >= ll) & (df[Curr_Segment] <= ul)).astype(int)
					bin_ranges.append(f"[{ll:.2f}, {ul:.2f}]")
					segment_names.append(f"Segment_{seg_number}")
				else:
					df[f"Segment_{seg_number}"] = ((df[Curr_Segment] >= ll) & (df[Curr_Segment] < ul)).astype(int)
					bin_ranges.append(f"[{ll:.2f}, {ul:.2f})")
					segment_names.append(f"Segment_{seg_number}")
				seg_number += 1		
			
			print(f"Segment '{Curr_Segment}' binned into {bins} bins with ranges: {bin_ranges}")
		else:
			df[Curr_Segment] = df[Curr_Segment].astype(str)
			segment_names = []
			for index, value in enumerate(segment_values):
				df[f"Segment_{index + 1}"] = (df[Curr_Segment] == value).astype(int)
				segment_names.append(f"Segment_{index + 1}")
			print(f"Segment '{Curr_Segment}' has {len(segment_values)} unique values: {segment_values}")
		
	 
	causal_inference_config = config['causal_inference']
	dp_variables_for_treatment = causal_inference_config['dp_variables_for_treatment']
	controls_formula = ' + '.join(controls)

	for curr_dp_variable in dp_variables_for_treatment:
		results = {}
		for seg in segment_names:
			df_seg = deepcopy(df)
			df_seg = df_seg[df_seg[seg] == 1]
			print(f"\nEstimating ATE and TOT for segment: {seg}")	
			# Covariate balance check
			for instrument in treatments_renamed:
				df_curr = deepcopy(df_seg)
				# only keep the rows which are control and the current treatment group
				df_curr = df_curr[df_curr['campaign_cohort'].isin(['control', instrument])]
				table_one = create_table_one(data=df_curr, treatment_col=instrument, features=controls)
				print(f"\nCovariate balance for {orignal_to_instrument_map[instrument]}:")
				print(table_one)
			# Formula
			controls_formula = ' + '.join(controls)
			formula = f'{curr_dp_variable} ~ ' + ' + '.join([f'treatment_{i+1}' for i in range(len(treatments))]) + f' + {controls_formula}'
			# Fit model with clustered standard errors at customer_id level
			try:
				model = smf.ols(formula, data=df_seg).fit(cov_type='cluster', cov_kwds={'groups': df['customer_id']})
			except:
				model = smf.ols(formula, data=df_seg).fit()  # fallback to regular OLS if clustering fails
			#print(f"'n ATE estimates with clustered SEs:")
			#print(model.summary().as_text())
			# Extract ATE estimates and clustered SEs
			temp = {}
			for instrument in treatments_renamed:
				ate_A = model.params[instrument]
				se_A = model.bse[instrument]
				temp[f"ATE_{orignal_to_instrument_map[instrument]}"] = (ate_A, se_A)

			if seg not in results:
				results[seg] = {}

			
			results[seg]['linear_model'] = temp

			'''
			2SLS estimation for TOT:
			First stage: treatment_sent_flag ~ instrument (variant) + controls
			Second stage: churned ~ predicted_treatment_sent_flag + controls
			
			'''
			for instrument in treatments_renamed:
				df_curr = deepcopy(df_seg)
				# only keep the rows which are control and the current treatment group
				df_curr = df_curr[df_curr['campaign_cohort'].isin(['control', instrument])]
				# Prepare exog and instrument matrices
				exog = ['treatment_sent_flag'] + controls
				instr = [instrument] + controls
				# 2SLS estimation
				endog = df_curr[curr_dp_variable]
				exog_df = sm.add_constant(df_curr[exog])
				instr_df = sm.add_constant(df_curr[instr])
				model = IV2SLS(endog, exog_df, instr_df)
				model_fit = model.fit()  # 'cov_type' not supported for IV2SLS.fit()
				#print(f"\n2SLS results for {instrument.replace('treatment_', 'Variant_')}:")
				#print(model_fit.summary().as_text())
				# Extract TOT estimate and SE
				tot = model_fit.params['treatment_sent_flag']
				se_tot = model_fit.bse['treatment_sent_flag']
				results_dict = {
					'TOT': tot,
					'SE_TOT': se_tot,
				}
				results[seg][orignal_to_instrument_map[instrument]] = results_dict

		
		# Nicely print the ITT and TOT results
		print(f"\nFinal ATE and TOT estimates for dependent variable: '{curr_dp_variable}'")
		for seg in segment_names:
			print(f"\nSegment: {seg}")
			for variant in treatments:
				ate = results[seg]['linear_model'][f'ATE_{variant}']
				tot = results[seg][variant]['TOT']
				se_ate = results[seg]['linear_model'][f'ATE_{variant}'][1]
				se_tot = results[seg][variant]['SE_TOT']
				print(f"{variant}: ATE = {ate[0]:.4f} (SE: {se_ate:.4f}), TOT = {tot:.4f} (SE: {se_tot:.4f})")

		print("\nSummary of significant differences between segments:")
		for variant in treatments:
			print(f"\nComparing segments for {variant}:")
			for i in range(len(segment_names)):
				for j in range(i+1, len(segment_names)):
					seg_i = segment_names[i]
					seg_j = segment_names[j]
					ate_i = results[seg_i]['linear_model'][f'ATE_{variant}']
					ate_j = results[seg_j]['linear_model'][f'ATE_{variant}']
					se_i = results[seg_i]['linear_model'][f'ATE_{variant}'][1]
					se_j = results[seg_j]['linear_model'][f'ATE_{variant}'][1]
					ci_i = (ate_i[0] - 1.96*se_i, ate_i[0] + 1.96*se_i)
					ci_j = (ate_j[0] - 1.96*se_j, ate_j[0] + 1.96*se_j)
					significant = not (ci_i[1] < ci_j[0] or ci_j[1] < ci_i[0])
					print(f"Segments {seg_i} vs {seg_j}: ATE difference significant? {'Yes' if significant else 'No'}")


def ate_t_learner():
	config = load_config('configs/default.yaml')
	data_path = config['paths']['processed_data']
	df = pd.read_parquet(data_path) if data_path.endswith('.parquet') else pd.read_csv(data_path)

	# Controls
	# Controls
	controls = config['causal_inference']['control_variables_for_treatment']
	campaign_col = config['variables']['treatment_col']

	columns_to_drop = []
	columns_to_add = []
	

	for col in controls:
		if col in df.columns:
			if df[col].dtype == 'object' or df[col].dtype.name == 'category':
				# replace space with underscore in column values to avoid issues with formula parsing
				df[col] = df[col].str.replace(' ', '_')
				# Apply one-hot encoding to the column
				encoder = OneHotEncoder()
				encoded_cols = encoder.fit_transform(df[[col]])
				# Create a DataFrame with the encoded columns and concatenate it with the original DataFrame
				encoded_df = pd.DataFrame(encoded_cols, columns=encoder.get_feature_names_out([col]), index=df.index)
				columns=encoder.get_feature_names_out([col])
				# replace = with space in column names to avoid issues with formula parsing
				columns = [c.replace('=', '_') for c in columns]
				columns = [c.replace('-', '_') for c in columns]
				# replace space with underscore in column names to avoid issues with formula parsing
				columns = [c.replace(' ', '_') for c in columns]
				encoded_df.columns = columns

				df = pd.concat([df.drop(columns=[col]), encoded_df], axis=1)
				columns_to_drop.append(col)
				columns_to_add.extend(encoded_df.columns.tolist())
		else:
			print(f"[WARNING] Control variable '{col}' not found in the DataFrame. Skipping this variable.")
			columns_to_drop.append(col)

	controls = [c for c in controls if c not in columns_to_drop] + columns_to_add
	# Covariate balance check
	# lower case all values in campaign_cohort to avoid case sensitivity issues
	df[campaign_col] = df[campaign_col].str.lower()

	treatments = df[campaign_col].unique().tolist()
	treatments = [t for t in treatments if t.lower() != 'control']

	# create a mapping from treatments value to instrument value (e.g. variant_a to treatment_a) to avoid issues with formula parsing
	treatment_map = {t: f"treatment_{i + 1}" for i, t in enumerate(treatments)}

	orignal_to_instrument_map = {treatment_map[t]: t for t in treatments}

	treatments_renamed = list(treatment_map.values())

	df[campaign_col] = df[campaign_col].map(treatment_map).fillna(df[campaign_col])  # map to treatment_x or keep as is if not in treatments

	for col in treatments_renamed:
		df[col] = (df[campaign_col] == col).astype(int)

	for instrument in treatments_renamed:
		df_curr = deepcopy(df)
		# only keep the rows which are control and the current treatment group
		df_curr = df_curr[df_curr[campaign_col].isin(['control', instrument])]
		table_one = create_table_one(data=df_curr, treatment_col=instrument, features=controls)
		print(f"\nCovariate balance for {orignal_to_instrument_map[instrument]}:")
		print(table_one)

	causal_inference_config = config['causal_inference']
	dp_variables_for_treatment = causal_inference_config['dp_variables_for_treatment']
	output_folder = config['paths']['outputs_dir']
	

	for curr_dp_variable in dp_variables_for_treatment:
		# Only binary treatment supported, so estimate ATE for Variant_A vs Control and Variant_B vs Control
		results = {}
		for instrument in treatments_renamed:
			df_curr = deepcopy(df)
			df_curr =  df_curr[df_curr[campaign_col].isin(['control', instrument])]

			df_treated = deepcopy(df_curr)
			df_control = deepcopy(df_curr)
			df_treated = df_treated[df_treated[instrument] == 1]
			df_control = df_control[df_control[instrument] == 0]

			model_treated = xgb.XGBRegressor()
			model_treated.fit(df_treated[controls], df_treated[curr_dp_variable])

			# Training the model for the untreated group
			model_control = xgb.XGBRegressor()
			model_control.fit(df_control[controls], df_control[curr_dp_variable])

			df_curr['pred_treated'] = model_treated.predict(df_curr[controls])
			df_curr['pred_control'] = model_control.predict(df_curr[controls])
			df_curr['treatment_effect'] = df_curr['pred_treated'] - df_curr['pred_control']

			ate_mean = df_curr['treatment_effect'].mean()
			ate_se = df_curr['treatment_effect'].std() / np.sqrt(len(df_curr))
			ate_ci_lower = ate_mean - 1.96 * ate_se
			ate_ci_upper = ate_mean + 1.96 * ate_se
			ate_t = (ate_mean, ate_ci_lower, ate_ci_upper)
			results[orignal_to_instrument_map[instrument]] = ate_t
			print(f"\nT-learner ATE estimate for {orignal_to_instrument_map[instrument]} vs Control : Dependent variable: '{curr_dp_variable}'")
			print(f'ATE estimate   : {ate_t[0]:.04f}')
			print(f'ATE lower bound: {ate_t[1]:.04f}')
			print(f'ATE upper bound: {ate_t[2]:.04f}')

			# append the ITE values to the original dataframe for further analysis if needed
			df_curr[f'ITE_{orignal_to_instrument_map[instrument]}'] = df_curr['treatment_effect']

			# Write the df_curr to a csv file for further analysis if needed
			#output_path = f"{output_folder}/ite_{orignal_to_instrument_map[instrument]}_{curr_dp_variable}.csv"
			#df_curr.to_csv(output_path, index=False)

			# Predict outcomes under treatment
			y_pred_treated = model_treated.predict(df[controls])
			# Predict outcomes under control
			y_pred_control = model_control.predict(df[controls])
			# Individual treatment effect (ITE) for each observation in full sample
			ite_full = y_pred_treated - y_pred_control
			# Save ITEs to output
			df_out = df.copy()
			df_out[f'ITE_{orignal_to_instrument_map[instrument]}'] = ite_full
			output_path_full = f"{output_folder}/ATE_Full_Sample_{orignal_to_instrument_map[instrument]}_{curr_dp_variable}.csv"
			df_out.to_csv(output_path_full, index=False)
			print(f"Full-sample ITEs for {curr_dp_variable} saved to {output_path_full}")




	

		





