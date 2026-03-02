import pandas as pd
import numpy as np
import re
from dateutil import parser
from src.utils import load_config, pprint_df
import os



def churn_rate(df,col):
	value_counts = df[col].value_counts().sort_index()
	df_value_counts = pd.DataFrame(value_counts)
	df_value_counts = df_value_counts.reset_index()
	df_value_counts.columns = ['Category', 'Frequency']
	df_value_counts['Category'] = df_value_counts['Category'].map({0: 'Not Churned', 1: 'Churned'})
	df_value_counts['Percentage'] = (df_value_counts['Frequency'] / df_value_counts['Frequency'].sum() * 100).round(2).astype(str) + '%'
	df_value_counts = df_value_counts[['Category', 'Frequency', 'Percentage']]
	return df_value_counts
 
def clean_currency(value):
	if pd.isnull(value):
		return np.nan
	# Remove currency symbols and spaces
	value = str(value).replace('€', '').replace('$', '').replace('EUR', '').replace('eur', '').replace('USD', '').replace('usd', '').strip()
		
	if value.count(',') > 0 and value.count('.') == 0:
		value = value.replace('.', '')  # Remove thousand sep if any
		value = value.replace(',', '.')
	elif value.count(',') > 0 and value.count('.') > 0:
		# Assume comma is thousand sep, dot is decimal
		value = value.replace(',', '')
	value = re.sub(r'[^0-9.\-]', '', value)
	try:
		return float(value)
	except Exception:
		return np.nan

def clean_percentage(value):
	if pd.isnull(value):
		return np.nan
	value = str(value).replace('%', '').replace(',', '.').strip()
	try:
		v = float(value)
		if v > 1:
			v = v / 100.0
		return v
	except Exception:
		return np.nan

def clean_boolean(value):
	if pd.isnull(value):
		return np.nan
	if str(value).lower() in ['1', 'y', 'yes', 'true', 't']:
		return 1
	if str(value).lower() in ['0', 'n', 'no', 'false', 'f']:
		return 0
	


def clean_date(value):
	if pd.isnull(value) or str(value).strip().lower() in ['na', 'n/a', 'none', '']:
		return np.nan
	s = str(value).strip()
	# Remove time component if present (e.g., 2022-03-04 00:00:00 -> 2022-03-04)
	s = re.split(r'[ T]', s)[0] if re.match(r'.*\d{2}:\d{2}:\d{2}', s) else s
 
	# Try to extract year (should be 2 or 4 digits and > 2000)
	year_match = re.findall(r'(\d{2,4})', s)
	year = None
	# If any 4-digit year exists, use the last one
	four_digit_years = [y for y in year_match if len(y) == 4 and y.startswith('20')]
	if four_digit_years:
		year = int(four_digit_years[-1])
	else:
		# Otherwise, search for a 2-digit year and convert to 20xx
		for y in reversed(year_match):
			if len(y) == 2 and y.isdigit():
				y2 = int(y)
				if 0 <= y2 <= 99:
					year = 2000 + y2
					break
	if year and year < 2000:
		return np.nan
	# Split by common delimiters
	parts = re.split(r'[-/ .]', s)
	parts = [p for p in parts if p.isdigit()]
	if year and len(parts) >= 3:
		# Find the year part and its index
		year_idx = [i for i, p in enumerate(parts) if (len(p) == 4 and p.startswith('20'))]
		if len(year_idx) == 0:
			year_idx = [i for i, p in enumerate(parts) if (len(p) == 2 and p.isdigit() and 2000+int(p) == year)]
		if year_idx:
			y_idx = year_idx[-1]
			# Remove year from parts
			y = parts.pop(y_idx)
			# Convert two-digit year to 20xx
			if len(y) == 2 and y.isdigit():
				y = str(2000 + int(y))
		else:
			y = str(year)
		# Now, first two numbers are month and day, but may need to flip
		if len(parts) >= 2:
			a, b = int(parts[0]), int(parts[1])
			# If a > 12, it's day/month/year; flip
			if a > 12:
				day, month = a, b
			else:
				month, day = a, b
			try:
				dt = pd.Timestamp(year=int(y), month=int(month), day=int(day))
				return dt.strftime('%Y-%m-%d')
			except Exception:
				return np.nan
	# Fallback to dateutil parser
	try:
		dt = parser.parse(s, dayfirst=False, yearfirst=False, fuzzy=True)
		if dt.year < 2000:
			return np.nan
		return dt.strftime('%Y-%m-%d')
	except Exception:
		return np.nan

def clean_add_ons(value):
	if pd.isnull(value):
		return np.nan
	s = str(value).strip()
	if s == '':
		return np.nan
	# Split by comma and sum. if comma fails, try some other delimiters
	delimiters = [',', ';', '|', '/']
	for delim in delimiters:
		parts = s.split(delim)
		try:
			return sum([int(p) for p in parts if p.isdigit()])
		except Exception:
			continue
	return np.nan


def parse_dates_list(val):
	if pd.isnull(val) or val in ['[]', 'None', 'NA', 'N/A', '']:
		return np.nan
	# Try to extract dates
	dates = re.findall(r'\d{4}-\d{2}-\d{2}|\d{2}-\d{2}-\d{4}|\d{2}/\d{2}/\d{2,4}|[A-Za-z]+ \d{4}', str(val))
	return [clean_date(d) for d in dates if clean_date(d) is not np.nan]

def win_features(row):
		wins = row.get('dates_of_wins', np.nan)		
		if not isinstance(wins, list):
			return pd.Series([0]*7)
		win_dates = pd.to_datetime(wins, errors='coerce').dropna()
		if len(win_dates) == 0:
			return pd.Series([0]*7)
		churned = row.get('churned')
		churned = bool(churned) 
		if churned:
			end_date = pd.to_datetime(row.get('churn_date'), format='%Y-%m-%d', errors='coerce')
		else:
			end_date = pd.to_datetime('2023-06-30', format='%Y-%m-%d', errors='coerce')

		subscription_date = pd.to_datetime(row.get('subscription_date'), format='%Y-%m-%d', errors='coerce')
		# Calculate number of wins in last 3 months, 6 months, 9 months, 12 months
		wins_in_last_3 = win_dates[win_dates >= end_date - pd.DateOffset(months=3)].size
		wins_in_last_6 = win_dates[win_dates >= end_date - pd.DateOffset(months=6)].size
		wins_in_last_9 = win_dates[win_dates >= end_date - pd.DateOffset(months=9)].size
		wins_in_last_12 = win_dates[win_dates >= end_date - pd.DateOffset(months=12)].size
		# Calculate time to first win and time since last win in months
		minimum_win_date = win_dates.min()
		max_win_date = win_dates.max()
		time_to_first_win = (minimum_win_date - subscription_date).days / 30
		time_since_last_win = (end_date -max_win_date).days / 30
		# Calculate average time between wins in months
		if len(win_dates) > 1:
			win_dates_sorted = win_dates.sort_values()
			time_between_wins = win_dates_sorted.diff().dropna().days.values / 30
			avg_time_between = time_between_wins.mean()
		else:
			avg_time_between = 0
  
		return pd.Series([wins_in_last_3, wins_in_last_6, wins_in_last_9, wins_in_last_12, time_to_first_win, time_since_last_win, avg_time_between])

# Post-cleaning checks for column types/consistency
def check_date_format(series, col):
	# All non-null values should match yyyy-mm-dd
	mask = series.dropna().astype(str).str.match(r'\d{4}-\d{2}-\d{2}$')
	if not mask.all():
		bad_vals = series[~mask].dropna().unique()
		print(f"[WARNING] Column '{col}' has non-standard date values: {bad_vals}")

def check_numeric(series, col):
	# All non-null values should be float or int
	if not pd.api.types.is_numeric_dtype(series):
		print(f"[WARNING] Column '{col}' is not numeric after cleaning.")

def check_boolean(series, col):
	# All non-null values should be bool
	if not series.dropna().map(lambda x: isinstance(x, (bool, np.bool_))).all():
		bad_vals = series[~series.dropna().map(lambda x: isinstance(x, (bool, np.bool_)))].unique()
		print(f"[WARNING] Column '{col}' has non-boolean values: {bad_vals}")

def check_list_of_dates(series, col):
	# All non-null values should be list of yyyy-mm-dd or nan
	for v in series.dropna():
		if not (isinstance(v, list) and all(isinstance(d, str) and re.match(r'\d{4}-\d{2}-\d{2}$', d) for d in v)):
			print(f"[WARNING] Column '{col}' has non-list-of-date values: {v}")

def preprocess_data():
	config = load_config('configs/default.yaml')
	raw_data_path = config['paths']['raw_data']
	processed_data_path = config['paths']['processed_data']
	
	# check if the output file already exists, if yes, print a warning and overwrite
	if os.path.exists(processed_data_path):
		print(f"[WARNING] Processed data file '{processed_data_path}' already exists and will be overwritten.")
		# Optionally, you could add a prompt here to ask the user if they want to continue or not
		response = input("Do you want to continue? (y/n): ")
		#response = 'n'  # default to 'n' to avoid accidental overwriting during testing
		if response.lower() == 'n':
			print("Operation cancelled.")
			return

			
	df = pd.read_csv(raw_data_path, na_values=['NA', 'N/A', 'None', 'missing', 'NaN', 'nan', ''] )

	# Standardize columns
	df = df.rename(columns=lambda x: x.strip().lower().replace(' ', '_').replace('-', '_'))

	# Clean columns
	df['subscription_date'] = df['subscription_date'].apply(clean_date)
	df['churn_date'] = df['churn_date'].apply(clean_date)
	df['observation_end_date'] = df['observation_end_date'].apply(clean_date)

	
	if 'web_sessions_90d_raw' in df.columns:
		# Extract unique values with pattern digit+
		unique_vals = df['web_sessions_90d_raw'].dropna().unique()
		digit_plus_vals = [v for v in unique_vals if isinstance(v, str) and re.match(r'^\d+\+$', v)]
		if digit_plus_vals:
			# Extract digits and find the smallest
			digits = [int(re.match(r'^(\d+)\+$', v).group(1)) for v in digit_plus_vals]
			min_digit = min(digits)
			# Replace all such values with "More than {min_digit}"
			df['web_sessions_90d_raw'] = df['web_sessions_90d_raw'].apply(lambda x: f'More than {min_digit}' if isinstance(x, str) and re.match(r'^\d+\+$', x) else x)
			# if there are any value which are neither numeric nor "More than {min_digit}", make them nan
			df['web_sessions_90d_raw'] = df['web_sessions_90d_raw'].apply(lambda x: x if (pd.isnull(x) or isinstance(x, (int, float)) or (isinstance(x, str) and x.startswith('More than'))) else np.nan)

	# Replace 99 with nan in service_contacts_12m and complaints_12m
	for col in ['service_contacts_12m', 'complaints_12m']:
		if col in df.columns:
			df[col] = df[col].apply(lambda x: np.nan if x == 99 else x)

	# Numeric columns (except add_ons)
	for col in ['participant_age', 'extra_draws_per_year', 'failed_payments_12m', 'lifetime_wins']:
		if col in df.columns:
			# Make sure to convert to numeric, coercing errors to nan
			df[col] = pd.to_numeric(df[col], errors='coerce')

	# Clean add_ons column
	if 'add_ons' in df.columns:
		df['add_ons'] = df['add_ons'].apply(clean_add_ons)

	# Currency columns
	for col in ['monthly_spend_estimated', 'offer_cost_eur', 'historic_revenue_12m', 'revenue_next_12m_observed']:
		if col in df.columns:
			df[col] = df[col].apply(clean_currency)

	# Percentage columns
	if 'donation_share_charity' in df.columns:
		df['donation_share_charity'] = df['donation_share_charity'].apply(clean_percentage)

	# Boolean columns
	for col in ['treatment_sent_flag', 'churned']:
		if col in df.columns:
			df[col] = df[col].apply(clean_boolean)

   
	# Set churn_date to 2023-06-30 for non-churned using consistent date formatting
	churn_date_filled = df['churn_date'].copy()
	churn_date_fill_value = pd.Timestamp(year=2023, month=6, day=30).strftime('%Y-%m-%d')
	churn_date_filled[df['churned'] != 1] = churn_date_fill_value
	# Parse dates
	sub_dates = pd.to_datetime(df['subscription_date'], errors='coerce')
	churn_dates = pd.to_datetime(churn_date_filled, errors='coerce')
	# Calculate difference in months using year/month arithmetic
	duration_months = (
		(churn_dates.dt.year - sub_dates.dt.year) * 12 +
		(churn_dates.dt.month - sub_dates.dt.month) +
		(churn_dates.dt.day - sub_dates.dt.day) / 30
	)
	df['churn_duration_months'] = duration_months.round(0).astype('Int64')  # Round to nearest month and convert to integer type that allows NA
	# number of samples dropped due to negative or zero churn duration
	num_dropped = df.shape[0] - df[df['churn_duration_months'] > 0].shape[0]
	print(f"Number of samples dropped due to negative or zero churn duration: {num_dropped}")
 	# Remove any negative durations which indicate data issues
	df = df[df['churn_duration_months'] > 0]  
 
	df['dates_of_wins'] = df['dates_of_wins'].apply(parse_dates_list)

	df[['num_wins_last_3m', 'num_wins_last_6m', 'num_wins_last_9m', 'num_wins_last_12m', 'time_to_first_win', 'time_since_last_win', 'avg_time_between_wins']] = df.apply(win_features, axis=1)

	# number of sample dropped due to negative time_to_first_win or time_since_last_win
	num_dropped_time = df.shape[0] - df[(df['time_to_first_win'] >= 0) & (df['time_since_last_win'] >= 0)].shape[0]
	print(f"Number of samples dropped due to negative time_to_first_win or time_since_last_win: {num_dropped_time}")
	df = df[(df['time_to_first_win'] >= 0) & (df['time_since_last_win'] >= 0)]  # Remove any samples with negative time_to_first_win or time_since_last_win which indicate data issues

	# Collecting required columns for modeling
	Required_cols = ['customer_id', 'subscription_id', 'subscription_date', 'participant_age', 'marketing_channel', 'country_code', 'extra_draws_per_year', 'add_ons','payment_method', 'failed_payments_12m',
					 'monthly_spend_estimated', 'donation_share_charity','web_sessions_90d_raw', 'service_contacts_12m', 'complaints_12m', 'lifetime_wins', 'campaign_cohort', 'treatment_sent_flag', 'offer_cost_eur', 'historic_revenue_12m', 
					 'revenue_next_12m_observed', 'churned', 'churn_date', 'churn_duration_months','num_wins_last_3m', 'num_wins_last_6m', 'num_wins_last_9m', 'num_wins_last_12m', 'time_to_first_win', 'time_since_last_win', 'avg_time_between_wins']

	print(f"# of observation after data processing: {df.shape[0]}")
	df = df[Required_cols]
	pprint_df(churn_rate(df,'churned'))


	independent_vars = [col for col in Required_cols if col not in ['customer_id', 'subscription_id', 'subscription_date', 'churn_date', 'churned']]

	Missing_count_df = pd.DataFrame(columns=['Column','Missing_Count'])
	for index, col in enumerate(independent_vars):
		missing_count = df[col].isna().sum()
		Missing_count_df.loc[index] = [col, missing_count]

	print("\nMissing values count in independent variables:")
	pprint_df(Missing_count_df)

	Missing_count_df = Missing_count_df.sort_values(by='Missing_Count', ascending=False)
	Drop_Top_Cols = 2
	top_missing_cols = Missing_count_df['Column'].tolist()[0:Drop_Top_Cols]
	independent_vars = [col for col in independent_vars if col not in top_missing_cols]
	print(f"\nDropping top {Drop_Top_Cols} columns with highest missing values: {top_missing_cols}")

	# drop rows with any missing values in required columns
	df = df[['customer_id', 'subscription_id', 'subscription_date', 'churn_date', 'churned'] + independent_vars]
	df = df.dropna(subset=independent_vars)
	print(f"# of observation after dropping missing values: {df.shape[0]}")	
	pprint_df(churn_rate(df,'churned'))

	treatment_col = config['variables']['treatment_col']
	if treatment_col in df.columns:
		df[treatment_col] = df[treatment_col].str.lower()

	
	# Save processed data
	if processed_data_path.endswith('.parquet'):
		df.to_parquet(processed_data_path, index=False)
	else:
		df.to_csv(processed_data_path, index=False)
