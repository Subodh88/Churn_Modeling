import yaml
from tabulate import tabulate
from datetime import datetime
import calendar

def load_config(config_path):
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)

def pprint_df(dframe,showindex=False):
	print(tabulate(dframe, headers='keys', tablefmt='psql', showindex=showindex))


def get_current_date_time():
    currentDay = datetime.now().day
    currentMonth = datetime.now().month
    currentYear = datetime.now().year
    month_eng = calendar.month_name[currentMonth]
    final_date = (month_eng + '_'+ str(currentDay) + '_'+ str(currentYear))
    now = datetime.now()
    date_time_str = now.strftime("%H_%M_%S")
    return final_date + '_' + date_time_str