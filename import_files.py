
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Sep 14 02:27:39 2024

"""
import os
pwd = str(os.path.dirname(os.path.abspath(__file__)))
os.chdir(pwd)

import pandas as pd
import numpy as np
import time
from datetime import datetime

from data_query import dirdatacsv
csvs = dirdatacsv()
from settings import model_settings
ms = model_settings()
settings = ms.import_model_settings()
security_settings = settings[0]['security_settings']
s = security_settings[5]
ticker = security_settings[0]
lower_moneyness = security_settings[1]
upper_moneyness = security_settings[2]
lower_maturity = security_settings[3]
upper_maturity = security_settings[4]
day_count = settings[0]['day_count']
calendar = settings[0]['calendar']
calculation_date = settings[0]['calculation_date']


file_time = time.time()
file_datetime = datetime.fromtimestamp(file_time)
time_tag = file_datetime.strftime('%H-%M-%S')
date_tag = file_datetime.strftime('%Y-%m-%d')
date_tag = '2024-09-13'
generic = f"{ticker} {date_tag} {time_tag}"

"""
# =============================================================================
                                                                   import files
                                                                   
                                                                         raw_ts
"""

rawtsname = [file for file in csvs if 'raw_ts' in file][0]
raw_ts = pd.read_csv(rawtsname).drop_duplicates()
raw_ts = raw_ts.rename(
    columns={raw_ts.columns[0]: 'Strike'}).set_index('Strike')
raw_ts.columns = raw_ts.columns.astype(int)
raw_ts = raw_ts.loc[
    lower_moneyness:upper_moneyness,
    lower_maturity:upper_maturity]
raw_ts = raw_ts.replace(0,np.nan)

"""
                                                               contract_details
"""

# contract_details_name = [
#     file for file in csvs if 'contract_details' in file][0]
# contract_details = pd.read_csv(contract_details_name).drop_duplicates()
# contract_details = contract_details.drop(
#     columns = contract_details.columns[0])

"""
                                                                         Derman
"""

# derman_ts_name = [file for file in csvs if 'derman_coefs' in file][0]
# derman_coefs = pd.read_csv(derman_ts_name).set_index('coef')


"""
# =============================================================================
#                                                                    save files

                                                                         raw_ts
"""

# raw_ts.drop_duplicates().to_csv(f"{generic} raw_ts.csv")
# raw_ts.drop_duplicates().to_csv(f"{generic} raw_ts.csv")

"""
                                                                         Derman
"""

# from routine_Derman import derman_coefs
# derman_coefs.drop_duplicates().to_csv(f"{generic} derman_coefs.csv")

"""
                                                               contract_details
"""

# from routine_collection import collect_directory_market_data
# contract_details = collect_directory_market_data()                                                             
# contract_details.drop_duplicates().to_csv(f"{generic} contract_details.csv")

print('files processed')


