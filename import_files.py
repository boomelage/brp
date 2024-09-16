
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Sep 14 02:27:39 2024

"""
import os
import sys
current_dir = os.path.dirname(os.path.abspath(__file__))
ts_data_dir = os.path.join(current_dir,'term_structure')
cd_data_dir = os.path.join(current_dir,'contract_details')
sys.path.append(ts_data_dir)
sys.path.append(cd_data_dir)
os.chdir(current_dir)

import pandas as pd
import numpy as np
import time
from datetime import datetime

from data_query import dirdatacsv
csvs = dirdatacsv()
from settings import model_settings
ms = model_settings()
settings = ms.import_model_settings()
dividend_rate = settings[0]['dividend_rate']
risk_free_rate = settings[0]['risk_free_rate']
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
generic = f"{ticker} {date_tag} {time_tag}"

"""
                                                                         raw_ts
"""

from routine_ivol_collection import raw_ts
# raw_ts.drop_duplicates().to_csv(f"{generic} raw_ts.csv")
print(f'\n{raw_ts}')


"""
                                                            Derman coefficients
"""

# from routine_Derman import derman_coefs
# derman_coefs.to_csv(f"{generic} derman_coefs.csv")
# print(f"\n\nDerman coefficients:\n{derman_coefs}")

"""
                                                               contract_details
"""

# from routine_collection import contract_details
# contract_details.drop_duplicates().to_csv(f"{generic} contract_details.csv")
# print(f'\n{contract_details}')




print('files processed')


