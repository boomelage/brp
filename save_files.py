#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Sep 14 02:04:00 2024

"""

import os
pwd = str(os.path.dirname(os.path.abspath(__file__)))
os.chdir(pwd)

import time
from datetime import datetime

ticker = r"SPX"

file_time = time.time()
file_datetime = datetime.fromtimestamp(file_time)
time_tag = file_datetime.strftime('%H-%M-%S')
date_tag = '2024-09-13'
generic = f"{ticker} {date_tag} {time_tag}"

from routine_ivol_collection import raw_ts
raw_ts.drop_duplicates().to_csv(f"{generic} raw_ts.csv")

from rountine_Derman import raw_ts, derman_ts, derman_coefs
derman_ts.drop_duplicates().to_csv(f"{generic} derman_ts.csv")
derman_coefs.drop_duplicates().to_csv(f"{generic} derman_coefs.csv")

# from routine_collection import collect_directory_market_data
# contract_details = collect_directory_market_data()
# contract_details.drop_duplicates().to_csv(f"{generic} contract_details.csv")

print('files saved')