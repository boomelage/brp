#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Sep 13 15:23:06 2024

"""
import os
pwd = str(os.path.dirname(os.path.abspath(__file__)))
os.chdir(pwd)

"""
"""

import time
import pandas as pd
import numpy as np
from datetime import datetime
from Derman import derman
from data_query import dirdata,dirdatacsv
csvs = dirdatacsv()
xlsxs = dirdata()
derman = derman()

ticker = r'SPX'

clean_ts_filename = csvs[1]
raw_ts_filname = csvs[0]
derman_coefs_filename = csvs[2]
# contract_details_filename = csvs[1]

"""
# =============================================================================

"""
# from routine_ivol_collection import raw_market_ts, clean_market_ts
# timestamp = time.time()
# file_time = datetime.fromtimestamp(timestamp)
# file_tag = file_time.strftime("%Y-%m-%d %H-%M-%S")
# clean_market_ts_name = f"{ticker} {file_tag} clean_ts.csv"
# raw_market_ts_name = f"{ticker} {file_tag} raw_ts.csv"
# raw_market_ts.to_csv(raw_market_ts_name)
# clean_market_ts.to_csv(clean_market_ts_name)

"""
# =============================================================================
                     loading Derman coefficients from OMON term strtucture data
"""

# derman_coefs = derman.get_derman_coefs()
# timestamp = time.time()
# file_time = datetime.fromtimestamp(timestamp)
# derman_filename = f"{ticker} {file_tag} derman_coefs.csv"
# derman_coefs.to_csv(derman_filename)

"""
# =============================================================================
                                                loading option data from market
"""

# from routine_collection import contract_details
# s = contract_details['spot_price'].unique()[0]
# K = contract_details['strike_price'].unique()
# T = contract_details['days_to_maturity'].unique()

# timestamp = time.time()
# file_time = datetime.fromtimestamp(timestamp)
# file_tag = file_time.strftime("%Y-%m-%d %H-%M-%S")
# derman_data_filename = f"{ticker} {file_tag} derman_data.csv"
# # contract_details.to_csv(derman_data_filename)

# from routine_ivol_collection import clean_market_ts


"""
# =============================================================================
                                                   loading option data from csv
"""

# contract_details = pd.read_csv(contract_details_filename)
# contract_details.index = contract_details[contract_details.columns[0]]
# contract_details = contract_details.drop(
#     columns = contract_details.columns[0]).reset_index(drop=True)
# S = np.sort(contract_details['spot_price'].unique().astype(int))
# K = np.sort(contract_details['strike_price'].unique().astype(int))
# T = np.sort(contract_details['days_to_maturity'].unique().astype(int))

"""
# =============================================================================
                                              importing term structure from csv                    
"""

clean_market_ts = pd.read_csv(clean_ts_filename)
clean_market_ts = clean_market_ts.set_index(clean_market_ts.columns[0])
# clean_market_ts.columns = clean_market_ts.columns.astype(int)
# atm_vol_df = clean_market_ts.loc[int(min(S)):int(max(S))]

"""
# =============================================================================
                                           loading Derman coefficients from csv
"""

from Derman import retrieve_derman_from_csv
derman_coefs, derman_maturities = retrieve_derman_from_csv(derman_coefs_filename)

"""
# =============================================================================
                                                 loading Derman historical data
"""

raw_market_ts = pd.read_csv(raw_ts_filname)


