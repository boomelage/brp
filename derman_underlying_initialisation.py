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

"""
# =============================================================================

"""
# from routine_ivol_collection import implied_vols
# timestamp = time.time()
# file_time = datetime.fromtimestamp(timestamp)
# ticker = r'SPX'
# file_tag = file_time.strftime("%Y-%m-%d %H-%M-%S")
# derman_ivols_filename = f"{ticker} {file_tag} derman_ts.csv"
# implied_vols.to_csv(derman_ivols_filename)

"""
# =============================================================================
                     loading Derman coefficients from OMON term strtucture data
"""

# derman_coefs = derman.get_derman_coefs()
# timestamp = time.time()
# file_time = datetime.fromtimestamp(timestamp)
# ticker = r'SPX'
# 
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
# ticker = r'SPX'
# file_tag = file_time.strftime("%Y-%m-%d %H-%M-%S")
# derman_data_filename = f"{ticker} {file_tag} derman_data.csv"
# # contract_details.to_csv(derman_data_filename)

# from routine_ivol_collection import implied_vols


"""
# =============================================================================
                                                   loading option data from csv
"""

contract_details = pd.read_csv(csvs[3])
contract_details.index = contract_details[contract_details.columns[0]]
contract_details = contract_details.drop(
    columns = contract_details.columns[0]).reset_index(drop=True)
S = np.sort(contract_details['spot_price'].unique().astype(int))
K = np.sort(contract_details['strike_price'].unique().astype(int))
T = np.sort(contract_details['days_to_maturity'].unique().astype(int))

"""
# =============================================================================
                                              importing term structure from csv                    
"""

implied_vols = pd.read_csv(csvs[2])
implied_vols = implied_vols.set_index(implied_vols.columns[0])
implied_vols.columns = implied_vols.columns.astype(int)
atm_vol_df = implied_vols.loc[int(min(S)):int(max(S))]

"""
# =============================================================================
                                           loading Derman coefficients from csv
"""

from Derman import retrieve_derman_from_csv
derman_coefs, derman_maturities = retrieve_derman_from_csv(dirdatacsv()[1])
