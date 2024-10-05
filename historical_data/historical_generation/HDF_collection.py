#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Oct  3 09:38:02 2024

"""
import os
import sys
import pandas as pd
import numpy as np
from datetime import datetime
from tqdm import tqdm
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
grandparent_dir = os.path.dirname(parent_dir)
sys.path.append(parent_dir)
sys.path.append(grandparent_dir)

from settings import model_settings
ms = model_settings()
os.chdir(current_dir)

print("\nimporting dataset(s)...\n")

# start_date = datetime.strptime("2007-01-01", "%Y-%m-%d")
# end_date = datetime.strptime("2007-01-5", "%Y-%m-%d")



barriers_dir = os.path.join(current_dir,'historical_barrier_generation')
sys.path.append(barriers_dir)
h5_file_path = os.path.join(barriers_dir,'SPX barriers.h5')

# vanillas_dir = os.path.join(current_dir,'historical_vanilla_generation')
# sys.path.append(vanillas_dir)
# h5_file_path = os.path.join(vanillas_dir,'SPX vanillas.h5')




with pd.HDFStore(h5_file_path, 'r') as hdf_store:
    keys = hdf_store.keys()
    
    
# keys = pd.Series(keys)
# date_pattern = r"(\d{4}_\d{2}_\d{2})"
# extracted_dates = keys.str.extract(date_pattern, expand=False)
# keys_dates = pd.to_datetime(
#     extracted_dates, format="%Y_%m_%d", errors='coerce')
# filtered_keys = keys[(keys_dates >= start_date) & (keys_dates <= end_date)]
# filtered_keys.tolist()


LOADING_KEYS = keys


bar = tqdm(desc='loading',total=len(LOADING_KEYS))
contracts_list = []
with pd.HDFStore(h5_file_path, 'r') as hdf_store: 
    for key in LOADING_KEYS:
        contracts_list.append(hdf_store.get(key))
        bar.update(1)
contracts = pd.concat(contracts_list, ignore_index=True)
bar.close()

print('\npreparing data...\n')

try:
    contracts.loc[:,'observed_price'] = ms.noisy_prices(
        contracts.loc[:,'barrier_price'])
except Exception:
    contracts.loc[:,'observed_price'] = ms.noisy_prices(
        contracts.loc[:,'heston_price'])

contracts = contracts[contracts['observed_price']>0]


contracts.loc[:,'moneyness'] = ms.vmoneyness(
    contracts['spot_price'],
    contracts['strike_price'],
    contracts['w']
    )


contracts = contracts[
    ((contracts['moneyness'] <= 0.1) & (contracts['moneyness'] >= -0.1))
    ]

contracts['calculation_date'] = pd.to_datetime(
    contracts['calculation_date'],
    format="%Y_%m_%d"
    )

contracts['expiration_date'] = pd.to_datetime(
    contracts['expiration_date'],
    "%Y_%m_%d"
    )


pd.set_option("display.max_columns",None)
print(f"\n{contracts.describe()}")

