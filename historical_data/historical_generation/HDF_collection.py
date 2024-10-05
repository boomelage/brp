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

start_date = datetime.strptime("2007-01-01", "%Y-%m-%d")
end_date = datetime.strptime("2012-12-31", "%Y-%m-%d")

h5_file_path = 'SPX vanillas.h5'
with pd.HDFStore(h5_file_path, 'r') as hdf_store:
    keys = hdf_store.keys()

keys = pd.Series(keys)
date_pattern = r"(\d{4}_\d{2}_\d{2})"
extracted_dates = keys.str.extract(date_pattern, expand=False)
keys_dates = pd.to_datetime(extracted_dates, format="%Y_%m_%d", errors='coerce')
filtered_keys = keys[(keys_dates >= start_date) & (keys_dates <= end_date)]
filtered_keys.tolist()


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


check = contracts[
    [
      'spot_price','strike_price', 'w','days_to_maturity',
     
        # 'barrier','barrier_type_name','barrier_price',

        'heston_price',
      ]
    ]

check.columns = [
    
    's','k','w','t',
    
    # 'B','type',
    
    'price'
    
    ]

pd.reset_option("display.max_rows")
pd.set_option("display.max_columns",None)
pd.set_option('display.float_format', '{:.6f}'.format)
print(f"\npreview:\n{check}")
print(f"\n{contracts.dtypes}")
print(f"\n{check.describe().iloc[1:]}")
print(f"\ntotal prices collected: {contracts.shape[0]}")

S = np.sort(contracts['spot_price'].unique())
K = np.sort(contracts['strike_price'].unique())
T = np.sort(contracts['days_to_maturity'].unique())
W = np.sort(contracts['w'].unique())
n_calls = contracts[contracts['w']=='call'].shape[0]
n_puts = contracts[contracts['w']=='put'].shape[0]

print(f"\n{contracts}")
print(f"\n{contracts.describe()}\n")
print(f"\nspot(s):\n{S}\n\nstrikes:\n{K}\n\nmaturities:\n{T}\n\ntypes:\n{W}")
try:
    print(f"\n{contracts['barrier_type_name'].unique()}")
except Exception:
    pass
print(f"\nmoneyness:\n{np.sort(contracts['moneyness'].unique())}")
print(f"\nnumber of calls, puts:\n{n_calls},{n_puts}")
print(f"\ntotal prices:\n{contracts.shape[0]}\n")
pd.reset_option("display.max_columns")

