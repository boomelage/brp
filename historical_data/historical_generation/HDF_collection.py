#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Oct  3 09:38:02 2024

"""
import os
import pandas as pd
from datetime import datetime
from tqdm import tqdm
current_dir = os.path.dirname(os.path.abspath(__file__))
os.chdir(current_dir)
print("\nimporting data...\n")


start_date = datetime.strptime("2000-01-01", "%Y-%m-%d")
end_date = datetime.strptime("2020-02-10", "%Y-%m-%d")

h5_file_path = 'SPX vanillas.h5'
with pd.HDFStore(h5_file_path, 'r') as hdf_store:
    keys = hdf_store.keys()

keys = pd.Series(keys)
date_pattern = r"(\d{4}_\d{2}_\d{2})"
extracted_dates = keys.str.extract(date_pattern, expand=False)
keys_dates = pd.to_datetime(extracted_dates, format="%Y_%m_%d", errors='coerce')
filtered_keys = keys[(keys_dates >= start_date) & (keys_dates <= end_date)]
filtered_keys.tolist()


LOADING_KEYS = filtered_keys

bar = tqdm(desc='loading',total=len(LOADING_KEYS))
contracts_list = []
with pd.HDFStore(h5_file_path, 'r') as hdf_store: 
    for key in LOADING_KEYS:
        contracts_list.append(hdf_store.get(key))
        bar.update(1)
contracts = pd.concat(contracts_list, ignore_index=True)
bar.close()

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

