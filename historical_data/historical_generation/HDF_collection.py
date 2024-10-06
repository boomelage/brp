#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Oct  3 09:38:02 2024

"""
import os
import sys
import pandas as pd
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

"""
data selection
"""


barriers_dir = os.path.join(current_dir,'hirtorical_barrier_generation')
sys.path.append(barriers_dir)
h5_file_path = os.path.join(barriers_dir,'SPX barriers.h5')

# vanillas_dir = os.path.join(current_dir,'historical_vanilla_generation')
# sys.path.append(vanillas_dir)
# h5_file_path = os.path.join(vanillas_dir,'SPX vanillas.h5')

# sparse_vanillas_dir = os.path.join(
#     current_dir,'historical_vanilla_generation_sparse')
# sys.path.append(sparse_vanillas_dir)
# h5_file_path = os.path.join(sparse_vanillas_dir,'SPX vanillas sparse.h5')


""""""
with pd.HDFStore(h5_file_path, 'r') as hdf_store:
    keys = hdf_store.keys()

""""""
  
"""
date filter
"""
start_date = datetime.strptime("2007-01-01", "%Y-%m-%d")
end_date = datetime.strptime("2007-01-05", "%Y-%m-%d")
keys = pd.Series(keys)
date_pattern = r"(\d{4}_\d{2}_\d{2})"
extracted_dates = keys.str.extract(date_pattern, expand=False)
keys_dates = pd.to_datetime(
    extracted_dates, format="%Y_%m_%d", errors='coerce')
filtered_keys = keys[(keys_dates >= start_date) & (keys_dates <= end_date)]
filtered_keys.tolist()



"""
loading selection
"""
LOADING_KEYS = keys


bar = tqdm(desc='loading',total=len(LOADING_KEYS))
contracts_list = []
with pd.HDFStore(h5_file_path, 'r') as hdf_store: 
    for key in LOADING_KEYS:
        contracts_list.append(hdf_store.get(key))
        bar.update(1)
contracts = pd.concat(contracts_list, ignore_index=True)
bar.close()
contracts.dtypes


print('\npreparing data...\n')

try:
    contracts = contracts[contracts['barrier_price']>0].copy()
    contracts.loc[:,'observed_price'] = ms.noisy_prices(
        contracts.loc[:,'barrier_price'])
except Exception:
    contracts = contracts[contracts['heston_price']>0].copy()
    contracts.loc[:,'observed_price'] = ms.noisy_prices(
        contracts.loc[:,'heston_price'])


contracts.loc[:,'moneyness'] = ms.vmoneyness(
    contracts['spot_price'],
    contracts['strike_price'],
    contracts['w']
    )

contracts = contracts.reset_index(drop=True)

pd.set_option("display.max_columns",None)
print(f"\n{contracts.describe()}")
print(f"\n{contracts.dtypes}\n")

import matplotlib.pyplot as plt
spots = contracts.set_index('calculation_date')['spot_price'].copy().drop_duplicates()
plt.figure()
plt.plot(spots,color='black')
plt.xticks(rotation=45)
plt.show()
plt.clf()

