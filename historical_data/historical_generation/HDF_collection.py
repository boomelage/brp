#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Oct  3 09:38:02 2024

"""
import os
import sys
import time
import pandas as pd
from datetime import datetime
from tqdm import tqdm
import matplotlib.pyplot as plt
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
grandparent_dir = os.path.dirname(parent_dir)
sys.path.append(parent_dir)
sys.path.append(grandparent_dir)

from model_settings import ms
os.chdir(current_dir)

print("\nimporting dataset(s)...\n")

"""
data selection
"""


barriers_dir = os.path.join(current_dir,'historical_barrier_generation')
sys.path.append(barriers_dir)
h5_file_path = os.path.join(barriers_dir,'SPX barriers.h5')



# vaillas_dir = os.path.join(current_dir,'historical_vanilla_generation')
# sys.path.append(vaillas_dir)
# h5_file_path = os.path.join(vaillas_dir,'SPX vanillas.h5')



""""""
collection_start_time = time.time()
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
if 'barrier_price' in contracts.columns:
    contracts = contracts[contracts['barrier_price']>0]
    contracts['observed_price'] = ms.noisyfier(contracts['barrier_price'])
    
elif 'heston_price' in contracts.columns:
    contracts = contracts[contracts['heston_price']>0]
    contracts['observed_price'] = ms.noisyfier(contracts['heston_price'])
else:
    raise ValueError('no price found in contracts dataset')

contracts = contracts.reset_index(drop=True)

pd.set_option("display.max_columns",None)
print(f"\n{contracts.describe()}")
print(f"\n{contracts.dtypes}\n")
collection_end_time = time.time()
collection_runtime = collection_end_time - collection_start_time
print(f"\nruntime: {round(collection_runtime,4)} seconds\n")



histogram_tolerance = 1
hist_contracts = contracts[
    'observed_price'][contracts['observed_price']>histogram_tolerance].copy()
percent_exclusion = 1-len(hist_contracts)/contracts['observed_price'].shape[0]

spots = contracts.set_index('calculation_date')['spot_price'].drop_duplicates()
plt.figure()
plt.plot(spots,color='black')
plt.xticks(rotation=45)
plt.title('SPX index time series considered')
plt.ylabel('spot price')
plt.yticks(rotation=45)
plt.show()
plt.clf()
plt.hist(
    hist_contracts,
    bins=int(round(len(contracts['observed_price'].values)**0.5,0))
    )
plt.title('distribution of observed option '
          f'prices greater than {histogram_tolerance}')
plt.xlabel('option price')
plt.ylabel(f'frequency ({round(percent_exclusion,2)}% of data excluded)')
plt.yticks(rotation = 45)
plt.show()
plt.clf()