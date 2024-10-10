# -*- coding: utf-8 -*-
"""
Created on Tue Oct  8 21:00:01 2024

@author: boomelage
"""
import os
current_dir = os.path.dirname(os.path.abspath(__file__))
os.chdir(current_dir)
from historical_alphaVantage_collection import chain
import pandas as pd
from tqdm import tqdm

chain_keys = []
for key in chain.keys():
    chain_keys.append(key)

store = pd.HDFStore(r'alphaVantage vanillas.h5')
bar = tqdm(total=len(chain_keys))
for date in chain_keys:
    hdf_datekey = str('date_'+key.replace('-','_'))
    
    store.append(
        f"{hdf_datekey}/raw_data",chain[date]['raw_data'],
        format='table', append=True
        )
    store.append(
        f"{hdf_datekey}/surface",chain[date]['surface'],
        format='table', append=True
        )
    store.append(
        f"{hdf_datekey}/hottest_contracts",chain[date]['hottest_contracts'],
        format='table', append=True
        )

    bar.update(1)
bar.close()

bar = tqdm(total=len(store.keys()))
for key in store.keys():
    cleaned = store[key].drop_duplicates(keep='first')
    store.put(key,cleaned,format='table',append=False)
    bar.update(1)
bar.close()
    
store.close()


