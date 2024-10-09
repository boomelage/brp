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
for key in chain_keys:
    hdf_datekey = str('date_'+key.replace('-','_'))
    store.append(
        f"call/contracts/{hdf_datekey}",chain[key]['call']['contracts'],
        format='table', append=True
        )
    store.append(
        f"call/surface/{hdf_datekey}",chain[key]['call']['surface'],
        format='table', append=True
        )
    store.append(
        f"put/contracts/{hdf_datekey}",chain[key]['put']['contracts']
        )
    store.append(
        f"put/surface/{hdf_datekey}",chain[key]['put']['surface']
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


