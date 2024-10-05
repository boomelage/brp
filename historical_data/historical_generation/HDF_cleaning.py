#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Oct  3 09:38:02 2024

"""
import os
import pandas as pd


current_dir = os.path.dirname(os.path.abspath(__file__))
os.chdir(current_dir)

h5_file_path = 'SPX vanillas.h5'

"""
get key series
"""

# with pd.HDFStore(h5_file_path, 'r') as hdf_store:
#     keys = hdf_store.keys()
# keys = pd.Series(keys)


"""

filtering keys


"""    
# from datetime import datetime
# start_date = datetime.strptime("2011-01-01", "%Y-%m-%d")
# end_date = datetime.strptime("2011-01-10", "%Y-%m-%d")
# extracted_dates = keys.str.extract(date_pattern, expand=False)
# extracted_dates
# keys_dates = pd.to_datetime(extracted_dates, format="%Y_%m_%d", errors='coerce')
# filtered_keys = keys[(keys_dates >= start_date) & (keys_dates <= end_date)]
# filtered_keys.tolist()


"""

rename keys


"""

# with pd.HDFStore(h5_file_path, 'a') as hdf_store:
#     original_key = '/put/date_2010-11-04'
#     data = hdf_store.get(original_key)
    
#     new_key = '/put/date_2010_11_04'
#     hdf_store[new_key] = data

#     del hdf_store[original_key] 

#     print(f"Modified key is now: {new_key}")
    


"""

apply specific cleaning procedures


"""

# from tqdm import tqdm
# bar = tqdm(desc='cleaning',total=len(keys))
# with pd.HDFStore(h5_file_path, 'a') as hdf_store: 
#     for key in keys:
#         data = hdf_store.get(key)
#         data['expiration_date'] = pd.to_datetime(data['expiration_date'])
#         data['expiration_date'] = data['expiration_date'].dt.strftime('%Y-%m-%d')
#         hdf_store.put(key, data, format='table', data_columns=True) 
#         bar.update(1)
        



# from datetime import datetime 
# with pd.HDFStore(h5_file_path, 'a') as hdf_store:
#     data = hdf_store.get(hdf_store.keys()[756])
#     for i, row in data.iterrows():
#         exp = data.at[i, 'expiration_date']
#         exp_dt = datetime.strptime(exp, "%d/%m/%Y")
#         data.at[i, 'expiration_date'] = exp_dt.strftime('%Y-%m-%d')
#     hdf_store.put(hdf_store.keys()[756], data, format='table', data_columns=True)


print('\ncleaning complete!')