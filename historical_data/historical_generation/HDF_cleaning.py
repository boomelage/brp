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

start_date = datetime.strptime("2011-01-01", "%Y-%m-%d")
end_date = datetime.strptime("2011-01-10", "%Y-%m-%d")

h5_file_path = 'SPX vanillas.h5'
with pd.HDFStore(h5_file_path, 'r') as hdf_store:
    keys = hdf_store.keys()

keys = pd.Series(keys)
date_pattern = r"(\d{4}_\d{2}_\d{2})"
extracted_dates = keys.str.extract(date_pattern, expand=False)
keys_dates = pd.to_datetime(extracted_dates, format="%Y_%m_%d", errors='coerce')
filtered_keys = keys[(keys_dates >= start_date) & (keys_dates <= end_date)]
filtered_keys.tolist()

bar = tqdm(desc='cleaning',total=len(keys))
with pd.HDFStore(h5_file_path, 'a') as hdf_store: 
    for key in keys:
        data = hdf_store.get(key)
        data['calculation_date'] = pd.to_datetime(data['calculation_date'])
        data['calculation_date'] = data['calculation_date'].dt.strftime('%Y-%m-%d')
        hdf_store.put(key, data, format='table', data_columns=True) 
        bar.update(1)
        
