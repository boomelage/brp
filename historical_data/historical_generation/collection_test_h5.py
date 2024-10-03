#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Oct  3 09:38:02 2024

"""
import os
import re
import pandas as pd
from datetime import datetime
from tqdm import tqdm
current_dir = os.path.dirname(os.path.abspath(__file__))
os.chdir(current_dir)

def collect_dataframes_from_h5(file_path, keys):
    bar = tqdm(total=len(keys),desc='collecting')
    contracts = pd.DataFrame()
    with pd.HDFStore(file_path, 'r') as hdf_store:
        for key in keys:
            if key in hdf_store:
                contracts = pd.concat(
                    [contracts, hdf_store.get(key)],
                    ignore_index=True
                    )
                bar.update(1)
            else:
                print(f"Key '{key}' not found in the HDF5 file.")
                bar.update(1)
        bar.close()
    return contracts





with pd.HDFStore('SPXvanillas.h5') as store:
    paths = store.keys()


"""
put/call filter
"""
# pahts = [path for path in paths if path.startswith('/put')]
# paths = [path for path in paths if path.startswith('/call')]

"""
date filter
"""
start_date = datetime.strptime("2010-01-01", "%Y-%m-%d")
end_date = datetime.strptime("2011-01-01", "%Y-%m-%d")


""""""
date_pattern = re.compile(r'date_(\d{4}_\d{2}_\d{2})')
filtered_paths = []
for path in paths:
    match = date_pattern.search(path)
    if match:
        date_str = match.group(1)
        date_obj = datetime.strptime(date_str, "%Y_%m_%d")
        if start_date <= date_obj <= end_date:
            filtered_paths.append(path)
""""""        



contracts = collect_dataframes_from_h5('SPXvanillas.h5', filtered_paths)
