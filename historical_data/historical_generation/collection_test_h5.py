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

def collect_dataframes_from_h5(
        paths,
        h5_file_path,
        start_date=None,
        end_date=None
        ):
    
    
    for path in paths:
        match = date_pattern.search(path)
        if match:
            date_str = match.group(1)
            date_obj = datetime.strptime(date_str, "%Y_%m_%d")
            if start_date <= date_obj <= end_date:
                paths.append(path)
    
    bar = tqdm(total=len(paths),desc='collecting')
    contracts = pd.DataFrame()
    with pd.HDFStore(h5_file_path, 'r') as hdf_store:
        for key in paths:
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





h5_file_path = 'SPXvanillas.h5'
start_date = datetime.strptime("2011-01-01", "%Y-%m-%d")
end_date = datetime.strptime("2011-01-06", "%Y-%m-%d")
date_pattern = re.compile(r'date_(\d{4}_\d{2}_\d{2})')



paths = []
with pd.HDFStore(h5_file_path) as store:
    paths = store.keys()

contracts = collect_dataframes_from_h5(
    paths, h5_file_path, start_date,end_date
    ).drop_duplicates().reset_index(drop=True)

check = contracts[
    [
    
     'spot_price','strike_price',
     
     # 'barrier','barrier_price','barrier_type_name',
     
     'heston_price',
     
     'w','days_to_maturity'
     
     ]
    ]

check.columns = [
    
    's','k','price',
    
    # 'B','type',
    
    'w','t'
    
    ]


pd.set_option("display.max_rows",None)
pd.set_option("display.max_columns",None)
print(f"\n{check.head(100)}\n")
pd.reset_option("display.max_rows")
pd.reset_option("display.max_columns")


