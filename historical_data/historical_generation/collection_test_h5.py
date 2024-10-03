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
        file_path,
        start_date=None,
        end_date=None
        ):
    
    with pd.HDFStore(file_path) as store:
        paths = store.keys()
        
    date_pattern = re.compile(r'date_(\d{4}_\d{2}_\d{2})')
    h5_keys = []
    
    for path in paths:
        match = date_pattern.search(path)
        if match:
            date_str = match.group(1)
            date_obj = datetime.strptime(date_str, "%Y_%m_%d")
            if start_date <= date_obj <= end_date:
                h5_keys.append(path)
    
    bar = tqdm(total=len(h5_keys),desc='collecting')
    contracts = pd.DataFrame()
    with pd.HDFStore(file_path, 'r') as hdf_store:
        for key in h5_keys:
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





h5filename = 'SPXbarriers.h5'

start_date = datetime.strptime("2000-01-01", "%Y-%m-%d")
end_date = datetime.strptime("2020-01-01", "%Y-%m-%d")


contracts = collect_dataframes_from_h5(
    h5filename,start_date,end_date
    ).drop_duplicates().reset_index(drop=True)


check = contracts[
    ['spot_price','strike_price','barrier',
     'barrier_price','barrier_type_name','w','days_to_maturity']
    ]

check.columns = ['s','k','B','price','type','w','t']
pd.set_option("display.max_rows",None)
pd.set_option("display.max_columns",None)
print(f"\n{check.sort_values(by='price')}\n")
pd.reset_option("display.max_rows")
pd.reset_option("display.max_columns")



