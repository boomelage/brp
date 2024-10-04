#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Oct  3 09:38:02 2024

"""
import os
import re
import pandas as pd
from datetime import datetime
current_dir = os.path.dirname(os.path.abspath(__file__))
os.chdir(current_dir)
date_pattern = re.compile(r'date_(\d{4}_\d{2}_\d{2})')


def collect_dataframes_from_h5(h5_file_path, start_date, end_date):
    contracts = pd.DataFrame()
    with pd.HDFStore(h5_file_path, 'r') as hdf_store:
        for key in hdf_store.keys():
            match = date_pattern.search(key)
            if match:
                date_str = match.group(1)
                date_obj = datetime.strptime(date_str, "%Y_%m_%d")
                if start_date <= date_obj <= end_date:
                    try:
                        contracts = pd.concat(
                            [contracts, hdf_store.get(key)], 
                            ignore_index=True
                            )
                    except KeyError:
                        print(f"Key '{key}' not found in the HDF5 file.")
    return contracts


h5_file_path = 'SPX barriers.h5'
start_date = datetime.strptime("2006-02-01", "%Y-%m-%d")
end_date = datetime.strptime("2020-02-10", "%Y-%m-%d")
date_pattern = re.compile(r'date_(\d{4}_\d{2}_\d{2})')


print("\nimporting data...\n")

contracts = collect_dataframes_from_h5(
    h5_file_path, start_date,end_date
    ).drop_duplicates().reset_index(drop=True)



check = contracts[
    [
      'spot_price','strike_price', 'w','days_to_maturity',
     
       'barrier','barrier_type_name','barrier_price',

       # 'heston_price',
       
      
      ]
    ]

check.columns = [
    
    's','k','w','t',
    
    'B','type',
    
    'price'
    
    ]

pd.reset_option("display.max_rows")
pd.set_option("display.max_columns",None)
pd.set_option('display.float_format', '{:.6f}'.format)
print(f"\npreview:\n{check}")
print(f"\n{contracts.dtypes}")
print(f"\n{check.describe().iloc[1:]}")
print(f"\ntotal prices collected: {contracts.shape[0]}")

