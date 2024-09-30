# -*- coding: utf-8 -*-
"""
Created on Thu Sep 26 01:57:11 2024

@author: boomelage
"""

import os
import sys
import pandas as pd
from tqdm import tqdm
from datetime import datetime
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
historical_data_dir = os.path.join(
    parent_dir,'historical_data','historical_vanillas')
sys.path.append(parent_dir)
sys.path.append(historical_data_dir)
from data_query import dirdatacsv
COLLECTION_DIRECTORY = historical_data_dir
os.chdir(COLLECTION_DIRECTORY)

csvs = dirdatacsv()

def filter_by_date(file_list, start_date=None, end_date=None):
    filtered_files = []
    
    start_date = datetime.strptime(start_date, '%Y-%m-%d') if start_date else None
    end_date = datetime.strptime(end_date, '%Y-%m-%d') if end_date else None
    
    for file_name in file_list:
        file_date_str = file_name.split()[0]
        file_date = datetime.strptime(file_date_str, '%Y-%m-%d')
        
        if (not start_date or file_date >= start_date) and (not end_date or file_date <= end_date):
            filtered_files.append(file_name)
    
    return filtered_files

csvs = filter_by_date(csvs, start_date='2010-01-01', end_date='2010-03-01')

print('\nloading data...\n')
price_counter = 0
file_bar = tqdm(
    desc='collecting',total=len(csvs),unit='files',postfix=price_counter,
    bar_format= str('{percentage:3.0f}% | {n_fmt}/{total_fmt} {unit} | {rate_fmt} '
    '| Elapsed: {elapsed} | Remaining: {remaining} | Prices: {postfix}'))
train_vanillas = pd.DataFrame()
for file in csvs:
    vanilla_subset = pd.read_csv(file)
    train_vanillas = pd.concat([train_vanillas,vanilla_subset],ignore_index=True)
    price_counter += vanilla_subset.shape[0]
    file_bar.postfix = price_counter
    file_bar.update(1)
file_bar.close()
train_vanillas = train_vanillas.drop(columns = train_vanillas.columns[0])




