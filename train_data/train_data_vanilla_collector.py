# -*- coding: utf-8 -*-
"""
Created on Thu Sep 26 01:57:11 2024

@author: boomelage
"""

import os
import sys
import re
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


print('\nloading data...\n')
price_counter = 0
file_bar = tqdm(
    desc='collecting',total=len(csvs),unit='files',postfix=price_counter,
    bar_format= str('{percentage:3.0f}% | {n_fmt}/{total_fmt} {unit} | {rate_fmt} '
    '| Elapsed: {elapsed} | Remaining: {remaining} | Prices: {postfix}'))
training_data = pd.DataFrame()
for file in csvs:
    train_subset = pd.read_csv(file)
    training_data = pd.concat([training_data,train_subset],ignore_index=True)
    price_counter += train_subset.shape[0]
    file_bar.postfix = price_counter
    file_bar.update(1)
file_bar.close()
training_data = training_data.drop(columns = training_data.columns[0])




