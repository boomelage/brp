# -*- coding: utf-8 -*-
"""
Created on Thu Sep 26 01:57:11 2024

@author: boomelage
"""

import os
import sys
import pandas as pd
from tqdm import tqdm
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
VANILLAS = os.path.join(
    parent_dir,'historical_data','historical_vanillas')
BARRIERS = os.path.join(
    parent_dir,'historical_data','historical_barriers')
sys.path.append(parent_dir)
sys.path.append(VANILLAS)
sys.path.append(BARRIERS)
from data_query import dirdatacsv


COLLECTION_DIRECTORY = VANILLAS

os.chdir(COLLECTION_DIRECTORY)

csvs = dirdatacsv()

print('\nloading data...\n')
price_counter = 0
file_bar = tqdm(
    desc='collecting', total=len(csvs), unit='files',
    postfix={'price count': int(price_counter)},
    )
train_contracts = pd.DataFrame()
for file in csvs:
    vanilla_subset = pd.read_csv(file)
    train_contracts = pd.concat([train_contracts,vanilla_subset],ignore_index=False)
    price_counter += vanilla_subset.shape[0]
    file_bar.set_postfix({'Price Count': price_counter})
    file_bar.update(1)
file_bar.close()


before_drop_count = train_contracts.shape[0]
train_contracts = train_contracts.iloc[:,1:].copy(
    ).drop_duplicates().reset_index(drop=True)
after_drop_count = train_contracts.shape[0]

print(f"\nduplicates dropped: {before_drop_count-after_drop_count}")

