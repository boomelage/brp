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
    desc='collecting', total=len(csvs), unit='files',
    postfix={'price count': price_counter},
    )
train_vanillas = pd.DataFrame()
for file in csvs:
    vanilla_subset = pd.read_csv(file)
    train_vanillas = pd.concat([train_vanillas,vanilla_subset],ignore_index=False)
    price_counter += vanilla_subset.shape[0]
    file_bar.set_postfix({'Price Count': price_counter})
    file_bar.update(1)
file_bar.close()


before_drop_count = train_vanillas.shape[0]
train_vanillas = train_vanillas.iloc[:,1:].copy(
    ).drop_duplicates().reset_index(drop=True)
after_drop_count = train_vanillas.shape[0]


print(f"\nduplicates dropped: {before_drop_count-after_drop_count}")

