# -*- coding: utf-8 -*-
"""
Created on Thu Sep 26 01:57:11 2024

@author: boomelage
"""

import os
import sys
import numpy as np
import pandas as pd
from tqdm import tqdm
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
grandparent_dir = os.path.dirname(parent_dir)
sys.path.append(parent_dir)
sys.path.append(grandparent_dir)
from data_query import dirdatacsv
from settings import model_settings, compute_moneyness
os.chdir(current_dir)

ms = model_settings()
csvs = dirdatacsv()

print('\nloading data...\n')

file_bar = tqdm(desc="file",total=len(csvs),unit='files',leave=True)
training_data = pd.DataFrame()
for file in csvs:
    train_subset = pd.read_csv(file)
    training_data = pd.concat([training_data,train_subset],ignore_index=True)
    file_bar.update(1)
file_bar.close()


training_data = training_data.drop(
    columns=training_data.columns[0]).drop_duplicates()
initial_count = training_data.shape[0]
"""
maturities filter
"""

# training_data = training_data[
    
#     (training_data['days_to_maturity']>=0)
#     &
#     (training_data['days_to_maturity']<=31)
    
#     ].reset_index(drop=True)


"""
type filter
"""

# training_data = training_data[training_data.loc[:,'w'] == 'put']


""""""
training_data = compute_moneyness(training_data)
""""""

"""
moneyness filter
"""
lower = -0.1
upper = 0.1

training_data = training_data[
    (training_data['moneyness'] >=  lower ) 
    &
    (training_data['moneyness'] <=  upper )
    ].reset_index(drop=True)


""""""
training_data = training_data[
    ['spot_price', 'strike_price', 'days_to_maturity', 'moneyness', 'w', 
     'theta', 'kappa', 'rho', 'eta','v0', 'heston_price', 'observed_price']
    ]

training_data = training_data.loc[
    training_data['observed_price'] >= 0.01*training_data['spot_price']
    ]

S = np.sort(training_data['spot_price'].unique())
K = np.sort(training_data['strike_price'].unique())
T = np.sort(training_data['days_to_maturity'].unique())
W = np.sort(training_data['w'].unique())
pd.set_option("display.max_columns",None)
print(f"\n{training_data.describe()}\n")
print(f"\nspot(s):\n{S}\n\nstrikes:\n{K}\n\nmaturities:\n{T}\n\ntypes:\n{W}\n")
print(f"\ninitial count:\n{initial_count}")
print(f"\ntotal prices:\n{training_data.shape[0]}")
pd.reset_option("display.max_columns")
