# -*- coding: utf-8 -*-
"""
Created on Thu Sep 26 01:57:11 2024

@author: boomelage
"""

import os
import sys
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
grandparent_dir = os.path.dirname(parent_dir)
sys.path.append(parent_dir)
sys.path.append(grandparent_dir)
import numpy as np
import pandas as pd
from data_query import dirdatacsv
from settings import model_settings, compute_moneyness
os.chdir(current_dir)

ms = model_settings()
csvs = dirdatacsv()

training_data = pd.DataFrame()
for file in csvs:
    train_subset = pd.read_csv(file)
    training_data = pd.concat([training_data,train_subset],ignore_index=True)
    
training_data['eta'] = training_data['eta'].combine_first(training_data['sigma'])
training_data = training_data.drop(columns='sigma')

training_data = training_data.drop(
    columns=training_data.columns[0]).drop_duplicates()

"""
maturities filter
"""

training_data = training_data[
    (abs(training_data['days_to_maturity'])>=0)&
    (abs(training_data['days_to_maturity'])<=9999)
    ].reset_index(drop=True)


"""
type filter
"""


training_data[training_data.loc[:,'w'] == 'Put']


""""""
training_data = compute_moneyness(training_data)
""""""

"""
moneyness filter
"""
training_data = training_data[
    (training_data['moneyness']>=-0.05)&
    (training_data['moneyness']<= 0.05)
    ].reset_index(drop=True)


""""""
training_data = training_data[
    [ 
     'spot_price', 'strike_price', 'days_to_maturity', 'moneyness', 
     'w', 'theta', 'kappa', 'rho', 'eta','v0', 
     'heston_price', 'observed_price' 
     ]
    ]

training_data = training_data.loc[
    training_data['observed_price'] >= 0.005 * training_data['spot_price']
    ]

S = np.sort(training_data['spot_price'].unique())
K = np.sort(training_data['strike_price'].unique())
T = np.sort(training_data['days_to_maturity'].unique())
W = np.sort(training_data['w'].unique())
pd.set_option("display.max_columns",None)
print(f"\n{training_data.describe()}\n")
print(f"\nspot(s):\n{S}\n\nstrikes:\n{K}\n\nmaturities:\n{T}\n\ntypes:\n{W}\n")
pd.reset_option("display.max_columns")
