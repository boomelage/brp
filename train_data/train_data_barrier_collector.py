# -*- coding: utf-8 -*-
"""
Created on Thu Sep 26 01:57:11 2024

@author: boomelage
"""

import os
import sys
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)
historical_data_dir = os.path.abspath(os.path.join(parent_dir,'historical_data','hist_outputs'))
spontaneous_data_directory = os.path.abspath(os.path.join(current_dir,'barriers'))

import pandas as pd
import numpy as np
from data_query import dirdatacsv
from settings import model_settings, compute_moneyness
ms = model_settings()


DATA_DIRECTORY = historical_data_dir

os.chdir(DATA_DIRECTORY)
csvs = dirdatacsv()


training_data = pd.DataFrame()
for file in csvs:
    train_subset = pd.read_csv(file)
    training_data = pd.concat([training_data,train_subset],ignore_index=True)


training_data = training_data.drop(
    columns=training_data.columns[0]).drop_duplicates()




"""
maturities filter
"""

# training_data = training_data[
#     (abs(training_data['days_to_maturity'])>=0)&
#     (abs(training_data['days_to_maturity'])<=31)
#     ].reset_index(drop=True)

"""
type filter
"""
 
training_data = training_data[training_data.loc[:,'barrier_type_name'] == 'DownOut']


""""""
training_data = compute_moneyness(training_data)
""""""
"""
moneyness filter
"""

training_data = training_data[
    (training_data['moneyness']>=-0.05)&
    (training_data['moneyness']<=-0.0)
    ].reset_index(drop=True)


""""""

training_data['moneyness_tag'] = ms.encode_moneyness(training_data['moneyness'])
training_data['moneyness_tag'] = ms.encode_moneyness(training_data['moneyness'])

training_data = training_data[
    [ 'spot_price', 'strike_price', 'days_to_maturity', 'moneyness','barrier', 
      'outin', 'w', 'updown', 'barrier_type_name', 'moneyness_tag',
      'theta', 'kappa', 'rho', 'eta','v0', 
      'heston_price', 'barrier_price', 'observed_price' ]
    ]


T = np.sort(training_data['days_to_maturity'].unique())
W = np.sort(training_data['barrier_type_name'].unique())
S = np.sort(training_data['spot_price'].unique())
K = np.sort(training_data['strike_price'].unique())

pd.set_option("display.max_columns",None)
print(f"\n{training_data.describe()}\n")
print(f"\nspot(s):\n{S}\n\nstrikes:\n{K}\n\nmaturities:\n{T}\n\ntypes:\n{W}\n")
pd.reset_option("display.max_columns")

os.chdir(parent_dir)