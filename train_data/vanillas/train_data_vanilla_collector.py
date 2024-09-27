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


file_bar = ms.make_tqdm_bar(desc="file", total=len(csvs), unit='files')

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

otm_lower = -0.5
otm_upper = -0.0

itm_lower =  0.0
itm_upper =  0.5

training_data = training_data[
    
    (
      (training_data['moneyness'] >= otm_lower) & 
      (training_data['moneyness'] <= otm_upper)
      )
   
    |
    
    (
      (training_data['moneyness'] >= itm_lower) & 
      (training_data['moneyness'] <= itm_upper)
      )

]

""""""

training_data = training_data.loc[
    training_data['observed_price'] >= 0.01*training_data['spot_price']
    ]

training_data['moneyness_tag'] = ms.encode_moneyness(training_data['moneyness'])

training_data = training_data[
    [
     'spot_price', 'strike_price', 'days_to_maturity', 
     'moneyness', 'moneyness_tag','w', 
     # 'theta', 'kappa', 'rho', 'eta', 'v0', 'heston_price', 
     'observed_price'
     ]
    ]

S = np.sort(training_data['spot_price'].unique())
K = np.sort(training_data['strike_price'].unique())
T = np.sort(training_data['days_to_maturity'].unique())
W = np.sort(training_data['w'].unique())
n_calls = training_data[training_data['w']=='call'].shape[0]
n_puts = training_data[training_data['w']=='put'].shape[0]

pd.set_option("display.max_columns",None)
print(f"\n{training_data.describe()}\n")
print(f"\nspot(s):\n{S}\n\nstrikes:\n{K}\n\nmaturities:\n{T}\n\ntypes:\n{W}")
print(f"\nnumber of calls, puts:\n{n_calls},{n_puts}")
print(f"\ninitial count:\n{initial_count}")
print(f"\ntotal prices:\n{training_data.shape[0]}\n")
pd.reset_option("display.max_columns")





