# -*- coding: utf-8 -*-
"""
Created on Thu Sep 26 01:57:11 2024

@author: boomelage
"""

import os
import sys
import numpy as np
import pandas as pd
from datetime import datetime
current_dir = os.path.dirname(os.path.abspath(__file__))
os.chdir(current_dir)
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)
from settings import model_settings
ms = model_settings()
from train_data_vanilla_collector import train_vanillas

print('\npreparing data...')

training_data = train_vanillas.copy()
initial_count = training_data.shape[0]

training_data['calculation_date'] = pd.to_datetime(
    training_data['calculation_date'])
training_data['expiration_date'] = pd.to_datetime(
    training_data['expiration_date'])

# training_data = training_data[
#     training_data['numpy_black_scholes']>0.000001*training_data['spot_price']
#     ]

training_data.loc[:,'moneyness'] = ms.vmoneyness(
    training_data['spot_price'],
    training_data['strike_price'],
    training_data['w']
    )

training_data.loc[:,'moneyness_tag'] = ms.encode_moneyness(
    training_data['moneyness']).astype(object)

training_data['observed_price'] = ms.noisy_prices(
    training_data['heston_price'])


"""
date filter
"""

# training_data = training_data[
    
#     (training_data['calculation_date']>=datetime(2010,2,1))
#     &
#     (training_data['calculation_date']<=datetime(2030,2,11))
    
#     ].reset_index(drop=True)


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


"""
moneyness filter
"""

otm_lower = -0.9
otm_upper = -0.01

itm_lower =  0.02
itm_upper =  0.9


# training_data = training_data[
    
#     (
#       (training_data['moneyness'] >= otm_lower) & 
#       (training_data['moneyness'] <= otm_upper)
#       )
   
    # |
    
    # (
    #   (training_data['moneyness'] >= itm_lower) & 
    #   (training_data['moneyness'] <= itm_upper)
    #   )

# ]


# training_data = training_data[training_data['moneyness_tag'] == str('otm')]


""""""

S = np.sort(training_data['spot_price'].unique())
K = np.sort(training_data['strike_price'].unique())
T = np.sort(training_data['days_to_maturity'].unique())
W = np.sort(training_data['w'].unique())
n_calls = training_data[training_data['w']=='call'].shape[0]
n_puts = training_data[training_data['w']=='put'].shape[0]

training_data = training_data[
    [
     'spot_price', 'strike_price', 'w', 'heston_price', 'days_to_maturity',
     'moneyness','30D', '60D', '3M', '6M', '12M', '18M', '24M', 'calculation_date',
     'expiration_date', 'moneyness_tag', 'observed_price'
     ]
    ]

pd.set_option("display.max_columns",None)
print(f"\n{training_data}")
print(f"\n{training_data.describe()}\n")
print(f"\nspot(s):\n{S}\n\nstrikes:\n{K}\n\nmaturities:\n{T}\n\ntypes:\n{W}")
print(f"\n{training_data['moneyness_tag'].unique()}")
print(f"\nmoneyness:\n{np.sort(training_data['moneyness'].unique())}")
print(f"\nnumber of calls, puts:\n{n_calls},{n_puts}")
print(f"\ninitial count:\n{initial_count}")
print(f"\ntotal prices:\n{training_data.shape[0]}\n")
pd.reset_option("display.max_columns")





