# -*- coding: utf-8 -*-
"""
Created on Tue Sep 17 15:46:57 2024

@author: boomelage
"""

import os
import sys
os.chdir(os.path.dirname(os.path.abspath(__file__)))
sys.path.append('term_structure')
sys.path.append('contract_details')
sys.path.append('misc')
import pandas as pd
from itertools import product
from settings import model_settings

ms = model_settings()
s = ms.s
ticker = ms.ticker
day_count = ms.day_count
calendar = ms.day_count
calculation_date = ms.day_count

def generate_features(K,T,s):
    features = pd.DataFrame(
        product(
            [s],
            K,
            T,
            ),
        columns=[
            "spot_price", 
            "strike_price",
            "days_to_maturity",
                  ])
    return features

from derman_test import call_dermans, put_dermans

def apply_derman_vols(row):
    t = row['days_to_maturity']
    moneyness = row['moneyness']
    
    if row['w'] == 'call':
        b = call_dermans.loc['b',t]
        atm_vol = call_dermans.loc['atm_vol',t]
        
    elif row['w'] == 'put':
        b = put_dermans.loc['b',t]
        atm_vol = put_dermans.loc['atm_vol',t]
        
    else:
        print('flag error')
    
    volatility = atm_vol + b*moneyness
    
    row['volatility'] = volatility
    return row



T = ms.T
call_K = ms.call_K[:4]
put_K = ms.put_K[-4:]

S = [ms.s]


n_k = int(1e2) #ms.n_k

import numpy as np

call_K_train = np.linspace(min(call_K),max(ms.calibration_K),n_k)

call_features = generate_features(call_K_train,T,s)
call_features['w'] = 'call'
call_features['moneyness'] = call_features['spot_price'] - call_features['strike_price']

put_K_train = np.linspace(min(put_K),max(put_K),n_k)

put_features = generate_features(put_K_train,T,s)
put_features['w'] = 'put'
put_features['moneyness'] = put_features['strike_price'] - put_features['spot_price']

features = pd.concat([put_features,call_features])

features['dividend_rate'] = 0.02
features['risk_free_rate'] = 0.04




from routine_calibration_global import heston_parameters

features['sigma'] = heston_parameters['sigma'].iloc[0]
features['theta'] = heston_parameters['theta'].iloc[0]
features['kappa'] = heston_parameters['kappa'].iloc[0]
features['rho'] = heston_parameters['rho'].iloc[0]
features['v0'] = heston_parameters['v0'].iloc[0]

# features = features.apply(apply_derman_vols,axis=1).reset_index(drop=True)
# features.describe()

from bivariate_interpolation import bilinear_vol_row


features = features.apply(bilinear_vol_row,axis=1)

from pricing import black_scholes_price, noisyfier, heston_price_vanilla_row
bs_features = features.apply(black_scholes_price,axis=1)

# pd.set_option('display.max_rows',None)
# pd.set_option('display.max_columns',None)
# pd.reset_option('display.max_rows')
# pd.reset_option('display.max_columns')

heston_features = bs_features.apply(heston_price_vanilla_row,axis=1)

ml_data = noisyfier(heston_features)

print(f"\n\ntraining dataset:\n{ml_data}")
