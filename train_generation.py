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


def generate_features(K,T,s,flag):
    features = pd.DataFrame(
        product(
            [s],
            K,
            T,
            flag
            ),
        columns=[
            "spot_price", 
            "strike_price",
            "days_to_maturity",
            "w"
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





from settings import model_settings
ms = model_settings()
import numpy as np
s = ms.s
call_K = ms.call_K[:5]
put_K = ms.put_K[-5:]

n_KT = int(1e2/2)

call_K_interp = np.linspace(min(call_K), max(call_K),n_KT)
put_K_interp = np.linspace(min(put_K),max(put_K),n_KT)

T = np.linspace(1,31,n_KT)

call_features = generate_features(call_K_interp, T, s, ['call'])
put_features = generate_features(put_K_interp, T, s, ['put'])


features = pd.concat([call_features,put_features],ignore_index=True).reset_index(drop=True)

def compute_moneyness_row(row):
    s = row['spot_price']
    k = row['strike_price']
    
    if row['w'] == 'call':
        call_moneyness = s-k
        row['moneyness'] = call_moneyness
        return row
    elif row['w'] == 'put':
        put_moneyness = k-s
        row['moneyness'] = put_moneyness
        return row
    else:
        raise ValueError('\n\n\nflag error')

features = features.apply(compute_moneyness_row,axis = 1)




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

from bilinear_interpolation import bilinear_vol_row
features = features.apply(bilinear_vol_row,axis=1)


from pricing import black_scholes_price, noisyfier, heston_price_vanilla_row
bs_features = features.apply(black_scholes_price,axis=1)






heston_features = bs_features.apply(heston_price_vanilla_row,axis=1)

ml_data = noisyfier(heston_features)

# pd.set_option('display.max_rows',None)

pd.set_option('display.max_columns',None)
print(f"\n\ntraining dataset:\n{ml_data}")
print(f"\n\ndescriptive statistics:\n{ml_data.describe()}")
pd.reset_option('display.max_columns')

# pd.reset_option('display.max_rows')

