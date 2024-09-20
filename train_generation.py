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
from tqdm import tqdm

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


n_strikes = int(10)
n_maturities = int(10)

n_contracts = int(n_maturities*n_maturities*2)

print(f"pricing {n_contracts} contracts...")

progress_bar = tqdm(total=2, desc="generatingFeatures", leave=True,
                    bar_format='{l_bar}{bar} | {n_fmt}/{total_fmt}')

call_K_interp = np.linspace(min(call_K), max(call_K),n_strikes)
put_K_interp = np.linspace(min(put_K),max(put_K),n_strikes)

T = np.unique(np.linspace(1,31,n_maturities).astype(int))
# T = ms.T

call_features = generate_features(call_K_interp, T, s, ['call'])
put_features = generate_features(put_K_interp, T, s, ['put'])

train_K = np.sort(np.array([put_K,call_K],dtype=int).flatten())

features = pd.concat(
    [call_features,put_features],ignore_index=True).reset_index(drop=True)

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

progress_bar.set_description(f'pricing{int(n_contracts)}contracts')
progress_bar.update(1)

from pricing import black_scholes_price, noisyfier, heston_price_vanilla_row
bs_features = features.apply(black_scholes_price,axis=1)

heston_features = bs_features.apply(heston_price_vanilla_row,axis=1)


ml_data = noisyfier(heston_features)


progress_bar.update(1)
progress_bar.close()

# pd.set_option('display.max_rows',None)

pd.set_option('display.max_columns',None)
print(f"\n\ntraining dataset:\n{ml_data}")
print(f"\n\ndescriptive statistics:\n{ml_data.describe()}")
print(f"\n\ntrain s: {s}, K:\n{train_K}")
pd.reset_option('display.max_columns')

# pd.reset_option('display.max_rows')

