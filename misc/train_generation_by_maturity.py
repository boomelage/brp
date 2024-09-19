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
import numpy as np
import pandas as pd
from itertools import product
from settings import model_settings
ms = model_settings()

settings = ms.import_model_settings()
s = ms.s
ticker = ms.ticker

day_count = settings[0]['day_count']
calendar = settings[0]['calendar']
calculation_date = settings[0]['calculation_date']
from derman_test import derman_coefs

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

        
def apply_derman_vols(row):
    s = row['spot_price']
    k = row['strike_price']
    t = row['days_to_maturity']
    b = derman_coefs.loc['b',t]
    atm_vol = derman_coefs.loc['atm_vol',t]
    
    if row['w'] == 'call':
        moneyness = k-s
    elif row['w'] == 'put':
        moneyness = s-k
    else:
        print('flag error')
    
    volatility = atm_vol + b*moneyness
    
    row['volatility'] = volatility
    return row



from routine_calibration_by_maturity import T_parameters


S = [ms.s]

features_dataset = pd.DataFrame()
T = derman_coefs.columns

print(f'generating {int(2*ms.n_k*len(T))} contract features')

from routine_ivol_collection import raw_ts

raw_ks = raw_ts.iloc[:,0].dropna().index
ub_k = max(raw_ks)
lb_k = min(raw_ks)
n_k =  int(1e4) # ms.n_k

K_calls = np.linspace(lb_k,s,n_k)
call_features = generate_features(K_calls,T,s)
call_features['w'] = 'call'
call_features['moneyness'] = call_features['strike_price']-call_features['spot_price']
call_features

K_puts = np.linspace(s,ub_k,n_k)
put_features = generate_features(K_puts,T,s)
put_features['w'] = 'put'
put_features['moneyness'] = put_features['spot_price']-put_features['strike_price']
put_features

features = pd.concat([put_features,call_features])

features = features.apply(apply_derman_vols,axis=1).reset_index(drop=True)

features['dividend_rate'] = 0.02
features['risk_free_rate'] = 0.04

def apply_hestons(row):
    t = row['days_to_maturity']
    row['sigma'] = T_parameters.loc[t,'sigma']
    row['theta'] = T_parameters.loc[t,'theta']
    row['kappa'] = T_parameters.loc[t,'kappa']
    row['rho'] = T_parameters.loc[t,'rho']
    row['v0'] = T_parameters.loc[t,'v0']
    return row

features = features.apply(apply_hestons,axis=1)

from pricing import black_scholes_price, heston_price_vanilla_row, noisyfier
bs_features = features.apply(black_scholes_price,axis=1)
heston_features = bs_features.apply(heston_price_vanilla_row,axis=1)

ml_data = noisyfier(heston_features)


pd.set_option("display.max_columns",None)
print(f"\n{ml_data.describe()}")
pd.reset_option("display.max_columns")

