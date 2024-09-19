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
dividend_rate = settings[0]['dividend_rate']
risk_free_rate = settings[0]['risk_free_rate']

security_settings = settings[0]['security_settings']
s = security_settings[5]

ticker = security_settings[0]
lower_moneyness = security_settings[1]
upper_moneyness = security_settings[2]
lower_maturity = security_settings[3]
upper_maturity = security_settings[4]

day_count = settings[0]['day_count']
calendar = settings[0]['calendar']
calculation_date = settings[0]['calculation_date']

from derman_test import derman_coefs

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



from routine_calibration_global import parameters


S = [ms.s]

features_dataset = pd.DataFrame()
flag = ['call','put']
T = derman_coefs.columns
n_k = int(1e5)
print(f'\ngenerating {len(S)*n_k*len(flag)*len(T)} contract features')


from routine_ivol_collection import raw_ts

for s in S:
    raw_ks = raw_ts.iloc[:,0].dropna().index
    u_k = max(raw_ks)
    l_k = min(raw_ks)
    K = np.linspace(l_k,u_k,n_k)
    features = generate_features(K,T,s,flag)
    
    features['dividend_rate'] = 0.02
    features['risk_free_rate'] = 0.04
    
    
    features['sigma'] = parameters['sigma']
    features['theta'] = parameters['theta']
    features['kappa'] = parameters['kappa']
    features['rho'] = parameters['rho']
    features['v0'] = parameters['v0']
    
    features = features.apply(apply_derman_vols,axis=1)
    
    features_dataset = pd.concat(
        [features_dataset, features],ignore_index=True)
    
features_dataset = features_dataset.reset_index(drop=True)

from pricing import black_scholes_price, heston_price_vanilla_row, noisyfier

priced_features = features_dataset.apply(black_scholes_price,axis=1)

priced_features = priced_features.apply(heston_price_vanilla_row,axis=1)

# priced_features['error'] = priced_features['heston_price']/priced_features['black_scholes']-1

ml_data = noisyfier(priced_features)


pd.set_option("display.max_columns",None)
print(f"\n{ml_data.describe()}")
pd.reset_option("display.max_columns")