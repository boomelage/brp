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



from routine_calibration_global import heston_parameters

from routine_calibration_generation import T, call_ks, put_ks
S = [ms.s]

features_dataset = pd.DataFrame()

n_k = int(1e1) #ms.n_k
print(f'generating {int(2*n_k*len(T))} contract_features')

import numpy as np

# call_ks = np.linspace(0.99*s,s,n_k)
call_features = generate_features(call_ks,T,s)
call_features['w'] = 'call'
call_features['moneyness'] = call_features['strike_price']-call_features['spot_price']
call_features

# put_ks = np.linspace(s,1.01*s,n_k)
put_features = generate_features(put_ks,T,s)
put_features['w'] = 'put'
put_features['moneyness'] = put_features['spot_price']-put_features['strike_price']
put_features

features = pd.concat([put_features,call_features])
features['dividend_rate'] = 0.02
features['risk_free_rate'] = 0.04

features['sigma'] = heston_parameters['sigma'].iloc[0]
features['theta'] = heston_parameters['theta'].iloc[0]
features['kappa'] = heston_parameters['kappa'].iloc[0]
features['rho'] = heston_parameters['rho'].iloc[0]
features['v0'] = heston_parameters['v0'].iloc[0]


features = features.apply(apply_derman_vols,axis=1).reset_index(drop=True)
features



from pricing import black_scholes_price, heston_price_vanilla_row, noisyfier
bs_features = features.apply(black_scholes_price,axis=1)

# bs_features
heston_features = bs_features.apply(heston_price_vanilla_row,axis=1)



# ml_data = noisyfier(heston_features)


pd.set_option("display.max_columns",None)
# ml_data[ml_data['heston_price']<0]
# print(f"\n{ml_data.describe()}")
# pd.reset_option("display.max_columns")
pd.reset_option("display.max_rows")


heston_features[heston_features['heston_price']<0]


