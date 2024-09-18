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

from derman_test import atm_volvec, derman_coefs

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
    atm_vol = atm_volvec[t]
    
    if row['w'] == 'call':
        moneyness = k-s
    elif row['w'] == 'put':
        moneyness = s-k
    else:
        print('flag error')
    
    volatility = atm_vol + b*moneyness
    
    row['volatility'] = volatility
    return row



from routine_calibration_global2 import heston_by_s
S = heston_by_s.index


features_dataset = pd.DataFrame()

for s in S:
    
    hestons = heston_by_s.loc[s]
    T = atm_volvec.index
    
    
    K = np.linspace(s*0.50,s*1.5,int(1e1))
    flag = ['call']
    features = generate_features(K,T,s,flag)
    
    features['dividend_rate'] = hestons.loc['dividend_rate']
    features['risk_free_rate'] = hestons.loc['risk_free_rate']
    
    features['sigma'] = hestons.loc['sigma']
    features['theta'] = hestons.loc['theta']
    features['kappa'] = hestons.loc['kappa']
    features['rho'] = hestons.loc['rho']
    features['v0'] = hestons.loc['v0']
    
    features = features.apply(apply_derman_vols,axis=1)
    
    features_dataset = pd.concat(
        [features_dataset, features],ignore_index=True)
    
features_dataset = features_dataset.reset_index(drop=True)


pd.set_option("display.max_columns",None)
features_dataset


# from pricing import heston_price_vanilla_row, noisyfier

# priced_features = features_dataset.apply(heston_price_vanilla_row,axis=1)

# ml_data = noisyfier(priced_features)


# pd.set_option("display.max_columns",None)
# print(f"\n{ml_data.describe()}")
# pd.reset_option("display.max_columns")