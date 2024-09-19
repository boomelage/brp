# -*- coding: utf-8 -*-
"""
Created on Mon Sep 16 21:30:51 2024

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
# pd.set_option('display.max_columns',None)
pd.reset_option('display.max_rows')


from settings import model_settings
ms = model_settings()

s = ms.s


from derman_test import derman_coefs,atm_volvec
from routine_ivol_collection import raw_ts

def apply_derman_vols(row):
    t = row['days_to_maturity']
    moneyness = row['moneyness']
    b = derman_coefs.loc['b',t]
    atm_vol = derman_coefs.loc['atm_vol',t]
    
    volatility = atm_vol + b*moneyness
    row['volatility'] = volatility
    
    return row

def generate_features(K,T,s):
    features = pd.DataFrame(
        product(
            [float(s)],
            K,
            T,
            ),
        columns=[
            "spot_price", 
            "strike_price",
            "days_to_maturity",
                  ])
    return features



raw_ks = raw_ts.iloc[:,0].dropna().index
T = np.sort(derman_coefs.columns.unique().astype(int))



put_ks = np.linspace(min(raw_ks),s,3)
call_ks = np.linspace(s,min(raw_ks),3)




calls = generate_features(call_ks, T, s)
calls = calls[calls['days_to_maturity'].isin(atm_volvec.index)].copy()
calls['w'] = 'call'
calls['moneyness'] = calls['strike_price'] - calls['spot_price']
calls = calls[calls['moneyness']<0]
calls = calls.apply(apply_derman_vols,axis=1)

puts = generate_features(put_ks, T, s)
puts = puts[puts['days_to_maturity'].isin(atm_volvec.index)].copy()
puts['w'] = 'put'
puts['moneyness'] = puts['spot_price'] - puts['strike_price']
puts = puts[puts['moneyness']<0]
puts = puts.apply(apply_derman_vols,axis=1)



features = pd.concat([calls,puts],ignore_index=True)

contract_details = features.copy()



contract_details['risk_free_rate'] = 0.04
contract_details['dividend_rate'] = 0.001

contract_details

