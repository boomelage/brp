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


from settings import model_settings
ms = model_settings()

s = ms.s
from routine_ivol_collection import raw_T
# from bicubic_interpolation import derman_coefs
from derman_test import derman_coefs



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





from bicubic_interpolation import ql_T,ql_K,ql_vols
import QuantLib as ql
i = ql.BilinearInterpolation(ql_T, ql_K, ql_vols)


def apply_interpolated_vol_row(row):
    k = row['strike_price']
    t = row['days_to_maturity']
    atm_vol =i(t,k, True)
    row['volatility'] = atm_vol
    return row

"""
5625, 5630, 5635, 5640, 5645, 5650
"""

call_K = np.array([5615, 5620, 5625, 5630],dtype=float)
put_K = np.array([5630, 5635, 5640, 5645],dtype=float)


T = raw_T

calls = generate_features(call_K, T, s)
calls = calls[calls['days_to_maturity'].isin(T)].copy()
calls['w'] = 'call'
calls['moneyness'] = calls['strike_price'] - calls['spot_price']





puts = generate_features(put_K, T, s)
puts = puts[puts['days_to_maturity'].isin(T)].copy()
puts['w'] = 'put'
puts['moneyness'] = puts['spot_price'] - puts['strike_price']
puts = puts.apply(apply_interpolated_vol_row,axis=1)
puts

features = pd.concat([calls,puts],ignore_index=True)


contract_details = features.copy()
contract_details['risk_free_rate'] = 0.04
contract_details['dividend_rate'] = 0.001


# pd.set_option('display.max_rows',None)
# pd.set_option('display.max_columns',None)

pd.reset_option('display.max_rows')
pd.reset_option('display.max_columns')

contract_details = contract_details.replace(0,np.nan)
contract_details = contract_details.dropna().reset_index(drop=True)
contract_details


