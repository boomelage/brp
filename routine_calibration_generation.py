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

call_K = ms.call_K[0:3]
put_K = ms.put_K[-3:]

call_T = ms.call_T
put_T = ms.put_T



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

"""

"""


calls = generate_features(call_K, call_T, s)
calls = calls[calls['days_to_maturity'].isin(call_T)].copy()
calls['w'] = 'call'
calls['moneyness'] = calls['spot_price'] - calls['strike_price']



puts = generate_features(put_K, put_T, s)
puts = puts[puts['days_to_maturity'].isin(put_T)].copy()
puts['w'] = 'put'
puts['moneyness'] = puts['strike_price'] - puts['spot_price']



"""
bivariate interpolation

"""

from bivariate_interpolation import ql_T,ql_K,ql_vols,plot_bicubic_rotate

import QuantLib as ql
i = ql.BilinearInterpolation(ql_T, ql_K, ql_vols)
def apply_interpolated_vol_row(row):
    k = row['strike_price']
    t = row['days_to_maturity']
    atm_vol =i(t,k, True)
    row['volatility'] = atm_vol
    return row

calls = calls.apply(apply_interpolated_vol_row,axis=1)
puts = puts.apply(apply_interpolated_vol_row,axis=1)


surf = plot_bicubic_rotate()


"""
Derman approximation
"""

# from derman_test import plot_derman_rotate, call_dermans, put_dermans, \
#     call_atmvols, put_atmvols
    
    
# def apply_derman_vols(row,coef_df,atm_vols):
    
#     t = row['days_to_maturity']
#     moneyness = row['moneyness']
#     b = call_dermans.loc['b',t]
#     atm_vol = call_dermans.loc['atm_vol',t]
#     volatility = atm_vol + b*moneyness
#     row['volatility'] = volatility
#     return row

# calls = calls.apply(
#     apply_derman_vols, axis=1, coef_df=call_dermans, atm_vols=call_atmvols)
# puts = puts.apply(
#     apply_derman_vols,axis = 1, coef_df=put_dermans, atm_vols = put_atmvols)
# surf = plot_derman_rotate()


# """
# wip
# """


features = pd.concat([calls,puts],ignore_index=True)
contract_details = features.copy()
contract_details['risk_free_rate'] = 0.04
contract_details['dividend_rate'] = 0.001


# pd.set_option('display.max_rows',None)
# pd.set_option('display.max_columns',None)

pd.reset_option('display.max_rows')
pd.reset_option('display.max_columns')

print(f"\ncontract details:\n{contract_details}")

