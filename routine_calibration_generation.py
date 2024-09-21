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
import pandas as pd
from itertools import product


from settings import model_settings
ms = model_settings()
s = ms.s
calibration_call_K = ms.calibration_call_K
calibration_put_K = ms.calibration_put_K
T = ms.T



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


calls = generate_features(calibration_call_K, T, s)
calls = calls[calls['days_to_maturity'].isin(T)].copy()
calls['w'] = 'call'
calls['moneyness'] = calls['spot_price'] - calls['strike_price']



puts = generate_features(calibration_put_K, T, s)
puts = puts[puts['days_to_maturity'].isin(T)].copy()
puts['w'] = 'put'
puts['moneyness'] = puts['strike_price'] - puts['spot_price']



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
#     apply_derman_vols, axis=1, coef_df=call_dermans, atm_vols = call_atmvols)
# puts = puts.apply(
#     apply_derman_vols,axis = 1, coef_df=call_dermans, atm_vols = call_atmvols)

"""
Bilinear interpolation of Derman surface
"""

# from bilinear_interpolation import bilinear_vol_row, plot_bilinear_rotate
# calls = calls.apply(bilinear_vol_row,axis=1)
# puts = puts.apply(bilinear_vol_row,axis=1)
# fig = plot_bilinear_rotate()

"""
Bicubic Spline interpolation of Derman surface
"""

from bicubic_interpolation import bicubic_vol_row, plot_bicubic_rotate
calls = calls.apply(bicubic_vol_row,axis=1)
puts = puts.apply(bicubic_vol_row,axis=1)
fig = plot_bicubic_rotate()

"""
wip
"""
features = pd.concat([calls,puts],ignore_index=True)
contract_details = features.copy()
contract_details['risk_free_rate'] = 0.04
contract_details['dividend_rate'] = 0.001


# pd.set_option('display.max_rows',None)
# pd.set_option('display.max_columns',None)
pd.reset_option('display.max_rows')
pd.reset_option('display.max_columns')
print(f"\ncalibration dataset:\n{contract_details}")





