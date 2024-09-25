# -*- coding: utf-8 -*-
"""
Created on Wed Sep 25 19:58:27 2024

@author: boomelage
"""

import os
import sys
os.chdir(os.path.dirname(os.path.abspath(__file__)))
sys.path.append('term_structure')
sys.path.append('contract_details')
sys.path.append('misc')
import pandas as pd
import numpy as np
import QuantLib as ql
from settings import model_settings
ms = model_settings()




"""
example calibtration

"""

from routine_calibration_generation import calibration_dataset
from routine_calibration_global import calibrate_heston

s = ms.s
calculation_date = ms.calculation_date

heston_parameters, performance_df = calibrate_heston(
    calibration_dataset, s, calculation_date)


test_features = calibration_dataset.copy()
test_features['dividend_rate'] = 0.02
test_features['risk_free_rate'] = 0.04
test_features['sigma'] = heston_parameters['sigma'].iloc[0]
test_features['theta'] = heston_parameters['theta'].iloc[0]
test_features['kappa'] = heston_parameters['kappa'].iloc[0]
test_features['rho'] = heston_parameters['rho'].iloc[0]
test_features['v0'] = heston_parameters['v0'].iloc[0]


test_features['w'] = ""

for i,row in test_features.iterrows():
    s = row['spot_price']
    k = row['strike_price']
    if k>=s:
        test_features.at[i,'w'] = 'call'
    else:
        test_features.at[i,'w'] = 'put'




test_features['ql_black_scholes'] = np.nan
for i,row in test_features.iterrows():
    s = row['spot_price']
    k = row['strike_price']
    t = int(row['days_to_maturity'])
    r = row['risk_free_rate']
    g = row['dividend_rate']
    volatility = row['volatility']
    w = row['w']

    print(f"\n{row}\n")    
    ql_bsp = ql_black_scholes(s,t,k,r,g,volatility,calculation_date,w)
    test_features.at[i,'ql_black_scholes'] =  ql_bsp
    


    
    

test_features['ql_heston_price'] = 0.00
for i, row in test_features.iterrows():
    s = row['spot_price']
    k = row['strike_price']
    t = int(row['days_to_maturity'])
    r = row['risk_free_rate']
    w = row['w']  
    g = row['dividend_rate']
    
    v0 = row['v0']
    kappa = row['kappa']
    theta = row['theta']
    sigma = row['sigma']
    rho = row['rho']
    
    h_price = ms.ql_heston_price(
            s,k,t,r,g,w,
            v0,kappa,theta,sigma,rho,
            calculation_date)
    test_features.at[i,'ql_heston_price'] = h_price



test_features['black_scholes_price'] = np.nan
for i, row in test_features.iterrows():
    s = row['spot_price']
    k = row['strike_price']
    t = int(row['days_to_maturity'])
    r = row['risk_free_rate']
    w = row['w']
    g = row['dividend_rate']
    volatility = row['volatility']
    price = ms.black_scholes_price(s, k, t, r, volatility, w)
    test_features.at[i,'black_scholes_price'] = price


pd.set_option("display.max_columns",None)
print(test_features)
pd.reset_option("display.max_columns",None)





# error_df = pd.concat(
#     [
#       personal_black_scholes_prices,
#       ql_personal_bs,
#       test_prices,
#       ],
#     axis = 1,
#     )
# error_df.columns = [
#         'test_black_scholes',
#         'test_ql_black_scholes',
#         'test_heston',
#         ]
# print(f"\n{error_df}\n")