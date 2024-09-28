# -*- coding: utf-8 -*-
"""
Created on Wed Sep 25 19:58:27 2024

"""

import os
import sys
os.chdir(os.path.dirname(os.path.abspath(__file__)))
sys.path.append('term_structure')
sys.path.append('contract_details')
sys.path.append('misc')
import pandas as pd
import numpy as np
from settings import model_settings
ms = model_settings()
import QuantLib as ql


"""
example calibtration

"""

from routine_calibration_generation import calibration_dataset
from routine_calibration_global import calibrate_heston

s = ms.s
calculation_date = ms.calculation_date
ql.Settings.instance().evaluationDate = calculation_date
g = 0.001
r = 0.04


heston_parameters, performance_df = calibrate_heston(
    calibration_dataset, s, r, g, calculation_date)

test_features = calibration_dataset.copy()
test_features['dividend_rate'] = g
test_features['risk_free_rate'] = r
test_features['eta'] = heston_parameters['eta']
test_features['theta'] = heston_parameters['theta']
test_features['kappa'] = heston_parameters['kappa']
test_features['rho'] = heston_parameters['rho']
test_features['v0'] = heston_parameters['v0']


for i,row in test_features.iterrows():
    s = row['spot_price']
    k = row['strike_price']
    if k>=s:
        test_features.at[i,'w'] = 'call'
    else:
        test_features.at[i,'w'] = 'put'
        

test_features['ql_heston_price'] = np.nan
test_features['ql_black_scholes'] = np.nan
test_features['black_scholes_price'] = np.nan

for i,row in test_features.iterrows():
    
    s = row['spot_price']
    k = row['strike_price']
    t = row['days_to_maturity']
    r = row['risk_free_rate']
    g = row['dividend_rate']
    volatility = row['volatility']
    w = row['w']
    v0 = row['v0']
    kappa = row['kappa']
    theta = row['theta']
    eta = row['eta']
    rho = row['rho']
    expiration_date = calculation_date + ql.Period(int(t),ql.Days)
    
    
    ql_bsp = ms.ql_black_scholes(
        s,k,r,g,
        volatility,w,
        calculation_date, 
        expiration_date
        )
    test_features.at[i,'ql_black_scholes'] =  ql_bsp
    
    
    h_price = ms.ql_heston_price(
            s,k,r,g,w,
            v0,kappa,theta,eta,rho,
            calculation_date,
            expiration_date)
    test_features.at[i,'ql_heston_price'] = h_price
    
    
    my_bs = ms.black_scholes_price(s, k, t, r, volatility, w)
    test_features.at[i,'black_scholes_price'] = my_bs
    

pd.set_option("display.max_columns",None)
print(test_features)
pd.reset_option("display.max_columns",None)


heston_parameters

