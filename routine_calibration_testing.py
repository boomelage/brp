# -*- coding: utf-8 -*-
"""
Created on Wed Sep 25 19:58:27 2024

"""

import QuantLib as ql
import pandas as pd
import numpy as np
from settings import model_settings
ms = model_settings()



def test_heston_calibration(
        test_features, heston_parameters,calculation_date,r,g
        ):
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
            
    S = test_features['spot_price']
    K = test_features['strike_price']
    W = test_features['w']
    test_features['moneyness'] = ms.vmoneyness(S,K,W)
    test_features['moneyness_tag'] = ms.encode_moneyness(test_features['moneyness'])
    
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
        
        
    print_test = test_features[['w', 'moneyness', 'ql_heston_price',
           'ql_black_scholes']].copy()
    
    print_test['relative_error'] = \
         (print_test['ql_black_scholes'] - print_test['ql_heston_price']
          )/print_test['ql_heston_price']
         
    test_avg = np.average(np.abs(np.array(print_test['relative_error'])))
    test_avg_print = f"{round(test_avg*100,4)}%"
    
    heston_parameters['relative_error'] = test_avg
    
    pd.set_option("display.max_columns",None)
    print(f"\ncalibration test:\n{print_test}\n"
          f"average absolute relative error: {test_avg_print}"
          f"\n{heston_parameters}\n")
    pd.reset_option("display.max_columns")
    return heston_parameters


