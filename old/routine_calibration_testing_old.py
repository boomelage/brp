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
import numpy as np
import QuantLib as ql
from settings import model_settings
ms = model_settings()

def test_heston_calibration(
        calibration_dataset,heston_parameters,performance_df,s):
    
    """
    function for testing calibration accuracy
    """
    test_features = calibration_dataset.copy()
    test_features['dividend_rate'] = 0.02
    test_features['risk_free_rate'] = 0.04
    test_features['sigma'] = heston_parameters['sigma'].iloc[0]
    test_features['theta'] = heston_parameters['theta'].iloc[0]
    test_features['kappa'] = heston_parameters['kappa'].iloc[0]
    test_features['rho'] = heston_parameters['rho'].iloc[0]
    test_features['v0'] = heston_parameters['v0'].iloc[0]
    test_features['heston_price'] = 0.00
    test_features['w'] = 'call'
    
    for i, row in test_features.iterrows():
        s = row['spot_price']
        k = row['strike_price']
        t = int(row['days_to_maturity'])
        r = row['risk_free_rate']
        g = row['dividend_rate']
        v0 = row['v0']
        kappa = row['kappa']
        theta = row['theta']
        sigma = row['sigma']
        rho = row['rho']
        w = row['w']
        
        date = ms.calculation_date + ql.Period(t,ql.Days)
        option_type = ql.Option.Call if w == 'call' else ql.Option.Put
        
        payoff = ql.PlainVanillaPayoff(option_type, k)
        exercise = ql.EuropeanExercise(date)
        european_option = ql.VanillaOption(payoff, exercise)
        flat_ts = ms.make_ts_object(r)
        dividend_ts = ms.make_ts_object(g)
        
        heston_process = ql.HestonProcess(
            flat_ts,dividend_ts, 
            ql.QuoteHandle(ql.SimpleQuote(s)), 
            v0, kappa, theta, sigma, rho)
        
        heston_model = ql.HestonModel(heston_process)
        
        engine = ql.AnalyticHestonEngine(heston_model)
        
        european_option.setPricingEngine(engine)
        
        h_price = european_option.NPV()
        test_features.at[i, 'heston_price'] = h_price
    
    test_features.at[0,'heston_price']
    
    black_scholes_prices = performance_df['black_scholes']
    calibration_prices = performance_df['model']
    test_prices = test_features['heston_price']
    error = test_prices/calibration_prices - 1
    error_series = pd.DataFrame({'relative_error':error})
    
    error_df = pd.concat(
        [
          black_scholes_prices,
          calibration_prices,
          test_prices,
          error_series
          ],
        axis = 1
        )
    error_df
    
    error_df.rename(columns={'model': 'calibration_price', 
                        'heston_price': 'test_price'}, inplace=True)
    
    avg = np.sum(
        np.abs(
            error_df['relative_error'])
        )*100/len(error_df['relative_error'])
    
    print(f"\nerrors:\n{error_df}")
    print(f"average absolute relative calibration testing error: {round(avg,4)}%")
    return error_df







"""
example calibtration
"""

from routine_calibration_generation import calibration_dataset
from routine_calibration_global import calibrate_heston

s = ms.s
calculation_date = ms.calculation_date

heston_parameters, performance_df = calibrate_heston(
    calibration_dataset, s, calculation_date)

error_df = test_heston_calibration(
        calibration_dataset,heston_parameters,performance_df,s)

