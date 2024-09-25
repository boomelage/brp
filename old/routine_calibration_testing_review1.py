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

def ql_black_scholes_row(row):
    r = row['risk_free_rate']
    s = row['spot_price']
    t = int(row['days_to_maturity'])
    k = row['strike_price']
    volatility = float(row['volatility'])
    calculation_date = row['calculation_date']
    w = row['w']
    option_type = ql.Option.Call if w == 'call' else ql.Option.Put
    expiration_date = calculation_date + ql.Period(t,ql.Days)
    flat_ts = ms.make_ts_object(r)
    initialValue = ql.QuoteHandle(ql.SimpleQuote(s))
    
    volTS = ql.BlackVolTermStructureHandle(
        ql.BlackConstantVol(
            expiration_date, 
            ms.calendar, 
            volatility, 
            ms.day_count
            )
        )
    
    process = ql.BlackScholesProcess(initialValue, flat_ts, volTS)
    engine = ql.AnalyticEuropeanEngine(process)
    
    payoff = ql.PlainVanillaPayoff(option_type, k)
    europeanExercise = ql.EuropeanExercise(expiration_date)
    european_option = ql.VanillaOption(payoff, europeanExercise)
    european_option.setPricingEngine(engine)
    
    row['ql_black_scholes'] = european_option.NPV()
    return row



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

test_features['calculation_date'] = ms.calculation_date
personal_black_scholes = test_features.apply(ms.black_scholes_price,axis=1)
ql_personal = test_features.apply(ql_black_scholes_row,axis=1)

personal_black_scholes_prices = personal_black_scholes['black_scholes']
ql_personal_bs = ql_personal['ql_black_scholes']
test_prices = test_features['heston_price']

error_df = pd.concat(
    [
      personal_black_scholes_prices,
      ql_personal_bs,
      test_prices,
      ],
    axis = 1,
    )
error_df.columns = [
        'test_black_scholes',
        'test_ql_black_scholes',
        'test_heston',
        ]
print(f"\n{error_df}\n")