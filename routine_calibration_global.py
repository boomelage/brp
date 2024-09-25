# -*- coding: utf-8 -*-
"""
Created on Mon Sep 16 19:47:34 2024


"""
import time
from datetime import datetime
start_time = time.time()
start_datetime = datetime.fromtimestamp(start_time)
start_tag = start_datetime.strftime("%c")
import os
import sys
current_dir = os.path.dirname(os.path.abspath(__file__))
os.chdir(current_dir)
sys.path.append('term_structure')
import QuantLib as ql
import numpy as np
import pandas as pd
from settings import model_settings
ms = model_settings()
day_count = ms.day_count
calendar = ms.calendar
s = ms.s
"""
calibrating a heston model to contracts from routine_calibration_generation.py
"""




def calibrate_heston(calibration_dataset, s, calculation_date):
    risk_free_rate = 0.05
    dividend_rate = 0.02
    flat_ts = ms.make_ts_object(risk_free_rate)
    dividend_ts = ms.make_ts_object(dividend_rate)

    heston_helpers = []

        
    S_handle = ql.QuoteHandle(ql.SimpleQuote(s))


    v0 = 0.01; kappa = 0.2; theta = 0.02; rho = -0.75; sigma = 0.5; 
    process = ql.HestonProcess(
        flat_ts,                
        dividend_ts,            
        S_handle,               
        v0,                # Initial volatility
        kappa,             # Mean reversion speed
        theta,             # Long-run variance (volatility squared)
        sigma,             # Volatility of the volatility
        rho                # Correlation between asset and volatility
    )

    model = ql.HestonModel(process)
    engine = ql.AnalyticHestonEngine(model)

    for row_idx, row in calibration_dataset.iterrows():
        t = row['days_to_maturity']
        
        date = calculation_date + ql.Period(int(t),ql.Days)
        dt = (date - calculation_date)
        p = ql.Period(dt, ql.Days)
        volatility = row['volatility']
        k = row['strike_price']
        
        helper = ql.HestonModelHelper(
            p, calendar, float(s), k, 
            ql.QuoteHandle(ql.SimpleQuote(volatility)), 
            flat_ts, 
            dividend_ts
            )
        helper.setPricingEngine(engine)
        heston_helpers.append(helper)
    
    lm = ql.LevenbergMarquardt(1e-8, 1e-8, 1e-8)
    
    
    model.calibrate(heston_helpers, lm,
                      ql.EndCriteria(1000, 50, 1.0e-8,1.0e-8, 1.0e-8))
    
    theta, kappa, sigma, rho, v0 = model.params()
    
    perfcols = ['market','model','relative_error']
    performance_np = np.zeros((calibration_dataset.shape[0],3),dtype=float)
    performance_df = pd.DataFrame(performance_np)
    performance_df.columns = perfcols
    
    for i in range(len(heston_helpers)):
        opt = heston_helpers[i]
        performance_df.loc[i,'market'] = opt.marketValue()
        performance_df.loc[i,'model'] = opt.modelValue()
        performance_df.loc[i,'relative_error'] = opt.modelValue() / opt.marketValue() - 1
    
    avg = np.sum(
        np.abs(
            performance_df.loc[i,'relative_error'])
        )*100/performance_df.shape[0]
    
    param_names = ['spot','theta', 'rho', 'kappa', 'sigma', 'v0', 'avg']
    
    heston_parameters_np = np.zeros((1,len(param_names)),dtype=float)
    heston_parameters = pd.DataFrame(heston_parameters_np)
    heston_parameters.columns = param_names
    
    heston_parameters['spot'] = s
    heston_parameters['theta'] = theta
    heston_parameters['rho'] = rho
    heston_parameters['kappa'] = kappa
    heston_parameters['sigma'] = sigma
    heston_parameters['v0'] = v0
    heston_parameters['avg'] = avg
    heston_parameters = heston_parameters.set_index('spot',drop=True)
    
    pd.set_option("display.max_columns",None)
    pd.set_option("display.max_rows",None),
    print(f'\n\ncalibration results:\n{performance_df}')
    print(f"average abs relative error: {round(100*avg,4)}%")
    print(f"\n{heston_parameters}")
    
    pd.reset_option("display.max_columns")
    pd.reset_option("display.max_rows")
    return heston_parameters, performance_df


