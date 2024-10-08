# -*- coding: utf-8 -*-
"""
Created on Mon Sep 16 19:47:34 2024

"""
import os
current_dir = os.path.dirname(os.path.abspath(__file__))
os.chdir(current_dir)
import QuantLib as ql
import numpy as np
import pandas as pd
from model_settings import ms

def calibrate_heston(
        calibration_dataset, 
        s,
        r,
        g,
        calculation_date,
        calendar = ms.calendar,
        day_count = ms.day_count
        ):

    ql.Settings.instance().evaluationDate = calculation_date
    flat_ts, dividend_ts = ms.ql_ts_rg(r, g, calculation_date)
    S_handle = ql.QuoteHandle(ql.SimpleQuote(s))

    heston_helpers = []
    v0 = 0.01; kappa = 0.2; theta = 0.02; rho = -0.75; eta = 0.5;
    process = ql.HestonProcess(
        flat_ts,                
        dividend_ts,            
        S_handle,               
        v0,                # Initial volatility
        kappa,             # Mean reversion speed
        theta,             # Long-run variance (volatility squared)
        eta,               # Volatility of the volatility
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
    
    theta, kappa, eta, rho, v0 = model.params()
    
    performance_columns = ['model','market','relative_error']
    performance_np = np.zeros(
        (calibration_dataset.shape[0],len(performance_columns)),
        dtype=float)
    performance = pd.DataFrame(performance_np)
    performance.columns = performance_columns
    
    for i, opt in enumerate(heston_helpers):
        performance.at[i,'model'] = opt.modelValue()
        performance.at[i,'market'] = opt.marketValue()
        performance.at[i,'relative_error'] = \
            (opt.modelValue()-opt.marketValue())/opt.modelValue()
    print('\ncalibration results:\n')
    print(performance)
    avg = np.average(np.abs(performance['relative_error']))
    print(f"average absolute relative error: {round(avg*100,4)}%")
    param_names = [
        'theta', 'rho', 'kappa', 'eta', 'v0', 'spot_price', 'relative_error']
    
    heston_parameters_np = np.zeros(len(param_names),dtype=float)
    heston_parameters = pd.Series(heston_parameters_np)
    heston_parameters.index = param_names
    
    heston_parameters['theta'] = theta
    heston_parameters['rho'] = rho
    heston_parameters['kappa'] = kappa
    heston_parameters['eta'] = eta
    heston_parameters['v0'] = v0
    heston_parameters['spot_price'] = s
    heston_parameters['relative_error'] = avg
    
    return heston_parameters


