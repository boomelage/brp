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
from settings import model_settings
ms = model_settings()

def calibrate_bates(
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

    bates_helpers = []
    v0 = 0.01; kappa = 0.2; theta = 0.02; rho = -0.75; eta = 0.5;
    lambda_ = 0.1; nu = 0.05; delta = 0.02;
    
    process = ql.BatesProcess(
        flat_ts,                
        dividend_ts,            
        S_handle,               
        v0,                # Initial volatility
        kappa,             # Mean reversion speed
        theta,             # Long-run variance (volatility squared)
        eta,               # Volatility of the volatility
        rho,               # Correlation between asset and volatility
        lambda_,
        nu,
        delta
    )

    model = ql.BatesModel(process)
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
        bates_helpers.append(helper)
    
    lm = ql.LevenbergMarquardt(1e-8, 1e-8, 1e-8)
    
    
    model.calibrate(bates_helpers, lm,
                      ql.EndCriteria(1000, 50, 1.0e-8,1.0e-8, 1.0e-8))
    
    theta, kappa, eta, rho, v0, lambda_, nu, delta= model.params()
    
    performance_columns = ['model','market','relative_error']
    performance_np = np.zeros(
        (calibration_dataset.shape[0],len(performance_columns)),
        dtype=float)
    performance = pd.DataFrame(performance_np)
    performance.columns = performance_columns
    
    for i, opt in enumerate(bates_helpers):
        performance.at[i,'model'] = opt.modelValue()
        performance.at[i,'market'] = opt.marketValue()
        performance.at[i,'relative_error'] = \
            (opt.modelValue()-opt.marketValue())/opt.modelValue()
    print('\ncalibration results:\n')
    print(performance)
    avg = np.average(np.abs(performance['relative_error']))
    print(f"average absolute relative error: {round(avg*100,4)}%")
    param_names = [
        'kappa', 'theta', 'rho', 'eta', 'v0', 
        'lambda_','nu','delta', 'spot_price'
        ]
    
    bates_parameters_np = np.zeros(len(param_names),dtype=float)
    bates_parameters = pd.Series(bates_parameters_np)
    bates_parameters.index = param_names
    
    bates_parameters['theta'] = theta
    bates_parameters['rho'] = rho
    bates_parameters['kappa'] = kappa
    bates_parameters['eta'] = eta
    bates_parameters['v0'] = v0
    bates_parameters['lambda_'] = lambda_
    bates_parameters['nu'] = nu
    bates_parameters['delta'] = delta
    bates_parameters['spot_price'] = s
    
    return bates_parameters


from routine_calibration_generation import calibration_dataset
calibration_dataset
s = 1277.92
r = 0.04
g = 0.02154
calculation_date = ql.Date.todaysDate()

bates_parameters = calibrate_bates(
        calibration_dataset, 
        s,
        r,
        g,
        calculation_date,
        calendar = ms.calendar,
        day_count = ms.day_count
        )

bates_parameters