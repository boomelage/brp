# -*- coding: utf-8 -*-
"""
Created on Mon Sep 16 19:47:34 2024

works for one spot only at the moment, be careful of the data loaded in
/contract_details

"""

def clear_all():
    globals_ = globals().copy()
    for name in globals_:
        if not name.startswith('_') and name not in ['clear_all']:
            del globals()[name]
clear_all()
import time
from datetime import datetime

start_time = time.time()
start_datetime = datetime.fromtimestamp(start_time)
start_tag = start_datetime.strftime("%c")

import os
import sys
from tqdm import tqdm

current_dir = os.path.dirname(os.path.abspath(__file__))
os.chdir(current_dir)
sys.path.append('term_structure')
sys.path.append('contract_details')

import QuantLib as ql
import numpy as np
import pandas as pd

from settings import model_settings
ms = model_settings()
settings = ms.import_model_settings()

day_count = settings[0]['day_count']
calendar = settings[0]['calendar']
calculation_date = settings[0]['calculation_date']

from derman_test import derman_coefs
from calibration_generation import contract_details




def apply_derman_vols(row):
    t = row['days_to_maturity']
    moneyness = row['moneyness']
    b = derman_coefs.loc['b',t]
    atm_vol = derman_coefs.loc['atm_vol',t]
    
    volatility = atm_vol + b*moneyness
    row['volatility'] = volatility
    
    return row



calls = contract_details.copy()
calls['w'] = 'call'
calls['moneyness'] = calls['strike_price'] - calls['spot_price']
calls = calls[calls['moneyness']<0]

puts = contract_details.copy()
puts['w'] = 'put'
puts['moneyness'] = puts['spot_price'] - puts['strike_price']
puts = puts[puts['moneyness']<0]


features = pd.concat([calls,puts],ignore_index=True)

calibration_dataset = features.apply(apply_derman_vols,axis=1)


calibration_dataset




s = ms.s
S_handle = ql.QuoteHandle(ql.SimpleQuote(s))



groupedby_t = calibration_dataset.groupby('days_to_maturity')

T = np.sort(calibration_dataset['days_to_maturity'].unique())

T_param_cols = ['v0','kappa','theta','sigma','rho']
T_param_np = np.zeros((len(T),len(T_param_cols)))
T_parameters = pd.DataFrame(T_param_np)
T_parameters.columns = T_param_cols
T_parameters.index = T

for t in T:
    group_t = groupedby_t.get_group(t)
    
    risk_free_rate = 0.05
    dividend_rate = 0.02
    flat_ts = ms.make_ts_object(risk_free_rate)
    dividend_ts = ms.make_ts_object(dividend_rate)
    
    date = calculation_date + ql.Period(int(t),ql.Days)
    dt = (date - calculation_date)
    p = ql.Period(dt, ql.Days)
    
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
    heston_helpers = []    

    for row_idx, row in group_t.iterrows():
        
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
                      ql.EndCriteria(500, 50, 1.0e-8,1.0e-8, 1.0e-8))
    
    theta, kappa, sigma, rho, v0 = model.params()
    
    perfcols = [
        'black_scholes',
        'heston',
        'relative_error']
    performance_np = np.zeros((group_t.shape[0],len(perfcols)),dtype=float)
    performance_df = pd.DataFrame(performance_np)
    performance_df.columns = perfcols
    
    for i in range(len(heston_helpers)):
        opt = heston_helpers[i]
        performance_df.loc[i,'black_scholes'] = opt.marketValue()
        performance_df.loc[i,'heston'] = opt.modelValue()
        performance_df.loc[i,'relative_error'] = \
            opt.modelValue() / opt.marketValue() - 1
        print(f"\n{performance_df}")
    
    T_parameters.loc[t,'theta'] = theta
    T_parameters.loc[t,'rho'] = rho
    T_parameters.loc[t,'kappa'] = kappa
    T_parameters.loc[t,'sigma'] = sigma
    T_parameters.loc[t,'v0'] = v0

pd.set_option("display.max_columns",None)
pd.set_option("display.max_rows",None)
print(f"\n{T_parameters}")
pd.reset_option("display.max_columns")
pd.reset_option("display.max_rows")









