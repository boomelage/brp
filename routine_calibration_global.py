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

from derman_test import derman_coefs, atm_volvec
from routine_collection import contract_details

calls = contract_details['calls']
puts = contract_details['puts']

def apply_derman_vols(row):
    t = row['days_to_maturity']
    moneyness = row['moneyness']
    b = derman_coefs.loc['b',t]
    atm_vol = atm_volvec[t]
    
    volatility = atm_vol + b*moneyness
    row['volatility'] = volatility
    
    return row

calls = calls[calls['days_to_maturity'].isin(atm_volvec.index)].copy()

calls['moneyness'] = \
    calls.loc[:,'spot_price'] - calls.loc[:,'strike_price'] 
calls = calls.loc[calls['moneyness']<0].copy()
calls = calls.apply(apply_derman_vols,axis=1)

puts = puts[puts['days_to_maturity'].isin(atm_volvec.index)].copy()

puts['moneyness'] = \
    puts.loc[:,'strike_price'] - puts.loc[:,'spot_price']
puts = puts.loc[puts['moneyness']<0].copy()
puts = puts.apply(apply_derman_vols,axis=1)


features = pd.concat([calls,puts],ignore_index=True).reset_index(drop=True)

calibration_dataset = features
s = ms.s
S_handle = ql.QuoteHandle(ql.SimpleQuote(s))

risk_free_rate = 0.05
dividend_rate = 0.02


flat_ts = ms.make_ts_object(risk_free_rate)
dividend_ts = ms.make_ts_object(dividend_rate)

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
                  ql.EndCriteria(500, 50, 1.0e-8,1.0e-8, 1.0e-8))

theta, kappa, sigma, rho, v0 = model.params()


perfcols = ['black_scholes','heston','relative_error']
performance_np = np.zeros((calibration_dataset.shape[0],3),dtype=float)
performance_df = pd.DataFrame(performance_np)
performance_df.columns = perfcols

for i in range(len(heston_helpers)):
    opt = heston_helpers[i]
    # print(f'{opt.marketValue()}')
    performance_df.loc[i,'black_scholes'] = opt.marketValue()
    performance_df.loc[i,'heston'] = opt.modelValue()
    performance_df.loc[i,'relative_error'] = opt.modelValue() / opt.marketValue() - 1


parameters = {
    'theta' : theta,
    'rho' : rho,
    'kappa' : kappa,
    'sigma' : sigma,
    'v0' : v0
    }

pd.set_option("display.max_columns",None)
# pd.set_option("display.max_rows",None)
print(f'\n\n{performance_df}')
print(f"    average abs relative error: {round(100*np.mean(performance_df.loc[i,'relative_error']),4)}%")
pd.reset_option("display.max_columns")
pd.reset_option("display.max_rows")

