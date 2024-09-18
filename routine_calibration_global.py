# -*- coding: utf-8 -*-
"""
Created on Mon Sep 16 19:47:34 2024

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
# print(f"\n{start_tag}")
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
dividend_rate = settings[0]['dividend_rate']
risk_free_rate = settings[0]['risk_free_rate']

security_settings = settings[0]['security_settings']
s = security_settings[5]

ticker = security_settings[0]
lower_moneyness = security_settings[1]
upper_moneyness = security_settings[2]
lower_maturity = security_settings[3]
upper_maturity = security_settings[4]

day_count = settings[0]['day_count']
calendar = settings[0]['calendar']
calculation_date = settings[0]['calculation_date']


def calibrate_heston_model(
        contracts, 
        pricing_error_tolerance = 0.02999999999
        ):
        dataset = contracts.copy()
    # Svec = dataset['spot_price'].unique()
    
    # heston_np = np.zeros((5,len(Svec)),dtype=float)
    # heston_df = pd.DataFrame(heston_np)
    # heston_df.index = ['theta', 'kappa', 'sigma', 'rho', 'v0']
    # heston_df.columns = Svec
    
        grouped = dataset.groupby('spot_price')
    
    # for s_idx, s in enumerate(Svec):
        s = 5485
        S_handle = ql.QuoteHandle(ql.SimpleQuote(s))
        calibration_set_s = grouped.get_group(s)
        flat_ts = ms.make_ts_object(ms.risk_free_rate)
        dividend_ts = ms.make_ts_object(ms.dividend_rate)

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
       
        for row_idx, row in calibration_set_s.iterrows():
            volatility = row['volatility']
            k = row['strike_price']
            t = row['days_to_maturity']
            date = calculation_date + ql.Period(int(t),ql.Days)
            dt = (date - calculation_date)
            p = ql.Period(dt, ql.Days)
            helper = ql.HestonModelHelper(
                p, calendar, float(s), k, ql.QuoteHandle(ql.SimpleQuote(
                    volatility)), flat_ts, dividend_ts)
            helper.setPricingEngine(engine)
            heston_helpers.append(helper)
        
        
        lm = ql.LevenbergMarquardt(1e-8, 1e-8, 1e-8)
        model.calibrate(heston_helpers, lm,
                          ql.EndCriteria(500, 50, 1.0e-8,1.0e-8, 1.0e-8))
        
        theta, kappa, sigma, rho, v0 = model.params()

        avg = 0.0
        
        for i in range(
                min(len(heston_helpers), calibration_set_s.shape[0])):
            opt = heston_helpers[i]
            
            err = ( opt.modelValue() / opt.marketValue() ) - 1
            print(f"\nerror: {100*err}%")
            print(f"bs = {opt.marketValue()}")
            print(f"heston = {opt.modelValue()}")
            avg += abs(err)
        
        # avg = avg*100.0/len(heston_helpers)
        
    
    # (theta, kappa, sigma, rho, v0)
        
    # heston_df.loc['theta',s] = theta
    # heston_df.loc['kappa',s] = kappa
    # heston_df.loc['sigma',s] = sigma
    # heston_df.loc['rho',s] = rho
    # heston_df.loc['v0',s] = v0
    
    # return heston_df
    

    

        
        
    

from routine_collection import contract_details  

puts = contract_details['puts']
puts['moneyness'] = \
    puts['strike_price'] - puts['spot_price']
puts = puts[puts['moneyness']<0]

calls = contract_details['calls']
calls['moneyness'] = \
    calls['spot_price'] - calls['strike_price'] 
calls = calls[calls['moneyness']<0]

features = pd.concat([calls,puts],ignore_index=True).reset_index(drop=True)

features
calibrate_heston_model(features)

# print(f'{parameters}')


# tqdm.write('#####CalibratingCalls#####')
# call_heston_parameters = calibrate_heston_model(contract_details['calls'])

# tqdm.write('#####CalibratingPuts#####')
# put_heston_parameters = calibrate_heston_model(contract_details['puts'])

# calibration_end = time.time()
# calibration_end_datetime = datetime.fromtimestamp(calibration_end)
# calibration_end_tag = calibration_end_datetime.strftime("%c")
# runtime = calibration_end-start_time

# print(f"\n\n#####calls#####\n{call_heston_parameters}")
# print(f"\n#####puts#####\n{put_heston_parameters}")
# print(f"\n{calibration_end_tag}")
# print(f"calibration runtime: {int(runtime)} seconds")