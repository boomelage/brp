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



def calibrate_heston_model(
        contracts, 
        pricing_error_tolerance = 0.02999999999
        ):
    dataset = contracts.copy()
    
    Svec = dataset['spot_price'].unique()
    
    output_columns = [
        'dividend_rate', 'risk_free_rate','volatility','sigma',
        'theta','kappa','rho','v0','error','black_scholes','heston']
    
    output_np = np.zeros((len(Svec),len(output_columns)),dtype=float)
    output_df = pd.DataFrame(output_np)
    output_df.columns = output_columns
    output_df.index = Svec
    
    progress_bar = tqdm(
        total=len(Svec), 
        desc="CalibratingBySpot", 
        unit="calibrations",
        leave=True)
    for s_idx, s in enumerate(Svec):
        
        
        calibration_dataset = dataset.groupby('spot_price').get_group(s)
        
        S_handle = ql.QuoteHandle(ql.SimpleQuote(s))
        
        risk_free_rate = ms.risk_free_rate
        dividend_rate = ms.risk_free_rate
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
                min(len(heston_helpers), calibration_dataset.shape[0])):
            opt = heston_helpers[i]
            err = (opt.modelValue() / max(
                        opt.marketValue(),0.0000000000001)
                                                            ) - 1
            avg += abs(err)
            
            avg += abs(err)
        
        avg = avg*100.0/len(heston_helpers)

        output_df.loc[s,'dividend_rate'] = dividend_rate
        output_df.loc[s,'risk_free_rate'] = risk_free_rate
        output_df.loc[s,'volatility'] = volatility
        output_df.loc[s,'sigma'] = sigma
        output_df.loc[s,'theta'] = theta
        output_df.loc[s,'kappa'] = kappa
        output_df.loc[s,'sigma'] = sigma
        output_df.loc[s,'rho'] = rho
        output_df.loc[s,'v0'] = v0
        output_df.loc[s,'error'] = avg/100
        output_df.loc[s,'black_scholes'] = opt.marketValue()
        output_df.loc[s,'heston'] = opt.modelValue()
        
        progress_bar.update(1)
    progress_bar.close()
    return output_df

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

params = calibrate_heston_model(features)
print(f'\n{params}')

