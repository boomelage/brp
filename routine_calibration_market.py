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
security_settings = settings[0]['security_settings']
day_count = settings[0]['day_count']
calendar = settings[0]['calendar']
calculation_date = settings[0]['calculation_date']



def calibrate_heston_model(
        contracts, 
        pricing_error_tolerance = 0.02999999999
        ):
    dataset = contracts.copy()
    Svec = dataset['spot_price'].unique()
    all_heston_parameters = pd.DataFrame()
    
    for s_idx, s in enumerate(Svec):
        

        
        S_handle = ql.QuoteHandle(ql.SimpleQuote(s))
        grouped = dataset.groupby(by='days_to_maturity')
        T = dataset['days_to_maturity'].unique()
        
        progress_bar = tqdm(total=len(T), desc=f"calibrating spot = {int(s)}", unit="calibration",leave=True)
        
        heston_np_s = np.zeros((len(T),13),dtype=float)
        heston_df_s = pd.DataFrame(heston_np_s)
        df_tag = str(f"s = {int(s)}")
        heston_df_s[df_tag] = T
        heston_df_s = heston_df_s.set_index(df_tag)
        heston_df_s.columns = [
            'spot_price', 'volatility', 'days_to_maturity', 'risk_free_rate',
            'dividend_rate','v0','kappa','theta','rho','sigma','error',
            'black_scholes','heston',]
        
        avg_t = 0.00
        for t_idx, t in enumerate(T):
            
            calibration_dataset = grouped.get_group(t).reset_index(drop=True)
            
            risk_free_rate = float(calibration_dataset['risk_free_rate'].loc[0])
            dividend_rate = float(calibration_dataset['dividend_rate'].loc[0])
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
            date = calculation_date + ql.Period(int(t),ql.Days)
            dt = (date - calculation_date)
            p = ql.Period(dt, ql.Days)
            
            for row_idx, row in calibration_dataset.iterrows():
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
                                                                ) - 1.
                avg += abs(err)
                
                avg += abs(err)
            
            avg = avg*100.0/len(heston_helpers)

            heston_df_s.loc[t,'spot_price'] = s
            heston_df_s.loc[t,'dividend_rate'] = s
            heston_df_s.loc[t,'risk_free_rate'] = s
            heston_df_s.loc[t,'volatility'] = volatility
            heston_df_s.loc[t,'sigma'] = sigma
            heston_df_s.loc[t,'theta'] = theta
            heston_df_s.loc[t,'kappa'] = kappa
            heston_df_s.loc[t,'sigma'] = sigma
            heston_df_s.loc[t,'rho'] = rho
            heston_df_s.loc[t,'v0'] = v0
            heston_df_s.loc[t,'error'] = avg/100
            heston_df_s.loc[t,'black_scholes'] = opt.marketValue()
            heston_df_s.loc[t,'heston'] = opt.modelValue()
            heston_df_s.loc[t,'days_to_maturity']  = t
            
            
            # tqdm.write("-"*40)
            # tqdm.write("Total Average Abs Error (%%) : %5.3f" % (avg))
            # tqdm.write(f"for spot = {int(s)}, {int(t)} day maturity")
            # tqdm.write("-"*40)
            avg_t += np.mean(heston_df_s['error'])
            progress_bar.set_postfix({"AccumulatedAbsError": f"{avg_t:.2f}%"})
            progress_bar.update(1)
            
        
        heston_parameters = heston_df_s.copy()
        
        progress_bar.close()
        
        # end_time = time.time()
        # end_datetime = datetime.fromtimestamp(end_time)
        # end_tag = end_datetime.strftime("%c")
        # tqdm.write(f"\n{end_tag}")
        
        all_heston_parameters = pd.concat(
            [all_heston_parameters,heston_parameters])
        
    all_heston_parameters = all_heston_parameters[~(
        all_heston_parameters['error']>pricing_error_tolerance)]
    all_heston_parameters = all_heston_parameters.sort_values(
        'error').reset_index(drop=True)
    
    
    
    # tqdm.write(f'\n{all_heston_parameters}')
    # runtime = end_time - start_time
    # tqdm.write(f"\ntotal time elapsed: {int(runtime)} seconds")
    
    return all_heston_parameters

from routine_collection import contract_details  

# tqdm.write('\n\ncalibrating calls...\n')
call_heston_parameters = calibrate_heston_model(contract_details['calls'])

# tqdm.write('\n\ncalibrating puts...\n')
put_heston_parameters = calibrate_heston_model(contract_details['puts'])

# tqdm.write(f"\nputs:\n{put_heston_parameters}")
# tqdm.write(f"\ncalls:\n{call_heston_parameters}")

calibration_end = time.time()
calibration_end_datetime = datetime.fromtimestamp(calibration_end)
calibration_end_tag = calibration_end_datetime.strftime("%c")
# print(f"\n{calibration_end_datetime}")
runtime = calibration_end-start_time
# print(f"calibration runtime: {int(runtime)} seconds")