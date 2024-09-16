# -*- coding: utf-8 -*-
"""
Created on Fri Sep 13 21:45:51 2024

"""
def clear_all():
    globals_ = globals().copy()
    for name in globals_:
        if not name.startswith('_') and name not in ['clear_all']:
            del globals()[name]
clear_all()
import os
import sys
current_dir = os.path.dirname(os.path.abspath(__file__))
os.chdir(current_dir)
sys.path.append('term_structure')


import time
import QuantLib as ql
import numpy as np
import pandas as pd



from settings import model_settings
ms = model_settings()
settings = ms.import_model_settings()
security_settings = settings[0]['security_settings']
s = security_settings[5]
day_count = settings[0]['day_count']
calendar = settings[0]['calendar']
calculation_date = settings[0]['calculation_date']

from derman_test import derman_ts

start_time = time.time()

S = [s]

heston_dicts = np.empty(len(S),dtype=object)

for s_idx, s in enumerate(S):
    
    ts_df = derman_ts
    K = ts_df.index
    T = ts_df.columns
    
    heston_np_s = np.zeros((6,len(T)),dtype=float)
    heston_df_s = pd.DataFrame(heston_np_s)
    df_s_name = str(f"S = {int(s)}")
    heston_df_s[df_s_name] = ['v0','kappa','theta','rho','sigma','error']
    heston_df_s = heston_df_s.set_index(df_s_name)
    heston_df_s.columns = T
    
    S_handle = ql.QuoteHandle(ql.SimpleQuote(s))
    derK = np.sort(ts_df.index).astype(float)
    derT = np.sort(ts_df.columns).astype(float)
    implied_vols_matrix = ms.make_implied_vols_matrix(derK, derT, ts_df)
    expiration_dates = ms.compute_ql_maturity_dates(derT)
    black_var_surface = ms.make_black_var_surface(
        expiration_dates, derK.astype(float), implied_vols_matrix)
    
    sets_for_maturities = np.empty(len(derT),dtype=object)
    for t_idx, t in enumerate(derT):
        risk_free_rate = 0.055
        dividend_rate = 0.01312
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
                
        for k_idx, k in enumerate(derK):
            sigma = black_var_surface.blackVol(dt/365.25, k) 
            helper = ql.HestonModelHelper(
                p,
                calendar,
                float(s),
                k,
                ql.QuoteHandle(ql.SimpleQuote(sigma)),
                flat_ts,
                dividend_ts)
            helper.setPricingEngine(engine)
            heston_helpers.append(helper)
        lm = ql.LevenbergMarquardt(1e-8, 1e-8, 1e-8)
        model.calibrate(heston_helpers, lm,
                          ql.EndCriteria(500, 50, 1.0e-8,1.0e-8, 1.0e-8))
        theta, kappa, sigma, rho, v0 = model.params()
        
        avg = 0.0
        time.sleep(0.005)
        
        for i in range(min(len(heston_helpers), len(K))):
            opt = heston_helpers[i]
            err = (opt.modelValue() / max(opt.marketValue(),0.001) - 1.0)
            
            avg += abs(err)
            
        avg = avg*100.0/len(heston_helpers)
        heston_df_s.loc['theta',t] = theta
        heston_df_s.loc['kappa',t] = kappa
        heston_df_s.loc['sigma',t] = sigma
        heston_df_s.loc['rho',t] = rho
        heston_df_s.loc['v0',t] = v0
        heston_df_s.loc['error',t] = avg/100
        
        heston_params = {
            'theta':theta, 
            'kappa':kappa, 
            'sigma':sigma, 
            'rho':rho, 
            'v0':v0,
            'error': (avg/100,f"{int(t)}D")
            }
        sets_for_maturities[t_idx] = heston_params
        
        print("-"*40)
        print("Total Average Abs Error (%%) : %5.3f" % (avg))
        print(f"for {int(t)} day maturity\n")
        for key, value in heston_params.items():
            print(f'{key}: {value}')

        print("-"*40)

    heston_dicts[s_idx] = sets_for_maturities

end_time = time.time()
runtime = int(end_time-start_time)


print('\nmaturities under \n1% abs error:')
tolerance = 0.01
for i, s in enumerate(heston_dicts):
    for j, t in enumerate(s):
        error = t['error']
        if error[0] < tolerance:
            print("-"*15)
            print(f'error: {round(error[0]*100,4)}%')
            print(f"maturity: {error[1]}")
            print("-"*15)
        else:
            pass


mask = heston_df_s.loc['error', :] < tolerance
heston_df = heston_df_s.loc[:, mask]

print(f"\n{heston_df}")

