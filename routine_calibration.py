# -*- coding: utf-8 -*-
"""
Created on Fri Sep 13 21:45:51 2024

"""

def clear_all():
    globals_ = globals().copy()  # Make a copy to avoid 
    for name in globals_:        # modifying during iteration
        if not name.startswith('_') and name not in ['clear_all']:
            del globals()[name]
clear_all()
import os
pwd = str(os.path.dirname(os.path.abspath(__file__)))
os.chdir(pwd)

import time
start_time = time.time()
import QuantLib as ql
import numpy as np

from settings import model_settings
ms = model_settings()
settings = ms.import_model_settings()
calculation_date = settings[0]['calculation_date']
day_count = settings[0]['day_count']
calendar = settings[0]['calendar']
security_settings = settings[0]['security_settings']
ticker = security_settings[0]
lower_strike = security_settings[1]
upper_strike = security_settings[2]
lower_maturity = security_settings[3]
upper_maturity = security_settings[4]
s = security_settings[5]

S = [s]


from routine_generation import rates_dict
from routine_Derman import derman_ts
ts_df = derman_ts
K = ts_df.index
T = ts_df.columns

heston_dicts = np.empty(len(S),dtype=object)
for s_idx, s in enumerate(S):
    S_handle = ql.QuoteHandle(ql.SimpleQuote(s))
    derK = np.sort(ts_df.index).astype(float)
    derT = np.sort(ts_df.columns).astype(float)
    implied_vols_matrix = ms.make_implied_vols_matrix(derK, derT, ts_df)
    expiration_dates = ms.compute_ql_maturity_dates(derT)
    black_var_surface = ms.make_black_var_surface(
        expiration_dates, derK.astype(float), implied_vols_matrix)
    
    sets_for_maturities = np.empty(len(derT),dtype=object)
    for t_idx, t in enumerate(derT):
        
        risk_free_rate = float(rates_dict['risk_free_rate'].loc[t,0])
        dividend_rate = float(rates_dict['dividend_rate'].loc[t,0])
        flat_ts = ql.YieldTermStructureHandle(ql.FlatForward(
            calculation_date, risk_free_rate, day_count))
        dividend_ts = ql.YieldTermStructureHandle(ql.FlatForward(
            calculation_date, dividend_rate, day_count))
        
        v0 = 0.01; kappa = 0.2; theta = 0.02; rho = -0.75; sigma = 0.5; 
        process = ql.HestonProcess(
            flat_ts,                
            dividend_ts,            
            S_handle,               
            v0,                     # Initial volatility
            kappa,                  # Mean reversion speed
            theta,                  # Long-run variance (volatility squared)
            sigma,                  # Volatility of the volatility
            rho                     # Correlation between asset and volatility
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
            err = (opt.modelValue() / opt.marketValue() - 1.0)
            
            avg += abs(err)
            
        avg = avg*100.0/len(heston_helpers)
        
        heston_params = {
            'theta':theta, 
            'kappa':kappa, 
            'sigma':sigma, 
            'rho':rho, 
            'v0':v0,
            'error':avg
            }
        sets_for_maturities[t_idx] = heston_params
        print("-"*40)
        print("Total Average Abs Error (%%) : %5.3f" % (avg))
        print('\nHeston model parameters:')
        for key, value in heston_params.items():
            print(f'{key}: {value}')
        print(f"\nfor {int(t)} day maturity")
        print("-"*40)
    heston_dicts[s_idx] = sets_for_maturities
end_time = time.time()
runtime = int(end_time-start_time)
print(f"total model runtime: {runtime} seconds")

