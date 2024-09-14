# -*- coding: utf-8 -*-
"""
Created on Fri Sep 13 21:45:51 2024

@author: boomelage
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

"""

calibration routine based on historical atm ivols using Derman's approximation
for otm ivols. atm ivol is momentarily a constant for simplicity. 


"""

import QuantLib as ql
import time
import numpy as np
import pandas as pd



from settings import model_settings
ms = model_settings()
settings = ms.import_model_settings()
dividend_rate = settings['dividend_rate']
risk_free_rate = settings['risk_free_rate']
calculation_date = settings['calculation_date']
day_count = settings['day_count']
calendar = settings['calendar']
flat_ts = settings['flat_ts']
dividend_ts = settings['dividend_ts']


from import_files import derman_ts, contract_details
groupedby_s = contract_details.groupby(by='strike_price')
S = [int(contract_details['spot_price'].unique()[1])]
K = derman_ts.index
T = derman_ts.columns



param_array_for_maturity = np.empty(len(T),dtype=object)
param_array_for_spot = np.empty(len(S),dtype=object)


for s_idx, s in enumerate(S):
    contract_details_for_s = groupedby_s.get_group(s)
    S_handle = ql.QuoteHandle(ql.SimpleQuote(float(s)))
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

    derK = np.sort(derman_ts.index).astype(float)
    derT = np.sort(derman_ts.columns).astype(float)
    
    implied_vols_matrix = ms.make_implied_vols_matrix(derK, derT, derman_ts)
    
    expiration_dates = ms.compute_ql_maturity_dates(derT)
        
    black_var_surface = ms.make_black_var_surface(
        expiration_dates, derK.astype(float), implied_vols_matrix)
    
    groupedby_sk = contract_details_for_s.groupby(by='strike_price')
    
    for t_idx, t in enumerate(derT):
        
        
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
        
        
        
        for k_idx, k in enumerate(derK):
            date = calculation_date + ql.Period(int(t),ql.Days)
            dt = (date - calculation_date)
            
            sigma = black_var_surface.blackVol(dt/365.25, k)  
            p = ql.Period(dt, ql.Days)
            
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
        print ("%15s %15s %15s %20s" % (
            "Strikes", "Market Value",
              "Model Value", "Relative Error (%)"))
        print ("="*70)
        for i in range(min(len(heston_helpers), len(K))):
            opt = heston_helpers[i]
            err = (opt.modelValue() / opt.marketValue() - 1.0)
            print(f"{K[i]:15.2f} {opt.marketValue():14.5f} "
                  f"{opt.modelValue():15.5f} {100.0 * err:20.7f}")
            avg += abs(err)
        print ("="*70)
        avg = avg*100.0/len(heston_helpers)
        print("-"*40)
        print("Total Average Abs Error (%%) : %5.3f" % (avg))
        heston_params = {
            'theta':theta, 
            'kappa':kappa, 
            'sigma':sigma, 
            'rho':rho, 
            'v0':v0,
            'error':avg
            }
        
        print('\nHeston model parameters:')
        for key, value in heston_params.items():
            print(f'{key}: {value}')
        print(f"\nfor {int(t)} days maturity")
        print("-"*40)
        print('\n\n')
        
        param_array_for_maturity[t_idx] =  heston_params
        
param_array_for_spot[s_idx] = param_array_for_maturity

heston_params = param_array_for_spot
