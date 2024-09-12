# -*- coding: utf-8 -*-
"""
Created on Mon Sep  9 00:17:45 2024

"""
def clear_all():
    globals_ = globals().copy()  # Make a copy to avoid 
    for name in globals_:        # modifying during iteration
        if not name.startswith('_') and name not in ['clear_all']:
            del globals()[name]
clear_all()
import os
import QuantLib as ql
import warnings
import numpy as np
import time
warnings.simplefilter(action='ignore')
pwd = str(os.path.dirname(os.path.abspath(__file__)))
os.chdir(pwd)

from settings import model_settings
ms = model_settings()
settings, ezprint = ms.import_model_settings()
dividend_rate = settings['dividend_rate']
risk_free_rate = settings['risk_free_rate']
calculation_date = settings['calculation_date']
day_count = settings['day_count']
calendar = settings['calendar']
flat_ts = settings['flat_ts']
dividend_ts = settings['dividend_ts']


# =============================================================================
                                                                # ivol and data

# filter out sections of incomplete term structure data and iterate 
# the routine through a list of maturities,strikes, black_var_surface for each
# individual ivoldf
from routine_ivol_collection import expiration_dates, strikes, S, black_var_surface, implied_vols_matrix


S_handle = ql.QuoteHandle(ql.SimpleQuote(S))

print("\n implied_vols_matrix:")
print(f"\n{implied_vols_matrix}")

# =============================================================================
                                                          # calibration routine

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

print(process)
heston_helpers = []

for current_index, date in enumerate(expiration_dates):
    print(f"\nCurrently calibrating for maturity: {date}")
    black_var_surface.setInterpolation("bicubic")
    for j, s in enumerate(strikes):
       t = day_count.yearFraction(calculation_date, date)
       sigma = black_var_surface.blackVol(t, s)  
       helper = ql.HestonModelHelper(
           ql.Period(int(t * 365), ql.Days),
           calendar, S, s,
           ql.QuoteHandle(ql.SimpleQuote(sigma)),
           flat_ts, dividend_ts
           )
       helper.setPricingEngine(engine)
       heston_helpers.append(helper)
    lm = ql.LevenbergMarquardt(1e-8, 1e-8, 1e-8)
    model.calibrate(heston_helpers, lm,
                     ql.EndCriteria(500, 50, 1.0e-8,1.0e-8, 1.0e-8))
    theta, kappa, sigma, rho, v0 = model.params()
    
    print (
        "\ntheta = %f, kappa = %f, "
        "sigma = %f, rho = %f, v0 = %f" % (theta, kappa, 
                                           sigma, rho, v0))
    avg = 0.0
    time.sleep(0.005)
    print ("%15s %15s %15s %20s" % (
        "Strikes", "Market Value",
          "Model Value", "Relative Error (%)"))
    print ("="*70)
    for i in range(min(len(heston_helpers), len(strikes))):
        opt = heston_helpers[i]
        err = (opt.modelValue() / opt.marketValue() - 1.0)
        print(f"{strikes[i]:15.2f} {opt.marketValue():14.5f} "
              f"{opt.modelValue():15.5f} {100.0 * err:20.7f}")
        avg += abs(err)  # accumulate the absolute error
    avg = avg*100.0/len(heston_helpers)
    print("-"*70)
    print("Total Average Abs Error (%%) : %5.3f" % (avg))
    heston_params = {
        'theta':theta, 
        'kappa':kappa, 
        'sigma':sigma, 
        'rho':rho, 
        'v0':v0
        }
    print('\nHeston model parameters:')
    for key, value in heston_params.items():
        print(f'{key}: {value}')        





                                                            