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

from derman_underlying_initialisation import derman_coefs,derman_maturities,\
    implied_vols, contract_details, S, K, T

from testing12 import derman_atm_dfs

from Derman import derman
derman = derman(derman_coefs=derman_coefs,implied_vols=implied_vols)

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


S = S
groupedby_s = contract_details.groupby(by='spot_price')

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

    print(process)
    heston_helpers = []
    
    
    derman_df_for_s = derman.make_derman_df_for_S(
        s, K, T, derman_atm_dfs[s_idx], contract_details, derman_coefs, \
            derman_maturities)

    derK = derman_df_for_s.index
    derT = derman_df_for_s.columns
    
    implied_vols_matrix = ql.Matrix(len(derK),len(derT),0.0)
    for i, k in enumerate(derK):
        for j, t in enumerate(derT):
            implied_vols_matrix[i][j] = derman_df_for_s.loc[k,t]
    print(implied_vols_matrix)
    
    expiration_dates = ms.compute_ql_maturity_dates(derT)
    
    
    black_var_surface = ms.make_black_var_surface(
        expiration_dates, derK.astype(float), implied_vols_matrix)
    
    groupedby_sk = contract_details_for_s.groupby(by='strike_price')
    
    for kk in derK:
        
    
    
    
    
    