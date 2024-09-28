# -*- coding: utf-8 -*-
"""
Created on Sat Sep 21 16:10:31 2024

generation routine
"""
import os
import sys
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)
sys.path.append(os.path.join(parent_dir,'train_data'))
sys.path.append(os.path.join(parent_dir,'term_structure'))
import time
import pandas as pd
import numpy as np
import QuantLib as ql
from tqdm import tqdm
from itertools import product
from datetime import datetime
from routine_calibration_global import calibrate_heston
from bicubic_interpolation import make_bicubic_functional, bicubic_vol_row
from train_generation_barriers import concat_barrier_features, generate_barrier_options
from settings import model_settings
ms = model_settings()
os.chdir(current_dir)
from routine_historical_collection import collect_historical_data
historical_data = collect_historical_data()

"""
# =============================================================================
                        historical generation routine
"""

    
# T = [
    
#     1,
#     7,
#     10,
#     14,
#     30,
#     90,
#     180,
#     360
    
#        ]
      
# T = np.arange(1,4,1)
n_strikes = 10
down_k_spread = 0.1
up_k_spread = 0.1

n_barriers = 30
barrier_spread = 0.005                  
n_barrier_spreads = 30



historical_barriers = pd.DataFrame()
for i, row in historical_data.iterrows():
    print(row)
    s = row['spot_price']
    g = row['dividend_rate']/100
    dtdate = row['date']
    calculation_date = ql.Date(dtdate.day,dtdate.month,dtdate.year)
    
    
    """
    ===========================================================================
    calibration
    """
    
    calibration_T = ms.derman_coefs.index.astype(int)
    
    atm_volvec = np.array(row[
        [
            '30D', '60D', '3M', '6M', '12M', 
            ]
        ]/100,dtype=float)
    
    atm_volvec = pd.Series(atm_volvec)
    atm_volvec.index = calibration_T
    
    n_hist_spreads = 10
    historical_spread = 0.005
    n_strikes = 10
    
    K = np.linspace(
        s*(1 - n_hist_spreads * historical_spread),
        s*(1 + n_hist_spreads * historical_spread),
        n_strikes)
    
    derman_ts = ms.make_derman_surface(s,K,calibration_T,ms.derman_coefs,atm_volvec)
       
    bicubic_vol = make_bicubic_functional(derman_ts,K.tolist(),calibration_T.tolist())
        
    calibration_dataset =  pd.DataFrame(
        product(
            [s],
            K,
            calibration_T,
            ),
        columns=[
            'spot_price', 
            'strike_price',
            'days_to_maturity',
                  ])
    
    calibration_dataset = calibration_dataset.apply(
        bicubic_vol_row, axis = 1, bicubic_vol = bicubic_vol)
    calibration_dataset = calibration_dataset.copy()
    calibration_dataset['risk_free_rate'] = 0.04
    
    r = 0.04
    
    heston_parameters, performance_df = calibrate_heston(
            calibration_dataset, 
            s,
            r,
            g,
            calculation_date
            )
    
    v0 = heston_parameters['v0']
    theta = heston_parameters['theta']
    kappa = heston_parameters['kappa']
    eta = heston_parameters['eta']
    rho = heston_parameters['rho']
    
    test_t = calibration_T[0]
    test_k = float(s*0.8)
    test_volatility =  float(atm_volvec[calibration_T[0]])
    test_w = 'call'
    expiration_date = calculation_date + ql.Period(int(test_t),ql.Days)
    bs = ms.ql_black_scholes(
            s,test_k,r,0.00,
            test_volatility,test_w,
            calculation_date, 
            expiration_date
            )
    heston = ms.ql_heston_price(
                s,test_k,r,0.00,test_w,
                v0,kappa,theta,eta,rho,
                calculation_date,
                expiration_date
                )
    my_bs = ms.black_scholes_price(s,test_k,test_t,r,test_volatility,test_w)
    
    tqdm.write(f"\nnumpy black scholes, quantlib bs, quantlib heston: "
          f"{round(my_bs,2)}, {round(bs,2)}, {round(heston,2)}")
    tqdm.write(
        f"\n{dtdate.strftime('%A %d %B %Y')} "
        f"{int(i)}/{historical_data.shape[0]}\n")
    
    """
    ===========================================================================
    data generation
    """
    T = calibration_T
    features = concat_barrier_features(
        s,T,g,heston_parameters,
        down_k_spread, up_k_spread, n_strikes,
        barrier_spread,n_barrier_spreads,n_barriers
            )
    
    training_data = generate_barrier_options(
            features, calculation_date, heston_parameters, g, 'hist_outputs')
    
    historical_option_data = pd.concat(
        [historical_barriers,training_data],
        ignore_index=True)
    
    print(f"\n{training_data}\n")
    print(f"\n{dtdate.strftime('%A %d %B %Y')}\n")
