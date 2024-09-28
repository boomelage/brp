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
import pandas as pd
import numpy as np
import QuantLib as ql
from itertools import product
from routine_calibration_global import calibrate_heston
from bicubic_interpolation import make_bicubic_functional, bicubic_vol_row
from train_generation_barriers import generate_barrier_options, concat_barrier_features
from settings import model_settings
ms = model_settings()
os.chdir(current_dir)
from routine_historical_collection import collect_historical_data


"""
# =============================================================================
                        historical generation routine
"""

historical_data = collect_historical_data()

total = historical_data.shape[0]
historical_option_data = pd.DataFrame()


training_data = pd.DataFrame()
for i, row in historical_data.iterrows():
    
    s = row['spot_price']
    g = row['dividend_rate']/100
    dtdate = row['date']
    calculation_date = ql.Date(dtdate.day,dtdate.month,dtdate.year)
    
    
    T = ms.derman_coefs.index.astype(int)
    
    atm_volvec = np.array(row[
        [
            '30D', '60D', '3M', '6M', '12M', 
            ]
        ]/100,dtype=float)
    
    atm_volvec = pd.Series(atm_volvec)
    atm_volvec.index = T
    
    
    """
    calibration dataset construction
    """
    
    n_hist_spreads = 10
    historical_spread = 0.005
    n_strikes = 10
    
    K = np.linspace(
        s*(1 - n_hist_spreads * historical_spread),
        s*(1 + n_hist_spreads * historical_spread),
        n_strikes)
    
    derman_ts = ms.make_derman_surface(s,K,T,ms.derman_coefs,atm_volvec)
       
    bicubic_vol = make_bicubic_functional(derman_ts,K.tolist(),T.tolist())
        
    calibration_dataset =  pd.DataFrame(
        product(
            [s],
            K,
            T,
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
    
    v0 = heston_parameters['v0'].iloc[0]
    theta = heston_parameters['theta'].iloc[0]
    kappa = heston_parameters['kappa'].iloc[0]
    eta = heston_parameters['eta'].iloc[0]
    rho = heston_parameters['rho'].iloc[0]
    
    t = T[0]
    
    
    k = float(s*0.8)
    volatility =  float(atm_volvec[T[0]])
    w = 'call'
    
    expiration_date = calculation_date + ql.Period(int(t),ql.Days)
    
    bs = ms.ql_black_scholes(
            s,k,r,0.00,
            volatility,w,
            calculation_date, 
            expiration_date
            )
    
    
    heston = ms.ql_heston_price(
                s,k,t,r,0.00,w,
                v0,kappa,theta,eta,rho,
                calculation_date,
                expiration_date
                )
    
    my_bs = ms.black_scholes_price(s,k,t,r,volatility,w)
    
    print(f"\nnumpy black scholes, quantlib bs, quantlib heston: "
          f"{int(my_bs)}, {int(bs)}, {int(heston)}")
    print(f"\n{dtdate.strftime('%A %d %B %Y')} {int(i+1)}/{int(total)}\n")
    """"""
    """
    data generation
    """
    T = [
        
        1,
        7,
        10,
        14,
        30,
        90,
        180,
        360
        
          ]
    
    n_strikes = 10
    down_k_spread = 0.1
    up_k_spread = 0.1
    
    n_barriers = 5
    barrier_spread = 0.005                  
    n_barrier_spreads = 20
    
    features = concat_barrier_features(
            s,K,T,g,heston_parameters,
            barrier_spread,n_barrier_spreads,n_barriers)
    
    training_data = generate_barrier_options(
        features, calculation_date, heston_parameters, g)
    historical_option_data = pd.concat(
        [historical_option_data,training_data],
        ignore_index=True)
    print(f"\n{training_data}\n")
    print(f"\n{dtdate.strftime('%A %d %B %Y')} {int(i+1)}/{int(total)}\n")
