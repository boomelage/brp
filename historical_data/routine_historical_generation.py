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
from train_generation_barriers import concat_barrier_features
from settings import model_settings
ms = model_settings()
os.chdir(current_dir)
from routine_historical_collection import collect_historical_data
historical_data = collect_historical_data()

"""
# =============================================================================
                        historical generation routine
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

n_spots = historical_data.shape[0]

n_maturities = len(T)

total = 2*n_spots*n_maturities*n_strikes*n_barriers



historical_bar = ms.make_tqdm_bar(
    desc="pricing",total=total, unit='contracts',leave=True)

historical_barriers = pd.DataFrame()
for i, row in historical_data.iterrows():
    
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
    
    v0 = heston_parameters['v0'].iloc[0]
    theta = heston_parameters['theta'].iloc[0]
    kappa = heston_parameters['kappa'].iloc[0]
    eta = heston_parameters['eta'].iloc[0]
    rho = heston_parameters['rho'].iloc[0]
    
    t = calibration_T[0]
    
    
    k = float(s*0.8)
    volatility =  float(atm_volvec[calibration_T[0]])
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
    
    tqdm.write(f"\nnumpy black scholes, quantlib bs, quantlib heston: "
          f"{int(my_bs)}, {int(bs)}, {int(heston)}")
    tqdm.write(f"\n{dtdate.strftime('%A %d %B %Y')} {int(i+1)}/{n_spots}\n")
    
    """
    ===========================================================================
    data generation
    """
    
    features = concat_barrier_features(
            s,K,T,g,heston_parameters,
            barrier_spread,n_barrier_spreads,n_barriers)
    
    features['eta'] = heston_parameters['eta'].iloc[0]
    features['theta'] = heston_parameters['theta'].iloc[0]
    features['kappa'] = heston_parameters['kappa'].iloc[0]
    features['rho'] = heston_parameters['rho'].iloc[0]
    features['v0'] = heston_parameters['v0'].iloc[0]
    features['heston_price'] = np.nan
    features['barrier_price'] = np.nan
    
    for i, row in features.iterrows():
        
        barrier_type_name = row['barrier_type_name']
        barrier = row['barrier']
        s = row['spot_price']
        k = row['strike_price']
        t = row['days_to_maturity']
        w = row['w']
        r = 0.04
        rebate = 0.
        
        v0 = heston_parameters['v0'].iloc[0]
        kappa = heston_parameters['kappa'].iloc[0] 
        theta = heston_parameters['theta'].iloc[0] 
        eta = heston_parameters['eta'].iloc[0] 
        rho = heston_parameters['rho'].iloc[0]
        expiration_date = calculation_date + ql.Period(int(t),ql.Days)
        
        heston_price = ms.ql_heston_price(
            s,k,t,r,g,w,
            v0,kappa,theta,eta,rho,
            calculation_date,
            expiration_date
            )
        features.at[i,'heston_price'] = heston_price
        
        barrier_price = ms.ql_barrier_price(
                s,k,t,r,g,calculation_date,w,
                barrier_type_name,barrier,rebate,
                v0, kappa, theta, eta, rho)
    
        features.at[i,'barrier_price'] = barrier_price
        
        historical_bar.update(1)
    
    training_data = features.copy()
    
    training_data = ms.noisyfier(training_data)
    
    pd.set_option("display.max_columns",None)
    print(f'\n{training_data}\n')
    print(f'\n{training_data.describe()}')
    pd.reset_option("display.max_columns")
    
    file_date = datetime(
        calculation_date.year(), 
        calculation_date.month(), 
        calculation_date.dayOfMonth())
    date_tag = file_date.strftime("%Y-%m-%d")
    file_time = datetime.fromtimestamp(time.time())
    file_time_tag = file_time.strftime("%Y-%m-%d %H%M%S")
    training_data.to_csv(os.path.join(
        'hist_outputs',f'barriers {date_tag} {file_time_tag}.csv'))
    
    historical_option_data = pd.concat(
        [historical_barriers,training_data],
        ignore_index=True)
    print(f"\n{training_data}\n")
    print(f"\n{dtdate.strftime('%A %d %B %Y')} {int(i+1)}/{n_spots}\n")
historical_bar.close()

