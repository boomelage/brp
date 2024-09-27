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
import pandas as pd
import numpy as np
import QuantLib as ql
from itertools import product
from routine_calibration_global import calibrate_heston
from bicubic_interpolation import make_bicubic_functional, bicubic_vol_row
from train_generation_barriers import generate_barrier_options
from settings import model_settings
from tqdm import tqdm
ms = model_settings()
os.chdir(current_dir)

from routine_historical_collection import collect_historical_data

def generate_features(K,T,s):
    features = pd.DataFrame(
        product(
            [float(s)],
            K,
            T
            ),
        columns=[
            "spot_price", 
            "strike_price",
            "days_to_maturity"
                  ])
    return features

def generate_initial_barrier_features(s,T,K,outins,updown,ws):
    features = pd.DataFrame(
        product(
            [s],
            T,
            K,
            [updown],
            outins,
            ws
            ),
        columns=[
            'spot_price', 
            'days_to_maturity',
            'strike_price',
            'updown',
            'outin',
            'w'
                  ])
    return features

outins = [
    
    'Out',
    'In'
    
    ]


ws = [
      'call',
      'put'
      ]

"""
# =============================================================================
                        historical generation routine
"""

historical_data = collect_historical_data()

total = historical_data.shape[0]
historical_option_data = pd.DataFrame()

hist_bar = ms.make_tqdm_bar(
    total=total, desc='generating', unit='days', leave=True)

training_data = pd.DataFrame()
for i, row in historical_data.iterrows():
    s = row['spot_price']
    dtdate = row['date']
    calculation_date = ql.Date(dtdate.day,dtdate.month,dtdate.year)
    expiry_dates = np.array([
            calculation_date + ql.Period(30,ql.Days), 
            calculation_date + ql.Period(60,ql.Days), 
            calculation_date + ql.Period(3,ql.Months), 
            calculation_date + ql.Period(6,ql.Months),
            calculation_date + ql.Period(12,ql.Months), 
            # calculation_date + ql.Period(18,ql.Months), 
            # calculation_date + ql.Period(24,ql.Months)
          ],dtype=object)
    
    T = expiry_dates - calculation_date
    g = row['dividend_rate']
    
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
    
    """
    NOTE: the make_bicubic_functional function takes the current volatility 
    surface estimated via Derman only for the coefficients
    """
    
    bicubic_vol = make_bicubic_functional(
        ms.derman_ts, 
        ms.derman_ts.index.tolist(), 
        ms.derman_ts.columns.tolist())
   
    calibration_dataset = generate_features(
        K, T, s)

    calibration_dataset = calibration_dataset.apply(
        bicubic_vol_row, axis = 1, bicubic_vol = bicubic_vol)
    calibration_dataset = calibration_dataset.copy()
    calibration_dataset['risk_free_rate'] = 0.04
    calibration_dataset['dividend_rate'] = g
    
    heston_parameters, performance_df = calibrate_heston(
        calibration_dataset, s, calculation_date)
    
    """"""
    T = [
        1,7,10,14,30,
        90,180,360
         ]

    n_strikes = 7
    down_k_spread = 0.1
    up_k_spread = 0.1

    n_barriers = 5
    barrier_spread = 0.005                  
    n_barrier_spreads = 20
    
    training_data = generate_barrier_options(
                n_strikes, down_k_spread, up_k_spread,
                n_barriers, barrier_spread, n_barrier_spreads,
                ms.calculation_date, T, ms.s, heston_parameters, 'hist_outputs'
                )
    
    historical_option_data = pd.concat(
        [historical_option_data,training_data],
        ignore_index=True)
    
    tqdm.write(dtdate.strftime("%Y%m%d"))
    hist_bar.update(1)
    
hist_bar.close()

historical_option_data