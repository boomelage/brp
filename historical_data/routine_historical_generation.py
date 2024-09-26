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
sys.path.append(os.path.join('term_structure',parent_dir))
import pandas as pd
import numpy as np
import QuantLib as ql
from itertools import product
from routine_calibration_global import calibrate_heston
from bicubic_interpolation import make_bicubic_functional, bicubic_vol_row
from settings import model_settings
ms = model_settings()
os.chdir(current_dir)
from routine_historical_collection import collect_historical_data
from train_generation_historical_barriers import generate_historical_barriers

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



"""
historical generation routine
"""
historical_data = collect_historical_data()

total = historical_data.shape[0]
historical_option_data = pd.DataFrame()

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
    
    
    # training_data_subset = generate_historical_barriers(
    #     s, calculation_date, g, heston_parameters)
    
    # print(training_data_subset['barrier_price'].unique())
    
    # training_data = pd.concat([training_data, training_data_subset],
    #                           ignore_index=True)
    


    
   