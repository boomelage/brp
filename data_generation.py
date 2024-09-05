#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Sep  4 17:22:26 2024

"""
import os
script_dir = os.path.dirname(os.path.abspath(__file__))
os.chdir(script_dir)

import pandas as pd
import numpy as np
from itertools import product
import QuantLib as ql
import math
from heston_calibration import calibrate_heston
from pricing import heston_price_vanillas, noisyfier

def generate_data_subset(S):
    spots = np.ones(1) * S
    
    lower_moneyness = 0.2
    upper_moneyness = 1.5
    n_strikes = 100
    K = np.linspace(S * lower_moneyness, S * upper_moneyness, n_strikes)
    
    n_maturities = 100
    # T = np.arange(3/12, 2.01, 1/12)
    T = np.linspace(3/12, 2.01, n_maturities)
    
    def generate_features():
        features = pd.DataFrame(
            product(spots, K, T),
            columns=[
                "spot_price", 
                "strike_price", 
                "years_to_maturity"
                     ]
        )
        return features
    
    features = generate_features()
    
    # min_vol = 0.28
    # max_vol = 0.4
    # n_vols = n_maturities * n_strikes
    # sigma = np.linspace(min_vol, max_vol, n_vols)
    # features['volatility'] = sigma
    # Define the number of lists and the range size
    

    lower_vol = 0.28
    upper_vol = 0.40
    n_vols = n_strikes
    
    data = [np.linspace(lower_vol, upper_vol, n_vols) for _ in range(n_vols)]
    data = [arr.tolist() for arr in data]
    
    flattened_data = [item for sublist in data for item in sublist]
    features['volatility'] = flattened_data
    
    
    r = 0.05
    features['risk_free_rate'] = r
    
    dividend_rate = 0.0
    features['dividend_rate'] = dividend_rate
    
    features['w'] = 1
    
    vanilla_params = features
    
    vanilla_params['calculation_date'] = ql.Date.todaysDate()
    vanilla_params['maturity_date'] = vanilla_params.apply(
        lambda row: row['calculation_date'] + ql.Period(
            int(math.floor(row['years_to_maturity'] * 365)), ql.Days), axis=1)
    
    
    # rates = np.ones(len(T))*r
    
    expiration_dates = vanilla_params['maturity_date'].unique()
    
    strikes = vanilla_params['strike_price'].unique()
    
    implied_vols = ql.Matrix(len(strikes), len(expiration_dates))
    
    calibrated_features = calibrate_heston(vanilla_params, dividend_rate, r, implied_vols, data)
    
    prices = heston_price_vanillas(calibrated_features)
    
    prices = noisyfier(prices)
    
    return prices

def generate_dataset(spots):
    data_subsets = []
    for spot in spots:
        spot = spot
        subset = generate_data_subset(spot)
        data_subsets.append(subset)
    dataset =pd.concat(data_subsets, ignore_index=True)
    return dataset



        
    
