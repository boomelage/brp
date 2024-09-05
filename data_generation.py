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
from pricing import heston_price_vanillas, BS_price_vanillas

S = 220
spots = np.ones(1) * S

lower_moneyness = 0.2
upper_moneyness = 1.5
nstrikes = 8
K = np.linspace(S * lower_moneyness, S * upper_moneyness, nstrikes)

# smallest_mat = 1/365
# biggest_mat = 1.5
# n_maturities = 24
# T = np.linspace(smallest_mat, biggest_mat, n_maturities)

T = np.arange(3/12, 2.01, 1/12)
n_maturities = len(T)

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

generate_features()

features = generate_features()

min_vol = 0.28
max_vol = 0.4
n_vols = n_maturities * nstrikes
sigma = np.linspace(min_vol, max_vol, n_vols)
features['volatility'] = sigma

r = 0.05
features['risk_free_rate'] = r

dividend_rate = 0.0
features['dividend_rate'] = dividend_rate

features['w'] = 1

# =============================================================================
# 
# =============================================================================

# vanilla_prices = BS_price_vanillas(features)
vanilla_prices = features

# =============================================================================
# 
# =============================================================================

vanilla_prices['calculation_date'] = ql.Date.todaysDate()
vanilla_prices['maturity_date'] = vanilla_prices.apply(
    lambda row: row['calculation_date'] + ql.Period(
        int(math.floor(row['years_to_maturity'] * 365)), ql.Days), axis=1)

data = vanilla_prices

row = data.iloc[0,:]

rates = np.ones(len(T))*r

expiration_dates = data['maturity_date'].unique()

strikes = data['strike_price'].unique()

implied_vols = ql.Matrix(len(strikes), len(expiration_dates))

calibrated_features = calibrate_heston(vanilla_prices, dividend_rate, r)

dataset = heston_price_vanillas(calibrated_features)

print(dataset)

