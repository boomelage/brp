#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Sep  4 17:22:26 2024

"""


import pandas as pd
import numpy as np
from itertools import product
import QuantLib as ql
import math
from pricing import BS_price_vanillas
S = 220
spots = np.ones(1) * S

lower_moneyness = 0.2
upper_moneyness = 1.5
nstrikes = 100
K = np.linspace(S * lower_moneyness, S * upper_moneyness, nstrikes)

# smallest_mat = 1/365
# biggest_mat = 1.5
# n_maturities = 20
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

min_vol = 0.01
max_vol = 0.8
n_vols = n_maturities * nstrikes
sigma = np.linspace(min_vol, max_vol, n_vols)

features['volatility'] = sigma
r = 0.01
features['risk_free_rate'] = r
features['w'] = 1

# r = np.linspace(0.005,0.05,n_maturities)
# for j in range (0,len(sigma)):
#     if features['years_to_maturity'][j] == T[i]:
#        features['risk_free_rate'][j] = r[i]


vanilla_prices = BS_price_vanillas(features)

vanilla_prices['dividend_rate'] = 0
vanilla_prices['v0'] = 0.04
vanilla_prices['kappa'] = 1.0
vanilla_prices['theta'] = 0.04
vanilla_prices['sigma'] = 0.2
vanilla_prices['rho'] = -0.5

vanilla_prices['calculation_date'] = ql.Date.todaysDate()
vanilla_prices['maturity_date'] = vanilla_prices.apply(
    lambda row: row['calculation_date'] + ql.Period(
        int(math.floor(row['years_to_maturity'] * 365)), ql.Days), axis=1)


row = vanilla_prices.iloc[0,:]

row['calculation_date']

data = vanilla_prices

rates = np.ones(len(T))*r

from borrowed_heston import QuantlibCalibration


params = QuantlibCalibration.fit_model(data)

print(f'\n{params}')
