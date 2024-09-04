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
S = 100
spots = np.ones(1) * S

lower_moneyness = 0.2
upper_moneyness = 1.5
nstrikes = 5
K = np.linspace(S * lower_moneyness, S * upper_moneyness, nstrikes)

smallest_mat = 1/252
biggest_mat = 1.5
n_maturities = 2
T = np.linspace(smallest_mat, biggest_mat, n_maturities)

def generate_features():
    features = pd.DataFrame(
        product(spots, K, T),
        columns=[
            "spot_price", 
            "strike_price", 
            "years_to_maturity"
                 ]
    )
    features['calculation_date'] = ql.Date.todaysDate()
    features['maturity_date'] = features.apply(
        lambda row: row['calculation_date'] + ql.Period(
            int(math.floor(row['years_to_maturity'] * 365)), ql.Days), axis=1)
    return features

generate_features()

features = generate_features()

print(features)

min_vol = 0.01
max_vol = 0.8
n_vols = n_maturities * nstrikes
sigma = np.linspace(min_vol, max_vol, n_vols)

features['volatility'] = sigma
features['risk_free_rate'] = 0.01
features['w'] = 1

print(features)
# r = np.linspace(0.005,0.05,n_maturities)
# for j in range (0,len(sigma)):
#     if features['years_to_maturity'][j] == T[i]:
#        features['risk_free_rate'][j] = r[i]


bs_vanilla_calls = BS_price_vanillas(features)

print(bs_vanilla_calls)

