#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Sep  3 00:19:49 2024

"""

import pandas as pd
import numpy as np
from itertools import product
import QuantLib as ql
import math
# from pricing import heston_price_vanillas

# lower_moneyness = 0.2
# upper_moneyness = 1.5
# nstrikes = 10
# S = np.linspace(80,100,10)
# T = np.arange(3/12, 2.01, 1/12)
# r = np.arange(0, 0.051, 0.01)
# sigma = np.arange(0.1, 0.81, 0.1)
# w = pd.DataFrame(np.array((-1,1)))

S = np.arange(40, 61)
K = np.arange(20, 91)
r = np.arange(0, 0.051, 0.01)
T = np.arange(3/12, 2.01, 1/12)
sigma = np.arange(0.1, 0.81, 0.1)

def generate_features():
    features = pd.DataFrame(
        product(S, K, T, r, sigma),
        columns=["spot_price", 
                 "strike_price", 
                 "risk_free_rate", 
                 "years_to_maturity", 
                 "volatility"]
    )
    features['calculation_date'] = ql.Date.todaysDate()
    features['maturity_date'] = features.apply(
        lambda row: row['calculation_date'] + ql.Period(
            int(math.floor(row['years_to_maturity'] * 365)), ql.Days), axis=1)
    return features

features = generate_features()
print(features)