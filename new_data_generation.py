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

lower_moneyness = 0.2
upper_moneyness = 1.5
nstrikes = 100

S = 100
r = np.arange(0, 0.051, 0.01)
T = np.arange(3/12, 2.01, 1/12)
sigma = np.arange(0.1, 0.81, 0.1)
w = pd.DataFrame(np.array((-1,1)))

def generate_features():
    spots = np.ones(nstrikes) * S
    K = np.linspace(S * lower_moneyness, S * upper_moneyness, nstrikes)
    features = pd.DataFrame(
        product(spots, K, r, T, sigma, w.values.flatten()),
        columns=["spot_price", 
                 "strike_price", 
                 "risk_free_rate", 
                 "years_to_maturity", 
                 "volatility", 
                 "w"]
    )
    features['calculation_date'] = ql.Date.todaysDate()
    features['maturity_date'] = features.apply(
        lambda row: row['calculation_date'] + ql.Period(
            int(math.floor(row['years_to_maturity'] * 365)), ql.Days), axis=1)
    return features
