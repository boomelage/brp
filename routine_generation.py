#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Sep  7 13:03:40 2024

"""
import os
import sys
os.chdir(os.path.dirname(os.path.abspath(__file__)))
sys.path.append('term_structure')
sys.path.append('contract_details')
sys.path.append('misc')
import numpy as np
import pandas as pd
from itertools import product
pd.set_option('display.max_columns',None)
pd.reset_option('display.max_rows')

"""

below is a routine which can take a dataset of spots, strikes, atm implied
volatility, risk_free_rate, dividend_yield, maturities, and momentarily a
static flag 'w' set to 1 indicating a call payoff

it is temporarily taking current option data with available volatilities.
the idea is that one can download long time series of at-the-money implied
volatiltities even from educational bloomberg terminals and approximate the
implied voilatility using Derman's method for any combination of spots, 
strikes, and maturities. in routine_generation.py, ther is a method to map 
dividend rates and risk free rates to the aforementioned combinations which 
can be massive when using vectors from timeseries data for the cartesian 
product. this would allow one to easily create large training datasets from 
rather sparse information. naturally, there are many assumptions underpinning 
the implied volatility being a functional form of maturity, strike, and spot.

"""

from settings import model_settings
ms = model_settings()
settings = ms.import_model_settings()
security_settings = settings[0]['security_settings']
s = security_settings[5]

"""
# =============================================================================
                                                           generation procedure
                                                           
                                                           
 computing Derman estimation of implied volatilities for available coefficients
 
"""

from derman_test import derman_coefs, atm_volvec, derman_ts
kUpper = int(max(derman_ts.index))
kLower = int(min(derman_ts.index))
Kitm = np.linspace(int(s*1.001),int(kUpper),50)
Kotm = np.linspace(int(kLower), int(s*0.999),50)
T = np.sort(derman_coefs.columns.unique().astype(int))
def generate_features(K,T,s):
    features = pd.DataFrame(
        product(
            [s],
            K,
            T,
            ),
        columns=[
            "spot_price", 
            "strike_price",
            "days_to_maturity",
                  ])
    return features


def generate_otm():
    otmfeatures = generate_features(Kotm, T, s)
    return otmfeatures
def generate_itm():
    itmfeatures = generate_features(Kitm, T, s)
    return itmfeatures

# from threadpooler import threadpooler
# functions = [generate_otm, generate_itm]
# results = threadpooler(functions)
# itmfeatures = results['generate_itm']['outcome']
# otmfeatures = results['generate_otm']['outcome']

itmfeatures = generate_itm()
otmfeatures = generate_otm()

features = pd.concat([itmfeatures,otmfeatures])
features['w'] = 1 # flag for call/put
features = features.dropna()


features['risk_free_rate'] = 0.05
features['dividend_rate'] = 0.05

def compute_derman_volatility_row(row):
    s = row['spot_price']  # Assuming s is spot_price (not defined in your function, but seems to be required)
    k = row['strike_price']  # Accessing strike_price directly
    t = row['days_to_maturity']  # Accessing days_to_maturity directly
    moneyness = s - k  # Calculate moneyness
    atm_value = atm_volvec[t]  # Look up ATM volatility for the given maturity
    b = derman_coefs.loc['b', t]  # Look up the coefficient for t
    volatility = atm_value + b * moneyness  # Compute volatility
    row['volatility'] = volatility
    return row

features = features.apply(compute_derman_volatility_row, axis=1)
features = features[~(features['volatility']<0)]

"""
                                                  generating synthetic dataset
"""

dataset = features.copy()
# from pricing import BS_price_vanillas,noisyfier

# option_prices = BS_price_vanillas(features)

# # from routine_calibration import heston_dicts
# # from pricing import heston_price_vanillas
# # # map appropriate parameters
# # option_prices = heston_price_vanillas(option_prices)

# dataset = noisyfier(option_prices)
# # dataset = dataset[~(
# #     dataset['observed_price']<0.01*dataset['spot_price']
# #     )]
# # dataset = dataset.dropna()
# dataset
# 

print('\ndata generated\n')
