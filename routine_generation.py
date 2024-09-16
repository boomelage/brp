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
pd.set_option('display.max_columns', None)
pd.reset_option('display.max_rows', None)
# pd.reset_option('display.max_columns', None)

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

from derman_test import derman_coefs, atm_volvec

def compute_derman_volatility_row(row,atm_volvec):
    try:
        t = int(row['days_to_maturity'])
        moneyness = row['spot_price'] - row['strike_price']
        atm_value = atm_volvec[t]
        alpha = derman_coefs.loc['alpha',t]
        b = derman_coefs.loc['b',t]
        derman_vol = atm_value + alpha + b*moneyness
        row['volatility'] = derman_vol
        return row
    except Exception:
        print(f'no coefficient for {t} day maturity')
        row['volatility'] = np.nan
        return row

Kitm = np.linspace(int(s+1),int(s*1.5),int(1e3))
Kotm = np.linspace(int(s*0.5), int(s-1),int(1e3))
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


from threadpooler import threadpooler
def generate_otm():
    otmfeatures = generate_features(Kotm, T, s)
    return otmfeatures
def generate_itm():
    itmfeatures = generate_features(Kitm, T, s)
    return itmfeatures
functions = [generate_otm, generate_itm]
results = threadpooler(functions)
itmfeatures = results['generate_itm']['outcome']
otmfeatures = results['generate_otm']['outcome']

features = pd.concat([itmfeatures,otmfeatures])
features = features.apply(
    lambda row: compute_derman_volatility_row(row, atm_volvec), axis=1)

features['w'] = 1 # flag for call/put
features = features.dropna()

dataset = features.copy()



"""
                                                  generating synthetic dataset
"""

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

print(f'\n{dataset}\n')
