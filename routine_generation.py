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
import QuantLib as ql
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
calculation_date = settings[0]['calculation_date']

"""
# =============================================================================
                                                           
                                                           
 computing Derman estimation of implied volatilities for available coefficients
 
"""
from import_files import raw_ts
from derman_test import derman_coefs, atm_volvec
from routine_calibration import heston_parameters
from pricing import BS_price_vanillas, noisyfier, heston_price_vanillas

kUpper = int(max(raw_ts.index))
kLower = int(min(raw_ts.index))
Kitm = np.linspace(int(s*1.001),int(kUpper),100000)
Kotm = np.linspace(int(kLower), int(s*0.999),100000)


# T = np.sort(derman_coefs.columns.unique().astype(int))
T = np.sort(heston_parameters.index)

def generate_features(K,T,s):
    features = pd.DataFrame(
        product(
            [s],
            K,
            T,
            [1,-1]
            ),
        columns=[
            "spot_price", 
            "strike_price",
            "days_to_maturity",
            "w"
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
features = features.dropna()


features['risk_free_rate'] = 0.05
features['dividend_rate'] = 0.00

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
features = features[~(features['volatility']<0)].reset_index(drop=True)

"""
                                                  generating synthetic dataset
"""
dataset = features.copy()


def map_heston_param(dataset):
    dataset['theta'] = dataset['days_to_maturity'].map(heston_parameters['theta'])
    dataset['kappa'] = dataset['days_to_maturity'].map(heston_parameters['kappa'])
    dataset['sigma'] = dataset['days_to_maturity'].map(heston_parameters['sigma'])
    dataset['rho'] = dataset['days_to_maturity'].map(heston_parameters['rho'])
    dataset['v0'] = dataset['days_to_maturity'].map(heston_parameters['v0'])
    return dataset
dataset = map_heston_param(features).dropna()



dataset = BS_price_vanillas(dataset)





# dataset['calculation_date'] = calculation_date
# dataset = dataset.apply(ms.compute_maturity_date,axis=1)
# dataset = heston_price_vanillas(dataset)


dataset = noisyfier(dataset)
dataset['error'] = (dataset['heston_price']-dataset['black_scholes'])/dataset['heston_price']
dataset = dataset[~(dataset['observed_price']<0.01*dataset['spot_price'])]
dataset = dataset[~(abs(dataset['error'])>0.01)]
dataset = dataset.dropna().reset_index(drop=True)

pd.reset_option('display.max_rows')
print(f'\ndata generated:\n{dataset}')
