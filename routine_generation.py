#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Sep  7 13:03:40 2024

"""
import os
pwd = str(os.path.dirname(os.path.abspath(__file__)))
os.chdir(pwd)
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
from pricing import BS_price_vanillas, noisyfier
from settings import model_settings
ms = model_settings()
settings = ms.import_model_settings()
security_settings = settings[0]['security_settings']
s = security_settings[5]
"""
# =============================================================================
                                                           generation procedure
"""

from routine_Derman import derman_coefs
from routine_collection import collect_directory_market_data
contract_details = collect_directory_market_data()

"""
                     generating features based on available Derman coefficients
"""

k = np.sort(contract_details['strike_price'].unique())
t = np.sort(derman_coefs.columns)
features = pd.DataFrame(
    product(
        [s],
        k,
        t,
        ),
    columns=[
        "spot_price", 
        "strike_price",
        "days_to_maturity",
              ])


"""
                                getting r and g pivots from current market_data
"""

details_indexed = contract_details.copy().set_index([
    'strike_price','days_to_maturity'])
rfrpivot = contract_details.pivot_table(
    values = 'risk_free_rate', 
    index = 'strike_price', 
    columns = 'days_to_maturity'
    )
dvypivot = contract_details.pivot_table(
    values = 'dividend_rate', 
    index = 'strike_price', 
    columns = 'days_to_maturity'
    )

rfrpivot

rvec = np.zeros(rfrpivot.shape[1],dtype=float)
rvec = pd.DataFrame(rvec)
rvec.index = rfrpivot.columns
for i, k in enumerate(rfrpivot.index):
    for j, t in enumerate(rfrpivot.columns):
        rvec.loc[t] = float(np.median(rfrpivot.loc[:, t].dropna().unique()))

gvec = np.zeros(dvypivot.shape[1],dtype=float)
gvec = pd.DataFrame(gvec)
gvec.index = dvypivot.columns
for i, k in enumerate(dvypivot.index):
    for j, t in enumerate(dvypivot.columns):
        gvec.loc[t] = float(np.median(dvypivot.loc[:, t].dropna().unique()))


rates_dict = {'risk_free_rate':rvec,'dividend_rate':gvec}

    # example
t = (rvec.index[45],0)
rt0 = rates_dict['risk_free_rate'].loc[t]
print(f'\nexample rt0: {rt0}\n')



"""
mapping appropriate rates
"""


def map_rate(ratename):
    for row in features.index:
        try:
            t = (int(features.iloc[row]['days_to_maturity']),0)
            features.loc[row,ratename] = rates_dict[ratename].loc[t]
        except Exception:
            features.loc[row,ratename] = np.nan
    return features
        
features = map_rate('risk_free_rate')
features = map_rate('dividend_rate')

"""
                              computing Derman estimation of implied volatility
"""
from routine_Derman import atm_vols

def compute_derman_volatility_row(row,atm_vols):
    try:
        t = int(row['days_to_maturity'])
        moneyness = row['spot_price'] - row['strike_price']
        atm_value = atm_vols[t]
        alpha = derman_coefs.loc['alpha',t]
        b = derman_coefs.loc['b',t]
        derman_vol = atm_value + alpha + b*moneyness
        row['volatility'] = derman_vol
        return row
    except Exception:
        row['volatility'] = np.nan
        return row

features = features.apply(
    lambda row: compute_derman_volatility_row(row, atm_vols), axis=1)

features['w'] = 1 # flag for call/put
features['calculation_date'] = contract_details['calculation_date']
features['maturity_date'] = contract_details['maturity_date']
features = features.dropna()

print(f'\n{features}\n')

"""
                                                    generating synthetic dataset
"""


option_prices = BS_price_vanillas(features)

# from routine_calibration import heston_dicts
# from pricing import heston_price_vanillas
# # map appropriate parameters
# option_prices = heston_price_vanillas(option_prices)

dataset = noisyfier(option_prices)
dataset = dataset[~(
    dataset['observed_price']<0.01*s
    )]
dataset = dataset.dropna()
dataset
print(f'\noriginal count: {contract_details.shape[0]}\n')
print(f'\nnew count: {dataset.shape[0]}')
print('\n')
print(f'{dataset.describe()}\n')
