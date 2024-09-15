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
from pricing import BS_price_vanillas, noisyfier


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

dividend_rate = settings['dividend_rate']
risk_free_rate = settings['risk_free_rate']
calculation_date = settings['calculation_date']
day_count = settings['day_count']
calendar = settings['calendar']
flat_ts = settings['flat_ts']
dividend_ts = settings['dividend_ts']
security_settings = settings['security_settings']
ticker = security_settings[0]
lower_strike = security_settings[1]
upper_strike = security_settings[2]
lower_maturity = security_settings[3]
upper_maturity = security_settings[4]
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
dvy_K = dvypivot.index
dvy_T = dvypivot.columns
dvy_np = np.zeros((1,len(dvy_T)))
dvys = pd.DataFrame(dvy_np)
dvys.columns = dvy_T
for t in dvy_T:
        dvys[t] = float(dvypivot.loc[:,t].dropna().unique()[0])
rfr_K = rfrpivot.index
rfr_T = rfrpivot.columns
rfr_np = np.zeros((1,len(rfr_T)))
rfrs = pd.DataFrame(rfr_np)
rfrs.columns = rfr_T
for t in rfr_T:
        rfrs[t] = float(rfrpivot.loc[:,t].dropna().unique()[0])

"""
mapping appropriate rates
"""
def map_rate(rate_series, ratename):
    for row in features.index:
        try:
            t = int(features.iloc[row]['days_to_maturity'])
            features.loc[row,ratename] = rate_series.loc[0,t]
        except Exception:
            features.loc[row,ratename] = np.nan
    return features
        
features = map_rate(rfrs, 'risk_free_rate')
features = map_rate(dvys, 'dividend_rate')


"""
                              computing Derman estimation of implied volatility
"""
atm_vol = 0.1312
def compute_derman_volatility_row(row,atm_vol):
    try:
        t = int(row['days_to_maturity'])
        moneyness = row['spot_price'] - row['strike_price']
        atm_vol = atm_vol
        alpha = derman_coefs.loc['alpha',t]
        b = derman_coefs.loc['b',t]
        derman_vol = atm_vol + alpha + b*moneyness
        row['volatility'] = derman_vol
        return row
    except Exception:
        row['volatility'] = np.nan
        return row

features = features.apply(
    lambda row: compute_derman_volatility_row(row, atm_vol), axis=1)



features['w'] = 1 # flag for call/put
features = features.dropna()

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
