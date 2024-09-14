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
import QuantLib as ql
from itertools import product
from settings import model_settings
from pricing import BS_price_vanillas, noisyfier


pd.set_option('display.max_columns', None)
pd.reset_option('display.max_rows', None)
# pd.reset_option('display.max_columns', None)



"""
# =============================================================================
                                                           generation procedure
"""

from import_files import contract_details, derman_coefs, derman_ts, spread_ts, raw_ts
calculation_date = ql.Date.todaysDate()


s = [np.sort(contract_details['spot_price'].unique().tolist())[0]]
k = derman_ts.index
t = derman_ts.columns
features = pd.DataFrame(
    product(
        s,
        k,
        t,
        ),
    columns=[
        "spot_price", 
        "strike_price",
        "days_to_maturity",
              ])

details_indexed = contract_details.copy().set_index([
    'strike_price','days_to_maturity'])
features = features[features['spot_price'] == s[0]]


def map_vol(row):
    row['volatility'] = derman_ts.loc[
        int(row['strike_price']),
        int(row['days_to_maturity'])
        ]
    return row

features = features.apply(lambda row: map_vol(row), axis=1)
features = features.dropna(axis=0).reset_index(drop=True)


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

features['w'] = 1
features = features.dropna()
print(f"\noriginal dataset:\n{contract_details}")
print(f"\nnew dataset:\n{features}")
print(f"\n{int(100*(features.shape[0]/contract_details.shape[0]-1))}% combinations gained")

# ms = model_settings()
# settings = ms.import_model_settings()
# dividend_rate = settings['dividend_rate']
# risk_free_rate = settings['risk_free_rate']
# calculation_date = settings['calculation_date']
# day_count = settings['day_count']
# calendar = settings['calendar']
# flat_ts = settings['flat_ts']
# dividend_ts = settings['dividend_ts']

# option_prices = BS_price_vanillas(features)
# # option_prices = heston_price_vanillas()
# dataset = noisyfier(option_prices)
# dataset = dataset.dropna()
# dataset
# print(dataset)
# print(dataset.describe())


# negative_columns = df.loc[:, (df < 0).any(axis=0)]
