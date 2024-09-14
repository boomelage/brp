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
from routine_collection import contract_details
from import_files import derman_ts

# pd.set_option('display.max_columns', None)
pd.reset_option('display.max_rows', None)
pd.reset_option('display.max_columns', None)
calculation_date = ql.Date.todaysDate()
s = [np.sort(contract_details['spot_price'].unique().tolist())[0]]


"""
                                                            generation function
"""

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

def map_vol(row,varname):
    row[varname] = derman_ts.loc[
        int(row['strike_price']),
        int(row['days_to_maturity'])
        ]
    return row

varname = 'volatility'
features = features.apply(lambda row: map_vol(row, varname), axis=1)
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


# features['w'] = 1



# ms = model_settings()
# settings = ms.import_model_settings()
# dividend_rate = settings['dividend_rate']
# risk_free_rate = settings['risk_free_rate']
# calculation_date = settings['calculation_date']
# day_count = settings['day_count']
# calendar = settings['calendar']
# flat_ts = settings['flat_ts']
# dividend_ts = settings['dividend_ts']

# option_prices = BS_price_vanillas(contract_details)
# # option_prices = heston_price_vanillas()
# dataset = noisyfier(option_prices)
# dataset = dataset.dropna()
# dataset
# print(dataset)
# print(dataset.describe())
