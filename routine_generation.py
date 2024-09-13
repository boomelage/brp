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

pd.set_option('display.max_columns', None)
pd.reset_option('display.max_rows', None)

calculation_date = ql.Date.todaysDate()

s = [np.median(contract_details['spot_price'].unique().tolist())]
k = contract_details['strike_price'].unique().tolist()
t = contract_details['days_to_maturity'].unique().tolist()
contract_details = pd.DataFrame(
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

data_for_pivots = contract_details[contract_details['spot_price'] == s[0]]



def map_var(varname):
    def make_varpivot(varname):
        pivot = data_for_pivots.pivot_table(
            index = 'strike_price', columns = 'days_to_maturity', values = varname)
        return pivot
    
    def map_var_by_row(varpivot, row, rowvar):
        try:
            return varpivot.loc[row['strike_price'], row[rowvar]]
        except KeyError:
            return np.nan
        except Exception:
            return np.nan
    
    contract_details[varname] = contract_details.apply(
        lambda row: map_var_by_row(make_varpivot(varname), row, 'days_to_maturity'), axis=1
    )
    

map_var('risk_free_rate')
map_var('volatility')
map_var('dividend_rate')
contract_details = contract_details.dropna(axis=0).reset_index(drop=True)
contract_details
contract_details['w'] = 1

contract_details

# ms = model_settings()

# settings, ezprint = ms.import_model_settings()
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
