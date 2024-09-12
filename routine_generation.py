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
from pricing import BS_price_vanillas, noisyfier
from routine_collection import collect_market_data

market_data = collect_market_data(r'SPXts.xlsx')

pd.set_option('display.max_columns', None)
calculation_date = ql.Date.todaysDate()

# v = market_data['volatility'].unique().tolist()
s = market_data['spot_price'].unique().tolist()
k = market_data['strike_price'].unique().tolist()
t = market_data['days_to_maturity'].unique().tolist()
g = market_data['dividend_rate'].unique().tolist()
r = market_data['risk_free_rate'].unique().tolist()
k = np.linspace(min(k),max(k),5)
# v = np.linspace(min(v),max(v),20)
contract_details = pd.DataFrame(
    product(
        s,
        k,
        # v,
        t,
        # g,
        # r,
        ),
    columns=[
        "spot_price", 
        "strike_price",
        # "volatility",
        "days_to_maturity",
        # "dividend_rate",
        # "risk_free_rate",
             ])

contract_details['risk_free_rate'] = np.average(r) #temporarily just an average
contract_details['dividend_rate'] = np.average(g) #temporarily just an average
contract_details['w'] = 1
contract_details['calculation_date'] = calculation_date

def compute_maturity_date(row):
    row['maturity_date'] = calculation_date + ql.Period(int(row['days_to_maturity']), ql.Days)
    return row
contract_details = contract_details.apply(compute_maturity_date, axis=1)

option_prices = BS_price_vanillas(contract_details)

dataset = noisyfier(option_prices)
dataset = dataset.dropna()
dataset
print(dataset)
print(dataset.describe())
