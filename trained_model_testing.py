# -*- coding: utf-8 -*-
"""
Created on Mon Sep 30 15:33:07 2024

@author: boomelage
"""

import os
import sys
import pandas as pd
import numpy as np
import QuantLib as ql
from itertools import product
current_dir = os.path.dirname(os.path.abspath(__file__))
os.chdir(current_dir)
sys.path.append(os.path.join(current_dir,'term_structure'))
from settings import model_settings
from train_main import model_fit, training_data
import Derman as derman
pd.set_option("display.max_columns",None)
print(f"{training_data.describe()}")
pd.reset_option("display.max")

ms = model_settings()

day = 5
print(f"\n{training_data.iloc[day]}\n")

dt_calc = training_data.iloc[day]['calculation_date']

trading_day = training_data.iloc[day]

v0,kappa,theta,eta,rho = trading_day['v0'], trading_day['kappa'], \
    trading_day['theta'], trading_day['eta'], trading_day['rho'],  

calculation_date = ql.Date(
    dt_calc.day,dt_calc.month,dt_calc.year
    )
r = 0.04
g = 0.001

volatility = trading_day['volatility']


s = trading_day['spot_price']


up_K = np.arange(
    int(s*1.01),
    int(s*1.05),
    1
    )

down_K = np.arange(
    int(s*0.95),
    int(s*0.99),
    1
    )

K = np.sort(np.array([down_K, up_K]).flatten().astype(float))


T = np.arange(
    min(training_data['days_to_maturity']),
    max(training_data['days_to_maturity']),
    7)

test_data = pd.DataFrame(
    product([s],K,T,),
    columns= ['spot_price','strike_price','days_to_maturity']
    )
test_data
test_data['predicted'] = model_fit.predict(test_data)


test_data['moneyness'] = ms.vmoneyness(s,test_data['strike_price'], 'put')
test_data['calculation_date'] = calculation_date
test_data['expiration_date'] = ms.vexpiration_datef(
    test_data['days_to_maturity'].tolist(),
    calculation_date,
    )

# test_data['numpy_black_scholes'] = ms.vector_black_scholes(
#     test_data['spot_price'], 
#     test_data['strike_price'], 
#     test_data['days_to_maturity'],
#     r, volatility, 
#     'put')

# test_data['ql_black_scholes'] = ms.vector_qlbs(
#     test_data['spot_price'],
#     test_data['strike_price'],
#     r, g, volatility, 'put',
#     calculation_date,
#     test_data['expiration_date']
#     )

test_data['heston_price'] = ms.vector_heston_price(
    test_data['spot_price'],
    test_data['strike_price'],
    r,g,'put',
    v0,kappa,theta,eta,rho,
    calculation_date,
    test_data['expiration_date']
    )

test_data['error'] = (
    test_data['predicted']/test_data['heston_price']
    )-1

test_data = test_data[
    [
     'spot_price', 'strike_price', 'days_to_maturity', 'predicted',
     'heston_price', 'error', 'moneyness', 'calculation_date', 'expiration_date'
     ]
    ].reset_index(drop=True)

avg = np.average(np.abs(test_data['error']))

pd.set_option("display.max_columns",None)
print(f"{test_data}")
print(f"\naverage absolute relative pricing error: {round(avg*100,2)}%\n")
pd.reset_option("display.max")
