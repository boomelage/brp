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

prediction_df = training_data.copy()


day = 5
trading_day = training_data.iloc[day]
print(f"\n{training_data.iloc[day]}\n")

dt_calc = trading_day['calculation_date']

v0,kappa,theta,eta,rho = trading_day['v0'], trading_day['kappa'], \
    trading_day['theta'], trading_day['eta'], trading_day['rho'],  


calculation_date = ql.Date(
    dt_calc.day,dt_calc.month,dt_calc.year
    )

expiration_date = calculation_date + ql.Period()

volatility = trading_day['volatility']


r = 0.04
g = 0.001

s = trading_day['spot_price']
k = s*0.98
t = trading_day['days_to_maturity']



S = np.linspace(s,s,1)
T = np.arange(t,180,1)
K = np.linspace(k,k,1)

prediction_features = pd.DataFrame(
    product(
        S,
        K,
        T
        ),
    columns = ['spot_price','strike_price','days_to_maturity']
    )

black_scholes = ms.vector_black_scholes(
    S,
    K,
    T,
    r,
    volatility,
    'put'
    )

predicted = model_fit.predict(prediction_features)

error = predicted/black_scholes - 1

prediction_df = pd.DataFrame(
    np.array([black_scholes,predicted,error]).transpose(),
    columns = ['black_scholes','predicted','error']
    )

avg_abs_relative_error = np.average(np.abs(prediction_df['error']))

print(f"\n{prediction_df}\n\naverage absolute relative error: "
      f"{round(avg_abs_relative_error*100,4)}%\n")