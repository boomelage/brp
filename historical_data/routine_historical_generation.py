# -*- coding: utf-8 -*-
"""
Created on Sat Sep 21 16:10:31 2024

generation routine
"""
import os
import sys
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)
import pandas as pd
import numpy as np
import QuantLib as ql
from itertools import product
from settings import model_settings
ms = model_settings()
from routine_historical_collection import historical_impvols
from bicubic_interpolation import bicubic_vol_row
from routine_calibration_global import calibrate_heston
# pd.set_option("display.max_rows",None)
pd.set_option("display.max_columns",None)

# pd.reset_option("display.max_rows")
# pd.reset_option("display.max_columns")

def generate_features(K,T,s):
    features = pd.DataFrame(
        product(
            [float(s)],
            K,
            T,
            ),
        columns=[
            "spot_price", 
            "strike_price",
            "days_to_maturity",
                  ])
    return features

def generate_train_features(K,T,s,flag):
    features = pd.DataFrame(
        product(
            [s],
            K,
            T,
            flag
            ),
        columns=[
            "spot_price", 
            "strike_price",
            "days_to_maturity",
            "w"
                  ])
    return features


"""
historical generation routine
"""

historical_impvols
    
row = historical_impvols.iloc[0]
s = row['spot_price']
dtdate = row['date']

calculation_date = ql.Date(dtdate.day,dtdate.month,dtdate.year)

expiry_dates = np.array([
       calculation_date + ql.Period(30,ql.Days), 
        calculation_date + ql.Period(60,ql.Days), 
        calculation_date + ql.Period(3,ql.Months), 
         calculation_date + ql.Period(6,ql.Months),
       #  calculation_date + ql.Period(12,ql.Months), 
       # calculation_date + ql.Period(18,ql.Months), 
       # calculation_date + ql.Period(24,ql.Months)
      ],dtype=object)
T = expiry_dates - calculation_date


"""
calibration
"""
call_K = np.linspace(s, s*1.01, 5)
put_K  =   np.arange(s*0.99, s, 5)

calls = generate_features(call_K, T, s)
calls = calls[calls['days_to_maturity'].isin(T)].copy()
calls['w'] = 'call'
calls['moneyness'] = calls['spot_price'] - calls['strike_price']

puts = generate_features(put_K, T, s)
puts = puts[puts['days_to_maturity'].isin(T)].copy()
puts['w'] = 'put'
puts['moneyness'] = puts['strike_price'] - puts['spot_price']

calls = calls.apply(bicubic_vol_row,axis=1)
puts = puts.apply(bicubic_vol_row,axis=1)

features = pd.concat([calls,puts],ignore_index=True)

features['dividend_rate'] = row['dividend_rate']
features['risk_free_rate'] = 0.04
heston_parameters = calibrate_heston(features,s)

"""
generation
"""
features['sigma'] = heston_parameters['sigma'].iloc[0]
features['theta'] = heston_parameters['theta'].iloc[0]
features['kappa'] = heston_parameters['kappa'].iloc[0]
features['rho'] = heston_parameters['rho'].iloc[0]
features['v0'] = heston_parameters['v0'].iloc[0]

heston_features = features.apply(ms.heston_price_vanilla_row,axis=1)

contract_details = heston_features.copy()

pricing_spread = 0.005
call_K_interp = np.linspace(s, s*(1+pricing_spread),100)
put_K_interp = np.linspace(s*(1-pricing_spread),s,100)

train_calls = generate_train_features(call_K,T,s,['call'])
train_puts = generate_train_features(put_K,T,s,['put'])



