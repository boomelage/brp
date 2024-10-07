# -*- coding: utf-8 -*-
"""
Created on Mon Sep 16 21:30:51 2024

@author: boomelage
"""

import os
import sys
os.chdir(os.path.dirname(os.path.abspath(__file__)))
sys.path.append('term_structure')
sys.path.append('contract_details')
sys.path.append('misc')
import pandas as pd
import numpy as np
import QuantLib as ql
from itertools import product
from model_settings import ms
import Derman as derman



"""

03/01/2012

30D,60D,6M,12M,18M,24M atm volatilities
19.7389,21.2123,21.9319,23.0063,23.6643,24.1647,24.4341

2.1514

1277.92

"""
g = 2.1514/100

r = 0.04

s =  1277.92

calculation_date = ql.Date(3,1,2012)

ql_T = [
     ql.Period(30,ql.Days),
     ql.Period(60,ql.Days),
     ql.Period(3,ql.Months),
     ql.Period(6,ql.Months),
     ql.Period(12,ql.Months),
     ql.Period(18,ql.Months),
     ql.Period(24,ql.Months)
     ]

expiration_dates = []
for t in ql_T:
    expiration_dates.append(calculation_date + t)
T = []
for date in expiration_dates:
    T.append(date - calculation_date)

print(derman.derman_coefs.index)
print(T)


T = [30,60,95,186,368]

atm_vols = [
    19.7389,21.2123,21.9319,23.0063,23.6643,
    ]
atm_vols = pd.Series(atm_vols,index=T)/100


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


spread = 0.05
initial_spread = 0.005
wing_size = 3 + 1

put_K = np.linspace(s*(1-spread),s*0.995,wing_size)

call_K = np.linspace(s*1.005,s*(1+spread),wing_size)

K = np.unique(np.array([put_K,call_K]).flatten())

T = [
      30,60,95,
      # 186,368
      ]

features = generate_features(
    K, T, s)
"""
with interpolation
"""

# bicubic_vol = ms.make_bicubic_functional(
#     s,
#     K.tolist(),
#     T,
#     atm_vols,
#     derman.derman_coefs,
#     )

# T = [1,7,14]



# features['volatility'] = np.nan
# for i, row in features.iterrows():
#     k = row['strike_price']
#     t = row['days_to_maturity']
#     features.at[i,'volatility'] = bicubic_vol(t,k,True)
    
"""
without
"""

features['volatility'] = np.nan
features['volatility'] = ms.derman_volatilities(
    s, 
    features['strike_price'], 
    features['days_to_maturity'], 
    features['days_to_maturity'].map(derman.derman_coefs), 
    features['days_to_maturity'].map(atm_vols)
    )


calibration_dataset = features.copy()
calibration_dataset['risk_free_rate'] = r
calibration_dataset['dividend_rate'] = g

pd.set_option('display.max_rows',None)
pd.set_option('display.max_columns',None)
print(f"\ncalibration dataset:\n{calibration_dataset}")
pd.reset_option('display.max_rows')
pd.reset_option('display.max_columns')















