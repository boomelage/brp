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
from itertools import product
from settings import model_settings
from Derman import raw_ts, derman_atm_vols, derman_s, derman_coefs
ms = model_settings()

T = derman_coefs.index
T
T = np.array([7,14,28,60,186,368],dtype=int)

s = derman_s
T = np.array([7,14,28],dtype=int)


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
K = np.linspace(s*(1-spread),s*(1+spread),5) 

vbicubic_vol = ms.make_bicubic_functional(
    derman_s,
    K,
    T,
    derman_atm_vols,
    derman_coefs,
    )

vbicubic_vol

calibration_T = [7,14,30]

features = generate_features(
    K, calibration_T, s)

features['volatility'] = np.nan
for i, row in features.iterrows():
    k = row['strike_price']
    t = row['days_to_maturity']
    features.at[i,'volatility'] = vbicubic_vol(t,k,True)
    
features
calibration_dataset = features.copy()
calibration_dataset['risk_free_rate'] = 0.04
calibration_dataset['dividend_rate'] = 0.001

pd.set_option('display.max_rows',None)
pd.set_option('display.max_columns',None)
print(f"\ncalibration dataset:\n{calibration_dataset}")
pd.reset_option('display.max_rows')
pd.reset_option('display.max_columns')




