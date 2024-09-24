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
from itertools import product
from bicubic_interpolation import make_bicubic_functional
from bicubic_interpolation import bicubic_vol_row
from settings import model_settings


ms = model_settings()
s = ms.s
T = ms.T.astype(int).tolist()
K = ms.calibration_K.astype(int).tolist()


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



bicubic_vol = make_bicubic_functional(ms.derman_ts, K, T)

features = generate_features(
    ms.calibration_K, T, s)

features = features.apply(bicubic_vol_row, axis = 1, bicubic_vol = bicubic_vol)
calibration_dataset = features.copy()
calibration_dataset['risk_free_rate'] = 0.04
calibration_dataset['dividend_rate'] = 0.001


pd.set_option('display.max_rows',None)
pd.set_option('display.max_columns',None)
print(f"\ncalibration dataset:\n{calibration_dataset}")
pd.reset_option('display.max_rows')
pd.reset_option('display.max_columns')




