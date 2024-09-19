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
import numpy as np
import pandas as pd
from itertools import product
# pd.set_option('display.max_columns',None)
pd.reset_option('display.max_rows')


from settings import model_settings
ms = model_settings()
settings = ms.import_model_settings()
security_settings = settings[0]['security_settings']
s = security_settings[5]


from derman_test import derman_coefs
from routine_ivol_collection import raw_ts

T = np.sort(derman_coefs.columns.unique().astype(int))
# K = np.linspace(5685, ms.upper_moneyness, 20)
K = raw_ts.iloc[:,0].dropna().index
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

contract_details = generate_features(K, T, s)
contract_details['risk_free_rate'] = 0.05
contract_details['dividend_rate'] = 0.05

contract_details