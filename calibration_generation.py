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

T = np.sort(derman_coefs.columns.unique().astype(int))
K = np.linspace(ms.lower_moneyness, ms.upper_moneyness, 5)
def generate_features(K,T,s):
    features = pd.DataFrame(
        product(
            [float(s)],
            K,
            T,
            ['call','put']
            ),
        columns=[
            "spot_price", 
            "strike_price",
            "days_to_maturity",
            "w"
                  ])
    return features

contract_details = generate_features(K, T, s)
contract_details['risk_free_rate'] = 0.05
contract_details['dividend_rate'] = 0.05

def compute_derman_volatility_row(row):
    s = row['spot_price']  # Assuming s is spot_price (not defined in your function, but seems to be required)
    k = row['strike_price']  # Accessing strike_price directly
    t = row['days_to_maturity']  # Accessing days_to_maturity directly
    moneyness = s - k  # Calculate moneyness
    atm_vol =  derman_coefs.loc['atm_vol', t]  # Look up ATM volatility for the given maturity
    b = derman_coefs.loc['b', t]  # Look up the coefficient for t
    volatility = atm_vol + b * moneyness  # Compute volatility
    row['volatility'] = volatility
    return row

contract_details = contract_details.apply(compute_derman_volatility_row, axis=1)
contract_details = contract_details[~(contract_details['volatility']<0)]
