# -*- coding: utf-8 -*-
"""
Created on Tue Sep 17 15:46:57 2024

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
import QuantLib as ql
from itertools import product
pd.set_option('display.max_columns',None)
# pd.set_option('display.max_rows',None)
pd.reset_option('display.max_rows')


from derman_test import derman_coefs, atm_volvec
from pricing import BS_price_vanillas, noisyfier, heston_price_vanillas
from routine_calibration_makret import all_heston_parameters
S = all_heston_parameters['spot_price'].unique()


byS = all_heston_parameters.copy().set_index('spot_price')
s = S[0]
df = byS.loc[s,:]


kUpper = s*1.2
kLower = s*0.8
K = np.linspace(kLower,kUpper,10000)
T = np.sort(all_heston_parameters['days_to_maturity'].unique().astype(int))

def generate_features(K,T,s):
    features = pd.DataFrame(
        product(
            [s],
            K,
            T,
            [1,-1]
            ),
        columns=[
            "spot_price", 
            "strike_price",
            "days_to_maturity",
            "w"
                  ])
    return features


features = generate_features(K, T, s)


parameters = all_heston_parameters.copy().set_index(
    ['spot_price','days_to_maturity'])


columns_to_map = [
    'volatility','v0', 'kappa', 'theta','rho',
    'sigma','black_scholes','heston']

dataset = features.merge(parameters[columns_to_map], 
                           left_on=['spot_price', 'days_to_maturity'], 
                           right_index=True, 
                           how='left')
dataset = noisyfier(dataset)

dataset