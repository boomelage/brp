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
from itertools import product
pd.set_option('display.max_columns',None)
# pd.set_option('display.max_rows',None)
pd.reset_option('display.max_rows')

from routine_collection import contract_details
from pricing import noisyfier
from routine_calibration_market import \
    put_heston_parameters, call_heston_parameters


def generate_features(K,T,s,flag):
    features = pd.DataFrame(
        product(
            [s],
            K,
            T,
            [flag]
            ),
        columns=[
            "spot_price", 
            "strike_price",
            "days_to_maturity",
            "w"
                  ])
    return features

def map_features(df,parameters,flag):
    
    S = np.sort(np.array(df['spot_price'].unique()))
    byS = df.groupby('spot_price')
    option_features = pd.DataFrame()
    
    for s_idx, s in enumerate(S):
        
        dfs = byS.get_group(s)
        K = np.linspace(s*0.5,s*1.5,int(1e5))
        T = np.sort(np.array(dfs['days_to_maturity'].unique()))
        
        initial_features = generate_features(K,T,s,flag)
        option_features = pd.concat([option_features,initial_features])
        
        feature_params = parameters.copy().set_index(
            ['spot_price','days_to_maturity'])
        columns_to_map = ['v0', 'kappa', 'theta','rho',
            'sigma','black_scholes','heston']
        mapped_features = option_features.merge(
            feature_params[columns_to_map], 
            left_on=['spot_price', 'days_to_maturity'], 
            right_index=True, how='left')
    return mapped_features
    
calls = contract_details['calls']
mapped_calls = map_features(calls, call_heston_parameters,"call")
mapped_calls = noisyfier(mapped_calls)

puts = contract_details['puts']  
mapped_puts = map_features(puts, put_heston_parameters,"put")
mapped_puts = noisyfier(mapped_puts)

features_dataset = pd.concat([mapped_calls,mapped_puts]).dropna(
    ).reset_index(drop=True)

features_dataset 
