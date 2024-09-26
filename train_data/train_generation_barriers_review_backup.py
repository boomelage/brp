#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Sep 10 17:55:16 2024

"""

import os
import sys
from tqdm import tqdm
import numpy as np
import pandas as pd
import time
from datetime import datetime
from itertools import product

current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)

from settings import model_settings
from routine_calibration_testing import heston_parameters

ms = model_settings()
calculation_date = ms.calculation_date
day_count = ms.day_count

os.chdir(current_dir)

def generate_features(K,T,s):
    features = pd.DataFrame(
        product(
            [s],
            K,
            T,
            ),
        columns=[
            "spot_price", 
            "strike_price",
            "days_to_maturity",
                  ])
    return features

start = time.time()

s = ms.s

"""
# =============================================================================
"""

title = 'barrier options'

# T = ms.T
T = [1,2,3]

n_strikes = 30

down_k_spread = 0.20
up_k_spread = 0.20


n_barriers = n_strikes
barrier_spread = 0.005
n_barrier_spreads = 5


n_contracts = len(T)*n_barriers*n_strikes*1

"""
# =============================================================================
                                up options
"""

up_K = np.linspace(
    s, 
    s*(1+up_k_spread*n_strikes),
    n_strikes)
initial_up_features = generate_features(up_K,T,s)
up_features = pd.DataFrame()
up_bar = tqdm(
    desc="generatingUps",
    total=initial_up_features.shape[0],
    unit='sets',
    leave=True)
for i, row in initial_up_features.iterrows():
    
    s = row['spot_price']
    k = row['strike_price']
    t = row['days_to_maturity']
    
    col_names = [
        'spot_price', 'strike_price', 'days_to_maturity','barrier','outin','w']
    strike_wise_np = np.zeros((n_barriers,len(col_names)),dtype=float)
    
    strike_wise_out = pd.DataFrame(strike_wise_np).copy()
    strike_wise_out.columns = col_names
    strike_wise_out['strike_price'] = k
    strike_wise_out['spot_price'] = s
    strike_wise_out['days_to_maturity'] = t
    strike_wise_out['w'] = 'calls'
    strike_wise_out['updown'] = 'Up'
    strike_wise_out['outin'] = 'Out'
    
    strike_wise_in = pd.DataFrame(strike_wise_np).copy()
    strike_wise_in.columns = col_names
    strike_wise_in['strike_price'] = k
    strike_wise_in['spot_price'] = s
    strike_wise_in['days_to_maturity'] = t
    strike_wise_in['w'] = 'calls'
    strike_wise_in['updown'] = 'Up'
    strike_wise_in['outin'] = 'In'
    
    barriers = np.linspace(
        k*(1+barrier_spread),
        k*(1+n_barrier_spreads*barrier_spread),
        n_barriers
        )
    
    strike_wise_in['barrier'] = barriers
    
    strike_wise_out['barrier'] = barriers
    
    strike_wise = pd.concat(
        [strike_wise_in, strike_wise_out],
        ignore_index=True)
    
    
    up_features = pd.concat(
        [up_features, strike_wise],
        ignore_index=True
        )
    up_bar.update(1)
up_bar.close()
    
"""
# =============================================================================
                                down options
"""

# down_K = np.linspace(
#     s*(1-down_k_spread),
#     s,
#     n_strikes
#     )
# initial_down_features = generate_features(down_K,T,s)
# down_features = pd.DataFrame()
# down_bar = tqdm(
#     desc="generatingDowns",
#     total=initial_down_features.shape[0],
#     unit='sets',
#     leave=True)
# for i, row in initial_down_features.iterrows():
    
#     s = row['spot_price']
#     k = row['strike_price']
#     t = row['days_to_maturity']
    
#     col_names = [
#         'spot_price', 'strike_price', 'days_to_maturity','barrier','outin','w']
#     strike_wise_np = np.zeros((n_barriers,len(col_names)),dtype=float)
    
#     strike_wise_out = pd.DataFrame(strike_wise_np).copy()
#     strike_wise_out.columns = col_names
#     strike_wise_out['strike_price'] = k
#     strike_wise_out['spot_price'] = s
#     strike_wise_out['days_to_maturity'] = t
#     strike_wise_out['w'] = 'put'
#     strike_wise_out['updown'] = 'Down'
#     strike_wise_out['outin'] = 'Out'
    
#     # strike_wise_in = pd.DataFrame(strike_wise_np).copy()
#     # strike_wise_in.columns = col_names
#     # strike_wise_in['strike_price'] = k
#     # strike_wise_in['spot_price'] = s
#     # strike_wise_in['days_to_maturity'] = t
#     # strike_wise_in['w'] = 'put'
#     # strike_wise_in['updown'] = 'Down'
#     # strike_wise_in['outin'] = 'In'
    
#     barriers = np.linspace(
#         k*(1-n_barrier_spreads*barrier_spread),
#         k*(1-barrier_spread),
#         n_barriers
#         )
    
#     # strike_wise_in['barrier'] = barriers
    
#     strike_wise_out['barrier'] = barriers
#     strike_wise = strike_wise_out
    
#     # strike_wise = pd.concat(
#     #     [strike_wise_in, strike_wise_out],
#     #     ignore_index=True)
    
#     down_features = pd.concat(
#         [down_features, strike_wise],
#         ignore_index=True
#         )
    
#     down_bar.update(1)
    
# down_bar.close()
    
"""  
# =============================================================================
"""

# features = pd.concat(
#     [up_features,down_features],
#     ignore_index=True)

features = up_features
features['barrier_type_name'] = features['updown'] + features['outin'] 
features['eta'] = heston_parameters['eta'].iloc[0]
features['theta'] = heston_parameters['theta'].iloc[0]
features['kappa'] = heston_parameters['kappa'].iloc[0]
features['rho'] = heston_parameters['rho'].iloc[0]
features['v0'] = heston_parameters['v0'].iloc[0]

features['barrier_price'] = np.nan
pricing_bar = tqdm(
    desc="pricing",
    total=features.shape[0],
    unit='contracts',
    leave=True)

for i, row in features.iterrows():
    
    barrier_type_name = row['barrier_type_name']
    barrier = row['barrier']
    s = row['spot_price']
    k = row['strike_price']
    t = row['days_to_maturity']
    r = 0.04
    g = 0.001
    rebate = 0.
    
    calculation_date = ms.calculation_date
    
    v0 = heston_parameters['v0'].iloc[0]
    kappa = heston_parameters['kappa'].iloc[0] 
    theta = heston_parameters['theta'].iloc[0] 
    eta = heston_parameters['eta'].iloc[0] 
    rho = heston_parameters['rho'].iloc[0]
    
    barrier_price = ms.ql_barrier_price(
            s,k,t,r,g,calculation_date,
            barrier_type_name,barrier,rebate,
            v0, kappa, theta, eta, rho)
    
    features.at[i,'barrier_price'] = barrier_price
    
    pricing_bar.update(1)
    

pricing_bar.close()

training_data = features.copy()

training_data = ms.noisyfier(training_data)

pd.set_option("display.max_columns",None)
print(f'\n{training_data}\n')
print(f'\n{training_data.describe()}\n')
pd.reset_option("display.max_columns")

file_time = datetime.fromtimestamp(time.time())
file_tag = file_time.strftime("%Y-%d-%m %H%M%S")
training_data.to_csv(os.path.join('barriers',f'barriers {file_tag}.csv'))
