# -*- coding: utf-8 -*-
"""
Created on Thu Sep 26 17:33:55 2024

@author: boomelage
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

start = time.time()

s = ms.s


"""
# =============================================================================
"""

title = 'barrier options'

# T = ms.T
T = [1,2]
n_strikes = 5

down_k_spread = 0.5
up_k_spread = 0.5


n_barriers = n_strikes
barrier_spread = 0.005
n_barrier_spreads = 3


n_contracts = len(T)*n_barriers*n_strikes*1


outins = [
    
    'Out',
    'In'
    
    ]

updowns = [
    'Up',
    'Down'
    ]

features = pd.DataFrame(
    product(
        [s],
        T,
        updowns
        ),
    columns=[
        "spot_price", 
        "days_to_maturity",
        "updown"
              ])


features =  features.set_index('updown')





"""
up features
"""

initial_up_features = features.loc['Up'].reset_index()

up_K = np.linspace(
    s, 
    s*(1+up_k_spread*n_strikes),
    n_strikes)

up_features = pd.DataFrame()
for i, row in initial_up_features.iterrows():
    s = row['spot_price']
    t = row['days_to_maturity']
    new_up_features = pd.DataFrame(
        product(
            [s],
            [t],
            up_K,
            ['Up'],
            outins
            ),
        columns=[
            "spot_price", 
            "days_to_maturity",
            "strike_price",
            "updown",
            "outin"
                  ])
    for j,row2 in new_up_features.iterrows():
        
        k = row2['strike_price']
        
        barriers = np.linspace(
            k*(1+barrier_spread),
            k*(1+n_barrier_spreads*barrier_spread),
            n_barriers
            )
        
        new_up_features_with_bar = pd.DataFrame(
            product(
                [s],
                [t],
                [k],
                barriers,
                ['Up'],
                outins,
                ['call']
                ),
            columns = [
                "spot_price", 
                "days_to_maturity",
                "strike_price",
                "barrier",
                "updown",
                "outin",
                "w"
                ])
        new_up_features_with_bar['moneyness'] = \
            new_up_features_with_bar.loc[:,'spot_price']/\
                new_up_features_with_bar.loc[:,'strike_price'] - 1
                
        up_features = pd.concat(
            [up_features, new_up_features_with_bar],
            ignore_index=True
            )


up_features
# """
# down features
# """

# initial_down_features = features.loc['Down'].reset_index()

# down_K = np.linspace(
#     s*(1-down_k_spread),
#     s,
#     n_strikes
#     )

# down_features = pd.DataFrame()
# for i, row in initial_down_features.iterrows():
#     s = row['spot_price']
#     t = row['days_to_maturity']
#     new_down_features = pd.DataFrame(
#         product(
#             [s],
#             [t],
#             down_K,
#             ['Down'],
#             outins
#             ),
#         columns=[
#             "spot_price", 
#             "days_to_maturity",
#             "strike_price",
#             "updown",
#             "outin"
#                   ])
#     for j,row2 in new_down_features.iterrows():
#         k = row2['strike_price']
        
#         barriers = np.linspace(
#             k*(1-n_barrier_spreads*barrier_spread),
#             k*(1-barrier_spread),
#             n_barriers
#             )
        
#         new_down_features_with_bar = pd.DataFrame(
#             product(
#                 [s],
#                 [t],
#                 [k],
#                 barriers,
#                 ['Down'],
#                 outins,
#                 ['put']
#                 ),
#             columns = [
#                 "spot_price", 
#                 "days_to_maturity",
#                 "strike_price",
#                 "barrier",
#                 "updown",
#                 "outin",
#                 "w"
#                 ])
#         down_features = pd.concat([down_features, new_down_features_with_bar],
#                                   ignore_index=True)
        

# features = pd.concat([down_features,up_features],ignore_index=True)

# features['barrier_type_name'] = features['updown'] + features['outin'] 
# features['eta'] = heston_parameters['eta'].iloc[0]
# features['theta'] = heston_parameters['theta'].iloc[0]
# features['kappa'] = heston_parameters['kappa'].iloc[0]
# features['rho'] = heston_parameters['rho'].iloc[0]
# features['v0'] = heston_parameters['v0'].iloc[0]

# features['barrier_price'] = np.nan
# pricing_bar = tqdm(
#     desc="pricing",
#     total=features.shape[0],
#     unit='contracts',
#     leave=True)

# for i, row in features.iterrows():
    
#     barrier_type_name = row['barrier_type_name']
#     barrier = row['barrier']
#     s = row['spot_price']
#     k = row['strike_price']
#     t = row['days_to_maturity']
#     r = 0.04
#     g = 0.001
#     rebate = 0.
    
#     calculation_date = ms.calculation_date
    
#     v0 = heston_parameters['v0'].iloc[0]
#     kappa = heston_parameters['kappa'].iloc[0] 
#     theta = heston_parameters['theta'].iloc[0] 
#     eta = heston_parameters['eta'].iloc[0] 
#     rho = heston_parameters['rho'].iloc[0]
    
#     barrier_price = ms.ql_barrier_price(
#             s,k,t,r,g,calculation_date,
#             barrier_type_name,barrier,rebate,
#             v0, kappa, theta, eta, rho)
    
#     features.at[i,'barrier_price'] = barrier_price
    
#     pricing_bar.update(1)
    

# pricing_bar.close()

# features

# training_data = features.copy()

# training_data = ms.noisyfier(training_data)

# pd.set_option("display.max_columns",None)
# print(f'\n{training_data}\n')
# print(f'\n{training_data.describe()}\n')
# pd.reset_option("display.max_columns")

# file_time = datetime.fromtimestamp(time.time())
# file_tag = file_time.strftime("%Y-%d-%m %H%M%S")
# training_data.to_csv(os.path.join('barriers',f'barriers {file_tag}.csv'))