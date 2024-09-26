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

pd.reset_option('display.max_rows')
"""
# =============================================================================
"""

title = 'barrier options'

# T = ms.T
T = [1,2,3]


n_strikes = 50
down_k_spread = 0.05
up_k_spread = 0.05


n_barriers = 50
barrier_spread = 0.0010                   
n_barrier_spreads = 5




outins = [
    
    'Out',
    'In'
    
    ]

updowns = [
    'Up',
    'Down'
    ]


def generate_initial_barrier_features(s,T,K,outins,updown,w):
    features = pd.DataFrame(
        product(
            [s],
            T,
            K,
            [updown],
            outins,
            [w]
            ),
        columns=[
            'spot_price', 
            'days_to_maturity',
            'strike_price',
            'updown',
            'outin',
            'w'
                  ])
    return features





"""
up features
"""


up_K = np.linspace(
    s, 
    s*(1+up_k_spread),
    n_strikes)

initial_up_features = generate_initial_barrier_features(
    s,T,up_K,outins,'Up','call')

initial_up_features

up_features = pd.DataFrame()

for i, row in initial_up_features.iterrows():
    s = row['spot_price']
    k = row['strike_price']
    t = row['days_to_maturity']
    outin = row['outin']
    updown = row['updown']
    w = row['w']
    
    barriers = np.linspace(
        k*(1+barrier_spread),
        k*(1+barrier_spread*n_barrier_spreads),
        n_barriers
        )    
    
    up_subset =  pd.DataFrame(
        product(
            [s],
            [t],
            [k],
            [updown],
            [outin],
            [w],
            barriers
            ),
        columns=[
            'spot_price', 
            'days_to_maturity',
            'strike_price',
            'updown',
            'outin',
            'w',
            'barrier'
                  ])
    
    up_subset['barrier_type_name'] = up_subset['updown']+up_subset['outin']
    
    print(f"\n{up_subset}\n")

    up_features = pd.concat(
        [up_features, up_subset],
        ignore_index = True)    
        

"""
down features
"""


down_K = np.linspace(
    s*(1-down_k_spread), 
    s,
    n_strikes)

initial_down_features = generate_initial_barrier_features(
    s,T,down_K,outins,'Down','put')

initial_down_features

down_features = pd.DataFrame()

for i, row in initial_down_features.iterrows():
    s = row['spot_price']
    k = row['strike_price']
    t = row['days_to_maturity']
    outin = row['outin']
    updown = row['updown']
    w = row['w']
    
    
    barriers = np.linspace(
        k*(1-barrier_spread*n_barrier_spreads),
        k*(1-barrier_spread),
        n_barriers
        )    
    
    down_subset =  pd.DataFrame(
        product(
            [s],
            [t],
            [k],
            [updown],
            [outin],
            [w],
            barriers
            ),
        columns=[
            'spot_price', 
            'days_to_maturity',
            'strike_price',
            'updown',
            'outin',
            'w',
            'barrier'
                  ])
    
    down_subset['barrier_type_name'] = down_subset['updown']+down_subset['outin']
    
    print(f"\n{down_subset}\n")
    
    down_features = pd.concat(
        [down_features, down_subset],
        ignore_index = True)    
        

features = pd.concat([down_features,up_features],ignore_index=True)

features['eta'] = heston_parameters['eta'].iloc[0]
features['theta'] = heston_parameters['theta'].iloc[0]
features['kappa'] = heston_parameters['kappa'].iloc[0]
features['rho'] = heston_parameters['rho'].iloc[0]
features['v0'] = heston_parameters['v0'].iloc[0]

features['barrier_price'] = np.nan

features
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







