# -*- coding: utf-8 -*-
"""
Created on Thu Sep 26 17:33:55 2024

@author: boomelage
"""

import os
import sys
import numpy as np
import pandas as pd
import time
from datetime import datetime
from itertools import product

current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)

from settings import model_settings

ms = model_settings()

os.chdir(current_dir)

start = time.time()

pd.reset_option('display.max_rows')

def make_barriers(s, updown, n_barriers, barrier_spread,n_barrier_spreads):
    if updown == "Up":
        flag = 1
    elif updown == "Down":
        flag = -1
    else:
        raise ValueError("updown error")
    barriers = np.linspace(
        s*(1+flag*barrier_spread*n_barrier_spreads),
        s*(1+flag*barrier_spread),
        n_barriers
        )
    return barriers

def generate_barrier_features(s,K,T,updown,barriers):
    
    barrier_features =  pd.DataFrame(
        product(
            [s],
            T,
            K,
            [updown],
            ['Out','In'],
            ['call','put'],
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
    
    barrier_features['barrier_type_name'] = \
        barrier_features['updown'] + barrier_features['outin']
    
    return barrier_features



def concat_barrier_features(
        s,K,T,g,heston_parameters,
        barrier_spread,n_barrier_spreads,n_barriers):
    
    barriers = make_barriers(s, 'Up', n_barriers, barrier_spread,n_barrier_spreads)
    
    up_features = generate_barrier_features(s,K,T,'Up',barriers)
    
            
    barriers = make_barriers(s, 'Down', n_barriers, barrier_spread,n_barrier_spreads)
    
    down_features = generate_barrier_features(s,K,T,'Down',barriers)
    
    
    features = pd.concat([down_features,up_features],ignore_index=True)
    
    return features




def generate_barrier_options(features, calculation_date, heston_parameters, g):
    features['eta'] = heston_parameters['eta'].iloc[0]
    features['theta'] = heston_parameters['theta'].iloc[0]
    features['kappa'] = heston_parameters['kappa'].iloc[0]
    features['rho'] = heston_parameters['rho'].iloc[0]
    features['v0'] = heston_parameters['v0'].iloc[0]
    features['heston_price'] = np.nan
    features['barrier_price'] = np.nan
    
    # pricing_bar = ms.make_tqdm_bar(
    #     desc="pricing",total=features.shape[0],unit='contracts',leave=False)
    
    for i, row in features.iterrows():
        
        barrier_type_name = row['barrier_type_name']
        barrier = row['barrier']
        s = row['spot_price']
        k = row['strike_price']
        t = row['days_to_maturity']
        w = row['w']
        r = 0.04
        rebate = 0.
        
        v0 = heston_parameters['v0'].iloc[0]
        kappa = heston_parameters['kappa'].iloc[0] 
        theta = heston_parameters['theta'].iloc[0] 
        eta = heston_parameters['eta'].iloc[0] 
        rho = heston_parameters['rho'].iloc[0]
        
        heston_price = ms.ql_heston_price(
            s,k,t,r,g,w,v0,kappa,theta,eta,rho,calculation_date
            )
        features.at[i,'heston_price'] = heston_price
        
        barrier_price = ms.ql_barrier_price(
                s,k,t,r,g,calculation_date,w,
                barrier_type_name,barrier,rebate,
                v0, kappa, theta, eta, rho)
    
        features.at[i,'barrier_price'] = barrier_price
        
    #     pricing_bar.update(1)
    # pricing_bar.close()
    
    training_data = features.copy()
    
    training_data = ms.noisyfier(training_data)
    
    pd.set_option("display.max_columns",None)
    print(f'\n{training_data}\n')
    print(f'\n{training_data.describe()}')
    pd.reset_option("display.max_columns")
    
    # file_date = datetime(
    #     calculation_date.year(), 
    #     calculation_date.month(), 
    #     calculation_date.dayOfMonth())
    # date_tag = file_date.strftime("%Y-%m-%d")
    # file_time = datetime.fromtimestamp(time.time())
    # file_time_tag = file_time.strftime("%Y-%m-%d %H%M%S")
    # training_data.to_csv(os.path.join(
    #     file_path,f'barriers {date_tag} {file_time_tag}.csv'))

    return training_data



