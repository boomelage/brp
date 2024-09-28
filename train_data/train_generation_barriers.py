# -*- coding: utf-8 -*-
"""
Created on Thu Sep 26 17:33:55 2024

"""

import os
import time
import numpy as np
import pandas as pd
import QuantLib as ql
from itertools import product
from datetime import datetime
from settings import model_settings
ms = model_settings()


def make_barriers(
        s, updown, n_barriers, barrier_spread,n_barrier_spreads):
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

def generate_barrier_features(s,K,T,barriers,updown,OUTIN,W):
    
    barrier_features =  pd.DataFrame(
        product(
            [s],
            K,
            barriers,
            T,
            [updown],
            OUTIN,
            W
            ),
        columns=[
            'spot_price', 
            'strike_price',
            'barrier',
            'days_to_maturity',
            'updown',
            'outin',
            'w'
                  ])
    
    barrier_features['barrier_type_name'] = \
        barrier_features['updown'] + barrier_features['outin']
    
    return barrier_features


def generate_barrier_options(
        features, calculation_date, heston_parameters, g, output_folder):
    
    spot_date = datetime(
        calculation_date.year(), 
        calculation_date.month(), 
        calculation_date.dayOfMonth())
    
    features['calculation_date'] = spot_date
    features['expiration_date'] = datetime(999,1,1)
    features['eta'] = heston_parameters['eta']
    features['theta'] = heston_parameters['theta']
    features['kappa'] = heston_parameters['kappa']
    features['rho'] = heston_parameters['rho']
    features['v0'] = heston_parameters['v0']
    features['heston_price'] = np.nan
    features['barrier_price'] = np.nan
    
    pricing_bar = ms.make_tqdm_bar(
        desc='pricing', unit='contracts', leave=True, total = features.shape[0]
        )
    
    for i, row in features.iterrows():
        
        barrier_type_name = row['barrier_type_name']
        barrier = row['barrier']
        s = row['spot_price']
        k = row['strike_price']
        t = row['days_to_maturity']
        w = row['w']
        barrier = row['barrier']
        barrier_type_name = row['barrier_type_name']
        v0 = row['v0']
        kappa = row['kappa']
        theta = row['theta']
        eta = row['eta']
        rho = row['rho']
        expiration_date = calculation_date + ql.Period(int(t),ql.Days)
        r = 0.04
        rebate = 0.
        
        heston_price = ms.ql_heston_price(
            s,k,r,g,w,
            v0,kappa,theta,eta,rho,
            calculation_date,
            expiration_date
            )
        
        features.at[i,'expiration_date'] = datetime(
            expiration_date.year(), 
            expiration_date.month(), 
            expiration_date.dayOfMonth()
            )
        
        features.at[i,'heston_price'] = heston_price
        
        barrier_price = ms.ql_barrier_price(
                s,k,t,r,g,calculation_date,w,
                barrier_type_name,barrier,rebate,
                v0, kappa, theta, eta, rho)
    
        features.at[i,'barrier_price'] = barrier_price
        pricing_bar.update(1)
    pricing_bar.close()
    training_data = features.copy()
    
    training_data = ms.noisyfier(training_data)
    
    pd.set_option("display.max_columns",None)
    print(f'\n{training_data}\n')
    print(f'\n{training_data.describe()}')
    pd.reset_option("display.max_columns")
    
    date_tag = spot_date.strftime("%Y-%m-%d")
    file_time = datetime.fromtimestamp(time.time())
    file_time_tag = file_time.strftime("%Y-%m-%d %H%M%S")
    training_data.to_csv(os.path.join(
        output_folder,f'barriers {date_tag} {file_time_tag}.csv'))

    return training_data



