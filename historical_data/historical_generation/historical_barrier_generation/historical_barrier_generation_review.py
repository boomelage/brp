# -*- coding: utf-8 -*-
"""
Created on Sat Sep 21 13:54:06 2024

@author: boomelage
"""
import os
import sys
import time
import pandas as pd
import numpy as np
import QuantLib as ql
from tqdm import tqdm
from itertools import product
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor

current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
grandparent_dir = os.path.dirname(parent_dir)
grandgrandparent_dir = os.path.dirname(grandparent_dir)
sys.path.append(parent_dir)
sys.path.append(grandparent_dir)
sys.path.append(grandgrandparent_dir)
from settings import model_settings
from data_query import dirdatacsv
ms = model_settings()
os.chdir(current_dir)


def generate_barrier_features(s,K,T,barriers,updown,outin,W):
    
    barrier_features =  pd.DataFrame(
        product(
            [s],
            K,
            barriers,
            T,
            [updown],
            [outin],
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
                  ]
        )
    
    barrier_features['barrier_type_name'] = \
        barrier_features['updown'] + barrier_features['outin']
    
    return barrier_features

def compute_segmented_barriers(
        features, rebate, r, g, calculation_date, heston_parameters):
    features['theta'] = heston_parameters['theta']
    features['kappa'] = heston_parameters['kappa']
    features['rho'] = heston_parameters['rho']
    features['eta'] = heston_parameters['eta']
    features['v0'] = heston_parameters['v0']
    
    features['rebate'] = rebate
    features['dividend_rate'] = g
    features['risk_free_rate'] = r
    
    features = features.dropna()
    
    features['calculation_date'] = calculation_date
    
    features['barrier_price'] = ms.vector_barrier_price(features)
    
    features['calculation_date'] = datetime(
        calculation_date.year(),
        calculation_date.month(),
        calculation_date.dayOfMonth()
        )
    features['expiration_date'] =  features['calculation_date']  + pd.to_timedelta(
            features['days_to_maturity'], unit='D')
    priced_features = features.copy()
    return priced_features
    

def generate_barrier_type(T,updown,W,rebate,r,g,heston_parameters):
    if updown == 'Up':
        udw = 1
    elif updown == 'Down':
        udw = -1
    else:
        raise ValueError('invalid Up/Down flag')
       
    
    K = np.linspace(
        s*0.9,
        s*1.1,
        50
        )
    
    barriers = np.linspace(
        s*(1+udw*0.01),
        s*(1+udw*0.5),
        5
        ).astype(float).tolist()
    

    def price_out_barriers(
            s, K, T, barriers, updown, W, rebate, r, g, calculation_date):
        
        out_features = generate_barrier_features(
            s, K, T, barriers, updown, 'Out', W)
        out_barriers = compute_segmented_barriers(
            out_features, rebate, r, g, calculation_date,heston_parameters)
        
        return out_barriers

    def price_in_barriers(
            s, K, T, barriers, updown, W, rebate, r, g, calculation_date):
        
        in_features = generate_barrier_features(
            s, K, T, barriers, updown, 'In', W)
        in_barriers = compute_segmented_barriers(
            in_features, rebate, r, g, calculation_date,heston_parameters)
        
        return in_barriers
    
    with ThreadPoolExecutor() as executor:
        future_out = executor.submit(
            price_out_barriers, 
            s, K, T, barriers, updown, W, rebate, r, g, calculation_date)
        future_in = executor.submit(
            price_in_barriers, 
            s, K, T, barriers, updown, W, rebate, r, g, calculation_date)

    out_barriers = future_out.result()
    in_barriers = future_in.result()
    
    barriers = pd.concat([out_barriers,in_barriers],ignore_index=True)

    
    return barriers


"""
###########
# routine #
###########
"""

historical_calibrated = pd.read_csv(dirdatacsv()[0])
historical_calibrated = historical_calibrated.iloc[:1,1:].copy(
    ).reset_index(drop=True)
historical_calibrated['date'] = pd.to_datetime(historical_calibrated['date'])

r = 0.04 
rebate = 0.
step = 1
atm_spread = 1
r = 0.04

T = [
    60,
    90,
    180,
    360,
    540,
    720
    ]
W = ['call','put']


bar = tqdm(
    total = historical_calibrated.shape[0],
    desc = 'generating',
    )
for rowi, row in historical_calibrated.iterrows():
    
    s = row['spot_price']
    g = row['dividend_rate']
    
    calculation_datetime = row['date']
    calculation_date = ql.Date(
        calculation_datetime.day,
        calculation_datetime.month,
        calculation_datetime.year)
    
    param_names = ['theta', 'kappa', 'rho', 'eta', 'v0']
    heston_parameters = pd.Series(
        historical_calibrated.loc[rowi, param_names].values,
        index = param_names)
    h5_key = str('date_'+ calculation_datetime.strftime("%Y_%m_%d"))
    
    def generate_up_barriers(T, W, rebate, r, g, heston_parameters):
        return generate_barrier_type(
            T, 'Up', W, rebate, r, g, heston_parameters)

    def generate_down_barriers(T, W, rebate, r, g, heston_parameters):
        return generate_barrier_type(
            T, 'Down', W, rebate, r, g, heston_parameters)

    with ThreadPoolExecutor() as executor:
        future_up = executor.submit(
            generate_up_barriers, T, W, rebate, r, g, heston_parameters)
        future_down = executor.submit(
            generate_down_barriers, T, W, rebate, r, g, heston_parameters)
        up_barriers = future_up.result()
        down_barriers = future_down.result()
    
    barriers = pd.concat([down_barriers,up_barriers],ignore_index=True)
    calls = barriers[barriers['w'] == 'call']
    puts = barriers[barriers['w'] == 'put']
    while True:
        with pd.HDFStore('SPX barriers review.h5') as store:
            try:
                store.append(
                    f'/call/{h5_key}', calls, format='table', append=True)
                store.append(
                    f'/put/{h5_key}', puts, format='table', append=True)
                break
            except Exception as e:
                raise KeyError(f"error in '{h5_key}': {e}"
                                f"\nretrying in 5 seconds...")
                time.sleep(5)
    bar.update(1)
bar.close()
