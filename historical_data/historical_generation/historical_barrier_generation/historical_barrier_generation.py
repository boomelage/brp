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
from tqdm import tqdm
from itertools import product
from datetime import datetime

def generate_barrier_features(s, K, T, barriers, updown, OUTIN, W):
    barrier_features = pd.DataFrame(
        product([s], K, barriers, T, [updown], OUTIN, W),
        columns=[
            'spot_price', 'strike_price', 'barrier', 'days_to_maturity',
            'updown', 'outin', 'w'
        ]
    )
    
    barrier_features['barrier_type_name'] = \
        barrier_features['updown'] + barrier_features['outin']
    
    return barrier_features


current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
grandparent_dir = os.path.dirname(parent_dir)
grandgrandparent_dir = os.path.dirname(grandparent_dir)
sys.path.append(parent_dir)
sys.path.append(grandparent_dir)
sys.path.append(grandgrandparent_dir)
from model_settings import ms
from data_query import dirdatacsv
os.chdir(current_dir)

"""
###########
# routine #
###########
"""

historical_calibrated = pd.read_csv(dirdatacsv()[0])
historical_calibrated = historical_calibrated.iloc[:,1:].copy(
    ).reset_index(drop=True)
historical_calibrated['date'] = pd.to_datetime(historical_calibrated['date'])

bar = tqdm(
    total = historical_calibrated.shape[0],
    desc = 'generating',
    leave = True
    )
for rowi, row in historical_calibrated.iterrows():
    s = row['spot_price']
    
    calculation_datetime = row['date']

    r = 0.04 
    rebate = 0.
    step = 1
    atm_spread = 1
    r = 0.04
    K = np.linspace(
        s*0.9,
        s*1.1,
        50
        )
    T = [
        60,
        90,
        180,
        360,
        540,
        720
        ]
    OUTIN = ['Out','In']
    W = ['call','put']
        
    
    barriers = np.linspace(
        s*0.5,s*0.99,
        5
        ).astype(float).tolist()
    down_features = generate_barrier_features(
        s, K, T, barriers, 'Down', OUTIN, W)
    
    
    barriers = np.linspace(
        s*1.01,s*1.5,
        5
        ).astype(float).tolist()
    up_features = generate_barrier_features(
        s, K, T, barriers, 'Up', OUTIN, W)

    
    features = pd.concat(
        [down_features,up_features],
        ignore_index = True
        )
    features['rebate'] = rebate
    features['dividend_rate'] = row['dividend_rate']
    features['risk_free_rate'] = r
    features.loc[:,'theta'] = historical_calibrated.loc[rowi,'theta']
    features.loc[:,'kappa'] = historical_calibrated.loc[rowi,'kappa']
    features.loc[:,'rho'] = historical_calibrated.loc[rowi,'rho']
    features.loc[:,'eta'] = historical_calibrated.loc[rowi,'eta']
    features.loc[:,'v0'] = historical_calibrated.loc[rowi,'v0']
    features['calculation_date'] = calculation_datetime
    
    # print(features)
    try:
       features['barrier_price'] = ms.vector_barrier_price(features)

       features['expiration_date'] =  calculation_datetime + pd.to_timedelta(
                features['days_to_maturity'], unit='D')
        
       h5_key = str('date_'+ calculation_datetime.strftime("%Y_%m_%d"))
        
       calls = features[features['w'] == 'call']
       puts = features[features['w'] == 'put']
       while True:
           with pd.HDFStore('SPX barriers.h5') as store:
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
    except Exception:
        pass
    bar.update(1)
bar.close()

