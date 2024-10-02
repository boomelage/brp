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

current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)
from settings import model_settings
from data_query import dirdatacsv
ms = model_settings()
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

historical_barriers = pd.DataFrame()

bar = tqdm(
    total = historical_calibrated.shape[0],
    desc = 'generating',
    unit = 'day'
    )
for rowi, row in historical_calibrated.iterrows():
    s = row['spot_price']
    
    calc_dtdate = row['date']
    
    calculation_date = ql.Date(
        calc_dtdate.day,calc_dtdate.month,calc_dtdate.year)
    
    r = 0.04 
    rebate = 0.
    step = 1
    atm_spread = 1
    r = 0.04
    K = np.linspace(
        s*0.9,
        s*1.1,
        5
        )
    T = [30,60,90,180,360]
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

    features[
        ['theta', 'kappa', 'rho', 'eta', 'v0']
        ] = historical_calibrated.loc[
            rowi, ['theta', 'kappa', 'rho', 'eta', 'v0']
            ].values
            
    features['calculation_date'] = calculation_date
    
    features['barrier_price'] = ms.vector_barrier_price(
            features['spot_price'],
            features['strike_price'],
            features['days_to_maturity'],
            features['risk_free_rate'],
            features['dividend_rate'],
            features['calculation_date'],
            features['w'],
            features['barrier_type_name'],
            features['barrier'],
            features['rebate'],
            features['kappa'],
            features['theta'],
            features['rho'],
            features['eta'],
            features['v0']
        )
    
    historical_barriers = pd.concat(
        [historical_barriers, features],
        ignore_index=True
        )
    
    file_time = datetime.fromtimestamp(
        time.time()).strftime("%Y-%m-%d %H-%M-%S")
    file_tag = f'{str(file_time)} barriers.csv'
    file_path = os.path.join(parent_dir,'historical_barriers',file_tag)
    features.to_csv(file_path)
    
    bar.update(1)
bar.close()
