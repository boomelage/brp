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



np.set_printoptions(precision=10, suppress=True)
pd.set_option("display.max_columns",None)
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
grandparent_dir = os.path.dirname(parent_dir)
sys.path.append(parent_dir)
sys.path.append(grandparent_dir)
sys.path.append(os.path.join(grandparent_dir,'train_data'))
from settings import model_settings

from data_query import dirdatacsv

ms = model_settings()

os.chdir(current_dir)

csvs = dirdatacsv()

"""
###########
# routine #
###########
"""

historical_calibrated = pd.read_csv(csvs[0])
historical_calibrated = historical_calibrated.iloc[:1,1:].copy(
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
    g = row['dividend_rate']
    r = 0.04
    calc_dtdate = row['date']
    
    calculation_date = ql.Date(
        calc_dtdate.day,calc_dtdate.month,calc_dtdate.year)
    
    
    rebate = 0.
    spread = s*0.2
    step = 1
    atm_spread = 1
    r = 0.04
    K = np.linspace(s*0.8,s*1.2,50).tolist()
    T = [30,60,90,180,360]
    OUTIN = ['Out','In']
    W = ['put']
    n_barriers = 5
    
    
    heston_parameters = historical_calibrated.loc[
        rowi,['theta', 'kappa', 'rho', 'eta', 'v0']
        ]
    
    barriers = np.linspace(s*0.5,s*0.99,n_barriers).tolist()
    down_features = generate_barrier_features(
        s, K, T, barriers, 'Down', OUTIN, W)
    
    barriers = np.linspace(s*1.01,s*1.5,n_barriers).tolist()
    up_features = generate_barrier_features(
        s, K, T, barriers, 'Up', OUTIN, W)
    
    features = pd.concat(
        [down_features,up_features],
        ignore_index = True
        )
    
    features['barrier_price'] = ms.vector_barrier_price(
            s,
            features['strike_price'],
            features['days_to_maturity'],
            r,g,calculation_date, 
            features['w'],
            features['barrier_type_name'],
            features['barrier'],
            rebate,
            heston_parameters['kappa'],
            heston_parameters['theta'],
            heston_parameters['rho'],
            heston_parameters['eta'],
            heston_parameters['v0']
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
    
historical_barriers