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

current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
grandparent_dir = os.path.dirname(parent_dir)
sys.path.append(parent_dir)
sys.path.append(grandparent_dir)
sys.path.append(os.path.join(grandparent_dir,'term_structure'))
from settings import model_settings
ms = model_settings()

os.chdir(current_dir)

from data_query import dirdatacsv
csvs = dirdatacsv()
historical_calibrated = pd.read_csv(csvs[0])
historical_calibrated = historical_calibrated.iloc[:,1:].copy(
    ).reset_index(drop=True)

historical_calibrated['date'] = pd.to_datetime(historical_calibrated['date'])

os.chdir(current_dir)

pd.set_option("display.max_columns",None)

"""
###########
# routine #
###########
"""


historical_training_data = pd.DataFrame()


bar = tqdm(
    total = historical_calibrated.shape[0],
    desc = 'generating',
    unit = 'day'
    )
for rowi, row in historical_calibrated.iterrows():
    
    calculation_date = row['date']
    ql_calc = ql.Date(
        calculation_date.day,calculation_date.month,calculation_date.year)
    
    s = row['spot_price']
    
    spread = 0.2
    K = np.linspace(
        s*(1-spread),
        s*(1+spread),
        int(
            (s-s*(1-spread*2))*3
            )
        )
    
    T = np.arange(30,360,1)
    
    features = pd.DataFrame(
        product(
            [s],
            K,
            T,
            ['call'],
            [ql_calc],
            ),
        columns=[
            'spot_price', 
            'strike_price',
            'days_to_maturity',
            'w',
            'calculation_date',
                  ]
        )
     
    
    features['expiration_date'] = ms.vexpiration_datef(
        features['days_to_maturity'],
        features['calculation_date']
        )
    
    
    features['kappa'] = row['kappa']
    features['theta'] = row['theta']
    features['rho'] = row['rho']
    features['v0'] = row['v0']
    features['eta'] = row['eta']
    features['dividend_rate'] = row['dividend_rate']
    features['risk_free_rate'] = 0.04
    
    
    features.loc[:,'heston_price'] = ms.vector_heston_price(
                features['spot_price'],
                features['strike_price'],
                features['days_to_maturity'],
                features['risk_free_rate'],
                features['dividend_rate'],
                features['w'],
                features['kappa'],
                features['theta'],
                features['rho'],
                features['eta'],
                features['v0'],
                features['calculation_date']
        )
    
    features = features[
        ['spot_price', 'strike_price', 'w', 'heston_price', 'days_to_maturity',
         'risk_free_rate', 'dividend_rate', 
         'kappa', 'theta', 'rho', 'eta', 'v0',
         'calculation_date', 'expiration_date']
        ]
    
    features['calculation_date'] = calculation_date
    features['expiration_date'] =  features[
        'calculation_date'] + pd.to_timedelta(
            features['days_to_maturity'], unit='D')

    features[
        ['calculation_date', 'expiration_date']
        ] = features[['calculation_date', 'expiration_date']].astype(str)
    features['days_to_maturity'] = features['days_to_maturity'].astype('int64')
    
    hist_file_date = str('date_'+calculation_date.strftime("%Y_%m_%d"))

    calls = features[features['w'] == 'call'].reset_index(drop=True)
    puts = features[features['w'] == 'put'].reset_index(drop=True)
    
    with pd.HDFStore('SPXvanillas.h5') as store:
        try:
            store.append(
                f'/call/{hist_file_date}', calls, format='table', append=True)
            store.append(
                f'/put/{hist_file_date}', puts, format='table', append=True)
        except Exception:
            print()
            print(store.select(f'/call/{hist_file_date}').columns)
            print()
            print(store.select(f'/call/{hist_file_date}').columns)
            print()
            print(features.columns)
            print()
            print(calls.dtypes)
            print()
            print(puts.dtypes)
            print()
            print(store.select(f'/call/{hist_file_date}').dtypes)
    bar.update(1)
bar.close()
    
