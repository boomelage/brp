# -*- coding: utf-8 -*-
"""
Created on Sat Sep 21 13:54:06 2024

@author: boomelage
"""
import os
import sys
import pandas as pd
import numpy as np
import QuantLib as ql
from tqdm import tqdm
from itertools import product
import time

current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
grandparent_dir = os.path.dirname(parent_dir)
grandgrandparent_dir = os.path.dirname(grandparent_dir)
sys.path.append(parent_dir)
sys.path.append(grandparent_dir)
sys.path.append(grandgrandparent_dir)
from settings import model_settings
ms = model_settings()

os.chdir(current_dir)

from data_query import dirdatacsv
csvs = dirdatacsv()
historical_calibrated = pd.read_csv(csvs[0])
historical_calibrated = historical_calibrated.iloc[:,1:].copy(
    ).reset_index(drop=True)

historical_calibrated['date'] = pd.to_datetime(
    historical_calibrated['date'],
    format='%d/%m/%Y')

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
    
    calculation_datetime = row['date']
    calculation_date = ql.Date(
        calculation_datetime.day,
        calculation_datetime.month,
        calculation_datetime.year)
    
    s = row['spot_price']
    
    spread = 0.2
    K = np.linspace(
        s*(1-spread),
        s*(1+spread),
        50
        )
    
    T = np.arange(30,360+1,7)

    features = pd.DataFrame(
        product(
            [s],
            K,
            T,
            ['call','put'],
            [calculation_date],
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
    
    
    features['heston_price'] = ms.vector_heston_price(features)
    
    features = features[
        ['spot_price', 'strike_price', 'w', 'heston_price', 'days_to_maturity',
          'risk_free_rate', 'dividend_rate', 
          'kappa', 'theta', 'rho', 'eta', 'v0',
          'calculation_date', 'expiration_date']
        ]
    
    features['calculation_date'] = calculation_datetime
    features['expiration_date'] =  calculation_datetime + pd.to_timedelta(
            features['days_to_maturity'], unit='D')
    
    hist_file_date = str('date_'+calculation_datetime.strftime("%Y_%m_%d"))
    calls = features[features['w'] == 'call'].reset_index(drop=True)
    puts = features[features['w'] == 'put'].reset_index(drop=True)
    while True:
        try:
            with pd.HDFStore('SPX vanillas sublime test.h5') as store:
                store.append(
                    f'/call/{hist_file_date}', calls, format='table', append=True)
                store.append(
                    f'/put/{hist_file_date}', puts, format='table', append=True)
            break 
        except Exception as e:
            print(f"Error encountered: {e}. Retrying in 5 seconds...")
            time.sleep(5)
            
    bar.update(1)
bar.close()

