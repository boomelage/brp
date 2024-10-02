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
np.set_printoptions(precision=10, suppress=True)

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
    ql_calc = ql.Date(calculation_date.day,calculation_date.month,calculation_date.year)
    
    s = row['spot_price']
    
    
    
    # spread = s*0.2
    # atm_spread = 0
    # step = 1
    # K = ms.make_K(s, spread, atm_spread, step)
    
    spread = 0.2
    K = np.linspace(
        s*(1-spread),
        s,
        int(
            (s-s*(1-spread))*2
            )
        )
    
    T = np.arange(
        30,
        180,
        1
        )
    
    features = pd.DataFrame(
        product(
            [s],
            K,
            T,
            ['put'],
            [ql_calc],
            [row['30D']],
            [row['60D']],
            [row['3M']],
            [row['6M']],
            [row['12M']],
            [row['18M']],
            [row['24M']],
            ),
        columns=[
            'spot_price', 
            'strike_price',
            'days_to_maturity',
            'w',
            'calculation_date',
            '30D', '60D', '3M', '6M', '12M', '18M', '24M'
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
                features['risk_free_rate'],
                features['dividend_rate'],
                features['w'],
                features['kappa'],
                features['theta'],
                features['rho'],
                features['eta'],
                features['v0'],
                features['calculation_date'],
                features['expiration_date']
        )
    
    features = features[
        [
          'spot_price', 'strike_price',  'w', 'heston_price',
          '30D', '60D', '3M', '6M', '12M', '18M', '24M', 
          'days_to_maturity', 'risk_free_rate', 'dividend_rate',
          'kappa', 'theta', 'rho', 'eta', 'v0',
          'calculation_date', 'expiration_date'
          ]
        ]
    
    features['calculation_date'] = calculation_date
    features['expiration_date'] =  features[
        'calculation_date'] + pd.to_timedelta(
            features['days_to_maturity'], unit='D')
    
    historical_training_data = pd.concat(
        [historical_training_data, features],
        ignore_index = True)
    
    hist_file_date = calculation_date.strftime("%Y-%m-%d") 
    file_datetime = datetime.fromtimestamp(time.time())
    file_tag = file_datetime.strftime("%Y-%m-%d %H%M%S") 
    file_path = os.path.join(
        parent_dir,
        r'historical_vanillas',
        str(hist_file_date+' '+file_tag+'.csv'),
        )
    
    features.to_csv(file_path)
    bar.update(1)
bar.close()
    
