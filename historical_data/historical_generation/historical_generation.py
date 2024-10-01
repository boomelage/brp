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

# row = historical_calibrated.iloc[0]

bar = tqdm(
    total = historical_calibrated.shape[0],
    desc = 'generating',
    unit = 'day'
    )
for rowi, row in historical_calibrated.iterrows():

    atm_vols = row[['30D', '60D', '3M', '6M', '12M', '18M', '24M']]    
    
    calculation_date = row['date']
    ql_calc = ql.Date(calculation_date.day,calculation_date.month,calculation_date.year)
    
    s = row['spot_price']
    
    spread = 0.2
    wing_size = 250
    
    put_K = np.linspace(
        s*(1-spread),
        s*0.995,
        wing_size)
    
    call_K = np.linspace(
        s*1.005,
        s*(1+spread),
        wing_size)
    
    K = np.unique(np.array([put_K,call_K]).flatten())
    
    T = np.arange(7,180,7)
    
    
    features = pd.DataFrame(
        product(
            [s],
            K,
            T,
            ['call','put'],
            [ql_calc],
            [atm_vols.iloc[0]],
            [atm_vols.iloc[1]],
            [atm_vols.iloc[2]],
            [atm_vols.iloc[3]],
            [atm_vols.iloc[4]],
            [atm_vols.iloc[5]],
            [atm_vols.iloc[6]],
            ),
        columns=[
            'spot_price', 
            'strike_price',
            'days_to_maturity',
            'w',
            'calculation_date',
            '30D', '60D', '3M', '6M', '12M', '18M', '24M'
                  ])
     
    
    features['expiration_date'] = ms.vexpiration_datef(
        features['days_to_maturity'],
        features['calculation_date']
        )
    
    theta = row['theta']
    kappa = row['kappa']
    rho = row['rho']
    eta = row['eta']
    v0 = row['v0']
    g = row['dividend_rate']
    r = 0.04
    
    features.loc[:,'heston_price'] = ms.vector_heston_price(
                s,
                features['strike_price'],
                r,g,
                features['w'],
                v0,kappa,theta,eta,rho,
                features['calculation_date'],
                features['expiration_date']
        )
    
    features = features[
        [
         'spot_price', 'strike_price',  'w','heston_price', 'days_to_maturity',
         '30D', '60D', '3M', '6M', '12M', '18M', '24M',
         'calculation_date', 'expiration_date'
         ]
        ]
    
    features['calculation_date'] = calculation_date
    
    features['expiration_date'] =  features['calculation_date'] + pd.to_timedelta(features['days_to_maturity'], unit='D')
    
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
    
