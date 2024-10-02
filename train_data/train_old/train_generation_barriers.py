# -*- coding: utf-8 -*-
"""
Created on Thu Sep 26 17:33:55 2024

"""

import sys
import time
import numpy as np
import pandas as pd
import QuantLib as ql
from itertools import product
from datetime import datetime
sys.path.append(r'E:/git/machine-learning-option-pricing')
from settings import model_settings
ms = model_settings()



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
        features, calculation_date, heston_parameters, output_folder):
    
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
        
        
        features.at[i,'expiration_date'] = datetime(
            expiration_date.year(), 
            expiration_date.month(), 
            expiration_date.dayOfMonth()
            )
        
        features.at[i,'heston_price'] = ms.ql_heston_price(
            s,k,t,r,g,w,
            kappa,theta,rho,eta,v0,
            calculation_date
            )
    
        features.at[i,'barrier_price'] = ms.ql_barrier_price(
            s,k,t,r,g,calculation_date, w,
            barrier_type_name,barrier,rebate,
            kappa,theta,rho,eta,v0
            )
        
    training_data = features.copy()
    
    # training_data = ms.noisyfier(training_data)
    
    pd.set_option("display.max_columns",None)
    print(f'\n{training_data}\n')
    print(f'\n{training_data.describe()}')
    pd.reset_option("display.max_columns")
    
    date_tag = spot_date.strftime("%Y-%m-%d")
    file_time = datetime.fromtimestamp(time.time())
    file_time_tag = file_time.strftime("%Y-%m-%d %H%M%S")
    # training_data.to_csv(os.path.join(
    #     output_folder,f'barriers {date_tag} {file_time_tag}.csv'))

    return training_data



caldf = pd.read_csv(r'E:/git/machine-learning-option-pricing/historical_data/historical_generation/SPX2007-2012_calibrated.csv')

caldf = caldf.iloc[:1]


calculation_date = ql.Date(3,1,2007)

from oneoff_calibration_testing import heston_parameters
s = caldf['spot_price'].iloc[0]
g = caldf['dividend_rate'].iloc[0]
r = 0.04
rebate = 0.
spread = s*0.2
step = 1
atm_spread = 1
r = 0.04


K = np.linspace(s*0.8,s*1.2,5).tolist()
T = [30,60,90]
updown = 'Down'
OUTIN = ['Out']
W = ['put']
barriers = np.linspace(s*0.5,s*0.99,5).tolist()

features = generate_barrier_features(s, K, T, barriers, updown, OUTIN, W)


features['dividend_rate'] = caldf['dividend_rate'].iloc[0]
features['risk_free_rate'] = 0.04

generate_barrier_options(
        features, calculation_date, heston_parameters, "")
pd.set_option("display.max_columns",None)
pd.set_option("display.max_rows",None)
features



