# -*- coding: utf-8 -*-
"""
Created on Tue Sep 17 15:46:57 2024

@author: boomelage
"""

import os
import sys
from tqdm import tqdm
import numpy as np
import pandas as pd
import time
from datetime import datetime
from itertools import product

current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)

from settings import model_settings
from routine_calibration_testing import heston_parameters

ms = model_settings()
calculation_date = ms.calculation_date
day_count = ms.day_count

os.chdir(current_dir)
s = ms.s

def generate_train_features(K,T,s,flag):
    features = pd.DataFrame(
        product(
            [s],
            K,
            T,
            flag
            ),
        columns=[
            "spot_price", 
            "strike_price",
            "days_to_maturity",
            "w"
                  ])
    return features

"""
# =============================================================================
#                       generating training dataset
                
"""
K = np.linspace(
    
    s*0.9,
    
    s*1.1,
    
    5
    )

T = ms.T

T = np.arange(min(T),max(T),1)

title = 'vanillas'

flags = ['put','call']

print(f"\ngenerating {2*len(K)*len(T)} contracts...")

"""
# =============================================================================
"""
features = generate_train_features(K, T, s, flags)

features['dividend_rate'] = 0.02
features['risk_free_rate'] = 0.04
features['eta'] = heston_parameters['eta'].iloc[0]
features['theta'] = heston_parameters['theta'].iloc[0]
features['kappa'] = heston_parameters['kappa'].iloc[0]
features['rho'] = heston_parameters['rho'].iloc[0]
features['v0'] = heston_parameters['v0'].iloc[0]

progress_bar = tqdm(desc="pricing",total=features.shape[0],unit= "contracts")

features['heston_price'] = 0.00
for i, row in features.iterrows():
    
    s = row['spot_price']
    k = row['strike_price']
    t = row['days_to_maturity']
    r = row['risk_free_rate']
    g = row['dividend_rate']
    
    v0 = row['v0']
    kappa = row['kappa']
    theta = row['theta']
    eta = row['eta']
    rho = row['rho']
    w = row['w']
    
    h_price = ms.ql_heston_price(s,k,t,r,g,w,
                                 v0,kappa,theta,eta,rho,
                                 ms.calculation_date)
    features.at[i,'heston_price'] = h_price
    
    progress_bar.update(1)

progress_bar.close()

training_data = features.copy()

training_data = ms.noisyfier(training_data)

print(f"\ntraining data:\n{training_data}\ndescriptive statistics:\n")
pd.set_option("display.max_rows",None)
pd.set_option("display.max_columns",None)
print(training_data.describe())
pd.reset_option("display.max_rows")
pd.reset_option("display.max_columns")
file_time = datetime.fromtimestamp(time.time())
file_tag = file_time.strftime("%Y-%d-%m %H%M%S")
training_data.to_csv(os.path.join('vanillas',f'vanillas {file_tag}.csv'))