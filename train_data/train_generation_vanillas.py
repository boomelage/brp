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

from settings import model_settings, compute_moneyness
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
    
    s*1.1,
    s*1.2,
    
    50
    )

# T = ms.T

T = np.arange(1,366,1)

title = 'vanillas'

flags = [
    # 'put',
    'call'
    ]

print(f"\ngenerating {len(flags)*len(K)*len(T)} contracts..."
      f"\nstrikes ({len(K)}):\n{K}\n\nmaturities ({len(T)}):"
      f"\n{T}\n\ntypes:\n{flags}")
"""
# =============================================================================
"""
features = generate_train_features(K, T, s, flags)

features = compute_moneyness(features)

g = 0.02
r = 0.04

print("\npricing contracts...\n")

w = features['w']
k = features['strike_price']
expiration_date = ms.compute_ql_maturity_dates(
    features['days_to_maturity'],calculation_date)
features['heston_price'] = ms.vector_heston_price(
            s,k,r,g,w,
            v0 = heston_parameters['v0'],
            kappa = heston_parameters['kappa'],
            theta = heston_parameters['theta'],
            eta = heston_parameters['eta'],
            rho = heston_parameters['rho'],
            calculation_date = calculation_date,
            expiration_date = expiration_date
    )

training_data = features.copy()

training_data = ms.noisyfier(training_data)

print(f"\n\ntraining data:\n{training_data}\ndescriptive statistics:\n")
pd.set_option("display.max_rows",None)
pd.set_option("display.max_columns",None)
print(training_data.describe())
pd.reset_option("display.max_rows")
pd.reset_option("display.max_columns")
file_time = datetime.fromtimestamp(time.time())
file_tag = file_time.strftime("%Y-%d-%m %H%M%S")
training_data.to_csv(os.path.join('vanillas',f'vanillas {file_tag}.csv'))
print("\ndata saved to csv!")