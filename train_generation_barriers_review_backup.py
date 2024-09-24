#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Sep 10 17:55:16 2024

"""

import os
import sys
os.chdir(os.path.dirname(os.path.abspath(__file__)))
sys.path.append('term_structure')
sys.path.append('contract_details')
sys.path.append('misc')
import QuantLib as ql
import numpy as np
from settings import model_settings
import pandas as pd
import time
pd.set_option("display.max_columns",None)
pd.reset_option("display.max_rows")
ms = model_settings()
from pricing import noisyfier
from tqdm import tqdm
from routine_calibration_testing import heston_parameters
calculation_date = ql.Date().todaysDate()
day_count = ql.Actual365Fixed()



def price_barrier_option_row(row,progress_bar):
    
    barrier_type_name = row['barrierType']
    try:
        if barrier_type_name == 'UpOut':
            barrierType = ql.Barrier.UpOut
        elif barrier_type_name == 'DownOut':
            barrierType = ql.Barrier.DownOut
        elif barrier_type_name == 'UpIn':
            barrierType = ql.Barrier.UpIn
        elif barrier_type_name == 'DownIn':
            barrierType = ql.Barrier.DownIn
        else:
            print('barrier flag error')

        t = row['days_to_maturity']
        calculation_date = ms.calculation_date
        expiration_date = calculation_date + ql.Period(int(t), ql.Days)
        
        K = row['strike_price']
        barrier = row['barrier']
        
        exercise = ql.EuropeanExercise(expiration_date)
        
        payoff = ql.PlainVanillaPayoff(ql.Option.Call, K)
        barrierOption = ql.BarrierOption(barrierType, barrier, rebate, payoff, exercise)
        barrierOption.setPricingEngine(engine)
        barrier_price = barrierOption.NPV()
        
        row['barrier_price'] = barrier_price
        progress_bar.update(1)
        return row
    except Exception:
        print(barrier_type_name)

"""
# Geometric Asian Option
rng = "pseudorandom" # could use "lowdiscrepancy"
numPaths = 100000

engine = ql.MCDiscreteArithmeticAPHestonEngine(hestonProcess, rng, requiredSamples=numPaths)
"""


import pandas as pd
from itertools import product
def generate_features(K,T,B,s):
    features = pd.DataFrame(
        product(
            [s],
            K,
            T,
            B,
            ['In','Out']
            ),
        columns=[
            "spot_price", 
            "strike_price",
            "days_to_maturity",
            "barrier",
            "outin"
                  ])
    return features

start = time.time()



s = ms.s

rebate = 0.

spotHandle = ql.QuoteHandle(ql.SimpleQuote(s))

v0, kappa, theta, sigma, rho = heston_parameters['v0'].iloc[0],heston_parameters['kappa'].iloc[0],\
    heston_parameters['theta'].iloc[0],heston_parameters['sigma'].iloc[0],heston_parameters['rho'].iloc[0]

flatRateTs = ms.make_ts_object(0.04)
flatDividendTs = ms.make_ts_object(0.04)

hestonProcess = ql.HestonProcess(
    flatRateTs, flatDividendTs, spotHandle, v0, kappa, theta, sigma, rho)

hestonModel = ql.HestonModel(hestonProcess)

engine = ql.FdHestonBarrierEngine(hestonModel)

# T = ms.T
T = [1]
pricing_spread = 0.005
n_spread_steps = 10
n_strikes = 10
K = np.linspace(
    s*(1+pricing_spread), 
    s*(1+n_spread_steps*pricing_spread),
    n_strikes)



"""
up options
"""
n_barriers = 5
max_barrier =  1.1
up_barriers  = np.linspace(s * 1.01, s * max_barrier, n_barriers)

up_features = generate_features(K,T,up_barriers,s)
up_features['updown'] = 'Up' 


"""
down options

"""
n_barreirs = 5
min_barrier = 0.9
down_barriers  = np.linspace(s * min_barrier, s*0.99, n_barreirs)

down_features = generate_features(K,T,down_barriers,s)
down_features['updown'] = 'Down' 

n_contracts = 4*n_barreirs*len(T)*len(K)

print(f"\ngenerating {n_contracts} contracts...\n")

features = pd.concat([up_features,down_features]).reset_index(drop=True)
features['sigma'] = heston_parameters['sigma'].iloc[0]
features['theta'] = heston_parameters['theta'].iloc[0]
features['kappa'] = heston_parameters['kappa'].iloc[0]
features['rho'] = heston_parameters['rho'].iloc[0]
features['v0'] = heston_parameters['v0'].iloc[0]
features['w'] = 'call'
features['moneyness'] = features['spot_price']/features['strike_price']
features['barrierType'] = features['updown'] + features['outin']

progress_bar = tqdm(
    desc="pricing",total=features.shape[0],unit="contracts",leave=False)
features = features.apply(price_barrier_option_row,axis=1,progress_bar=progress_bar)
progress_bar.close()

training_data = noisyfier(features)

pd.set_option("display.max_columns",None)
print(f'\n{training_data}\n')
pd.reset_option("display.max_columns")
