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
def generate_features(K,T,s):
    features = pd.DataFrame(
        product(
            [s],
            K,
            T,
            ),
        columns=[
            "spot_price", 
            "strike_price",
            "days_to_maturity",
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


"""
# =============================================================================
                    generating OTM call barrier options
"""


# T = ms.T
# spread = 0.005
# n_spreads = 10
# n_strikes = 300
# up_K = np.linspace(
#     s*(1+spread), 
#     s*(1+n_spreads*spread),
#     n_strikes)

# down_K = np.linspace(
#     s*(1-spread), 
#     s*(1-n_spreads*spread),
#     n_strikes
#     )

# n_barriers = n_strikes
# barrier_spread = spread
# n_barrier_spreads = n_spreads


"""
small example
"""
T = [1]
spread = 0.005
n_spreads = 5
n_strikes = 2
up_K = np.linspace(
    s*(1+spread), 
    s*(1+n_spreads*spread),
    n_strikes)

down_K = np.linspace(
    s*(1-spread), 
    s*(1-n_spreads*spread),
    n_strikes
    )

n_barriers = 2
barrier_spread = 0.005
n_barrier_spreads = 2


"""
# =============================================================================
                                up options
"""
initial_up_features = generate_features(up_K,T,s)
up_features = pd.DataFrame()
up_bar = tqdm(
    desc="ups",
    total=initial_up_features.shape[0],
    unit='sets',
    leave=True)
for i, row in initial_up_features.iterrows():
    s = row['spot_price']
    k = row['strike_price']
    t = row['days_to_maturity']
    col_names = [
        'spot_price', 'strike_price', 'days_to_maturity','barrier','outin','w']
    strike_wise_np = np.zeros((n_barriers,len(col_names)),dtype=float)
    strike_wise_out = pd.DataFrame(strike_wise_np).copy()
    strike_wise_in = pd.DataFrame(strike_wise_np).copy()
    strike_wise_out.columns = col_names
    strike_wise_in.columns = col_names
    strike_wise_out['strike_price'] = k
    strike_wise_in['strike_price'] = k
    strike_wise_out['spot_price'] = s
    strike_wise_in['spot_price'] = s
    strike_wise_out['days_to_maturity'] = t
    strike_wise_in['days_to_maturity'] = t
    
    
    strike_wise_out['w'] = 'call'
    strike_wise_out['updown'] = 'Up'
    strike_wise_out['outin'] = 'Out'
    
    
    strike_wise_in['w'] = 'call'
    strike_wise_in['updown'] = 'Up'
    strike_wise_in['outin'] = 'In'
    
    barriers = np.linspace(
        k*(1+barrier_spread),
        k*(1+n_barrier_spreads*barrier_spread),
        n_barriers
        )
    
    strike_wise_in['barrier'] = barriers
    strike_wise_out['barrier'] = barriers
    
    strike_wise = pd.concat(
        [strike_wise_in, strike_wise_out],
        ignore_index=True)
    
    up_features = pd.concat(
        [up_features, strike_wise],
        ignore_index=True
        )
    up_bar.update(1)
up_bar.close()
    
"""
# =============================================================================
                                down options
"""

initial_down_features = generate_features(down_K,T,s)
down_features = pd.DataFrame()
down_bar = tqdm(
    desc="downs",
    total=initial_down_features.shape[0],
    unit='sets',
    leave=True)
for i, row in initial_down_features.iterrows():
    s = row['spot_price']
    k = row['strike_price']
    t = row['days_to_maturity']
    col_names = [
        'spot_price', 'strike_price', 'days_to_maturity','barrier','outin','w']
    strike_wise_np = np.zeros((n_barriers,len(col_names)),dtype=float)
    strike_wise_out = pd.DataFrame(strike_wise_np).copy()
    strike_wise_in = pd.DataFrame(strike_wise_np).copy()
    strike_wise_out.columns = col_names
    strike_wise_in.columns = col_names
    strike_wise_out['strike_price'] = k
    strike_wise_in['strike_price'] = k
    strike_wise_out['spot_price'] = s
    strike_wise_in['spot_price'] = s
    strike_wise_out['days_to_maturity'] = t
    strike_wise_in['days_to_maturity'] = t
    
    strike_wise_out['w'] = 'put'
    strike_wise_out['updown'] = 'Down'
    strike_wise_out['outin'] = 'Out'
    
    
    strike_wise_in['w'] = 'put'
    strike_wise_in['updown'] = 'Down'
    strike_wise_in['outin'] = 'In'
    
    barriers = np.linspace(
        k*(1-barrier_spread),
        k*(1-n_barrier_spreads*barrier_spread),
        n_barriers
        )
    
    strike_wise_in['barrier'] = barriers
    strike_wise_out['barrier'] = barriers
    
    strike_wise = pd.concat(
        [strike_wise_in, strike_wise_out],
        ignore_index=True)
    
    down_features = pd.concat(
        [down_features, strike_wise],
        ignore_index=True
        )
    down_bar.update(1)
down_bar.close()
    
    
"""  
# =============================================================================
"""

features = pd.concat(
    [up_features,down_features],
    ignore_index=True)

features['barrierType'] = features['updown'] + features['outin'] 
features['sigma'] = heston_parameters['sigma'].iloc[0]
features['theta'] = heston_parameters['theta'].iloc[0]
features['kappa'] = heston_parameters['kappa'].iloc[0]
features['rho'] = heston_parameters['rho'].iloc[0]
features['v0'] = heston_parameters['v0'].iloc[0]

features = features.sort_values(by='days_to_maturity',ascending=False)
pricing_bar = tqdm(
    desc="pricing",total=features.shape[0],unit="contracts",leave=True)
features = features.apply(price_barrier_option_row,axis=1,progress_bar=pricing_bar)


pricing_bar.close()

training_data = noisyfier(features)

pd.set_option("display.max_columns",None)
print(f'\n{training_data}\n')
pd.reset_option("display.max_columns")
