#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Sep 10 17:55:16 2024

"""
import QuantLib as ql
import numpy as np
from settings import model_settings
import pandas as pd
import time
pd.set_option("display.max_columns",None)
pd.reset_option("display.max_rows")
ms = model_settings()
from routine_calibration_global import calibrate_heston
from pricing import noisyfier
calculation_date = ql.Date().todaysDate()
day_count = ql.Actual365Fixed()



def price_barrier_option_row(row):
    
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

from routine_calibration_generation import contract_details
s = ms.s
heston_parameters = calibrate_heston(contract_details, s, calculation_date)

t = 7
k = ms.s
barrier = ms.s-ms.s+1
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

T = ms.T
pricing_spread = 0.002
K = np.linspace(s*(1+pricing_spread), s*(1+3*pricing_spread),5)

n = 200

"""
up options
"""

max_barrier =  1.1
up_barriers  = np.linspace(s * 1.01, s * max_barrier, n)

up_features = generate_features(K,T,up_barriers,s)
up_features['updown'] = 'Up' 


"""
down options
"""

min_barrier = 0.9
down_barriers  = np.linspace(s * min_barrier, s*0.99, n)

down_features = generate_features(K,T,down_barriers,s)
down_features['updown'] = 'Down' 

n_contracts = 4*n*len(T)*len(K)

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
features = features.apply(price_barrier_option_row,axis=1)

training_data = noisyfier(features)

pd.set_option("display.max_columns",None)
print(f'\n{training_data}\n')
pd.reset_option("display.max_columns")

end_time = time.time()

runtime = end_time - start

print(f"runtime: {round(runtime,4)} seconds")





