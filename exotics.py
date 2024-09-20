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
from routine_calibration_global import heston_parameters
today = ql.Date().todaysDate()
day_count = ql.Actual365Fixed()


flatRateTs = ql.YieldTermStructureHandle(ql.FlatForward(today, 0.05, day_count))
calendar = ql.UnitedStates(ql.UnitedStates.GovernmentBond)
flatDividendTs = ql.YieldTermStructureHandle(ql.FlatForward(today, 0.02, day_count))
flatVolTs = ql.BlackVolTermStructureHandle(ql.BlackConstantVol(today, calendar, 0.2, day_count))


# bsm = ql.BlackScholesProcess(spotHandle, flatRateTs, flatVolTs)
# engine = ql.AnalyticBarrierEngine(bsm)




T = 1
K = 100.
barrier = 1000.
rebate = 0.
today = ql.Date().todaysDate()
maturity = today + ql.Period(int(T*365), ql.Days)

barrierType = ql.Barrier.UpOut
barrierType = ql.Barrier.DownOut
barrierType = ql.Barrier.UpIn
barrierType = ql.Barrier.DownIn

# exercise = ql.AmericanExercise(today, maturity, True)
s = 100

spotHandle = ql.QuoteHandle(ql.SimpleQuote(s))
v0, kappa, theta, sigma, rho = heston_parameters['v0'].iloc[0],heston_parameters['kappa'].iloc[0],\
    heston_parameters['theta'].iloc[0],heston_parameters['sigma'].iloc[0],heston_parameters['rho'].iloc[0]

hestonProcess = ql.HestonProcess(
    flatRateTs, flatDividendTs, spotHandle, v0, kappa, theta, sigma, rho)

hestonModel = ql.HestonModel(hestonProcess)

engine = ql.FdHestonBarrierEngine(hestonModel)


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

        # # s = row['spot_price']
        # v0 = row['v0']
        # kappa = row['kappa']
        # theta = row['theta'] 
        # sigma = row['sigma'] 
        # rho  = row['rho']
        # spotHandle = ql.QuoteHandle(ql.SimpleQuote(s))
        
        
        K = row['strike_price']
        barrier = row['barrier']
        
        exercise = ql.EuropeanExercise(maturity)
        
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

# T = np.array(ms.T,dtype=float)/365
# T = np.arange(1/12, 2.01, 3/12)
T = [7/365]

K = np.linspace(s*0.5, s*1.5, 10)

min_barrier = 0.4
down_barriers  = np.linspace(s * min_barrier, s*0.99, 10)

"""
                                                                     up options
"""

max_barrier =  1.6
up_barriers  = np.linspace(s * 1.01, s * max_barrier, 10)

up_features = generate_features(K,T,up_barriers,s)
up_features['updown'] = 'Up' 
down_features = generate_features(K,T,down_barriers,s)
down_features['updown'] = 'Down' 
features = pd.concat([up_features,down_features]).reset_index(drop=True)

features['sigma'] = heston_parameters['sigma'].iloc[0]
features['theta'] = heston_parameters['theta'].iloc[0]
features['kappa'] = heston_parameters['kappa'].iloc[0]
features['rho'] = heston_parameters['rho'].iloc[0]
features['v0'] = heston_parameters['v0'].iloc[0]
features['barrierType'] = features['updown'] + features['outin']
features = features.apply(price_barrier_option_row,axis=1)


print(f'\n{features}\n')


end_time = time.time()

runtime = end_time - start

print(f"runtime: {round(runtime,4)} seconds")