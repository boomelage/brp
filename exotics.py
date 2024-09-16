#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Sep 10 17:55:16 2024

"""
import QuantLib as ql

today = ql.Date().todaysDate()

spotHandle = ql.QuoteHandle(ql.SimpleQuote(100))
flatRateTs = ql.YieldTermStructureHandle(ql.FlatForward(today, 0.05, ql.Actual365Fixed()))
calendar = ql.UnitedStates(ql.UnitedStates.GovernmentBond)
day_count = ql.Actual365Fixed()
flatDividendTs = ql.YieldTermStructureHandle(ql.FlatForward(today, 0.02, day_count))
flatVolTs = ql.BlackVolTermStructureHandle(ql.BlackConstantVol(today, calendar, 0.2, ql.Actual365Fixed()))


# bsm = ql.BlackScholesProcess(spotHandle, flatRateTs, flatVolTs)
# engine = ql.AnalyticBarrierEngine(bsm)


v0, kappa, theta, sigma, rho = 0.01, 2.0, 0.01, 0.01, 0.0

hestonProcess = ql.HestonProcess(flatRateTs, flatDividendTs, spotHandle, v0, kappa, theta, sigma, rho)

hestonModel = ql.HestonModel(hestonProcess)

engine = ql.FdHestonBarrierEngine(hestonModel)

T = 1
K = 100.
barrier = 1000.
rebate = 0.
today = ql.Date().todaysDate()
maturity = today + ql.Period(int(T*365), ql.Days)
barrierType = ql.Barrier.UpOut
# barrierType = ql.Barrier.DownOut
# barrierType = ql.Barrier.UpIn
# barrierType = ql.Barrier.DownIn
# exercise = ql.AmericanExercise(today, maturity, True)
exercise = ql.EuropeanExercise(maturity)

def price_barrier_option(T, K, barrier, rebate, barrierType, engine, exercise):
    
    payoff = ql.PlainVanillaPayoff(ql.Option.Call, K)

    barrierOption = ql.BarrierOption(barrierType, barrier, rebate, payoff, exercise)
    
    barrierOption.setPricingEngine(engine)
    
    barrier_price = barrierOption.NPV()
    
    return barrier_price

try:
    price = price_barrier_option(T, K, barrier, rebate, barrierType, engine, exercise)
except Exception:
    price = None
print(f'\nbarrier: {price}')


from pricing import heston_price_one_vanilla
heston_price = heston_price_one_vanilla(1, today, K, maturity, spotHandle.value(), 0.05, 0.02, v0, kappa, theta, sigma, rho)    
print(f'vanilla: {heston_price}')



# Geometric Asian Option
rng = "pseudorandom" # could use "lowdiscrepancy"
numPaths = 100000

engine = ql.MCDiscreteArithmeticAPHestonEngine(hestonProcess, rng, requiredSamples=numPaths)


s = 100

import pandas as pd
from itertools import product
def generate_features(K,T,B,s):
    features = pd.DataFrame(
        product(
            [s],
            K,
            T,
            B,
            ['in','out']
            ),
        columns=[
            "spot_price", 
            "strike_price",
            "days_to_maturity",
            "barrier",
            "outin"
                  ])
    return features


import numpy as np


K = np.linspace(s*0.5, s*1.5, 10)
T = np.arange(1/12, 2.01, 3/12)
"""
                                                                   down options
"""

min_barrier = 0.4
down_barriers  = np.linspace(s * min_barrier, s*0.99, 5)

"""
                                                                     up options
"""

max_barrier =  1.6
up_barriers  = np.linspace(s * 1.01, s * max_barrier, 5)

up_features = generate_features(K,T,up_barriers,s)
up_features['updown'] = 'up' 
down_features = generate_features(K,T,down_barriers,s)
down_features['updown'] = 'down' 
features = pd.concat([up_features,down_features]).reset_index(drop=True)

print(f'\n{features}\n')



import numpy as np

from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler





transformers=[
    ('encoding',OneHotEncoder(),['outin','updown'])
    ]     

preprocessor = ColumnTransformer(transformers=transformers)

# =============================================================================
# """
# # devise a data generation technique for barrier option data
# # 
# # fix heston calibration -> calibrate against collected/generated data, not black surface
# # barrier -> barrier type, barrier columns
# # record runtime of calibration+pricing out of sample vs using the prediction method in sklearn
#     # greeks
#     # portfolio
#     # 
# 
# """
# =============================================================================



