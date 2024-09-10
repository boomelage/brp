#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Sep 10 17:55:16 2024

@author: doomd
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
barrier = 110.
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
    
"""
# Geometric Asian Option
rng = "pseudorandom" # could use "lowdiscrepancy"
numPaths = 100000

engine = ql.MCDiscreteArithmeticAPHestonEngine(hestonProcess, rng, requiredSamples=numPaths)

"""

import numpy as np
# down and out

barrier  = 70
spot = np.arange(barrier,barrier*2,1)
strike = np.arange(0.5*barrier,2*barrier,1)

# devise a data generation technique for barrier option data
# 
# fix heston calibration -> calibrate against collected/generated data, not black surface
# barrier -> barrier type, barrier columns
# record runtime of calibration+pricing out of sample vs using the prediction method in sklearn
    # greeks
    # portfolio
    # 





