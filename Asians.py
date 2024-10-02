# -*- coding: utf-8 -*-
"""
Created on Wed Oct  2 13:52:21 2024

"""
import QuantLib as ql

today = ql.Date().todaysDate()
riskFreeTS = ql.YieldTermStructureHandle(
    ql.FlatForward(today, 0.05, ql.Actual365Fixed()))
dividendTS = ql.YieldTermStructureHandle(ql.FlatForward(
    today, 0.01, ql.Actual365Fixed()))
initialValue = ql.QuoteHandle(ql.SimpleQuote(100))
k = 80
option_type = ql.Option.Call
v0 = 0.01 
kappa = 0.2
theta = 0.02
rho = -0.75
eta = 0.5
hestonProcess = ql.HestonProcess(
    riskFreeTS, dividendTS, initialValue, v0, kappa, theta, eta, rho)

rng = "pseudorandom" # could use "lowdiscrepancy"
numPaths = 100000


engine = ql.MCDiscreteGeometricAPHestonEngine(
    hestonProcess, rng, requiredSamples=numPaths)

periods = [
    ql.Period("6M"), ql.Period("12M"), ql.Period("18M"), ql.Period("24M")
    ]

pastFixings = 0 # Empty because this is a new contract
asianFutureFixingDates = [today + period for period in periods]
asianExpiryDate = today + periods[-1]


vanillaPayoff = ql.PlainVanillaPayoff(option_type, k)
europeanExercise = ql.EuropeanExercise(asianExpiryDate)

"""
mc arithmetic discrete
"""

arithmeticRunningAccumulator = 0.0

arithmeticAverage = ql.Average().Arithmetic
discreteArithmeticAsianOption = ql.DiscreteAveragingAsianOption(
    arithmeticAverage, arithmeticRunningAccumulator, pastFixings, 
    asianFutureFixingDates, vanillaPayoff, europeanExercise)

discreteArithmeticAsianOption.setPricingEngine(engine)

"""
mc geometric discrete
"""

geometricRunningAccumulator = 1.0

geometricAverage = ql.Average().Geometric
discreteGeometricAsianOption = ql.DiscreteAveragingAsianOption(
    geometricAverage, geometricRunningAccumulator, pastFixings, 
    asianFutureFixingDates, vanillaPayoff, europeanExercise)

discreteGeometricAsianOption.setPricingEngine(engine)

"""
continuous

continuousGeometricAsianOption = ql.ContinuousAveragingAsianOption(
    geometricAverage, vanillaPayoff, europeanExercise)
"""


discreteGeometricAsianOption.NPV()