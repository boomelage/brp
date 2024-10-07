# -*- coding: utf-8 -*-
"""
Created on Wed Oct  2 13:52:21 2024

"""
import QuantLib as ql
from settings import model_settings
ms = model_settings()

"""
# =============================================================================

inputs

"""
calculation_date = ql.Date.todaysDate()

r = 0.05
g = 0.02
s = 100.00
k = 80.00

option_type = ql.Option.Call

v0 = 0.01 
kappa = 0.2
theta = 0.02
rho = -0.75
eta = 0.5

rng = "pseudorandom" # could use "lowdiscrepancy"
numPaths = 100000

periods = [
    ql.Period("6M"), ql.Period("12M"), ql.Period("18M"), ql.Period("24M")
    ]
pastFixings = 0 # Empty because this is a new contract

"""
# =============================================================================

settings

"""
ql.Settings.instance().evaluationDate = calculation_date

s0 = ql.QuoteHandle(ql.SimpleQuote(s))

asianFutureFixingDates = [calculation_date + period for period in periods]

asianExpiryDate = calculation_date + periods[-1]

vanillaPayoff = ql.PlainVanillaPayoff(option_type, k)
europeanExercise = ql.EuropeanExercise(asianExpiryDate)

flat_r, flat_g = ms.ql_ts_rg(
    r,g,calculation_date
    )

hestonProcess = ql.HestonProcess(
    flat_r, flat_g, s0, 
    v0, kappa, theta, eta, rho)

engine = ql.MCDiscreteGeometricAPHestonEngine(
    hestonProcess, rng, requiredSamples=numPaths)

"""
# =============================================================================

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



geo = discreteGeometricAsianOption.NPV()
arith = discreteArithmeticAsianOption.NPV()
print(f"\narithmetic: {arith}\ngeometric: {geo}\n")





# """
# continuous

# continuousGeometricAsianOption = ql.ContinuousAveragingAsianOption(
#     geometricAverage, vanillaPayoff, europeanExercise)
# """



