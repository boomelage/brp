# -*- coding: utf-8 -*-
"""
Created on Sat Oct  5 23:49:34 2024

"""

import QuantLib as ql
import matplotlib.pyplot as plt

today = ql.Date().todaysDate()
riskFreeTS = ql.YieldTermStructureHandle(ql.FlatForward(today, 0.05, ql.Actual365Fixed()))
dividendTS = ql.YieldTermStructureHandle(ql.FlatForward(today, 0.01, ql.Actual365Fixed()))
initialValue = ql.QuoteHandle(ql.SimpleQuote(100))

v0, kappa, theta, rho, sigma = 0.003, 1.2, 0.3, -1, 0.2
hestonProcess = ql.HestonProcess(riskFreeTS, dividendTS, initialValue, v0, kappa, theta, sigma, rho)

timestep = 100
length = 1
numPaths = 100
dimension = hestonProcess.factors()
times = ql.TimeGrid(length, timestep)

rng = ql.UniformRandomSequenceGenerator(dimension * timestep, ql.UniformRandomGenerator())
sequenceGenerator = ql.GaussianRandomSequenceGenerator(rng)
pathGenerator = ql.GaussianMultiPathGenerator(hestonProcess, list(times), sequenceGenerator, False)

paths = [[] for i in range(dimension)]
for i in range(numPaths):
    samplePath = pathGenerator.next()
    values = samplePath.value()
    
    plt.plot(values[0])

    for j in range(dimension):
        paths[j].append([x for x in values[j]])
        
