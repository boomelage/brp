# -*- coding: utf-8 -*-
"""
Created on Sat Oct  5 23:49:34 2024

"""

import QuantLib as ql
import matplotlib.pyplot as plt



def plot_heston_trajectories(r,g,heston_parameters,length,timesteps,num_paths):
    
    today = ql.Date().todaysDate()
    riskFreeTS = ql.YieldTermStructureHandle(
        ql.FlatForward(today, r, ql.Actual365Fixed()))
    dividendTS = ql.YieldTermStructureHandle(
        ql.FlatForward(today, g, ql.Actual365Fixed()))
    
    v0 = heston_parameters['v0']
    kappa = heston_parameters['kappa']
    theta = heston_parameters['theta']
    rho = heston_parameters['rho']
    eta = heston_parameters['eta']
    s = heston_parameters['spot_price']

    initialValue = ql.QuoteHandle(ql.SimpleQuote(s))
    hestonProcess = ql.HestonProcess(
        riskFreeTS, dividendTS, initialValue, v0, kappa, theta, eta, rho)

    
    dimension = hestonProcess.factors()
    times = ql.TimeGrid(length, timesteps)
    
    rng = ql.UniformRandomSequenceGenerator(
        dimension * timesteps, ql.UniformRandomGenerator())
    sequenceGenerator = ql.GaussianRandomSequenceGenerator(rng)
    pathGenerator = ql.GaussianMultiPathGenerator(
        hestonProcess, list(times), sequenceGenerator, False)
    
    paths = [[] for i in range(dimension)]
    for i in range(num_paths):
        samplePath = pathGenerator.next()
        values = samplePath.value()
        
        plt.plot(values[0])
    
        for j in range(dimension):
            paths[j].append([x for x in values[j]])
            


