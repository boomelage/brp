#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Sep  8 03:09:26 2024
"""
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm

def plot_vol_surface(dataset, ivol_table, implied_vols_matrix, 
                     black_var_surface, strikes, maturities):
    plt.rcParams['figure.figsize']=(6,4)
    plt.style.use("dark_background")
    strikes_grid = strikes
    expiry = 0.3643 # years
    implied_vols = [black_var_surface.blackVol(expiry, s)
                    for s in strikes_grid] # can interpolate here
    actual_data = ivol_table[4]
    fig, ax = plt.subplots()
    ax.plot(strikes_grid, implied_vols, label="Black Surface")
    ax.plot(strikes, actual_data, "o", label="Actual")
    ax.set_xlabel("Strikes", size=9)
    ax.set_ylabel("Vols", size=9)
    legend = ax.legend(loc="upper right")
    fig.show()
    
    plottmin = (min(maturities)+0.05)/365
    plottmax = (max(maturities)-0.05)/365
    plot_maturities = np.linspace(plottmin, plottmax, len(maturities))
    
    X, Y = np.meshgrid(strikes, plot_maturities)
    
    Z = np.array([[black_var_surface.blackVol(y, x) for x in strikes] for y in plot_maturities])
    
    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')
    
    surf = ax.plot_surface(X, Y, Z, rstride=1, cstride=1, cmap=cm.coolwarm, linewidth=0.1)
    fig.colorbar(surf, shrink=0.35, aspect=6)
    
    ax.set_xlabel("Strikes", size=9)
    ax.set_ylabel("Maturities (Years)", size=9)
    ax.set_zlabel("Volatility", size=9)
    
    plt.show()
    plt.cla()
    plt.clf()
    plt.style.use('default')


