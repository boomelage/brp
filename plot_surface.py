#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Sep 14 17:26:58 2024

"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
def plot_volatility_surface(black_var_surface, K, T, elev=30, azim=120):
    plt.rcParams['figure.figsize']=(15,7)
    K = K.astype(int)
    plot_maturities = np.sort(T/365).astype(float)
    plot_strikes = np.sort(K).astype(float)
    X, Y = np.meshgrid(plot_strikes, plot_maturities)
    Z = np.array([[
        black_var_surface.blackVol(y, x) for x in plot_strikes] 
        for y in plot_maturities])
    
    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')
    ax.view_init(elev=elev, azim=azim)  
    surf = ax.plot_surface(X,Y,Z, rstride=1, cstride=1, cmap=cm.coolwarm,
                    linewidth=0.1)
    fig.colorbar(surf, shrink=0.5, aspect=5)
    
    ax.set_xlabel("Strikes", size=9)
    ax.set_ylabel("Maturities (Years)", size=9)
    ax.set_zlabel("Volatility", size=9)
    
    plt.show()
    plt.cla()
    plt.clf()
    return fig

def plot_term_structure(
        K,
        real_ts,
        est_ts,
        title
        ):
    plt.rcParams['figure.figsize']=(6,4)
    K = K.astype(int)
    fig, ax = plt.subplots()
    ax.plot(K, est_ts, label="Derman")
    ax.plot(K, real_ts, "o", label="Actual")
    ax.legend = ax.legend(loc="upper right")
    ax.set_title(title)
    plt.show()
    plt.cla()
    plt.clf()
    return fig