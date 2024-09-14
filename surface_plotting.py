#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Sep 14 17:26:58 2024

@author: doomd
"""

import numpy as np
import matplotlib.pyplot as plt
plt.rcParams['figure.figsize']=(15,7)
plt.style.use("dark_background")
from matplotlib import cm

def plot_volatility_surface(black_var_surface, K, T):
    plot_maturities = np.sort(T/365).astype(float)
    plot_strikes = np.sort(K).astype(float)
    X, Y = np.meshgrid(plot_strikes, plot_maturities)
    Z = np.array([[
        black_var_surface.blackVol(y, x) for x in plot_strikes] 
        for y in plot_maturities])
    
    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')
    
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
        target_maturity,
        real_ts,
        est_ts
        ):
    fig, ax = plt.subplots()
    real_target_ivols = real_ts[target_maturity]
    est_taget_ivols = est_ts[target_maturity]
    ax.plot(K, est_taget_ivols, label="Derman")
    ax.plot(K, real_target_ivols, "o", label="Actual")
    legend = ax.legend(loc="upper right")
    plt.show()
    plt.cla()
    plt.clf()