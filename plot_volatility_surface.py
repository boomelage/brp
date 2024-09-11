#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Sep 11 13:29:48 2024

"""
# =============================================================================
                                                  # plotting volatility surface

import matplotlib.pyplot as plt
plt.rcParams['figure.figsize']=(15,7)
plt.style.use("dark_background")
from matplotlib import cm
import re
import pandas as pd
import numpy as np
import os

expiry = 1
def plot_volatility_surface(
        outputs_path, ticker, ivoldf,strikes,maturities,black_var_surface):
    target_maturity_ivols = ivoldf[expiry]
    fig, ax = plt.subplots()
    ax.plot(strikes, target_maturity_ivols, label="Black Surface")
    ax.plot(strikes, target_maturity_ivols, "o", label="Actual")
    ax.set_xlabel("Strikes", size=9)
    ax.set_ylabel("Vols", size=9)
    ax.legend(loc="upper right")
    fig.show()
    
    plot_maturities = pd.Series(maturities) / 365.25
    plot_strikes = pd.Series(strikes)
    X, Y = np.meshgrid(plot_maturities, plot_strikes)
    Z = np.array([[
        black_var_surface.blackVol(y, x) for x in plot_maturities] 
        for y in plot_strikes])
    
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
    # timestamp = re.search(r'[^ ]+$', outputs_path).group(0)
    # plot_path = os.path.join(outputs_path, f"{ticker} ts {timestamp}.png")
    # plt.savefig(plot_path,dpi=600)
    
