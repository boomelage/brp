# -*- coding: utf-8 -*-
"""
Created on Fri Oct  4 11:11:51 2024

@author: boomelage
"""
import os
import sys
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
os.chdir(current_dir)
sys.path.append(parent_dir)
import QuantLib as ql
from settings import model_settings
from routine_ivol_collection import raw_puts, raw_calls
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
ms = model_settings()


ts = raw_puts
ts = ts.loc[
    ts.iloc[:,0].dropna().index,
    :
        ]

interp_test = ts.fillna(0)

K = interp_test.index.tolist()
T = interp_test.columns.tolist()


SIG = ql.Matrix(len(K),len(T),0)
for i, k in enumerate(K):
    for j, t in enumerate(T):
        SIG[i][j] = interp_test.loc[k,t]

bicubic_vol = ql.BicubicSpline(T, K, SIG)


T = np.linspace(min(T),max(T),500)
K = np.linspace(min(K),max(K),500)

plt.rcParams['figure.figsize']=(16,7)
plot_maturities = np.sort(np.array(T,dtype=float)/365)
plot_strikes = np.sort(np.array(K,dtype=float))
X, Y = np.meshgrid(plot_strikes, plot_maturities)
Z = np.array([[
    bicubic_vol(y, x, True) for x in plot_strikes] 
    for y in plot_maturities])

azims = np.arange(0,361,15)
for azim in azims:
    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')
    ax.view_init(elev=20, azim=azim)  
    surf = ax.plot_surface(X,Y,Z, rstride=1, cstride=1, cmap=cm.coolwarm,
                    linewidth=0.1)
    fig.colorbar(surf, shrink=0.5, aspect=10)
    
    ax.set_xlabel("Strikes", size=9)
    ax.set_ylabel("Maturities (Years)", size=9)
    ax.set_zlabel("Volatility", size=9)
    plt.tight_layout() 
    plt.show()
    plt.cla()
    plt.clf()
