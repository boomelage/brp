#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Oct  8 12:02:31 2024

@author: doomd
"""
import os
import sys
import pandas as pd
import numpy as np
import QuantLib as ql
from scipy import interpolate
import matplotlib.pyplot as plt
from matplotlib import cm

current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)

os.chdir(current_dir)
from historical_alphaVantage_collection import chain, start_date


for key in chain.keys():
    print(key)
    
ivol_df = chain['2020-01-02']['puts']['surface']

ivol_df = ivol_df.dropna(how='all',axis=0).dropna(how='all',axis=1)

strikes = ivol_df.iloc[:,0].dropna().index
ivol_df = ivol_df.loc[strikes,:].copy()
T = ivol_df.columns.tolist()
K = ivol_df.index.tolist()
ivol_array = ivol_df.to_numpy()
x = np.arange(0, ivol_array.shape[1])
y = np.arange(0, ivol_array.shape[0])
#mask invalid values
array = np.ma.masked_invalid(ivol_array)
xx, yy = np.meshgrid(x, y)

x1 = xx[~array.mask]
y1 = yy[~array.mask]
newarr = array[~array.mask]

GD1 = interpolate.griddata((x1, y1), newarr.ravel(),
                          (xx, yy),
                            method='cubic')


vol_surf = pd.DataFrame(
    GD1,
    index = K,
    columns = T
).copy()

# vol_surf = ivol_df.loc[:,ivol_df.columns>0].dropna(how='any', axis=1).copy()
# K = vol_surf.index.tolist()
# T = vol_surf.columns.tolist()

# vol_matrix = ql.Matrix(len(K),len(T),0.0)
# for i,k in enumerate(K):
#     for j,t in enumerate(T):
#         vol_matrix[i][j] = float(vol_surf.loc[k,t])

# bicubic_vol = ql.BicubicSpline(T,K,vol_matrix)


# K = np.linspace(
#     min(K),
#     max(K),
#     50
# )
# T = np.linspace(
#     1,
#     10,
#     50
# )

# KK,TT = np.meshgrid(K,T)

# V = np.array(
#     [[bicubic_vol(float(t),float(k),False) for k in K] for t in T]
#     )

# plt.rcParams['figure.figsize']=(7,5)

# fig = plt.figure()
# ax = fig.add_subplot(projection='3d')
# ax.view_init(elev=20, azim=120)  
# surf = ax.plot_surface(KK,TT,V, rstride=1, cstride=1, cmap=cm.coolwarm,
#                 linewidth=0.1)
# fig.colorbar(surf, shrink=0.3, aspect=5)

# ax.set_xlabel("Strike", size=9)
# ax.set_ylabel("Maturity", size=9)
# ax.set_zlabel("Volatility", size=9)

# plt.tight_layout()
# plt.show()
# plt.cla()
# plt.clf()  
  
  
  
  
  
  
  
  
  
  
  
  
  
        