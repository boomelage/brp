#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Sep 12 20:52:50 2024

"""

import pandas as pd
from data_query import dirdatacsv
import numpy as np
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt

data_files = dirdatacsv()

file = data_files[0]
ts = pd.read_csv(file)

ts = ts.set_index(ts.iloc[:,0]).drop(columns = ts.columns[0])
ts = ts.astype(float)
ts.columns = ts.columns.astype(int)
ts.index = ts.index.astype(int)


ts = ts.loc[
    
    5540:5640,
    7:15
    
    ]
ts = ts.loc[:, (ts != 0).any(axis=0)]

ks = ts.index.tolist()
mats = ts.columns.tolist()



def compute_derman_ivol(maturity):
    TSatmat = ts.loc[:,maturity]
    
    strikes = ts.index
    
    S = int(np.median(strikes))
    
    S
    
    x = np.array(strikes - S,dtype=float)
    
    atmvol = np.median(TSatmat)
    
    y = np.array(TSatmat - atmvol,dtype=float)
    
    model = LinearRegression()\
    
    x = x.reshape(-1,1)
    model.fit(x,y)
    
    b = model.coef_[0]
    alpha = model.intercept_
    
    derman_ivol = model.predict(x)/100
    
    derman_ivol = derman_ivol*b+alpha+atmvol
    
    return derman_ivol


derman_surface = np.empty(len(mats),dtype=object)
for i, maturity in enumerate(mats):
    # print(maturity)
    derman_ivol = compute_derman_ivol(maturity)
    derman_surface[i] = derman_ivol
    
dermandf = pd.DataFrame(index=ks, columns=mats)


for i, maturity in enumerate(mats):
    for j, k in enumerate(ks):
        dermandf.loc[k,maturity] = derman_surface[i][j]
        

Ks = ks
Ts = mats
implied_vols = dermandf

import QuantLib as ql
implied_vols_matrix = ql.Matrix(len(Ks),len(Ts),0)
for i, strike in enumerate(Ks):
    for j, maturity in enumerate(Ts):
        implied_vols_matrix[i][j] = implied_vols.loc[strike,maturity]

print(implied_vols_matrix)
        
        
from settings import model_settings
ms = model_settings()
settings, ezprint = ms.import_model_settings()
dividend_rate = settings['dividend_rate']
risk_free_rate = settings['risk_free_rate']
calculation_date = settings['calculation_date']
day_count = settings['day_count']
calendar = settings['calendar']
flat_ts = settings['flat_ts']
dividend_ts = settings['dividend_ts']

expiration_dates = ms.compute_ql_maturity_dates(Ts)
S = np.median(Ks)
black_var_surface = ql.BlackVarianceSurface(
    calculation_date, calendar,
    expiration_dates, Ks,
    implied_vols_matrix, day_count)

import matplotlib.pyplot as plt
plt.rcParams['figure.figsize']=(15,7)
plt.style.use("dark_background")
from matplotlib import cm
import numpy as np
import os

target_maturity_ivols = implied_vols.loc[:,]
fig, ax = plt.subplots()
ax.plot(Ks, target_maturity_ivols, label="Black Surface")
ax.plot(Ks, target_maturity_ivols, "o", label="Actual")
ax.set_xlabel("Strikes", size=9)
ax.set_ylabel("Vols", size=9)
ax.legend(loc="upper right")
fig.show()

plot_maturities = np.array(Ts,dtype=float)/365.25
moneyness = np.array(Ks,dtype=float)
X, Y = np.meshgrid(plot_maturities, moneyness)
Z = np.array(
    [
    black_var_surface.blackVol(x, y) for x, y in zip(X.flatten(), Y.flatten())
      ])
Z = Z.reshape(X.shape)
fig = plt.figure()
ax = fig.add_subplot(projection='3d')
surf = ax.plot_surface(
    X, Y, Z, rstride=1, cstride=1, cmap=cm.coolwarm, linewidth=0.1)
fig.colorbar(surf, shrink=0.5, aspect=5)
ax.set_xlabel("Maturities", size=9)
ax.set_ylabel("Strikes", size=9)
ax.set_zlabel("Implied Volatility", size=9)
ax.view_init(elev=30, azim=-35)
plt.show() 
plt.cla()
plt.clf()
      
