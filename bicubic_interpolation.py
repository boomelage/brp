#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Sep 19 14:17:18 2024

@author: doomd
"""

import os
import sys
os.chdir(os.path.dirname(os.path.abspath(__file__)))
sys.path.append('term_structure')
sys.path.append('contract_details')
sys.path.append('misc')

import numpy as np
from sklearn.linear_model import LinearRegression
from settings import model_settings
import pandas as pd


ms = model_settings()
from derman_test import atm_volvec
from routine_calibration_generation import T, call_ks, put_ks


K = np.array((call_ks, put_ks), dtype=float)
K = np.unique(K.flatten())

atm_volvec = atm_volvec.loc[T]


s = ms.s


from routine_ivol_collection import raw_ts

derman_coefs_np = np.zeros((2,len(T)),dtype=float)
derman_coefs = pd.DataFrame(derman_coefs_np)
derman_coefs.columns = T
derman_coefs.index = ['b','atm_vol']

for t in T:
    try:
        t = int(t)
        term_struct = raw_ts.loc[:,t].dropna()
        
        K_reg = term_struct.index
        x = np.array(K_reg  - s, dtype=float)
        y = np.array(term_struct - atm_volvec[t],dtype=float)
    
        model = LinearRegression(fit_intercept=False)
        x = x.reshape(-1,1)
        model.fit(x,y)
        b = model.coef_[0]

        derman_coefs.loc['b',t] = b
        derman_coefs.loc['atm_vol',t] = atm_volvec[t]
    except Exception:
        print(f'error: t = {t}')
    
    

vols_vector = [
       [ 
       derman_coefs.loc['atm_vol',t] + \
           derman_coefs.loc['b',t] * (ms.s-k) \
               for t in T
        ] for k in K
       ]
df = pd.DataFrame(vols_vector, columns=T, index=K)


import QuantLib as ql

ql_T = ql.Array(list(T))
ql_K = ql.Array(list(K))
ql_vols = ql.Matrix(len(K),len(T),0.00)

for i, k in enumerate(ql_K):
    for j, t in enumerate(ql_T):
        ql_vols[i][j] = df.loc[k,t]
     


i = ql.BicubicSpline(ql_T, ql_K, ql_vols)

TT = T
KK = np.linspace(s*0.6,s*1.5,len(TT))

bilinear_ts = pd.DataFrame(
    [[i(t,k, True) for t in TT] for k in KK],
    columns=TT,
    index=KK)

ql_bilinear = ql.Matrix(len(bilinear_ts.index),len(bilinear_ts.columns),0.00)


for i, k in enumerate(KK):
    for j, t in enumerate(TT):
        ql_bilinear[i][j] = bilinear_ts.loc[k,t]


from plot_surface import plot_volatility_surface

expiration_dates = ms.compute_ql_maturity_dates(TT)
black_var_surface = ms.make_black_var_surface(expiration_dates, KK, ql_bilinear)

azims = np.arange(0,720,1)
import time
for a in azims:
    # time.sleep()
    a
    fig = plot_volatility_surface(black_var_surface, KK, TT, elev=30, azim=a)