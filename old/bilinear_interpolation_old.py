#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Sep 19 14:17:18 2024

"""

import os
import sys
os.chdir(os.path.dirname(os.path.abspath(__file__)))
sys.path.append('term_structure')
sys.path.append('contract_details')
sys.path.append('misc')

import numpy as np
import pandas as pd
import QuantLib as ql

from settings import model_settings
ms = model_settings()
atm_volvec = ms.call_atmvols
T = ms.T
s = ms.s

from derman_test import derman_test_ts, real_test_ts


ts_df = derman_test_ts
K = ts_df.index
ql_T = ql.Array(list(T))
ql_K = ql.Array(K.tolist())
ql_vols = ql.Matrix(len(K),len(T),0.00)

for i, k in enumerate(ql_K):
    for j, t in enumerate(ql_T):
        ql_vols[i][j] = ts_df.loc[k,t]

bilin_vol = ql.BilinearInterpolation(ql_T, ql_K, ql_vols)

def bilinear_vol(t,k):
    vol = bilin_vol(t,k, True) 
    return vol

def bilinear_vol_row(row):
    row['volatility'] = bilinear_vol(row['days_to_maturity'], row['strike_price'])
    return row


TT = T
KK = K
KK = np.linspace(min(K),max(K),1000)

bilinear_ts = pd.DataFrame(
    [[bilin_vol(t,k, True) for t in TT] for k in KK],
    columns=TT,
    index=KK)

ql_bilinear = ql.Matrix(len(bilinear_ts.index),len(bilinear_ts.columns),0.00)


for i, k in enumerate(KK):
    for j, t in enumerate(TT):
        ql_bilinear[i][j] = bilinear_ts.loc[k,t]

from plot_surface import plot_rotate

expiration_dates = ms.compute_ql_maturity_dates(TT)
black_var_surface = ms.make_black_var_surface(expiration_dates, KK, ql_bilinear)

def plot_bilinear_rotate():
    fig = plot_rotate(black_var_surface, KK, TT, title='Bilinear interpolation of volatility surface approimated via Derman')
    return fig