#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Sep 11 19:08:30 2024

"""
import os
pwd = str(os.path.dirname(os.path.abspath(__file__)))
os.chdir(pwd)
from data_query import dirdata, dirdatacsv
csvs = dirdatacsv()
xlsxs = dirdata()
import pandas as pd
import numpy as np
import time
from datetime import datetime

pd.set_option('display.max_rows',None)
pd.set_option('display.max_columns',None)
# pd.reset_option('display.max_rows')
# pd.reset_option('display.max_columns')


data_files = xlsxs
raw_market_ts = pd.DataFrame()
for file in data_files:
    df = pd.read_excel(file)
    df.columns = df.loc[1]
    df = df.iloc[2:,:].reset_index(drop=True)
    callvols = df.set_index('Strike')
    df_strikes = df.index.tolist()
    df_maturities = df['DyEx'].loc[df_strikes[0]].unique().tolist()
    callvols = pd.concat([df.iloc[:, i:i+2] for i in range(
        0, df.shape[1], 4)], axis=1)
    raw_market_ts = pd.concat(
        [raw_market_ts,callvols])
raw_market_ts = raw_market_ts.set_index('Strike')
raw_market_ts = raw_market_ts.reset_index()

ts_columns = []
ivm_count = 1
dyex_count = 1

for col in raw_market_ts.columns:
    if col == 'IVM':
        ts_columns.append(f'IVM_{ivm_count}')
        ivm_count += 1
    elif col == 'DyEx':
        ts_columns.append(f'DyEx_{dyex_count}')
        dyex_count += 1
    else:
        ts_columns.append(col)

raw_market_ts.columns = ts_columns



# Step 1: Reshape the DataFrame using pd.melt
df_melted = pd.melt(raw_market_ts, 
                    id_vars=['Strike'], 
                    value_vars=['IVM_1', 'IVM_2', 'IVM_3', 'IVM_4'],
                    var_name='IVM_label', 
                    value_name='IVM')

df_melted_dyex = pd.melt(raw_market_ts, 
                         id_vars=['Strike'], 
                         value_vars=['DyEx_1', 'DyEx_2', 'DyEx_3', 'DyEx_4'],
                         var_name='DyEx_label', 
                         value_name='DyEx')
df_combined = pd.concat(
    [df_melted[['Strike', 'IVM']], df_melted_dyex['DyEx']], axis=1)
# Step 2: Drop rows where DyEx or IVM is NaN
df_combined = df_combined.dropna()
# Step 3: Set Strike and DyEx as the MultiIndex
df_indexed = df_combined.set_index(['Strike', 'DyEx'])
df_indexed = df_indexed.sort_index()

Ts = np.sort(df_combined["DyEx"].unique())  
Ts = Ts[Ts > 0]
Ks = np.sort(df_combined["Strike"].unique())
raw_ts_np = np.zeros((len(Ks) , len(Ts)), dtype=float)

for i, k in enumerate(Ks):
    for j, t in enumerate(Ts):
        try:
            raw_ts_np[i][j] = df_indexed.loc[(k, t), 'IVM'].iloc[0]
        except Exception:
            raw_ts_np[i][j] = np.nan
        
raw_ts_df = pd.DataFrame(raw_ts_np)
raw_ts_df.columns = Ts
raw_ts_df = raw_ts_df.set_index(Ks)

raw_ts = raw_ts_df.dropna(how = 'all', axis = 0)
raw_ts = raw_ts.dropna(how = 'all', axis = 1)
atm_vols = raw_ts.dropna()


"""
# =============================================================================
                                           cleaning the term structure manually
"""


strike_spread = raw_ts.iloc[:,0].dropna().index
spot = float(np.median(strike_spread))

spread_ts = raw_ts.loc[strike_spread,:]
spread_ts = spread_ts.fillna(0)

spread_ts = spread_ts.loc[
    :
        ,
    :
        ]

T = np.sort(spread_ts.columns)
K = np.sort(spread_ts.index)
s = np.median(K)


from Derman import derman
derman = derman()

def compute_derman_coefs(T,K,ts_df):
    derman_coefs = {}
    for i, k in enumerate(K):
        for j, t in enumerate(T):
            b, alpha, atmvol, derman_ivols = derman.compute_derman_ivols(t,ts_df)
            derman_coefs[t] = [b, alpha, atmvol]
    derman_coefs = pd.DataFrame(derman_coefs)
    derman_coefs['coef'] = ['b','alpha','atmvol']
    derman_coefs.set_index('coef',inplace = True)
    return derman_coefs

derman_coefs = compute_derman_coefs(T,K,spread_ts)

derman_maturities = np.sort(derman_coefs.columns)

"""
# =============================================================================
                                    Derman reconstruction of volatility surface
"""

derman_ts_np = np.zeros((len(K),len(derman_maturities)),dtype=float)
derman_ts = pd.DataFrame(derman_ts_np)
derman_ts.index = K
derman_ts.columns = derman_maturities



for i, k in enumerate(K):
    moneyness = k - s
    for j, t in enumerate(derman_maturities):
        derman_ts.iloc[i,j] = (
            derman_coefs.loc['alpha',t] + atm_vols[t] + \
                derman_coefs.loc['b',t] * moneyness
                )

from settings import model_settings
ms = model_settings()
derman_vol_matrix = ms.make_implied_vols_matrix(K, derman_maturities, derman_ts)

print(derman_vol_matrix)

expiration_dates = ms.compute_ql_maturity_dates(derman_maturities)

derman_surface = ms.make_black_var_surface(
    expiration_dates, K.astype(float), derman_vol_matrix)

def plot_volatility_surface(black_var_surface = derman_surface,
                            K = K,
                            T = T,
                            ts_df = spread_ts,
                            target_maturity = 3):
    import matplotlib.pyplot as plt
    plt.rcParams['figure.figsize']=(15,7)
    plt.style.use("dark_background")
    from matplotlib import cm
    target_maturity_ivols = ts_df[target_maturity]
    fig, ax = plt.subplots()
    ax.plot(K, target_maturity_ivols, label="Black Surface")
    ax.plot(K, target_maturity_ivols, "o", label="Actual")
    ax.set_xlabel("Strikes", size=9)
    ax.set_ylabel("Vols", size=9)
    ax.legend(loc="upper right")
    fig.show()
    
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


fig = plot_volatility_surface()
