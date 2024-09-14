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

# pd.set_option('display.max_rows',None)
# pd.set_option('display.max_columns',None)
pd.reset_option('display.max_rows')
pd.reset_option('display.max_columns')

"""
# =============================================================================
                                                                 importing data
"""

# from routine_generation import rfrpivot, dvypivot
# raw_ts = dvypivot
# raw_ts = rfrpivot

from import_files import imported_ts, derman_coefs, derman_ts, spread_ts, raw_ts
raw_ts = spread_ts

raw_ts = raw_ts.dropna(how = 'all')
raw_ts = raw_ts.dropna(how = 'all', axis = 1)
raw_ts = raw_ts.drop_duplicates()
atm_vols = raw_ts.dropna()


"""
# =============================================================================
                                           cleaning the term structure manually
"""
spot_spread = np.array(raw_ts.loc[:,34].dropna().index)
s = int(np.median(spot_spread))
raw_ts = raw_ts.dropna(axis=1, subset=[s])

strike_spread = spot_spread
spread_ts = raw_ts.loc[min(strike_spread):max(strike_spread),:]
spread_ts = spread_ts.fillna(0.000000)
T = np.sort(spread_ts.columns)
K = np.sort(spread_ts.index)

from Derman import derman
derman = derman()

def compute_derman_coefs(T,K,ts_df):
    derman_coefs = {}
    for i, k in enumerate(K):
        for j, t in enumerate(T):
            b, alpha, atm_value, derman_ivols = derman.compute_derman_ivols(t,ts_df)
            derman_coefs[t] = [b, alpha, atm_value]
    derman_coefs = pd.DataFrame(derman_coefs)
    derman_coefs['coef'] = ['b','alpha','atm_value']
    derman_coefs.set_index('coef',inplace = True)
    return derman_coefs


derman_coefs = compute_derman_coefs(T,K,spread_ts)
derman_maturities = np.sort(derman_coefs.columns)

derman_ts_np = np.zeros((len(K),len(derman_maturities)),dtype=float)
derman_ts = pd.DataFrame(derman_ts_np)
derman_ts.index = K
derman_ts.columns = derman_maturities

derman_ts

"""
# =============================================================================
                                                    applying Derman estimations
# """
for i, k in enumerate(K):
    moneyness = k - s
    for j, t in enumerate(derman_maturities):
        k = int(k)
        t = int(t)
        derman_ts.loc[k,t] = (
            derman_coefs.loc['alpha',t] + derman_coefs.loc['atm_value',t] + \
            derman_coefs.loc['b',t] * moneyness
        )



from settings import model_settings
ms = model_settings()
implied_vols_matrix = ms.make_implied_vols_matrix(
    K, derman_maturities, derman_ts)

# print(implied_vols_matrix)

expiration_dates = ms.compute_ql_maturity_dates(derman_maturities)

black_var_surface = ms.make_black_var_surface(
    expiration_dates, K.astype(float), implied_vols_matrix)



def plot_volatility_surface(black_var_surface = black_var_surface,
                            K = K,
                            T = T,
                            ts_df = spread_ts,
                            target_maturity = derman_maturities[0]):
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

for t in derman_maturities:
    print(t)