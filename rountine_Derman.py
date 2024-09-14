#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# """
# Created on Wed Sep 11 19:08:30 2024

# """  

import os
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
os.chdir(parent_dir)

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


# from routine_ivol_collection import raw_ts

from import_files import raw_ts

raw_ts = raw_ts.dropna(how = 'all')
raw_ts = raw_ts.dropna(how = 'all', axis = 1)
raw_ts = raw_ts.drop_duplicates()
atm_vols = raw_ts.dropna()


"""
# =============================================================================
                                            cleaning the term structure manually
"""
spot_spread = np.array(raw_ts.loc[:,3].dropna().index)
s = 5630
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


"""
# =============================================================================
                                                    applying Derman estimations
"""
for i, k in enumerate(K):
    moneyness = k - s
    for j, t in enumerate(derman_maturities):
        k = int(k)
        t = int(t)
        derman_ts.loc[k,t] = (
            derman_coefs.loc['alpha',t] + derman_coefs.loc['atm_value',t] + \
            derman_coefs.loc['b',t] * moneyness
        )

negative_dermans = derman_ts.copy().loc[:, (derman_ts < 0).any(axis=0)]
negative_dermans
derman_ts = derman_ts.drop(columns=negative_dermans.columns)
derman_maturities = derman_ts.columns
K = derman_ts.index


from settings import model_settings
ms = model_settings()
implied_vols_matrix = ms.make_implied_vols_matrix(
    K, derman_maturities, derman_ts)

print(implied_vols_matrix)

expiration_dates = ms.compute_ql_maturity_dates(derman_maturities)

black_var_surface = ms.make_black_var_surface(
    expiration_dates, K.astype(float), implied_vols_matrix)

from surface_plotting import plot_volatility_surface, plot_term_structure

# fig = plot_volatility_surface(
#     black_var_surface, K, derman_maturities)

for t in derman_maturities:
    fig = plot_term_structure(K,t,spread_ts,derman_ts)

# for t in derman_maturities:
#     print(t)