#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# """
# Created on Wed Sep 11 19:08:30 2024

# """  

import os
pwd = os.path.dirname(os.path.abspath(__file__))
os.chdir(pwd)
import time
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from data_query import dirdata, dirdatacsv
from Derman import derman
derman = derman()
from surface_plotting import plot_volatility_surface, plot_term_structure
csvs = dirdatacsv()
xlsxs = dirdata()


from settings import model_settings
ms = model_settings()
settings = ms.import_model_settings()
dividend_rate = settings['dividend_rate']
risk_free_rate = settings['risk_free_rate']
calculation_date = settings['calculation_date']
day_count = settings['day_count']
calendar = settings['calendar']
flat_ts = settings['flat_ts']
dividend_ts = settings['dividend_ts']
security_settings = settings['security_settings']
ticker = security_settings[0]
lower_strike = None
upper_strike = None
lower_maturity = None
upper_maturity = None
s = security_settings[5]

# pd.set_option('display.max_rows',None)
# pd.set_option('display.max_columns',None)
pd.reset_option('display.max_rows')
pd.reset_option('display.max_columns')



from import_files import raw_ts
# trimmed_ts = raw_ts.dropna(axis=1, subset=[s])


trimmed_ts = raw_ts.dropna(how = 'all')
trimmed_ts = trimmed_ts.dropna(how = 'all', axis = 1)
trimmed_ts = trimmed_ts.drop_duplicates()


trimmed_ts = trimmed_ts.loc[
    lower_strike:upper_strike,
    lower_maturity:upper_maturity
    ]


    

trimmed_ts = trimmed_ts.fillna(0.000000)

atm_vols = trimmed_ts.loc[s]
T = np.sort(trimmed_ts.columns)
K = np.sort(trimmed_ts.index)


def compute_derman_coefs(T,K,ts_df):
    derman_coefs = {}
    for i, k in enumerate(K):
        for j, t in enumerate(T):
            atm_value = atm_vols[t]
            b, alpha, derman_ivols = derman.compute_derman_ivols(s, t, trimmed_ts, atm_value)
            if b < 0:
                b = b
            else:
                b = 0
            derman_coefs[t] = [b, alpha, atm_value]
    derman_coefs = pd.DataFrame(derman_coefs)
    derman_coefs['coef'] = ['b','alpha','atm_value']
    derman_coefs.set_index('coef',inplace = True)
    return derman_coefs

derman_coefs = compute_derman_coefs(T,K,trimmed_ts)
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
T = derman_ts.columns
K = derman_ts.index


expiration_dates = ms.compute_ql_maturity_dates(T)
implied_vols_matrix = ms.make_implied_vols_matrix(K, T, derman_ts)
black_var_surface = ms.make_black_var_surface(expiration_dates, K, implied_vols_matrix)
fig = plot_volatility_surface(black_var_surface, K, T)
for t in T:
    time.sleep(0.05)
    fig = plot_term_structure(K, t, trimmed_ts, derman_ts)
    plt.cla()
    plt.clf()

