#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Created on Wed Sep 11 19:08:30 2024

"""  
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
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
from data_query import dirdatacsv
csvs = dirdatacsv()
rawtsname = [file for file in csvs if 'raw_ts' in file][0]
raw_ts = pd.read_csv(rawtsname).drop_duplicates()
raw_ts = raw_ts.rename(
    columns={raw_ts.columns[0]: 'Strike'}).set_index('Strike')
raw_ts.columns = raw_ts.columns.astype(int)
raw_ts = raw_ts.loc[
    lower_strike:upper_strike,
    lower_maturity:upper_maturity]



"""
script start
"""


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


def compute_one_derman_coef(ts_df, s, t, K):
    TSatmat = ts_df.loc[:,t]
    atm_value = np.median(TSatmat)
    strikes = ts_df.index
    x = np.array(strikes - s,dtype=float)
    y = np.array(TSatmat - atm_value,dtype=float)
    model = LinearRegression()
    x = x.reshape(-1,1)
    model.fit(x,y)
    b = model.coef_[0]
    alpha = model.intercept_
    derman_ivols = model.predict(x)
    derman_ivols = derman_ivols*b + alpha + atm_value
    return b, alpha, derman_ivols

def compute_derman_coefs(ts_df, s, T, K):
    derman_coefs = {}
    for i, k in enumerate(K):
        for j, t in enumerate(T):
            atm_value = atm_vols[t]
            b, alpha, derman_ivols = compute_one_derman_coef(ts_df, s, t, K)
            if b < 0:
                b = b
            else:
                b = 0
            derman_coefs[t] = [b, alpha, atm_value]
    derman_coefs = pd.DataFrame(derman_coefs)
    derman_coefs['coef'] = ['b','alpha','atm_value']
    derman_coefs.set_index('coef',inplace = True)
    return derman_coefs

derman_coefs = compute_derman_coefs(trimmed_ts, s, T, K)
derman_maturities = np.sort(derman_coefs.columns)


"""
# =============================================================================
                                                    applying Derman estimations
"""

derman_ts_np = np.zeros((len(K),len(derman_maturities)),dtype=float)
derman_ts = pd.DataFrame(derman_ts_np)
derman_ts.index = K
derman_ts.columns = derman_maturities

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

print('term structure approximated')
