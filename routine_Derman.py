#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Created on Wed Sep 11 19:08:30 2024

"""
import os
pwd = os.path.dirname(os.path.abspath(__file__))
os.chdir(pwd)
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression

from settings import model_settings
ms = model_settings()
settings = ms.import_model_settings()

day_count = settings[0]['day_count']
calendar = settings[0]['calendar']
calculation_date = settings[0]['calculation_date']
security_settings = settings[0]['security_settings']
s = security_settings[5]
ticker = security_settings[0]
lower_moneyness = security_settings[1]
upper_moneyness = security_settings[2]
lower_maturity = security_settings[3]
upper_maturity = security_settings[4]

from import_ts import raw_ts

"""
script start
"""

trimmed_ts = raw_ts.dropna(how = 'all')
trimmed_ts = trimmed_ts.dropna(how = 'all', axis = 1)
trimmed_ts = trimmed_ts.drop_duplicates()
trimmed_ts = trimmed_ts[
    (trimmed_ts.index > lower_moneyness) &
    (trimmed_ts.index < upper_moneyness)
]

trimmed_ts = trimmed_ts.loc[:,lower_maturity:upper_maturity]


atm_vols = trimmed_ts.loc[s]
atm_vols = atm_vols.dropna()
T = np.sort(atm_vols.index)
K = np.sort(trimmed_ts.index)

def compute_one_derman_coef(ts_df, s, t, atm_value):
    
    term_struct = ts_df.loc[:,t].dropna()
    K_reg = term_struct.index
    
    x = np.array(K_reg  - s,dtype=float)
    y = np.array(term_struct  - atm_value,dtype=float)
    
    model = LinearRegression()
    x = x.reshape(-1,1)
    model.fit(x,y)
        
    b = model.coef_[0]
    alpha = model.intercept_

    return b, alpha

def compute_derman_coefs(raw_ts, s, T, K, atm_vols):
    derman_coefs = {}
    for i, k in enumerate(K):
        for j, t in enumerate(T):
            atm_value = atm_vols[t]
            b, alpha = compute_one_derman_coef(raw_ts, s, t, atm_value)
            derman_coefs[t] = [b, alpha, atm_value]
    derman_coefs = pd.DataFrame(derman_coefs)
    derman_coefs['coef'] = ['b','alpha','atm_value']
    derman_coefs.set_index('coef',inplace = True)
    return derman_coefs

derman_coefs = compute_derman_coefs(raw_ts, s, T, K, atm_vols)


derman_ts_np = np.zeros((len(K),len(T)),dtype=float)
derman_ts = pd.DataFrame(derman_ts_np)
derman_ts.index = K
derman_ts.columns = T

for i, k in enumerate(K):
    moneyness = k - s
    for j, t in enumerate(T):
        k = int(k)
        t = int(t)
        derman_ts.loc[k,t] = (
            derman_coefs.loc['alpha',t] + derman_coefs.loc['atm_value',t] + \
            derman_coefs.loc['b',t] * moneyness
        )


print('\nterm structure approximated')
