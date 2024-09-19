# -*- coding: utf-8 -*-
"""
Created on Mon Sep 16 14:15:10 2024

"""
# =============================================================================
import os
import sys
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
os.chdir(current_dir)
sys.path.append(parent_dir)
# =============================================================================
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
# =============================================================================
from settings import model_settings
ms = model_settings()
raw_T = ms.raw_T 
raw_K = ms.raw_K
atm_volvec = ms.atm_volvec
s = ms.s
raw_ts = ms.model_vol_ts
# =============================================================================
"""
computing Derman coefficients
"""


from plot_surface import plot_rotate, plot_term_structure
T = raw_T
K = raw_K

derman_coefs_np = np.zeros((2,len(T)),dtype=float)
derman_coefs = pd.DataFrame(derman_coefs_np)
derman_coefs['t days'] = ['b','atm_vol']
derman_coefs = derman_coefs.set_index('t days')
derman_coefs.columns = T


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


print(f'\n{derman_coefs}')

"""
surface maker

"""

def make_derman_surface(
        K=None, atm_volvec=atm_volvec, derman_coefs=derman_coefs, s = s):
    T = derman_coefs.columns
    derman_ts_np = np.zeros((len(K),len(T)),dtype=float)
    derman_ts = pd.DataFrame(derman_ts_np)
    derman_ts.index = K
    derman_ts.columns = T
    
    for i, k in enumerate(K):
        moneyness = k-s
        for j, t in enumerate(T):
            derman_ts.loc[k,t] = (
                derman_coefs.loc['atm_vol',t] + \
                derman_coefs.loc['b',t] * moneyness
            )
        derman_ts = derman_ts[~(derman_ts<0)].dropna(how="any",axis=0)
    return derman_ts



"""
testing approximation fit
"""
def plot_derman_test():
    K_test = raw_ts.index
    derman_test_ts = make_derman_surface(K = K_test)
    
    raw_test_ts = raw_ts.copy().loc[derman_test_ts.index,derman_test_ts.columns]
    fig = plot_term_structure(K_test, raw_test_ts,derman_test_ts,title="Derman approximation of volatility versus market obs")
    return fig

"""
creating vol surface

"""

def plot_derman_rotate():
    upper_moneyness = s*1.2
    lower_moneyness = s*0.8
    
    n_K = 20
    K = np.linspace(int(lower_moneyness),int(upper_moneyness),int(n_K)).astype(int)
    
    derman_ts = make_derman_surface(K=K)
    
    T = derman_ts.columns.astype(float)
    K = derman_ts.index
    T = T[(
            ( T > 0 )
            &
            ( T < 37000 )
    )]
    
    expiration_dates = ms.compute_ql_maturity_dates(T)
    
    implied_vols_matrix = ms.make_implied_vols_matrix(K, T, derman_ts)
    black_var_surface = ms.make_black_var_surface(
        expiration_dates, K, implied_vols_matrix)
    
    fig = plot_rotate(
        black_var_surface,K,T,'Derman approximation of volatility surface')
    return fig


