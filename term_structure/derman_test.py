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

raw_calls = ms.raw_calls
raw_puts = ms.raw_puts

call_T = ms.call_T
call_K = ms.call_K

put_T = ms.put_T
put_K = ms.put_K

call_atmvols = ms.call_atmvols
put_atmvols = ms.put_atmvols

raw_vols = ms.raw_vols

s = ms.s

# =============================================================================
"""
computing Derman coefficients
"""


def compute_derman_coefficients(s,T,K,atm_volvec,raw_ts,flag):
    
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
    return derman_coefs


call_dermans = compute_derman_coefficients(s,call_T,call_K,call_atmvols,raw_calls,'call')
put_dermans = compute_derman_coefficients(s,put_T,put_K,put_atmvols,raw_puts,'put')


print(f'\n\n\ncall coefs:\n{call_dermans}')
print(f'\n\n\nput coefs:\n{put_dermans}')


from plot_surface import plot_rotate, plot_term_structure


"""
surface maker

"""

def make_derman_surface(atm_volvec, derman_coefs, K, s = ms.s):
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


derman_callvols = make_derman_surface(call_atmvols, call_dermans, call_K)
derman_putvols = make_derman_surface(put_atmvols, put_dermans, put_K)

otm_derman_vols = pd.concat([derman_putvols, derman_callvols])

"""
testing approximation fit
"""


"""
creating vol surface

"""

def plot_derman_rotate():
    upper_moneyness = s*1.2
    lower_moneyness = s*0.8
    
    n_K = 20
    K = np.linspace(int(lower_moneyness),int(upper_moneyness),int(n_K)).astype(int)
    
    derman_ts = otm_derman_vols
    
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


