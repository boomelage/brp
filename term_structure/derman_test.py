# -*- coding: utf-8 -*-
"""
Created on Mon Sep 16 14:15:10 2024

@author: boomelage
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
settings = ms.import_model_settings()
security_settings = settings[0]['security_settings']
s = security_settings[5]
# =============================================================================
"""
computing Derman coefficients
"""

from import_files import raw_ts

atm_volvec = raw_ts.copy().loc[s].dropna()
T = atm_volvec.index
derman_coefs_np = np.zeros((2,len(T)),dtype=float)
derman_coefs = pd.DataFrame(derman_coefs_np)
derman_coefs['t days'] = ['alpha','b']
derman_coefs = derman_coefs.set_index('t days')
derman_coefs.columns = T
for t in T:
    t = int(t)
    term_struct = raw_ts.copy()
    term_struct = term_struct.loc[:,t].dropna()
    K_reg = term_struct.index
    x = np.array(K_reg  - s, dtype=float)
    y = np.array(term_struct  - atm_volvec[t],dtype=float)
    model = LinearRegression(fit_intercept=False)
    x = x.reshape(-1,1)
    model.fit(x,y)
    b = model.coef_[0]
    alpha = model.intercept_
    derman_coefs.loc['alpha',t] = alpha
    derman_coefs.loc['b',t] = b

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
                derman_coefs.loc['alpha',t] + atm_volvec[t] + \
                derman_coefs.loc['b',t] * moneyness
            )
        derman_ts = derman_ts[~(derman_ts<0)].dropna(how="any",axis=0)
    return derman_ts



"""
testing approximation fit
"""

K_test = raw_ts.index
derman_test_ts = make_derman_surface(K = K_test)

raw_test_ts = raw_ts.copy().loc[derman_test_ts.index,derman_test_ts.columns]
from plot_derman import plot_derman_fit
plot_derman_fit(derman_test_ts, raw_test_ts)


"""
creating vol surface

"""

upper_moneyness = s*1.5
lower_moneyness = s*0.85

n_K = 50
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


"""
plotting vol surface

"""
from plot_surface import plot_volatility_surface
fig = plot_volatility_surface(black_var_surface, K, T)



