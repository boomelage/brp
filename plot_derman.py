#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Sep 15 18:17:14 2024

"""

def clear_all():
    globals_ = globals().copy()
    for name in globals_:
        if not name.startswith('_') and name not in ['clear_all']:
            del globals()[name]
clear_all()

import os
pwd = os.path.dirname(os.path.abspath(__file__))
os.chdir(pwd)



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

from routine_Derman import derman_ts, trimmed_ts
from plot_surface import plot_volatility_surface, plot_term_structure
def plot_derman_fit():
    T = derman_ts.columns
    for t in T:
        plot_ts = trimmed_ts.loc[:,t].dropna()
        plot_derman = derman_ts.loc[:,t]
        plot_derman = plot_derman[plot_ts.index]
        K = plot_derman.index
        title = f"t = {t} days"
        plot_term_structure(K, plot_ts, plot_derman,title)
    T_sur = T[(T>=lower_maturity)&
              (T<=upper_maturity)]
    K = derman_ts.index
    K_sur = K[(K>=lower_moneyness)&
              (K<=upper_moneyness)]
    
    expiration_dates = ms.compute_ql_maturity_dates(T_sur)
    implied_vols_matrix = ms.make_implied_vols_matrix(K_sur, T_sur, derman_ts)
    black_var_surface = ms.make_black_var_surface(
        expiration_dates, K_sur, implied_vols_matrix)
    plot_volatility_surface(black_var_surface, K_sur, T_sur)
    
    
plot_derman_fit()