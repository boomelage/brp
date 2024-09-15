# -*- coding: utf-8 -*-
"""
Created on Sun Sep 15 14:44:49 2024
                                          plotting the Derman coefficients' fit
"""
from settings import model_settings
ms = model_settings()
import matplotlib.pyplot as plt
from routine_Derman import derman_ts, trimmed_ts
from surface_plotting import plot_volatility_surface, plot_term_structure
def plot_derman_fit():
    T = derman_ts.columns
    K = derman_ts.index
    expiration_dates = ms.compute_ql_maturity_dates(T)
    implied_vols_matrix = ms.make_implied_vols_matrix(K, T, derman_ts)
    black_var_surface = ms.make_black_var_surface(
        expiration_dates, K, implied_vols_matrix)
    fig = plot_volatility_surface(black_var_surface, K, T)
    
    for t in T:
        fig = plot_term_structure(K, t, trimmed_ts, derman_ts)
        plt.cla()
        plt.clf()