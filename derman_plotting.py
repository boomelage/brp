# -*- coding: utf-8 -*-
"""
Created on Sun Sep 15 14:44:49 2024
                                          plotting the Derman coefficients' fit
"""
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
    T = T[T>=8]
    for t in T:
        plot_ts = trimmed_ts.loc[:,t].dropna()
        plot_derman = derman_ts.loc[:,t]
        plot_derman = plot_derman[plot_ts.index]
        K = plot_derman.index
        fig = plot_term_structure(K, plot_ts, plot_derman)
        plt.cla()
        plt.clf()
