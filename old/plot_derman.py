#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Sep 15 18:17:14 2024

"""
import time
from plot_surface import plot_term_structure
def plot_derman_fit(derman_ts,trimmed_ts):
    T = derman_ts.columns
    for t in T:
        time.sleep(0.05)
        plot_ts = trimmed_ts.loc[:,t].dropna()
        plot_derman = derman_ts.loc[:,t]
        plot_derman = plot_derman[plot_ts.index]
        K = plot_derman.index
        title = f"t = {t} days"
        plot_term_structure(K, plot_ts, plot_derman,title)


