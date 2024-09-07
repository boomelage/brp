#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Sep  7 13:03:40 2024

"""
import os
script_dir = os.path.dirname(os.path.abspath(__file__))
os.chdir(script_dir)
import numpy as np
from bloomberg_ivols import read_bbiv_table
# =============================================================================
                                                                     # Settings
dividend_rate = 0.00
risk_free_rate = 0.05

pricing_range = 0.1

ticker = 'AAPL'
bb_table_path = r'22000 AAPL.xlsx'
current_spot = 220.00
tl_strike = 195.00 
tl_ivol_q = 41.2680358886719
shortest_maturity = 1/12
longest_maturity = 2.01
maturity_step = 1/12
spots_subdivision = 1
strikes_subdivision = 3

spotmin = int(current_spot/(1+pricing_range))
spotmax = int(current_spot*(1+pricing_range))
nspots = int(spots_subdivision*(spotmax-spotmin))

# nspots = 1
# lower_moneyness = 0.5
# upper_moneyness = 1.5
# n_strikes = 8

lower_moneyness = tl_strike/current_spot
upper_moneyness = current_spot/tl_strike
n_strikes = int((strikes_subdivision)*(current_spot*upper_moneyness-\
                                        current_spot*lower_moneyness))

tl_ivol = tl_ivol_q/100
spots = np.linspace(spotmin,spotmax,nspots)
T = np.arange(shortest_maturity, longest_maturity, maturity_step)
n_maturities = len(T)

ivol_table = read_bbiv_table(bb_table_path)
print(ivol_table)
