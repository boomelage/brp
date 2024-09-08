#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Sep  7 13:03:40 2024

"""
import os
script_dir = os.path.dirname(os.path.abspath(__file__))
os.chdir(script_dir)
import numpy as np
from data_generation import data_generation
from generate_ivols import generate_ivol_table

# =============================================================================
                                                                     # Settings
risk_free_rate = 0.00
dividend_rate = 0.00

pricing_range = 0.1

ticker = 'AAPL'
bb_table_path = r'22000 AAPL.xlsx'
current_spot = 220.00
tl_strike = 195.00 
tl_ivol_q = 41.2680358886719

shortest_maturity = 14/365.25
longest_maturity = 2*52*7/365.25
maturity_step = 7.52

# shortest_maturity = 1/12
# longest_maturity =2.01
# maturity_step = 1/12

spots_subdivision = 5
strikes_subdivision = 7

spotmin = int(current_spot/(1+pricing_range))
spotmax = int(current_spot*(1+pricing_range))
nspots = int(spots_subdivision*(spotmax-spotmin))

# nspots = 1
# lower_moneyness = 0.5
# upper_moneyness = 1.5
# n_strikes = 7

lower_moneyness = tl_strike/current_spot
upper_moneyness = current_spot/tl_strike
n_strikes = int((strikes_subdivision)*(current_spot*upper_moneyness-\
                                        current_spot*lower_moneyness))

tl_ivol = tl_ivol_q/100
spots = np.linspace(spotmin,spotmax,nspots)
T = np.arange(shortest_maturity, longest_maturity, maturity_step)
n_maturities = len(T)

decay_rate = 1/(10*n_strikes*n_maturities)
row_decay = decay_rate/10

ivol_table = generate_ivol_table(n_maturities, n_strikes, tl_ivol, 
                                  decay_rate, row_decay)

multiple_S = np.linspace(90,110,5)


dg = data_generation(lower_moneyness=lower_moneyness, upper_moneyness=upper_moneyness,
                     T=T, n_maturities=n_maturities, n_strikes=n_strikes, tl_ivol=tl_ivol,
                     risk_free_rate=risk_free_rate, dividend_rate=dividend_rate)


def generate_syntetic_data():
    option_data = dg.generate_data_subset(current_spot)
    
    option_data,flat_ts,dividend_ts,spot,expiration_dates, \
        black_var_surface,strikes,day_count,calculation_date, calendar, \
            implied_vols_matrix = dg.prepare_calibration(
                ivol_table, option_data, dividend_rate, risk_free_rate)
    
    option_prices = dg.generate_dataset(ivol_table, lower_moneyness, 
                                        upper_moneyness, n_strikes, 
                                        n_maturities, T, tl_ivol, 
                                        risk_free_rate, dividend_rate, 
                                        current_spot)       
            
    return option_prices

option_prices = generate_syntetic_data()





# def multiple_spot_synthetic_dataset():
#     multiple_spots = []
#     for i in range(0, len(multiple_S)):
#         # spot_counter = i + 1
#         # of_total = len(multiple_S)
#         current_spot = multiple_S[i]
#         option_data = dg.generate_data_subset(current_spot)
        
#         option_data, flat_ts, dividend_ts, spot, expiration_dates, \
#             black_var_surface, strikes, day_count, calculation_date, calendar, \
#             implied_vols_matrix = dg.prepare_calibration(
#                 ivol_table, option_data, dividend_rate, risk_free_rate)
        
#         prices_parameters = dg.generate_dataset(
#             ivol_table, lower_moneyness, upper_moneyness, n_strikes, 
#             n_maturities, T, tl_ivol, risk_free_rate, dividend_rate, 
#             current_spot)
        
#         multiple_spots.append(prices_parameters)
        
#     return multiple_spots
