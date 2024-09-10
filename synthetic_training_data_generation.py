#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Sep  7 13:03:40 2024

"""
import os
pwd = str(os.path.dirname(os.path.abspath(__file__)))
os.chdir(pwd)
import numpy as np

# =============================================================================
                                                                     # Settings
risk_free_rate = 0.00
dividend_rate = 0.00

pricing_range = 0.1

ticker = 'AAPL'
bb_table_path = r'22000 AAPL.xlsx'
S = 5410
tl_strike = 195.00 
tl_ivol_q = 41.2680358886719

shortest_maturity = 14/365
longest_maturity = 2*52*7/365
maturity_step = 7/365

# shortest_maturity = 1/12
# longest_maturity =2.01
# maturity_step = 1/12

spots_subdivision = 5
strikes_subdivision = 7

spotmin = int(S/(1+pricing_range))
spotmax = int(S*(1+pricing_range))
nspots = int(spots_subdivision*(spotmax-spotmin))

# nspots = 1
# lower_moneyness = 0.5
# upper_moneyness = 1.5
# n_strikes = 7

lower_moneyness = tl_strike/S
upper_moneyness = S/tl_strike
n_strikes = int((strikes_subdivision)*(S*upper_moneyness-\
                                        S*lower_moneyness))

spots = np.linspace(spotmin,spotmax,nspots)
T = np.arange(shortest_maturity, longest_maturity, maturity_step)
n_maturities = len(T)

from data_generation import generate_dataset

dataset = generate_dataset(S, lower_moneyness, upper_moneyness, n_strikes, risk_free_rate, T, dividend_rate) 


# =============================================================================
# # """
# # MISGUIDED
# # """
# # """
# # def generate_syntetic_subset():
# #     option_data = dg.generate_data_subset(S)
#     
# #     option_data,flat_ts,dividend_ts,spot,expiration_dates, \
# #         black_var_surface,strikes,day_count,calculation_date, calendar, \
# #             implied_vols_matrix = dg.prepare_calibration(
# #                 ivol_table, option_data, dividend_rate, risk_free_rate)   
#             
# #     heston_params = calibrate_heston(
# #         option_data, flat_ts, dividend_ts, spot, expiration_dates, 
# #         black_var_surface, strikes, day_count, calculation_date, calendar, 
# #         dividend_rate, implied_vols_matrix)
#     
# #     heston_params = calibrate_heston(
# #         option_data, flat_ts, dividend_ts, spot, expiration_dates, 
# #         black_var_surface, strikes, day_count, calculation_date, calendar, 
# #         dividend_rate, implied_vols_matrix)
# 
# #     # Generate vanilla options and noisy dataset
# #     heston_vanillas = heston_price_vanillas(heston_params)
# #     dataset = noisyfier(heston_vanillas)
# 
# #     return dataset
# 
# 
# 
# 
# 
# 
# # # multiple_S = np.linspace(90,110,5)
# # # def multiple_spot_synthetic_dataset():
# # #     multiple_spots = []
# # #     for i in range(0, len(multiple_S)):
# # #         # spot_counter = i + 1
# # #         # of_total = len(multiple_S)
# # #         S = multiple_S[i]
# # #         option_data = dg.generate_data_subset(S)
#         
# # #         option_data, flat_ts, dividend_ts, spot, expiration_dates, \
# # #             black_var_surface, strikes, day_count, calculation_date, calendar, \
# # #             implied_vols_matrix = dg.prepare_calibration(
# # #                 ivol_table, option_data, dividend_rate, risk_free_rate)
#         
# # #         prices_parameters = dg.generate_dataset(
# # #             ivol_table, lower_moneyness, upper_moneyness, n_strikes, 
# # #             n_maturities, T, tl_ivol, risk_free_rate, dividend_rate, 
# # #             S)
#         
# # #         multiple_spots.append(prices_parameters)
#         
# # #     return multiple_spots
# # """
# =============================================================================
