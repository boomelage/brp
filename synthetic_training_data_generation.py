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
risk_free_rate = 0.05
dividend_rate = 0.00

pricing_range = 0.1

S = 5410

from ivolmat_from_market import extract_ivol_matrix_from_market
implied_vols_matrix, strikes, maturities, callvols = \
    extract_ivol_matrix_from_market(r'SPXts.xlsx')

S = [S]
K = strikes
T = np.array(maturities,dtype=float)/365

K = np.linspace(min(K), max(K), 10000)
T = np.linspace(min(T), max(T),150)

from data_generation import generate_dataset
contract_details = generate_dataset(S, K, T, risk_free_rate, dividend_rate) 

from pricing import BS_price_vanillas, heston_price_vanillas, noisyfier

contract_details['volatility'] = 0.2

from calibration_routine import heston_params
contract_details['v0'] = heston_params['v0']
contract_details['kappa'] = heston_params['kappa']
contract_details['sigma'] = heston_params['sigma']
contract_details['rho'] = heston_params['rho']
contract_details['theta'] = heston_params['theta']


bs_vanillas = BS_price_vanillas(contract_details)

heston_bs_vanillas = heston_price_vanillas(bs_vanillas)
dataset = noisyfier(heston_bs_vanillas)
dataset



















# =============================================================================
# # """
# # MISGUIDED BY MISUNDERSTANDING
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
