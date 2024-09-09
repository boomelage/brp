# -*- coding: utf-8 -*-
"""
Created on Mon Sep  9 00:17:45 2024
calibration_routine
@author: boomelage
"""

import os
pwd = str(os.path.dirname(os.path.abspath(__file__)))
os.chdir(pwd)
from data_query import dirdata
import QuantLib as ql
import warnings
warnings.simplefilter(action='ignore')
import numpy as np


dividend_rate = 0.005
risk_free_rate = 0.05

calculation_date = ql.Date.todaysDate()
day_count = ql.Actual365Fixed()
day_count = ql.Actual365Fixed()
calendar = ql.UnitedStates(m=1)
ql.Settings.instance().evaluationDate = calculation_date
dividend_yield = ql.QuoteHandle(ql.SimpleQuote(dividend_rate))
dividend_rate = dividend_yield
flat_ts = ql.YieldTermStructureHandle(ql.FlatForward(
    calculation_date, risk_free_rate, day_count))
dividend_ts = ql.YieldTermStructureHandle(ql.FlatForward(
    calculation_date, dividend_rate, day_count))



     

                                                                  #------------
                                                                  # market data


data_files = dirdata()


f
calls, strikes, maturities, ivols = fetch_bbdata(data_files, calculation_date)
n_maturities, n_strikes, S, expiration_dates, implied_vol_matrix = \
    make_ivol_matrix(strikes,maturities,ivols,calculation_date)



# Ks = strikes
# minK = min(Ks)
# maxK = max(Ks)
# lower_moneyness = minK/S
# upper_moneyness = maxK/S
# shortest_maturity = min(maturities)
# longest_maturity = max(maturities)

# K = strikes
# T = maturities

# nTs = (longest_maturity - shortest_maturity)*3
# nKs = (maxK-minK)*5
# data_T = np.linspace(shortest_maturity, longest_maturity,nTs)
# data_K = np.linspace(minK,maxK,nKs)


# option_data = calls.copy()
# option_data.columns = ['strike_price',
#                         'volatility',
#                         'days_to_matuirty',
#                         'risk_free_rate',
#                         'maturity_date']
# option_data['spot_price'] = S
# option_data['calculation_date'] = calculation_date
# option_data





# =============================================================================
                                                     # calibrating Heston model

# black_var_surface = ql.BlackVarianceSurface(
#     calculation_date, calendar,
#     expiration_dates, K,
#     implied_vol_matrix, day_count)

# from heston_calibration import calibrate_heston
# heston_params = calibrate_heston(
#     option_data,flat_ts,dividend_ts, S, expiration_dates, 
#     black_var_surface, K, day_count,calculation_date, calendar, 
#     dividend_rate, implied_vol_matrix)

#                                                                       #--------
#                                                                       # pricing
# from pricing import heston_price_vanillas, noisyfier
# heston_vanillas = heston_price_vanillas(heston_params)
# dataset = noisyfier(heston_vanillas)

# dataset



# =============================================================================
                                                 # plotting volatility surfance

# import matplotlib.pyplot as plt
# from matplotlib import cm


# plt.rcParams['figure.figsize']=(6,4)
# plt.style.use("dark_background")

# expiry = 1/365
# target_maturity_ivols = ivols.iloc[:,0].to_numpy()


# implied_vols = [black_var_surface.blackVol(expiry, k)
#                 for k in K]

# fig, ax = plt.subplots()
# ax.plot(K, target_maturity_ivols, label="Black Surface")
# ax.plot(K, target_maturity_ivols, "o", label="Actual")
# ax.set_xlabel("Strikes", size=9)
# ax.set_ylabel("Vols", size=9)
# legend = ax.legend(loc="upper right")
# fig.show()


# plot_maturities = maturities/365.25
# X, Y = np.meshgrid(K, plot_maturities)
# Z = np.array([[black_var_surface.blackVol(y, x) for x in K] for y in plot_maturities])

# fig = plt.figure()
# ax = fig.add_subplot(projection='3d')

# surf = ax.plot_surface(X, Y, Z, rstride=1, cstride=1, cmap=cm.coolwarm, linewidth=0.1)
# fig.colorbar(surf, shrink=0.35, aspect=6)

# ax.set_xlabel("Strikes", size=9)
# ax.set_ylabel("Maturities (Years)", size=9)
# ax.set_zlabel("Volatility", size=9)

# plt.show()
# plt.cla()
# plt.clf()