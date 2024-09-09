# -*- coding: utf-8 -*-
"""
Created on Mon Sep  9 00:17:45 2024
calibration_routine
@author: boomelage
"""

import os
import QuantLib as ql
import warnings
import numpy as np
import pandas as pd
warnings.simplefilter(action='ignore')
pwd = str(os.path.dirname(os.path.abspath(__file__)))
os.chdir(pwd)
from data_query import dirdata


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

from make_ivol_table import dfcalls, ivol_table, maturities, strikes, S,\
    n_maturities, n_strikes

from make_ivol_matrix import make_ivol_matrix

expiration_dates, implied_vol_matrix = make_ivol_matrix(
    strikes, maturities, ivol_table, calculation_date, n_strikes, n_maturities)


minK = min(strikes)
maxK = max(strikes)
lower_moneyness = minK/S
upper_moneyness = maxK/S
shortest_maturity = min(maturities)
longest_maturity = max(maturities)



# =============================================================================
                                          # generating black volatility surface

black_var_surface = ql.BlackVarianceSurface(
    calculation_date, calendar,
    expiration_dates, strikes,
    implied_vol_matrix, day_count)

# =============================================================================
                                                         # constructing dataset

option_data = pd.DataFrame()
option_data_spot_column = np.ones(len(dfcalls))*S
option_data['spot_price'] = option_data_spot_column
option_data['strike_price'] = dfcalls['Strike']
option_data['volatility'] = dfcalls['IVM']
option_data['risk_free_rate'] = dfcalls['Rate']
option_data['dividend_rate'] = dividend_rate
option_data['w'] = 1
option_data['days_to_maturity'] = dfcalls['DyEx']
option_data['calculation_date'] = calculation_date

def calculate_maturity_date(row, calc_date):
    return calc_date + ql.Period(int(row['days_to_maturity']), ql.Days)

option_data['maturity_date'] = option_data.apply(calculate_maturity_date, 
                                                 calc_date=calculation_date, 
                                                 axis=1)

# =============================================================================
                                                           # heston calibration 

from heston_calibration import calibrate_heston
heston_params = calibrate_heston(
    option_data,flat_ts,dividend_ts, S, expiration_dates, 
    black_var_surface, strikes, day_count,calculation_date, calendar, 
    dividend_rate, implied_vol_matrix)

# =============================================================================
                                                            # generating prices

from pricing import heston_price_vanillas, noisyfier
heston_vanillas = heston_price_vanillas(heston_params)

dataset = noisyfier(heston_vanillas)

# =============================================================================
                                                 # plotting volatility surfance

import matplotlib.pyplot as plt
from matplotlib import cm


plt.rcParams['figure.figsize']=(6,4)
plt.style.use("dark_background")

expiry = 10/365
target_maturity_ivols = ivol_table[0]


implied_vols = [black_var_surface.blackVol(expiry, k)
                for k in strikes]

fig, ax = plt.subplots()
ax.plot(strikes, target_maturity_ivols, label="Black Surface")
ax.plot(strikes, target_maturity_ivols, "o", label="Actual")
ax.set_xlabel("Strikes", size=9)
ax.set_ylabel("Vols", size=9)
legend = ax.legend(loc="upper right")
fig.show()


plot_maturities = np.array(maturities/365.25).astype(float)
plot_strikes = np.array(strikes).astype(float)
X, Y = np.meshgrid(plot_strikes, plot_maturities)
Z = np.array([[
    black_var_surface.blackVol(y, x) for x in plot_strikes] 
    for y in plot_maturities])

fig = plt.figure()
ax = fig.add_subplot(projection='3d')

surf = ax.plot_surface(
    X, Y, Z, rstride=1, cstride=1, cmap=cm.coolwarm, linewidth=0.1)
fig.colorbar(surf, shrink=0.35, aspect=6)

ax.set_xlabel("Strikes", size=9)
ax.set_ylabel("Maturities (Years)", size=9)
ax.set_zlabel("Volatility", size=9)

plt.show()
plt.cla()
plt.clf()

