# -*- coding: utf-8 -*-
"""
Created on Mon Sep  9 00:17:45 2024
calibration_routine
@author: boomelage
"""


def clear_all():
    globals_ = globals().copy()  # Make a copy to avoid modifying during iteration
    for name in globals_:
        if not name.startswith('_') and name not in ['clear_all']:
            del globals()[name]
clear_all()

import os
import QuantLib as ql
import warnings
import numpy as np
import pandas as pd
pd.reset_option('display.max_rows')
pd.reset_option('display.max_columns')
warnings.simplefilter(action='ignore')
pwd = str(os.path.dirname(os.path.abspath(__file__)))
os.chdir(pwd)
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


# =============================================================================
                                                               # simple example
                                                               
"""
expiration_dates = [ql.Date(9,12,2021), ql.Date(9,1,2022), ql.Date(9,2,2022),
                    ql.Date(9,3,2022), ql.Date(9,4,2022), ql.Date(9,5,2022),
                    ql.Date(9,6,2022), ql.Date(9,7,2022), ql.Date(9,8,2022),
                    ql.Date(9,9,2022), ql.Date(9,10,2022), ql.Date(9,11,2022),
                    ql.Date(9,12,2022), ql.Date(9,1,2023), ql.Date(9,2,2023),
                    ql.Date(9,3,2023), ql.Date(9,4,2023), ql.Date(9,5,2023),
                    ql.Date(9,6,2023), ql.Date(9,7,2023), ql.Date(9,8,2023),
                    ql.Date(9,9,2023), ql.Date(9,10,2023), ql.Date(9,11,2023)]
calculation_date = expiration_dates[11] - ql.Period(365,ql.Days)

maturities = np.empty(len(expiration_dates))
for i in range(len(expiration_dates)):
    maturities[i] = expiration_dates[i]-calculation_date


strikes = [527.50, 560.46, 593.43, 626.40, 659.37, 692.34, 725.31, 758.28]
ivol_table = [
[0.37819, 0.34177, 0.30394, 0.27832, 0.26453, 0.25916, 0.25941, 0.26127],
[0.3445, 0.31769, 0.2933, 0.27614, 0.26575, 0.25729, 0.25228, 0.25202],
[0.37419, 0.35372, 0.33729, 0.32492, 0.31601, 0.30883, 0.30036, 0.29568],
[0.37498, 0.35847, 0.34475, 0.33399, 0.32715, 0.31943, 0.31098, 0.30506],
[0.35941, 0.34516, 0.33296, 0.32275, 0.31867, 0.30969, 0.30239, 0.29631],
[0.35521, 0.34242, 0.33154, 0.3219, 0.31948, 0.31096, 0.30424, 0.2984],
[0.35442, 0.34267, 0.33288, 0.32374, 0.32245, 0.31474, 0.30838, 0.30283],
[0.35384, 0.34286, 0.33386, 0.32507, 0.3246, 0.31745, 0.31135, 0.306],
[0.35338, 0.343, 0.33464, 0.32614, 0.3263, 0.31961, 0.31371, 0.30852],
[0.35301, 0.34312, 0.33526, 0.32698, 0.32766, 0.32132, 0.31558, 0.31052],
[0.35272, 0.34322, 0.33574, 0.32765, 0.32873, 0.32267, 0.31705, 0.31209],
[0.35246, 0.3433, 0.33617, 0.32822, 0.32965, 0.32383, 0.31831, 0.31344],
[0.35226, 0.34336, 0.33651, 0.32869, 0.3304, 0.32477, 0.31934, 0.31453],
[0.35207, 0.34342, 0.33681, 0.32911, 0.33106, 0.32561, 0.32025, 0.3155],
[0.35171, 0.34327, 0.33679, 0.32931, 0.3319, 0.32665, 0.32139, 0.31675],
[0.35128, 0.343, 0.33658, 0.32937, 0.33276, 0.32769, 0.32255, 0.31802],
[0.35086, 0.34274, 0.33637, 0.32943, 0.3336, 0.32872, 0.32368, 0.31927],
[0.35049, 0.34252, 0.33618, 0.32948, 0.33432, 0.32959, 0.32465, 0.32034],
[0.35016, 0.34231, 0.33602, 0.32953, 0.33498, 0.3304, 0.32554, 0.32132],
[0.34986, 0.34213, 0.33587, 0.32957, 0.33556, 0.3311, 0.32631, 0.32217],
[0.34959, 0.34196, 0.33573, 0.32961, 0.3361, 0.33176, 0.32704, 0.32296],
[0.34934, 0.34181, 0.33561, 0.32964, 0.33658, 0.33235, 0.32769, 0.32368],
[0.34912, 0.34167, 0.3355, 0.32967, 0.33701, 0.33288, 0.32827, 0.32432],
[0.34891, 0.34154, 0.33539, 0.3297, 0.33742, 0.33337, 0.32881, 0.32492]]


implied_vol_matrix = ql.Matrix(len(strikes),len(ivol_table))

for i in range(len(ivol_table)):
    for j in range(len(strikes)):
        implied_vol_matrix[j][i] = ivol_table[i][j]
"""  
# =============================================================================
                                                       # collecting market data
lower_bound_strike = 5460
upper_bound_strike = 5675
# lower_bound_maturity = 1
# upper_bound_maturity = 370

from new_collect_market_data import new_market_data_collection
nmdc = new_market_data_collection()
market_data = nmdc.new_concat_market_data()
market_data = market_data.reset_index()

market_data = market_data[market_data['Strike'] >= lower_bound_strike]
market_data = market_data[market_data['Strike'] <= upper_bound_strike]

# market_data = market_data[market_data['DyEx'] >= lower_bound_maturity]
# market_data = market_data[market_data['DyEx'] <= upper_bound_maturity]

maturities = market_data['DyEx'].unique().tolist()
strikes = market_data['Strike'].unique().tolist()

# =============================================================================
                                            # market data vol matrix generation
                                            
ivol_table = nmdc.new_make_ivol_table(market_data)

expiration_dates = np.empty(len(maturities), dtype=object)
for i, maturity in enumerate(maturities):
    expiration_dates[i] = calculation_date + ql.Period(int(maturity), ql.Days)
 
                                               
from make_ivol_matrix import make_ivol_matrix
implied_vol_matrix = make_ivol_matrix(
    strikes, expiration_dates, ivol_table, calculation_date, len(strikes),
    len(maturities))

print(implied_vol_matrix)

# =============================================================================
                                          # generating black volatility surface

black_var_surface = ql.BlackVarianceSurface(
    calculation_date, calendar,
    expiration_dates, strikes,
    implied_vol_matrix, day_count)

# =============================================================================
                                                           # heston calibration
S = np.median(strikes)                                                         
option_data = pd.DataFrame()
option_data_spot_column = np.ones(len(market_data))*S
option_data['spot_price'] = option_data_spot_column
option_data['strike_price'] = market_data['Strike']
option_data['volatility'] = market_data['IVM']
option_data['risk_free_rate'] = market_data['Rate']
option_data['dividend_rate'] = dividend_rate
option_data['w'] = 1
option_data['days_to_maturity'] = market_data['DyEx']
option_data['calculation_date'] = calculation_date
def compute_maturity_date(row):
    return row['calculation_date'] + ql.Period(int(row['days_to_maturity']), ql.Days)
option_data = option_data.dropna()
option_data['maturity_date'] = option_data.apply(compute_maturity_date, axis=1)



from heston_calibration import calibrate_heston
heston_params = calibrate_heston(
    option_data, flat_ts,dividend_ts, S, expiration_dates, 
    black_var_surface, strikes, day_count,calculation_date, calendar, 
    dividend_rate)


# heston_params.to_csv(r'option_data_spx_heston_params_test.csv')


# =============================================================================
                                                           # importing from csv

"""
heston_params = pd.read_csv(r'option_data_spx_heston_params_test.csv')
heston_params = heston_params.drop(columns = heston_params.columns[0])
heston_params = heston_params.drop(columns = ['calculation_date',
                                              'maturity_date'])
heston_params['calculation_date'] = calculation_date
heston_params = heston_params.dropna()
heston_params['maturity_date'] = heston_params.apply(
    compute_maturity_date, axis=1)


def compute_maturity_date(row):
    return row['calculation_date'] + \
        ql.Period(int(row['days_to_maturity']), ql.Days)

def convert_to_ql_quote(row, variable):
    row[variable] = ql.QuoteHandle(ql.SimpleQuote(row[variable]))
    return row

heston_params = heston_params.apply(
    lambda row: convert_to_ql_quote(row, 'risk_free_rate'), axis=1)
heston_params = heston_params.apply(
    lambda row: convert_to_ql_quote(row, 'spot_price'), axis=1)
heston_params = heston_params.apply(
    lambda row: convert_to_ql_quote(row, 'strike_price'), axis=1)
heston_params = heston_params.apply(
    lambda row: convert_to_ql_quote(row, 'dividend_rate'), axis=1)
"""

# =============================================================================
                                                            # generating prices

from pricing import heston_price_vanilla_row


hprice = heston_price_vanilla_row(heston_params.loc[0])




# from pricing import heston_price_vanillas, noisyfier
# heston_vanillas = heston_price_vanillas(heston_params)
# heston_vanillas
# dataset = noisyfier(heston_vanillas)

# =============================================================================
                                                 # plotting volatility surfance

import matplotlib.pyplot as plt
plt.rcParams['figure.figsize']=(15,7)
plt.style.use("dark_background")
from matplotlib import cm

expiry = 0.043806
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

plot_maturities = pd.Series(maturities) / 365.25
plot_strikes = pd.Series(strikes)
X, Y = np.meshgrid(plot_strikes, plot_maturities)
Z = np.array([[
    black_var_surface.blackVol(y, x) for x in plot_strikes] 
    for y in plot_maturities])

fig = plt.figure()
ax = fig.add_subplot(projection='3d')

surf = ax.plot_surface(X,Y,Z, rstride=1, cstride=1, cmap=cm.coolwarm,
                linewidth=0.1)
fig.colorbar(surf, shrink=0.5, aspect=5)

ax.set_xlabel("Strikes", size=9)
ax.set_ylabel("Maturities (Years)", size=9)
ax.set_zlabel("Volatility", size=9)

plt.show()
plt.cla()
plt.clf()
