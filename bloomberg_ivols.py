# -*- coding: utf-8 -*-
"""
Created on Sat Sep  7 15:38:52 2024

"""
import os
pwd = str(os.path.dirname(os.path.abspath(__file__)))
os.chdir(pwd)
files = os.listdir(pwd)
excel_files = [file for file in files if file.endswith('.xlsx')]
import numpy as np
import pandas as pd
from itertools import product
import QuantLib as ql
import math
from heston_calibration import calibrate_heston
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
plt.rcParams['figure.figsize']=(15,7)
plt.style.use("dark_background")
from matplotlib import cm


risk_free_rate = 0.00
dividend_rate = 0.00

bbivols = pd.read_excel(str(excel_files[0]))

bbivols.columns = bbivols.iloc[1]
bbivols = bbivols.drop([0,1])
bbivols = bbivols.reset_index(drop=True)
bbivols = bbivols.sort_values(by = 'Strike')
bbivols = bbivols.dropna()
maturities = bbivols.filter(like='DyEx').iloc[0].unique()
strikes = bbivols.filter(like='Strike').to_numpy().astype(float).flatten()

bbiv_df = bbivols.filter(like='IVM')
bbivs = bbiv_df.to_numpy()/100
n_strikes = len(bbivs)
n_maturities = int(len(bbivs[0])/2)

ivol_table = np.empty(n_maturities,dtype=object)

for i in range(n_maturities):
    ivol_table[i] = []
    for j in range(n_strikes):
        ivol_table[i].append((bbivs[j][2*i] + bbivs[j][2*i+1])/2)  

S = [np.median(strikes)]
K = strikes
T = maturities
def generate_features():
    features = pd.DataFrame(
        product(S, K, T),
        columns=[
            "spot_price", 
            "strike_price", 
            "years_to_maturity"
                  ])
    return features
features = generate_features()
features["years_to_maturity"] = features["years_to_maturity"]/365
features['risk_free_rate'] = risk_free_rate
features['dividend_rate'] = dividend_rate
features['w'] = 1
option_data = features
option_data['calculation_date'] = ql.Date.todaysDate()
option_data['maturity_date'] = option_data.apply(
    lambda row: row['calculation_date'] + ql.Period(
        int(math.floor(row['years_to_maturity'] * 365)), ql.Days), axis=1)




spot = float(S[0])


calculation_date = ql.Date.todaysDate()
expiration_dates = []
for maturity in maturities:
    expiration_date = calculation_date + ql.Period(int(maturity), ql.Days)  # Adjust conversion as needed
    expiration_dates.append(expiration_date)
day_count = ql.Actual365Fixed()
day_count = ql.Actual365Fixed()
calendar = ql.UnitedStates(m=1)
ql.Settings.instance().evaluationDate = calculation_date
dividend_yield = ql.QuoteHandle(ql.SimpleQuote(dividend_rate))
dividend_rate = dividend_yield
flat_ts = ql.YieldTermStructureHandle(ql.FlatForward(calculation_date, risk_free_rate, day_count))
dividend_ts = ql.YieldTermStructureHandle(ql.FlatForward(calculation_date, dividend_rate, day_count))


implied_vols_matrix = ql.Matrix(n_strikes,n_maturities,float(0))
                
for i in range(n_strikes):
    for j in range(n_maturities):
        implied_vols_matrix[i][j] = ivol_table[j][i]

print(implied_vols_matrix)

black_var_surface = ql.BlackVarianceSurface(
    calculation_date, calendar,
    expiration_dates, K,
    implied_vols_matrix, day_count)

heston_params = calibrate_heston(option_data,flat_ts,dividend_ts,spot ,expiration_dates,
    black_var_surface,strikes,day_count,calculation_date, calendar,
        dividend_rate, implied_vols_matrix)


# fig, ax = plt.subplots()
# ax.plot(strikes, ivol_table, label="Black Surface")
# ax.plot(strikes, ivol_table, "o", label="Actual")
# ax.set_xlabel("Strikes", size=12)
# ax.set_ylabel("Vols", size=12)
# legend = ax.legend(loc="upper right")
# plot_years = np.arange(0, 2, 0.1)
# plot_strikes = np.arange(535.0, 750.0, 1.0)
# fig = plt.figure()

# ax = fig.add_subplot(projection='3d')
# X, Y = np.meshgrid(plot_strikes, plot_years)

# Z = np.array([black_var_surface.blackVol(y, x)
#               for xr, yr in zip(X, Y)
#                   for x, y in zip(xr,yr) ]
#              ).reshape(len(X), len(X[0]))

# surf = ax.plot_surface(X,Y,Z, rstride=1, cstride=1, cmap=cm.coolwarm,
#                 linewidth=0.1)
# fig.colorbar(surf, shrink=0.5, aspect=5)

# plt.show()