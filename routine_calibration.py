# -*- coding: utf-8 -*-
"""
Created on Mon Sep  9 00:17:45 2024

"""
def clear_all():
    globals_ = globals().copy()  # Make a copy to avoid 
    for name in globals_:        # modifying during iteration
        if not name.startswith('_') and name not in ['clear_all']:
            del globals()[name]
clear_all()
import os
import QuantLib as ql
import warnings
import numpy as np
import pandas as pd
warnings.simplefilter(action='ignore')
pwd = str(os.path.dirname(os.path.abspath(__file__)))
os.chdir(pwd)

                                            # QuantLib pricing settings/objects
dividend_rate = 0.005
risk_free_rate = 0.05
S  = 5400
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


def run_heston_calibration():
    from ivolmat_from_market import extract_ivol_matrix_from_market
    
    implied_vol_matrix, strikes, maturities, ivoldf = \
        extract_ivol_matrix_from_market(r'SPXts.xlsx')      
    
    expiration_dates = np.empty(len(maturities), dtype=object)
    for i, maturity in enumerate(maturities):
        expiration_dates[i] = calculation_date + ql.Period(maturity, ql.Days)



    black_var_surface = ql.BlackVarianceSurface(
        calculation_date, calendar,
        expiration_dates, strikes,
        implied_vol_matrix, day_count)


    from heston_calibration import calibrate_heston
    heston_params = calibrate_heston(
        flat_ts, dividend_ts, S, expiration_dates, 
        black_var_surface, strikes, day_count, calculation_date, calendar, 
        dividend_rate)
    
    return heston_params, ivoldf, black_var_surface, strikes, maturities

heston_params, ivoldf, black_var_surface, strikes, maturities \
    = run_heston_calibration()



# =============================================================================
                                                  # plotting volatility surface

import matplotlib.pyplot as plt
plt.rcParams['figure.figsize']=(15,7)
plt.style.use("dark_background")
from matplotlib import cm

expiry = 2/365
target_maturity_ivols = ivoldf[1]

def plot_volatility_surface():
    # implied_vols = [black_var_surface.blackVol(expiry, k)
    #                 for k in strikes]
    
    fig, ax = plt.subplots()
    ax.plot(strikes, target_maturity_ivols, label="Black Surface")
    ax.plot(strikes, target_maturity_ivols, "o", label="Actual")
    ax.set_xlabel("Strikes", size=9)
    ax.set_ylabel("Vols", size=9)
    # legend = ax.legend(loc="upper right")
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

plot_volatility_surface()
