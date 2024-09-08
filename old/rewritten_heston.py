#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Sep  5 11:30:54 2024

"""
from datetime import datetime
import warnings
import numpy as np
import pandas as pd
from nelson_siegel_svensson.calibrate import calibrate_nss_ols
import matplotlib.pyplot as plt
import QuantLib as ql
import math

def fit_model_data(data):
    
    YIELDS = np.array(
        [5.47, 5.48, 5.52, 5.46, 5.40, 5.16, 4.87, 4.62, 4.48, 4.47, 4.47, 4.68, 4.59]
    )
    YIELD_MATURITIES = np.array(
        [1 / 12, 2 / 12, 3 / 12, 4 / 12, 6 / 12, 1, 2, 3, 5, 7, 10, 20, 30]
    )
    strikes = data["strike_price"].unique().tolist()  # List of unique strike prices
    maturities = data["years_to_maturity"].unique().tolist()  # List of unique maturities
    imp_vols = data.pivot(index='strike_price', columns='years_to_maturity', values='volatility').to_numpy()
    expiration_dates = data["maturity_date"].unique().tolist()  # Convert dates to datetime objects
    spot_price = data["spot_price"].iloc[0]  # Assuming spot price is constant or take the first value

    # strikes = data["strike_price"]
    # maturities = data["years_to_maturity"]
    # imp_vols = data["volatility"]
    # expiration_dates = data["maturity_date"]
    # spot_price = data["spot_price"]

    day_count = (
        ql.Actual365Fixed()
    )  # This line ensures that the number of days are set to 365
    calendar = ql.UnitedStates(
        ql.UnitedStates.GovernmentBond
    )  # Calendar set according to the American standard

    today = datetime.today()
    calculation_date = ql.Date(today.day, today.month, today.year)

    ql.Settings.instance().evaluationDate = (
        calculation_date  # This line is used to set a pivot for today's date.
    )
    # Converting the `yield_maturities` into `Quantlib.Date` object.
    dates = [
        calculation_date + ql.Period(int(maturity * 12), ql.Months)
        for maturity in YIELD_MATURITIES
    ]

    # Calibrating the interest rates with respect to the yield maturities.
    zero_curve = ql.ZeroCurve(dates, YIELDS, day_count, calendar)
    zero_curve_handle = ql.YieldTermStructureHandle(zero_curve)

    # Calculating dividend yield.
    dividend_rate = 0.0
    dividend_ts = ql.YieldTermStructureHandle(
        ql.FlatForward(calculation_date, dividend_rate, day_count)
    )
    
    # Convert expiration_dates to QuantLib Date objects
    expiration_dates_ql = expiration_dates
    
    # imp_vols_df = imp_vols.reshape((len(maturities.unique()), len(strikes))).T
    imp_vols_df = imp_vols
    imp_vols_df = pd.DataFrame(imp_vols_df, index=strikes, columns=maturities)
    
    # Now you can fill the QuantLib matrix
    imp_vols_quantlib = ql.Matrix(len(strikes), len(maturities))
    for i, strike in enumerate(strikes):
        for j, maturity in enumerate(maturities):
            imp_vols_quantlib[i][j] = imp_vols_df.loc[strike, maturity]
    
    # Create the BlackVarianceSurface
    black_var_surface = ql.BlackVarianceSurface(
        calculation_date,
        calendar,
        expiration_dates_ql,
        strikes,
        imp_vols_quantlib,
        day_count,
        )

    # Initial guess for the Heston parameters.
    # v0, kappa, theta, rho, sigma = 0.01, 0.2, 0.02, -0.75, 0.5
    v0, kappa, theta, rho, sigma = 0.01, 0.02, 0.03, -0.04, 0.05
    process = ql.HestonProcess(
        zero_curve_handle,
        dividend_ts,
        ql.QuoteHandle(ql.SimpleQuote(spot_price)),
        v0,
        kappa,
        theta,
        sigma,
        rho,
    )

    model = ql.HestonModel(process)
    engine = ql.AnalyticHestonEngine(model)
    heston_helpers = []
    black_var_surface.setInterpolation("bicubic")

    for i, date in enumerate(expiration_dates):
        for j, s in enumerate(strikes):
            if i < imp_vols_quantlib.rows() and j < imp_vols_quantlib.columns():
                sigma = imp_vols_quantlib[i][j]
                t = math.floor((date - calculation_date)/365.0)
                p = ql.Period(t, ql.Days)
                sigma = imp_vols_quantlib[i][j]
                helper = ql.HestonModelHelper(
                    p,
                    calendar,
                    spot_price,
                    s,
                    ql.QuoteHandle(ql.SimpleQuote(sigma)),
                    zero_curve_handle,
                    dividend_ts,
                )
                helper.setPricingEngine(engine)
                heston_helpers.append(helper)
            # else:
                # print(f"Skipping out-of-bounds indices i={i}, j={j}")

    lm = ql.LevenbergMarquardt(1e-8, 1e-8, 1e-8)
    model.calibrate(
        heston_helpers, lm, ql.EndCriteria(500, 50, 1.0e-8, 1.0e-8, 1.0e-8)
    )
    
    # Extract the calibrated parameters
    calibrated_params = model.params()
    return calibrated_params