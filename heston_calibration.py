#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Sep  5 11:54:40 2024

@author: doomd
"""
import os
script_dir = os.path.dirname(os.path.abspath(__file__))
os.chdir(script_dir)
import QuantLib as ql
import time

def calibrate_heston(option_data,flat_ts,dividend_ts,spot,expiration_dates,
    black_var_surface,strikes,day_count,calculation_date, calendar,
        dividend_rate, implied_vols_matrix):
    heston_params = option_data.copy()
    # initial guesses
    v0 = 0.01; kappa = 0.2; theta = 0.02; rho = -0.75; sigma = 0.5;
    
    process = ql.HestonProcess(flat_ts, dividend_ts,
                               ql.QuoteHandle(ql.SimpleQuote(spot)),
                               v0, kappa, theta, sigma, rho)
    model = ql.HestonModel(process)
    engine = ql.AnalyticHestonEngine(model)
    heston_helpers = []
    
    # loop through all maturities and perform calibration for each one
    for current_index, date in enumerate(expiration_dates):
        print(f"\nCurrently calibrating for maturity: {date}")
        black_var_surface.setInterpolation("bicubic")
        for j, s in enumerate(strikes):
           t = day_count.yearFraction(calculation_date, date)
           sigma = black_var_surface.blackVol(t, s)  
           
           helper = ql.HestonModelHelper(
               ql.Period(int(t * 365), ql.Days),
               calendar, spot, s,
               ql.QuoteHandle(ql.SimpleQuote(sigma)),
               flat_ts, dividend_ts
           )
           helper.setPricingEngine(engine)
           heston_helpers.append(helper)
            
        lm = ql.LevenbergMarquardt(1e-8, 1e-8, 1e-8)
        model.calibrate(heston_helpers, lm,
                         ql.EndCriteria(500, 50, 1.0e-8,1.0e-8, 1.0e-8))
        theta, kappa, sigma, rho, v0 = model.params()
        
        print (
            "\ntheta = %f, kappa = %f, "
            "sigma = %f, rho = %f, v0 = %f" % (theta, kappa, 
                                               sigma, rho, v0))
        avg = 0.0
        
        # print ("%15s %15s %15s %20s" % (
        #     "Strikes", "Market Value",
        #      "Model Value", "Relative Error (%)"))
        # print ("="*70)
            
        for i in range(min(len(heston_helpers), len(strikes))):
            opt = heston_helpers[i]
            err = (opt.modelValue() / opt.marketValue() - 1.0)
        
            # print(f"{strikes[i]:15.2f} {opt.marketValue():14.5f} "
            #       f"{opt.modelValue():15.5f} {100.0 * err:20.7f}")
            
            avg += abs(err)  # accumulate the absolute error
        
        avg = avg*100.0/len(heston_helpers)
        print("-"*70)
        print("Total Average Abs Error (%%) : %5.3f" % (avg))
        
    heston_params['dividend_rate'] = dividend_rate
    heston_params['v0'] = v0
    heston_params['kappa'] = kappa
    heston_params['theta'] = theta
    heston_params['sigma'] = sigma
    heston_params['rho'] = rho
    
    heston_params
    return heston_params

    