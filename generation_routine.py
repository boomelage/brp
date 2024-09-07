#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Sep  7 13:22:14 2024

"""     
from data_generation import data_generation
from heston_calibration import calibrate_heston
from pricing import heston_price_vanillas, noisyfier

def generate_dateset(ivol_table,lower_moneyness, upper_moneyness,
    n_strikes, n_maturities, T, tl_ivol, risk_free_rate, dividend_rate,
        current_spot):
    # Create an instance of the data_generation class
    dg = data_generation(lower_moneyness=lower_moneyness,
                         upper_moneyness=upper_moneyness,
                         n_strikes=n_strikes,
                         n_maturities=n_maturities,
                         T=T,
                         tl_ivol=tl_ivol,
                         risk_free_rate = risk_free_rate,
                         dividend_rate = dividend_rate)
    
    # Call generate_data_subset method with current_spot
    option_data = dg.generate_data_subset(current_spot)
    
    option_data,flat_ts,dividend_ts,spot,expiration_dates, \
        black_var_surface,strikes,day_count,calculation_date, calendar, \
            implied_vols_matrix = dg.prepare_calibration(ivol_table, 
                                                         option_data, 
                                                         dividend_rate, 
                                                         risk_free_rate)
            
    heston_params = calibrate_heston(option_data,flat_ts,dividend_ts,spot,
                                     expiration_dates, black_var_surface,
                                     strikes,day_count,calculation_date, 
                                     calendar, dividend_rate, 
                                     implied_vols_matrix)
    
    heston_vanillas = heston_price_vanillas(heston_params)
    
    dataset = noisyfier(heston_vanillas)
    return dataset

