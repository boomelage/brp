#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Sep  7 13:22:14 2024

"""
from market_settings import lower_moneyness, upper_moneyness, \
    n_strikes, n_maturities, T, tl_ivol, risk_free_rate, dividend_rate, \
        current_spot
        
from data_generation import data_generation

dg = data_generation(lower_moneyness = lower_moneyness,
                     upper_moneyness = upper_moneyness,
                     n_strikes = n_strikes,
                     n_maturities = n_maturities,
                     T = T,
                     tl_ivol = tl_ivol,
                     risk_free_rate = risk_free_rate,
                     dividend_rate = dividend_rate,
                     current_spot = current_spot 
                     )

ivol_table, option_data = dg.generate_data_subset(current_spot)

ivol_table

# calibrated_subset = data_generation.calibrate_data_subset(ivol_table, option_data)

# calibrated_subset