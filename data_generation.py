#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Sep  4 17:22:26 2024

"""
import pandas as pd
import numpy as np
from itertools import product
import QuantLib as ql
import math
from heston_calibration import calibrate_heston_vanilla
from pricing import heston_price_vanillas, noisyfier
from generate_ivols import generate_ivol_table
# =============================================================================
                                                                     # Settings
dividend_rate = 0.0
r = 0.05

pricing_range = 1.1

ticker = 'AAPL'
current_spot = 220.00
tl_strike = 195.00 
tl_ivol_q = 41.2680358886719
shortest_maturity = 14/365
longest_maturity = 2*52*7/365
maturity_step = 7/365
spots_subdivision = 3

spotmin = int(current_spot/pricing_range)
spotmax = int(current_spot*pricing_range)
# nspots = int(spots_subdivision*(spotmax-spotmin))
nspots = 3

lower_moneyness = tl_strike/current_spot
upper_moneyness = current_spot/tl_strike
# n_strikes = int((spots_subdivision+2)*(current_spot*upper_moneyness-\
#                                        current_spot*lower_moneyness))
n_strikes = 15

# =============================================================================
tl_ivol = tl_ivol_q/100
spots = np.linspace(spotmin,spotmax,nspots)
T = np.arange(shortest_maturity, longest_maturity, maturity_step)
n_maturities = len(T)

def generate_data_subset(S,counter,of_total_spots):
    spots = np.ones(1) * S
    K = np.linspace(S*lower_moneyness, S*upper_moneyness, n_strikes)
    def generate_features():
        features = pd.DataFrame(
            product(spots, K, T),
            columns=[
                "spot_price", 
                "strike_price", 
                "years_to_maturity"
                     ])
        return features
    features = generate_features()
    n_lists = n_maturities
    n_elements = n_strikes
    decay_rate = 1/(10*n_maturities*n_strikes)
    row_decay = decay_rate/10
    data = generate_ivol_table(n_lists, n_elements, tl_ivol, 
                               decay_rate, row_decay)
    features['risk_free_rate'] = r
    features['dividend_rate'] = dividend_rate
    features['w'] = 1
    vanilla_params = features
    vanilla_params['calculation_date'] = ql.Date.todaysDate()
    vanilla_params['maturity_date'] = vanilla_params.apply(
        lambda row: row['calculation_date'] + ql.Period(
            int(math.floor(row['years_to_maturity'] * 365)), ql.Days), axis=1)
    expiration_dates = vanilla_params['maturity_date'].unique()
    strikes = vanilla_params['strike_price'].unique()
    implied_vols = ql.Matrix(len(strikes), len(expiration_dates))
    calibrated_features = calibrate_heston_vanilla.calibrate_heston(
        vanilla_params, dividend_rate, r, implied_vols, data, counter, 
        of_total_spots, n_strikes, nspots, n_maturities)
    prices = heston_price_vanillas(calibrated_features)
    dataset = noisyfier(prices)
    return dataset

def generate_dataset():
    data_subsets = []
    counter_spot = 0
    for spot in spots:
        counter_spot = counter_spot + 1
        of_total_spots = len(spots)
        spot = spot
        subset = generate_data_subset(spot, counter_spot, of_total_spots)
        data_subsets.append(subset)
    dataset = pd.concat(data_subsets, ignore_index=True)
    return dataset