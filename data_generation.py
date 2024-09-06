#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Sep  4 17:22:26 2024

"""
import os
script_dir = os.path.dirname(os.path.abspath(__file__))
os.chdir(script_dir)
import pandas as pd
import numpy as np
from itertools import product
import QuantLib as ql
import math
from heston_calibration import calibrate_heston
from pricing import heston_price_vanillas, noisyfier
from generate_ivols import generate_ivol_table
# =============================================================================
                                                                     # Settings
tl_ivol = 0.357
dividend_rate = 0.0
r = 0.05

spotmin = 90
spotmax = 120

lower_moneyness = 0.5
upper_moneyness = 1.5
n_strikes = 500

shortest_maturity = 1/12
longest_maturity = 2.01
maturity_step = 1/12
nspots = 3*(spotmax-spotmin)

tl_ivol = 0.357
dividend_rate = 0.0
r = 0.05

# =============================================================================

spots = np.linspace(spotmin,spotmax,nspots)
T = np.arange(shortest_maturity, longest_maturity, maturity_step)
n_maturities = len(T)

def generate_data_subset(S,counter,of_total):
    spots = np.ones(1) * S
    K = np.linspace(S * lower_moneyness, S * upper_moneyness, n_strikes)
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
            # GENERATING IVOLS
    n_lists = n_maturities
    n_elements = n_strikes
    decay_rate = 1/(10*n_maturities*n_strikes)
    row_decay = decay_rate/10
    data = generate_ivol_table(n_lists, n_elements, tl_ivol, 
                               decay_rate, row_decay)
    
    # data = [
    # [0.37819, 0.34177, 0.30394, 0.27832, 0.26453, 0.25916, 0.25941, 0.26127],
    # [0.3445, 0.31769, 0.2933, 0.27614, 0.26575, 0.25729, 0.25228, 0.25202],
    # [0.37419, 0.35372, 0.33729, 0.32492, 0.31601, 0.30883, 0.30036, 0.29568],
    # [0.37498, 0.35847, 0.34475, 0.33399, 0.32715, 0.31943, 0.31098, 0.30506],
    # [0.35941, 0.34516, 0.33296, 0.32275, 0.31867, 0.30969, 0.30239, 0.29631],
    # [0.35521, 0.34242, 0.33154, 0.3219, 0.31948, 0.31096, 0.30424, 0.2984],
    # [0.35442, 0.34267, 0.33288, 0.32374, 0.32245, 0.31474, 0.30838, 0.30283],
    # [0.35384, 0.34286, 0.33386, 0.32507, 0.3246, 0.31745, 0.31135, 0.306],
    # [0.35338, 0.343, 0.33464, 0.32614, 0.3263, 0.31961, 0.31371, 0.30852],
    # [0.35301, 0.34312, 0.33526, 0.32698, 0.32766, 0.32132, 0.31558, 0.31052],
    # [0.35272, 0.34322, 0.33574, 0.32765, 0.32873, 0.32267, 0.31705, 0.31209],
    # [0.35246, 0.3433, 0.33617, 0.32822, 0.32965, 0.32383, 0.31831, 0.31344],
    # [0.35226, 0.34336, 0.33651, 0.32869, 0.3304, 0.32477, 0.31934, 0.31453],
    # [0.35207, 0.34342, 0.33681, 0.32911, 0.33106, 0.32561, 0.32025, 0.3155],
    # [0.35171, 0.34327, 0.33679, 0.32931, 0.3319, 0.32665, 0.32139, 0.31675],
    # [0.35128, 0.343, 0.33658, 0.32937, 0.33276, 0.32769, 0.32255, 0.31802],
    # [0.35086, 0.34274, 0.33637, 0.32943, 0.3336, 0.32872, 0.32368, 0.31927],
    # [0.35049, 0.34252, 0.33618, 0.32948, 0.33432, 0.32959, 0.32465, 0.32034],
    # [0.35016, 0.34231, 0.33602, 0.32953, 0.33498, 0.3304, 0.32554, 0.32132],
    # [0.34986, 0.34213, 0.33587, 0.32957, 0.33556, 0.3311, 0.32631, 0.32217],
    # [0.34959, 0.34196, 0.33573, 0.32961, 0.3361, 0.33176, 0.32704, 0.32296],
    # [0.34934, 0.34181, 0.33561, 0.32964, 0.33658, 0.33235, 0.32769, 0.32368],
    # [0.34912, 0.34167, 0.3355, 0.32967, 0.33701, 0.33288, 0.32827, 0.32432],
    # [0.34891, 0.34154, 0.33539, 0.3297, 0.33742, 0.33337, 0.32881, 0.32492]]
    
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
    calibrated_features = calibrate_heston(vanilla_params, dividend_rate, r, 
                                           implied_vols, data, 
                                           counter,of_total, n_strikes, nspots, 
                                           n_maturities)
    prices = heston_price_vanillas(calibrated_features)
    dataset = noisyfier(prices)
    return dataset

def generate_dataset():
    data_subsets = []
    counter = 0
    for spot in spots:
        counter = counter + 1
        of_total = len(spots)
        spot = spot
        subset = generate_data_subset(spot, counter, of_total)
        data_subsets.append(subset)
    dataset = pd.concat(data_subsets, ignore_index=True)
    return dataset