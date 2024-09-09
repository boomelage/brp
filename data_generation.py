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

def generate_dataset(
        S, lower_moneyness, upper_moneyness, nKs, risk_free_rate, data_T, 
        dividend_rate):
    subset_spot = np.ones(1) * S
    K = np.linspace(S * lower_moneyness, S * upper_moneyness, 
                    nKs)
    def generate_features():
        features = pd.DataFrame(
            product(subset_spot, K, data_T),
            columns=[
                "spot_price", 
                "strike_price", 
                "days_to_maturity"
                     ])
        return features
    features = generate_features()
    features['risk_free_rate'] = risk_free_rate
    features['dividend_rate'] = dividend_rate
    features['w'] = 1
    option_data = features
    option_data['calculation_date'] = ql.Date.todaysDate()
    option_data['maturity_date'] = option_data.apply(
        lambda row: row['calculation_date'] + ql.Period(
            int(math.floor(row['days_to_maturity'])), ql.Days), axis=1)
    return option_data
