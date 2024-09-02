#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Sep  2 23:26:57 2024

@author: doomd
"""

import pandas as pd
import numpy as np
import math
import os
import QuantLib as ql
from itertools import product
script_dir = os.path.dirname(os.path.abspath(__file__))
os.chdir(script_dir)

def generate_features():
    S = np.linspace(50, 500, 100)
    K = np.linspace(25, 750, 50)
    r = np.arange(0, 0.051, 0.01)
    T = np.arange(3/12, 2.01, 1/12)
    sigma = np.arange(0.1, 0.81, 0.1)
    w = np.array((-1,1))

    features = pd.DataFrame(
        product(S, K, r, T, sigma, w),
        columns=["spot_price", 
                  "strike_price", 
                  "risk_free_rate", 
                  "years_to_maturity", 
                  "volatility", 
                  "w"]
    )

    feature_names = features.copy()
    feature_names = features.columns
    return features, feature_names



def generate_small_features():
    S = np.array((90, 100, 110))
    K = np.array((80, 100, 120))
    r = np.ones(3)*0.05
    T = np.ones(3)
    sigma = np.array((0.1, 0.81))
    w = np.array((-1,1))

    features = pd.DataFrame(
        product(S, K, r, T, sigma, w),
        columns=["spot_price", 
                  "strike_price", 
                  "risk_free_rate", 
                  "years_to_maturity", 
                  "volatility", 
                  "w"]
    )
    feature_names = features.copy()
    feature_names = features.columns
    return features, feature_names
    


def generate_qldates(features):
    features['calculation_date'] = ql.Date.todaysDate()
    features['maturity_date'] = features.apply(
        lambda row: row['calculation_date'] + ql.Period(
            int(math.floor(row['years_to_maturity'] * 365)), ql.Days), axis=1)
    return features


