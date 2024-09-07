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
from generate_ivols import generate_ivol_table


class data_generation():
    def __init__(self, lower_moneyness=None, upper_moneyness=None,
                 T=None, n_maturities = None, n_strikes = None, tl_ivol = None,
                 risk_free_rate = None, dividend_rate = None):
        self.lower_moneyness = lower_moneyness
        self.upper_moneyness = upper_moneyness
        self.T = T
        self.n_maturities = n_maturities
        self.n_strikes = n_strikes
        self.tl_ivol = tl_ivol
        self.risk_free_rate = risk_free_rate
        self.dividend_rate = dividend_rate

    def generate_data_subset(self,S):
        subset_spot = np.ones(1) * S
        K = np.linspace(S * self.lower_moneyness, S * self.upper_moneyness, 
                        self.n_strikes)
        def generate_features():
            features = pd.DataFrame(
                product(subset_spot, K, self.T),
                columns=[
                    "spot_price", 
                    "strike_price", 
                    "years_to_maturity"
                         ])
            return features
        features = generate_features()
        features['risk_free_rate'] = self.risk_free_rate
        features['dividend_rate'] = self.dividend_rate
        features['w'] = 1
        option_data = features
        option_data['calculation_date'] = ql.Date.todaysDate()
        option_data['maturity_date'] = option_data.apply(
            lambda row: row['calculation_date'] + ql.Period(
                int(math.floor(row['years_to_maturity'] * 365)), ql.Days), axis=1)
        return option_data
        
    
    def prepare_calibration(self, ivol_table, option_data, dividend_rate, risk_free_rate):
        day_count = ql.Actual365Fixed()
        calendar = ql.UnitedStates(m=1)
        
        calculation_date = option_data['calculation_date'][0]
        spot = option_data['spot_price'][0]
        ql.Settings.instance().evaluationDate = calculation_date
        
        
        dividend_yield = ql.QuoteHandle(ql.SimpleQuote(dividend_rate))
        dividend_rate = dividend_yield
        flat_ts = ql.YieldTermStructureHandle(ql.FlatForward(calculation_date, risk_free_rate, day_count))
        dividend_ts = ql.YieldTermStructureHandle(ql.FlatForward(calculation_date, dividend_rate, day_count))
        
        
        expiration_dates = option_data['maturity_date'].unique()
        
        strikes = option_data['strike_price'].unique()
    
        implied_vols = ql.Matrix(len(strikes), len(expiration_dates))
        
        for i in range(implied_vols.rows()):
            for j in range(implied_vols.columns()):
                implied_vols[i][j] = ivol_table[j][i]
        black_var_surface = ql.BlackVarianceSurface(
            calculation_date, calendar,
            expiration_dates, strikes,
            implied_vols, day_count)
        
        implied_vols_matrix = ql.Matrix(len(strikes), len(expiration_dates))
        for i in range(implied_vols_matrix.rows()):
            for j in range(implied_vols_matrix.columns()):
                implied_vols_matrix[i][j] = ivol_table[j][i]  
                
        return option_data,flat_ts,dividend_ts,spot,expiration_dates, \
            black_var_surface,strikes,day_count,calculation_date, calendar, \
                implied_vols_matrix

    



