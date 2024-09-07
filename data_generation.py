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
from heston_calibration import heston_calibration
from pricing import heston_price_vanillas, noisyfier

class data_generation():
    def __init__(self, lower_moneyness=None, upper_moneyness=None,
                 option_data=None, T=None, n_maturities = None,
                 n_strikes = None, tl_ivol = None, risk_free_rate = None,
                 dividend_rate = None):
        self.lower_moneyness = lower_moneyness
        self.upper_moneyness = upper_moneyness
        self.option_data = option_data
        self.T = T
        self.n_maturities = n_maturities
        self.n_strikes = n_strikes
        self.tl_ivol = tl_ivol
        self.risk_free_rate = risk_free_rate
        self.dividend_rate = dividend_rate
        
    def generate_data_subset(self,S):
        subset_spot = np.ones(1) * S
        K = np.linspace(S * self.lower_moneyness, S * self.upper_moneyness, self.n_strikes)
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
        n_lists = self.n_maturities
        n_elements = self.n_strikes
        decay_rate = 1/(10*self.n_maturities*self.n_strikes)
        row_decay = decay_rate/10
        ivol_table = generate_ivol_table(n_lists, n_elements, self.tl_ivol, 
                                   decay_rate, row_decay)
        features['risk_free_rate'] = self.risk_free_rate
        features['dividend_rate'] = self.dividend_rate
        features['w'] = 1
        option_data = features
        option_data['calculation_date'] = ql.Date.todaysDate()
        option_data['maturity_date'] = option_data.apply(
            lambda row: row['calculation_date'] + ql.Period(
                int(math.floor(row['years_to_maturity'] * 365)), ql.Days), axis=1)
        return ivol_table, option_data
        
    
    
    
    def calibrate_data_subset(self, ivol_table, option_data):
        option_data,flat_ts,dividend_ts,spot,expiration_dates, \
            black_var_surface,strikes,day_count,calculation_date, calendar, \
               implied_vols_matrix = \
                    heston_calibration.prepare_heston_calibration(
                        ivol_table,
                        option_data, 
                        self.dividend_rate,
                        self.risk_free_rate)
        
        heston_params = heston_calibration.calibrate_heston(option_data,
            flat_ts,dividend_ts,spot,expiration_dates,black_var_surface,
            strikes,day_count,calculation_date, calendar,self.dividend_rate, 
            implied_vols_matrix)
        
        prices = heston_price_vanillas(heston_params)
        
        calibrated_subset = noisyfier(prices)
        return calibrated_subset
    
    
    
    def calibrate_dataset(self,spots):
        data_subsets = []
        counter_spot = 0
        for spot in spots:
            counter_spot = counter_spot + 1
            spot = spot
            subset = data_generation.calibrate_data_subset(spot,self.option_data)
            data_subsets.append(subset)
        dataset = pd.concat(data_subsets, ignore_index=True)
        return dataset


