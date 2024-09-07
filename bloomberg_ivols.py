# -*- coding: utf-8 -*-
"""
Created on Sat Sep  7 15:38:52 2024

@author: boomelage
"""
import os
os.chdir(os.path.dirname(os.path.abspath(__file__)))
import numpy as np
import pandas as pd
from itertools import product
import QuantLib as ql
import math

risk_free_rate = 0.00
dividend_rate = 0.00

bbivols = pd.read_excel(r'22000 AAPL.xlsx')

bbivols.columns = bbivols.iloc[1]

bbivols = bbivols.drop([0,1])
bbivols = bbivols.reset_index(drop=True)
bbivols = bbivols.dropna()

maturities = bbivols.filter(like='DyEx').iloc[0].unique()
strikes = bbivols.filter(like='Strike').to_numpy()

n_maturities = len(maturities)
n_strikes = len(strikes)

bbiv_df = bbivols.filter(like='IVM')
bbivs = bbiv_df.to_numpy()

n_bbivs = bbivs.shape[0]

ivol_table = np.empty(n_maturities, dtype=object)
for i in np.arange(1,n_bbivs,2):
    ivol_table[int((i+1)/2)-1] = (bbivs[i]+bbivs[i+1])/2



S = [int(np.median(strikes))]
K = strikes
T = maturities
def generate_features():
    features = pd.DataFrame(
        product(S, K, T),
        columns=[
            "spot_price", 
            "strike_price", 
            "years_to_maturity"
                 ])
    return features
features = generate_features()
features["years_to_maturity"] = features["years_to_maturity"]/365
features['risk_free_rate'] = risk_free_rate
features['dividend_rate'] = dividend_rate
features['w'] = 1
option_data = features
option_data['calculation_date'] = ql.Date.todaysDate()
option_data['maturity_date'] = option_data.apply(
    lambda row: row['calculation_date'] + ql.Period(
        int(math.floor(row['years_to_maturity'] * 365)), ql.Days), axis=1)

prepare_calibration(self, ivol_table, option_data, dividend_rate, risk_free_rate):
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