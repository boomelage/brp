#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Sep  8 14:59:04 2024

A function that collects option data given there is an even number of columns
equally split between for calls and puts repsectively

"""
import os
pwd = str(os.path.dirname(os.path.abspath(__file__)))
os.chdir(pwd)
import pandas as pd
from datapwd import dirdata
import QuantLib as ql
import warnings
warnings.simplefilter(action='ignore')
data_files = dirdata()
import numpy as np
from heston_calibration import calibrate_heston
from pricing import heston_price_vanillas, noisyfier
calculation_date = ql.Date.todaysDate()
from surface_plotting import plot_vol_surface

# pd.set_option('display.max_rows', None)  # Display all rows
# pd.set_option('display.max_columns', None)  # Display all columns

pd.reset_option('display.max_rows')
pd.reset_option('display.max_columns')

dividend_rate = 0.00
risk_free_rate = 0.00
    
calls = pd.DataFrame()
puts = pd.DataFrame()
for file in data_files:
    octo = pd.read_excel(file)
    octo = octo.dropna()
    octo.columns = octo.iloc[0]
    octo = octo.drop(index = 0).reset_index().drop(
        columns = 'index')
    splitter = int(octo.shape[1]/2)
    octoputs = octo.iloc[:,:-splitter]
    octocalls = octo.iloc[:,:splitter]
    
    octocalls.loc[:,'w'] = 1
    calls = pd.concat([calls, octocalls], ignore_index=True)

calls = calls.sort_values(by='DyEx')
calls['DyEx'] = calls['DyEx'].astype(int)
calls['IVM'] = calls['IVM']/100
calls['maturity_date'] = calls.apply(
    lambda row: calculation_date + ql.Period(
        int(row['DyEx']/365), ql.Days), axis=1)
og_calls = calls.copy()

calls

maturities_days = calls['DyEx'].unique()

expiration_dates = np.empty(len(maturities_days),dtype=object)

for i in range(len(expiration_dates)):
    expiration_dates[i] = calculation_date + \
        ql.Period(int(maturities_days[i]), ql.Days)

expiration_dates

# ivol table generation
ivols = calls.copy().reset_index().drop(columns = ['index','w'])

ivols

def group_by_maturity(ivols):
    grouped = ivols.groupby('DyEx')
    group_arrays = []
    for _, group in grouped:
        group_array = group.values
        group_arrays.append(group_array)
    ivol_table = np.array(group_arrays, dtype=object)
    return ivol_table


ivol_table = group_by_maturity(ivols)

ivol_table
n_maturities = len(ivol_table)
n_strikes = len(ivol_table[0])

implied_vols_matrix = ql.Matrix(n_strikes,n_maturities,float(0))

for i in range(n_maturities):
    for j in range(n_strikes):
        implied_vols_matrix[j][i] = ivol_table[i][j][1]

maxK = int(max(calls['Strike']))
minK = int(min(calls['Strike']))
nKs = 7
S = np.median(calls['Strike'].unique())
K = np.linspace(minK,maxK,nKs)
T = expiration_dates
strikes = K
spot = S
maturities = expiration_dates

day_count = ql.Actual365Fixed()
day_count = ql.Actual365Fixed()
calendar = ql.UnitedStates(m=1)
ql.Settings.instance().evaluationDate = calculation_date
dividend_yield = ql.QuoteHandle(ql.SimpleQuote(dividend_rate))
dividend_rate = dividend_yield
flat_ts = ql.YieldTermStructureHandle(ql.FlatForward(
    calculation_date, risk_free_rate, day_count))
dividend_ts = ql.YieldTermStructureHandle(ql.FlatForward(
    calculation_date, dividend_rate, day_count))

features = og_calls.copy()
features['years_to_maturity'] = features['DyEx']/365
features['risk_free_rate'] = features['Rate']
features['volatility'] = features['IVM']
features['dividend_rate'] = dividend_rate
features = features.drop(columns = ['Rate','IVM','DyEx'])
option_data = features
option_data['spot_price'] = S
option_data['calculation_date'] = ql.Date.todaysDate()
option_data['maturity_date'] = option_data.apply(
    lambda row: row['calculation_date'] + ql.Period(
        int(row['years_to_maturity'] * 365), ql.Days), axis=1)

black_var_surface = ql.BlackVarianceSurface(
    calculation_date, calendar,
    expiration_dates, K,
    implied_vols_matrix, day_count)

heston_params = calibrate_heston(
    option_data,flat_ts,dividend_ts, spot, expiration_dates, 
    black_var_surface,strikes, day_count,calculation_date, calendar, 
    dividend_rate, implied_vols_matrix)

heston_vanillas = heston_price_vanillas(heston_params)

target_maturity = 1
target_mat_ivols = ivols[ivols['DyEx']==target_maturity]['IVM']

target_mat_ivols
surface = plot_vol_surface(heston_vanillas, target_mat_ivols, implied_vols_matrix, black_var_surface, strikes, maturities_days,target_maturity)


