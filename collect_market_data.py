# -*- coding: utf-8 -*-
"""
Created on Mon Sep  9 11:28:35 2024

@author: boomelage
"""

import os
pwd = str(os.path.dirname(os.path.abspath(__file__)))
os.chdir(pwd)
import pandas as pd
import numpy as np                                                     
pd.set_option('future.no_silent_downcasting', True)
import QuantLib as ql
from pricing import heston_price_vanillas, noisyfier
from data_query import dirdata

dividend_rate = 0.00
risk_free_rate = 0.00

calculation_date = ql.Date.todaysDate()
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

def clean_data(file):
    df = pd.read_excel(file)
    df.columns = df.loc[0]
    df = df.iloc[1:,:]
    df['DvYd'] = df['DvYd'].fillna(0).infer_objects(copy=False)
    df = df.dropna().copy()
    df = df.drop_duplicates()
    df = df.reset_index(drop=True)
    splitter = int(df.shape[1]/2)
    dfcalls_subset = df.iloc[:,:splitter]
    subset_spot_price = np.median(np.array(dfcalls_subset['Strike'].unique()))
    dfcalls_subset['spot_price'] = subset_spot_price
    dfcalls_subset.rename(columns={'Strike': 'strike_price'}, inplace=True)
    dfcalls_subset.rename(columns={'IVM': 'volatility'}, inplace=True)
    dfcalls_subset.rename(columns={'DyEx': 'days_to_maturity'}, inplace=True)
    dfcalls_subset.rename(columns={'Rate': 'risk_free_rate'}, inplace=True)
    dfcalls_subset.rename(columns={'DvYd': 'dividend_rate'}, inplace=True)
    dfcalls_subset['calculation_date'] = ql.Date.todaysDate()
    def calculate_maturity_date(row, calc_date):
        return calc_date + ql.Period(int(row['days_to_maturity']), ql.Days)
    dfcalls_subset['maturity_date'] = dfcalls_subset.apply(
        calculate_maturity_date,calc_date=ql.Date.todaysDate(),axis=1)
    dfcalls_subset['years_to_maturity'] = dfcalls_subset['days_to_maturity']/365
    
    
    
    

    return dfcalls_subset


def concat_data(data_files):
    dataset = pd.DataFrame()  
    for file in data_files:
        try:
            dfcalls_subset = clean_data(file)
            dataset = pd.concat([dataset, dfcalls_subset], ignore_index=True)
        except Exception as e:
            print(f'problem with {file}: {e}')

    return dataset

dataset = concat_data(dirdata())

dataset