# -*- coding: utf-8 -*-
"""
Created on Mon Sep  9 01:00:34 2024

"""

import os
pwd = str(os.path.dirname(os.path.abspath(__file__)))
os.chdir(pwd)
import pandas as pd
from datapwd import dirdata
import QuantLib as ql
import warnings
warnings.simplefilter(action='ignore')
import numpy as np
filename = dirdata()[0]

# pd.set_option('display.max_rows', None)  # Display all rows
# pd.set_option('display.max_columns', None)  # Display all columns

pd.reset_option('display.max_rows')
pd.reset_option('display.max_columns')

calculation_date = ql.Date.todaysDate()

raw = pd.read_excel(filename)

raw = raw.dropna()
raw.columns = raw.iloc[0]
raw = raw.iloc[1:,:]
strikes = raw['Strike'].to_numpy()
maturities = raw['DyEx'].iloc[0].to_numpy()
ivols = raw['IVM']



def collect_implied_vol_matrix(strikes,maturities,ivols):
    
    
    n_maturities = len(maturities)
    n_strikes = len(strikes)
    S = np.median(strikes)
    

    expiration_dates = np.empty(len(maturities),dtype=object)
    for i in range(len(expiration_dates)):
        expiration_dates[i] = calculation_date + \
            ql.Period(int(maturities[i]), ql.Days)


    implied_vol_matrix = ql.Matrix(n_strikes,n_maturities,float(0))
    for i in range(n_strikes):
        for j in range(n_maturities):
            implied_vol_matrix[i][j] = ivols.iloc[i,j]
          

    return n_maturities, n_strikes, S, expiration_dates, implied_vol_matrix

n_maturities, n_strikes, S, expiration_dates, implied_vol_matrix = \
    collect_implied_vol_matrix(strikes,maturities,ivols)
