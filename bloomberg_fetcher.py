#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Sep  8 14:59:04 2024

A function that collects option data given there is an even number of columns
equally split between for calls and puts repsectively

"""
import pandas as pd
import numpy as np
import QuantLib as ql

# def fetch_bbdata(data_files, calculation_date):
    
calculation_date = ql.Date.todaysDate()
from data_query import dirdata    
data_files = dirdata()
calls = pd.DataFrame()
# puts = pd.DataFrame()
for file in data_files:
    octo = pd.read_excel(file)
    octo = octo.dropna()
    octo.columns = octo.iloc[0]
    octo = octo.drop(index = 0).reset_index().drop(
        columns = 'index')
    splitter = int(octo.shape[1]/2)
    # octoputs = octo.iloc[:,:-splitter]
    octocalls = octo.iloc[:,:splitter]
    
    octocalls.loc[:,'w'] = 1
    calls = pd.concat([calls, octocalls], ignore_index=True)



calls = calls.sort_values(by='DyEx')


calls
# calls['DyEx'] = calls['DyEx'].astype(int)
# calls['IVM'] = calls['IVM']/100
# calls['maturity_date'] = calls.apply(
#     lambda row: calculation_date + ql.Period(
#         int(row['DyEx']/365), ql.Days), axis=1)

# ivols = octocalls['IVM'].to_numpy()

# strikes = octocalls['Strike'].unique()
# n_strikes = len(strikes)

# maturities = octocalls['DyEx'].unique().astype(int)
# n_maturities = len(maturities)

# ivol_table = np.empty(n_maturities,dtype=object)

# for i in range(n_maturities):
#     strikevols = np.empty(n_strikes)
#     for j in range(n_strikes):
        
        




    # return calls, strikes, maturities, ivols