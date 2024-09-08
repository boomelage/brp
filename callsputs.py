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


pd.set_option('display.max_rows', None)  # Display all rows
pd.set_option('display.max_columns', None)  # Display all columns

# pd.reset_option('display.max_rows')
# pd.reset_option('display.max_columns')


def clean_data():
    calls = pd.DataFrame()
    puts = pd.DataFrame()
    for file in data_files:
        octo = pd.read_excel(f"{str(file)}")
        octo = octo.dropna()
        octo.columns = octo.iloc[0]
        octo = octo.drop(index = 0).reset_index().drop(
            columns = 'index')
        splitter = int(octo.shape[1]/2)
        octoputs = octo.iloc[:,:-splitter]
        octocalls = octo.iloc[:,:splitter]
        octocalls.loc[:,'w'] = 1
        octoputs.loc[:,'w'] = -1
        calls = pd.concat([calls, octocalls], ignore_index=True)
        puts = pd.concat([puts, octoputs], ignore_index=True)
        calls = calls.sort_values(by = 'Strike')
        puts = puts.sort_values(by = 'Strike')
    calls['IVM'] = calls['IVM']/100
    puts['IVM'] = puts['IVM']/100
    return calls, puts

def group_by_maturity(df):
    grouped = df.groupby('DyEx')
    group_arrays = []
    for _, group in grouped:
        group_array = group.values
        group_arrays.append(group_array)
    final_array = np.array(group_arrays, dtype=object)
    return final_array



# full option data
calls, puts = clean_data()


# ivol table generation

callvols = calls.copy().drop(columns = ['w','Rate'])
groupedmat = group_by_maturity(callvols)
n_maturities = int(len(groupedmat))
n_strikes = int(len(groupedmat[0]))

implied_vols_matrix = ql.Matrix(n_strikes,n_maturities,float(0))

for i in range(n_maturities):
    for j in range(n_strikes):
        implied_vols_matrix[j][i] = groupedmat[i][j][1]




print(implied_vols_matrix)


