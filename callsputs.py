#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Sep  8 14:59:04 2024

"""
import os
pwd = str(os.path.dirname(os.path.abspath(__file__)))
os.chdir(pwd)
import numpy as np
import pandas as pd
from itertools import product
import QuantLib as ql
import math
import warnings
warnings.simplefilter(action='ignore')

# Adjust pandas settings to display all rows and columns
pd.set_option('display.max_rows', None)  # Display all rows
pd.set_option('display.max_columns', None)  # Display all columns

# Optionally, reset the settings to default
# pd.reset_option('display.max_rows')
# pd.reset_option('display.max_columns')

# from heston_calibration import calibrate_heston
# from pricing import heston_price_vanillas, noisyfier

from datapwd import dirdata
data_files = dirdata()

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
    return calls, puts

calls, puts = clean_data()


    
    
    
    
    
    
    
    
    

