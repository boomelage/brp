#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Sep 11 19:08:30 2024

"""
import os
import sys
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
os.chdir(current_dir)
sys.path.append(parent_dir)

from data_query import dirdata
import pandas as pd
import numpy as np

xlsxs = dirdata()
data_files = xlsxs

pd.set_option('display.max_columns',None)

raw_call_vols = pd.DataFrame()
raw_put_vols = pd.DataFrame()



for file in data_files:
    df = pd.read_excel(file, engine= 'openpyxl')
    
    df.columns = df.loc[1]
    
    df = df.iloc[2:,:].set_index('Strike',drop=True)
    
    IVMs = df.loc[:,'IVM']
    
    n_raw_cols = IVMs.shape[1]
    
    
    
    raw_maturities = pd.unique(df['DyEx'].values.ravel())
 
    
    call_col_idxs = np.arange(0,n_raw_cols,2)
    
    put_col_idxs = np.arange(1,n_raw_cols,2)
    
        
    callvols = IVMs.iloc[:,call_col_idxs].copy()
    
    
    callvols.columns = raw_maturities
    
    # putvols = IVMs.iloc[:,put_col_idxs].copy()
    # putvols.columns = raw_maturities
    
    
    # raw_call_vols = pd.concat(
        
    #     [raw_call_vols, callvols], axis = 0
        # )
        
        

    


# print(f"\n\ncalls:\n{callvols}")
# print(f"\n\nputs:\n{putvols}")

