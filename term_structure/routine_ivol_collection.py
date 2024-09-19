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

raw_call_ts = pd.DataFrame()
raw_put_ts = pd.DataFrame()
for file in data_files:
    df = pd.read_excel(file, engine= 'openpyxl')
    df.columns = df.loc[1]
    df = df.iloc[2:,:].reset_index(drop=True)
    
    df = df.set_index('Strike')
    
    df_strikes = df.index.tolist()
    
    df_maturities = pd.unique(df['DyEx'].values.ravel())
    
    
    
    callvols = pd.concat(
        [df.iloc[:, i:i+2] for i in range(0, df.shape[1], 4)], axis=1)
    callvols['Strike'] = df_strikes
    
    putvols = pd.concat(
        [df.iloc[:, i:i+2] for i in range(1, df.shape[1], 4)], axis=1)
    putvols['Strike'] = df_strikes
    
    
    raw_call_ts, raw_put_ts = pd.concat(
        [raw_call_ts,callvols]), pd.concat([raw_put_ts,putvols])
    
    
    
raw_put_ts

raw_market_ts = raw_put_ts.copy()
def clean_raw_ts(raw_market_ts):
    ts_columns = []
    ivm_count = 1
    dyex_count = 1
    
    for col in raw_market_ts.columns:
        if col == 'IVM':
            ts_columns.append(f'IVM_{ivm_count}')
            ivm_count += 1
        elif col == 'DyEx':
            ts_columns.append(f'DyEx_{dyex_count}')
            dyex_count += 1
        else:
            ts_columns.append(col)
    
    raw_market_ts.columns = ts_columns
    
    
    # Step 1: Reshape the DataFrame using pd.melt
    df_melted = pd.melt(raw_market_ts, 
                        id_vars=['Strike'], 
                        value_vars=['IVM_1', 'IVM_2', 'IVM_3', 'IVM_4'],
                        var_name='IVM_label', 
                        value_name='IVM')
    
    df_melted_dyex = pd.melt(raw_market_ts, 
                              id_vars=['Strike'], 
                              value_vars=['DyEx_1', 'DyEx_2', 'DyEx_3', 'DyEx_4'],
                              var_name='DyEx_label', 
                              value_name='DyEx')
    df_combined = pd.concat(
        [df_melted[['Strike', 'IVM']], df_melted_dyex['DyEx']], axis=1)
    df_combined = df_combined.dropna()
    df_indexed = df_combined.set_index(['Strike', 'DyEx'])
    df_indexed = df_indexed.sort_index()
    df_indexed
    
    
    T = np.sort(df_combined["DyEx"].unique())  
    T = T[T > 0]
    Ks = np.sort(df_combined["Strike"].unique())
    raw_ts_np = np.zeros((len(Ks) , len(T)), dtype=float)
    
    for i, k in enumerate(Ks):
        for j, t in enumerate(T):
            try:
                raw_ts_np[i][j] = df_indexed.loc[(k, t), 'IVM'].iloc[0]
            except Exception:
                raw_ts_np[i][j] = np.nan
            
    raw_ts_df = pd.DataFrame(raw_ts_np)
    raw_ts_df.columns = T
    raw_ts = raw_ts_df.set_index(Ks)
    
    raw_ts = raw_ts.replace(0,np.nan)
    raw_ts = raw_ts.dropna(how = 'all', axis = 0)
    raw_ts = raw_ts.dropna(how = 'all', axis = 1)
    raw_ts = raw_ts/100
    raw_ts.columns = raw_ts.columns.astype(int)
    raw_ts.index = raw_ts.index.astype(int)
    
    pd.set_option('display.max_rows',None)
    pd.set_option('display.max_columns',None)
    print(f'\nterm structure collected:\n\n{raw_ts}\n')
    pd.reset_option('display.max_rows')
    pd.reset_option('display.max_columns')
    os.chdir(parent_dir)
    return raw_ts

raw_calls = clean_raw_ts(raw_call_ts)
raw_puts = clean_raw_ts(raw_put_ts)