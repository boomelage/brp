# -*- coding: utf-8 -*-
"""
Created on Mon Sep  9 11:28:35 2024

@author: boomelage
"""

import os
pwd = str(os.path.dirname(os.path.abspath(__file__)))
os.chdir(pwd)
from data_query import dirdata
import pandas as pd
import numpy as np


pd.set_option('display.max_rows', None)  # To display all rows
pd.set_option('display.max_columns', None)  # To display all columns

# =============================================================================
                                                           # obtaining filename
data_files = dirdata()

# =============================================================================
                                                                     # cleaning 

def clean_data(file):
    df = pd.read_excel(file)
    df = df.dropna()
    df.columns = df.loc[0]
    
    df = df.iloc[1:,:].reset_index(drop=True)
    
    splitter = int(df.shape[1]/2)
    dfcalls_subset = df.iloc[:,:splitter]
    
    dfcalls_subset.loc[:,'Strike'] = dfcalls_subset.loc[:,'Strike'].astype(int)
    dfcalls_subset.loc[:,'DyEx'] = dfcalls_subset.loc[:,'DyEx'].astype(int)
    dfcalls_subset.loc[:,'IVM'] = dfcalls_subset.loc[:,'IVM']/100
    
    return dfcalls_subset

# =============================================================================
                                                           # concatinating data
def concat_data(data_files):
    dfcalls = pd.DataFrame()
    for file in data_files:
        dfcalls_subset = clean_data(file)
        dfcalls.concat(dfcalls_subset)
    return dfcalls

# =============================================================================
def make_ivol_vector(dfcalls):                              
    maturities = np.unique(np.sort(dfcalls['DyEx']))
    strikes = np.unique(np.sort(dfcalls['Strike']))
    S = int(np.median(strikes))
    
    n_maturities = len(maturities)
    n_strikes = len(strikes)
    
    n_maturities = len(maturities)
    n_strikes = len(strikes)
    
    ivol_table = np.empty(n_maturities, dtype=object)  
    for i in range(n_maturities):
        ivol_table[i] = np.empty(n_strikes)
    
    
    dfts = dfcalls.groupby('DyEx')
    
    
    for i in range(n_maturities):
        maturity = maturities[i]
        dfts_at_maturity = dfts.get_group(maturity)
        dfts_at_maturity = dfts_at_maturity.sort_values(by='Strike')
        dfts_at_maturity = dfts_at_maturity.drop(
            columns = ['DyEx','Rate','Strike'])
        dfts_at_maturity = dfts_at_maturity.to_numpy().flatten()
        for j in range(n_strikes): 
                ivol_table[i][j] = dfts_at_maturity[j]
            
    return dfcalls, maturities, strikes, S, n_maturities, n_strikes, ivol_table

