# -*- coding: utf-8 -*-
"""
Created on Mon Sep  9 11:28:35 2024

@author: boomelage
"""

import os
pwd = str(os.path.dirname(os.path.abspath(__file__)))
os.chdir(pwd)
import numpy as np

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
            
    return maturities, strikes, S, n_maturities, n_strikes, ivol_table



