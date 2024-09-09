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
file = data_files[0]


# =============================================================================
                                                                     # cleaning 
df = pd.read_excel(file)
df = df.dropna()
df.columns = df.loc[0]

df = df.iloc[1:,:].reset_index(drop=True)

splitter = int(df.shape[1]/2)
dfcalls = df.iloc[:,:splitter]

dfcalls['DyEx'] = dfcalls['DyEx'].astype(int)
dfcalls['Strike'] = dfcalls['Strike'].astype(int)


# =============================================================================
                                                       # extracting option data
dfcalls


# =============================================================================
                                               # extracting term structure data
                                               
maturities = np.unique(np.sort(dfcalls['DyEx']))
strikes = np.unique(np.sort(dfcalls['Strike']))
S = int(np.median(strikes))

n_maturities = len(maturities)
n_strikes = len(strikes)










dfts = dfcalls.groupby('DyEx')
ivol_table = np.empty(n_maturities,dtype=object)
for i in range(n_maturities):
    maturity = maturities[i]
    dfts_at_maturity = dfts.get_group(maturity)
    dfts_at_maturity = dfts_at_maturity.sort_values(by='Strike')
    dfts_at_maturity = dfts_at_maturity.drop(columns = ['DyEx','Rate','Strike'])
    dfts_at_maturity = dfts_at_maturity.to_numpy().flatten()
    ivols_at_maturity = np.empty(n_strikes)
    for j in range(n_strikes):
        ivols_at_maturity[j] = dfts_at_maturity[j]
    ivol_table[i] = ivols_at_maturity
        

# .pivot(index = 'DyEx',  columns = 'Strike', values = 'IVM')
# ivol_table = np.empty(n_maturities,dtype=object)
# for i in range(n_maturities):
#     ivols = np.empty(n_strikes)
#     for j in range(n_strikes):

#         ivols[i] =  dfts.get_group(maturities[i])[]


# dfts