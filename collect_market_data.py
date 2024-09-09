# -*- coding: utf-8 -*-
"""
Created on Mon Sep  9 11:28:35 2024

@author: boomelage
"""

import os
pwd = str(os.path.dirname(os.path.abspath(__file__)))
os.chdir(pwd)
import pandas as pd
from data_query import dirdata

# pd.set_option('display.max_rows', None)  # To display all rows
# pd.set_option('display.max_columns', None)  # To display all columns
pd.reset_option('display.max_rows')  # To display all rows
pd.reset_option('display.max_columns')  # To display all columns

# =============================================================================
                                                                     # cleaning 

def clean_data(file):
    df = pd.read_excel(file)
    df.columns = df.loc[0]
    df = df.iloc[1:,:]
    
    df['DvYd'] =  df['DvYd'].fillna(0)
    df = df.dropna().copy()
    df = df.drop_duplicates()
    df = df.reset_index(drop=True)
    splitter = int(df.shape[1]/2)
    dfcalls_subset = df.iloc[:,:splitter]
    
    return dfcalls_subset


# =============================================================================
                                                           # concatinating data

def concat_data(data_files):
    dataset = pd.DataFrame()  
    for file in data_files:
        dfcalls_subset = clean_data(file)
        dataset = pd.concat([dataset, dfcalls_subset], ignore_index=True)
        # dfcalls_subset['Strike'] = dfcalls_subset['Strike']
        # dfcalls_subset['DyEx'] = dfcalls_subset['DyEx']
        # dfcalls_subset['IVM'] = dfcalls_subset['IVM']/100
        # dfcalls_subset['Rate'] = dfcalls_subset['Rate']/100
        # dfcalls_subset['DvYd'] = dfcalls_subset['DvYd']/100
    return dataset


subset = clean_data(dirdata()[0])
subset


