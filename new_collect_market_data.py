#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Sep 10 12:40:38 2024

This class collects market data exported from the 'calls' tab in OMON

"""
import os
pwd = str(os.path.dirname(os.path.abspath(__file__)))
os.chdir(pwd)
import pandas as pd
import numpy as np

class option_data_from_market():
    def __init__(self,data_files):
        self.data_files = data_files
        self.market_data = pd.DataFrame()
        
    def collect_call_data(self, file):
        
        df = pd.read_excel(file)

        df = df.dropna().reset_index(drop=True)

        df.columns = df.loc[0]

        df = df.iloc[1:,:]

        df = df.astype(float)

        splitter = int(len(df.columns)/2)

        calls = df.iloc[:,:splitter]

        calls = calls[~(calls['DyEx'] < 1)]
    
        
        calls['spot_price'] = np.median(calls['Strike'])
        calls['volatility'] = calls['IVM'] / 100
        calls['dividend_rate'] = calls['DvYd'] / 100
        calls['risk_free_rate'] = calls['Rate'] / 100
        calls['days_to_expiry'] = calls['DyEx'].astype(int)
        calls['strike_price'] = calls['Strike'].astype(int)
        calls['w'] = 1
        calls = calls.drop(columns = ['IVM','DvYd','Rate','DyEx','Strike'])

        print(f"\nfile: {file}")
        print(calls.columns)
        print(calls['days_to_expiry'].unique())
        print(f"count: {len(calls['days_to_expiry'].unique())}")
        print(calls['strike_price'].unique())
        print(f"count: {len(calls['strike_price'].unique())}")

        return calls
# =============================================================================
# """
#     def collect_put_data(self, file):
#         
#         df = pd.read_excel(file)
# 
#         df = df.dropna().reset_index(drop=True)
# 
#         df.columns = df.loc[0]
# 
#         df = df.iloc[1:,:]
# 
#         df = df.astype(float)
# 
#         splitter = int(len(df.columns)/2)
# 
#         puts = df.iloc[:,splitter:]
# 
#         puts = puts[~(puts['DyEx'] < 1)]
#         
#         puts['DyEx'] = puts['DyEx'].astype(int)
#         puts['IVM'] = puts['IVM'] / 100
#         puts['DvYd'] = puts['DvYd'] / 100
#         puts['Rate'] = puts['Rate'] / 100
# 
#         print(f"\nfile: {file}")
#         print(puts.columns)
#         print(puts['DyEx'].unique())
#         print(f"count: {len(puts['DyEx'].unique())}")
#         print(puts['Strike'].unique())
#         print(f"count: {len(puts['Strike'].unique())}")
# 
#         return puts
# """
# =============================================================================
    
    def concat_option_data(self):
        market_data = pd.DataFrame()
        for file in self.data_files:
            df = self.collect_call_data(file)
            market_data = pd.concat([market_data, df], ignore_index=True)
            market_data = market_data.sort_values(by='days_to_expiry')
            market_data = market_data.reset_index(drop=True)
        return market_data

from data_query import dirdata
data_files = dirdata(r'SPXts.xlsx')
file = data_files[0]

nmdc = option_data_from_market(data_files=data_files)

market_data = nmdc.concat_option_data()



