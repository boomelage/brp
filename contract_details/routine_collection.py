#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Sep 10 12:40:38 2024

This class collects market data exported from the 'calls/puts' tab in OMON

"""

import os
import sys
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
os.chdir(current_dir)
sys.path.append(parent_dir)

import pandas as pd
import numpy as np
import QuantLib as ql
from data_query import dirdata
xlsxs = dirdata()

pd.set_option('display.max_columns', None)
# pd.set_option('display.max_rows', None)
pd.reset_option('display.max_rows')
# pd.reset_option('display.max_columns')

class routine_collection():
    def __init__(self):
        self.data_files = xlsxs
        self.market_data = pd.DataFrame()
        self.excluded_file = None
        
    def collect_call_data(self,file):
        try:
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
            calls['days_to_maturity'] = calls['DyEx'].astype(int)
            calls['strike_price'] = calls['Strike'].astype(int)
            calls['w'] = 1
            calls = calls.drop(columns = ['IVM','DvYd','Rate','DyEx','Strike'])
    
            print(f"\nfile: {file}")
            print(calls['days_to_maturity'].unique())
            print(f"maturities count: {len(calls['days_to_maturity'].unique())}")
            print(calls['strike_price'].unique())
            print(f"strikes count: {len(calls['strike_price'].unique())}")
    
            return calls
        except Exception:
            error_tag = f'file error: {file}'
            print('\n')
            print("#"*len(error_tag))
            print("-"*len(error_tag))
            print(error_tag)
            print("-"*len(error_tag))
            print("#"*len(error_tag))
            
    def collect_data(self,file):
        try:
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
            calls['days_to_maturity'] = calls['DyEx'].astype(int)
            calls['strike_price'] = calls['Strike'].astype(int)
            calls['w'] = 'call'
            calls = calls.drop(columns = ['IVM','DvYd','Rate','DyEx','Strike'])
            print(f"\nfile: {file}")
            print(calls['days_to_maturity'].unique())
            print(f"maturities count: {len(calls['days_to_maturity'].unique())}")
            print(calls['strike_price'].unique())
            print(f"strikes count: {len(calls['strike_price'].unique())}")
            puts = df.iloc[:,splitter:]
            puts = puts[~(puts['DyEx'] < 1)]
            puts['spot_price'] = np.median(puts['Strike'])
            puts['volatility'] = puts['IVM'] / 100
            puts['dividend_rate'] = puts['DvYd'] / 100
            puts['risk_free_rate'] = puts['Rate'] / 100
            puts['days_to_maturity'] = puts['DyEx'].astype(int)
            puts['strike_price'] = puts['Strike'].astype(int)
            puts['w'] = 'put'
            puts = puts.drop(columns = ['IVM','DvYd','Rate','DyEx','Strike'])
            print(f"\nfile: {file}")
            print(puts['days_to_maturity'].unique())
            print(f"maturities count: {len(puts['days_to_maturity'].unique())}")
            print(puts['strike_price'].unique())
            print(f"strikes count: {len(puts['strike_price'].unique())}")
            return calls, puts
        
        except Exception:
            error_tag = f'file error: {file}'
            print('\n')
            print("#"*len(error_tag))
            print("-"*len(error_tag))
            print(error_tag)
            print("-"*len(error_tag))
            print("#"*len(error_tag))


    def concat_option_data(self):
        market_data = pd.DataFrame()
        for file in self.data_files:
            df = self.collect_call_data(file)
            market_data = pd.concat([market_data, df], ignore_index=True)
            market_data = market_data.sort_values(by='days_to_maturity')
            market_data = market_data.reset_index(drop=True)
        return market_data
    
    def concat_data(self):
        market_calls = pd.DataFrame()
        market_puts = pd.DataFrame()
        for file in self.data_files:
            try:
                calls, puts = self.collect_data(file)
                market_calls = pd.concat([market_calls,calls])
                market_puts = pd.concat([market_puts,puts])
                
                # market_data = market_data.sort_values(by='days_to_maturity')
                # market_data = market_data.reset_index(drop=True)
            except Exception:
                error_tag = f'file error: {file}'
                print('\n')
                print("#"*len(error_tag))
                print("-"*len(error_tag))
                print(error_tag)
                print("-"*len(error_tag))
                print("#"*len(error_tag))
        market_calls = market_calls.reset_index(drop=True)
        market_puts = market_puts.reset_index(drop=True)
        return {'calls':market_calls,'puts':market_puts}
    

rc = routine_collection()

contract_details = rc.concat_data()