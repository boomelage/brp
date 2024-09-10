#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Sep 10 12:40:38 2024

This class collects market data exported from the 'calls' tab in OMON

"""

from data_query import dirdata
import pandas as pd
import numpy as np

pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None)

# pd.reset_option('display.max_rows')
# pd.reset_option('display.max_columns')

data_files = dirdata()

data_files

class new_market_data_collection():
    def __init__(self):
        self.new_collect_market_data
        
    def new_collect_market_data(file):
        df = pd.read_excel(file)
        
        df = df.dropna().reset_index(drop=True)
        
        df = df[~(df['DyEx'] < 1)]
        
        df['DyEx']= df['DyEx'].astype(int)
        
        df['IVM']= df['IVM']/100
        
        df['DvYd']= df['DvYd']/100
        
        df['Rate']= df['Rate']/100
        
        return df
    
    
    def new_concat_market_data(self, data_files):
        market_data = pd.DataFrame()
        for file in data_files:
            df = self.new_collect_market_data(file)
            market_data = pd.concat([market_data, df], ignore_index=True)
            market_data = market_data.sort_values(by='DyEx')
            print(np.array(market_data['Strike'].unique()))
            print(file)
        market_data = market_data.reset_index(drop=True)
        market_data = market_data.set_index('Strike')
        return market_data
    
    def new_make_ivol_table(self, market_data):
        market_data = self.new_concat_market_data(data_files)
        ivol_table = np.empty(len(market_data['DyEx'].unique()),dtype=object)
        market_data_for_maturities = market_data.groupby('DyEx')
        
        maturities = np.array(market_data['DyEx'].unique(),dtype=int)
        
        
        for i, maturity in enumerate(maturities):  
            market_data_for_maturity = market_data_for_maturities.get_group(maturity)
            strikes = np.array(market_data_for_maturity.index)
            ivol_vector_for_maturity = np.zeros(len(strikes), dtype=float)
            
            for j, strike in enumerate(strikes):
                ivm_value = market_data_for_maturity.loc[strike, 'IVM']
                
                # If 'IVM' returns a series, take the first value or use aggregation
                if isinstance(ivm_value, pd.Series):
                    ivm_value = ivm_value.iloc[0]
                ivol_vector_for_maturity[j] = ivm_value
            ivol_vector_for_maturity = np.array(ivol_vector_for_maturity, dtype=float)
            ivol_table[i] = ivol_vector_for_maturity
        
        
        return market_data, ivol_table


