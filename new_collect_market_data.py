#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Sep 10 12:40:38 2024

This class collects market data exported from the 'calls' tab in OMON

"""

from data_query import dirdata
import pandas as pd
import numpy as np

class new_market_data_collection():
    def __init__(self):
        self.data_files = dirdata()
        self.market_data = pd.DataFrame()
        
    def new_collect_market_data(self, file):
        df = pd.read_excel(file)
        
        df = df.dropna().reset_index(drop=True)
        
        df = df[~(df['DyEx'] < 1)]
        
        df['DyEx']= df['DyEx'].astype(int)
        
        df['IVM']= df['IVM']/100
        
        df['DvYd']= df['DvYd']/100
        
        df['Rate']= df['Rate']/100
        
        print(f"\nfile:{file}")
        print(df.columns)
        print(df['DyEx'].unique())
        print(f"count: {len(df['DyEx'].unique())}")
        print(df['Strike'].unique())
        print(f"count: {len(df['Strike'].unique())}")
        
        return df
    
    
    def new_concat_market_data(self):
        market_data = pd.DataFrame()
        for file in self.data_files:
            df = self.new_collect_market_data(file)
            market_data = pd.concat([market_data, df], ignore_index=True)
            market_data = market_data.sort_values(by='DyEx')
        
        market_data = market_data.set_index('Strike')
        return market_data
    
    
    def new_make_ivol_table(self, market_data):
        maturities = market_data['DyEx'].unique()
        strikes = market_data['Strike'].unique()
        n_maturities = len(maturities)
        n_strikes = len(strikes)
        ivol_table = np.empty(n_maturities,dtype=object)
        market_data_for_maturities = market_data.groupby('DyEx')
               
        for i, maturity in enumerate(maturities):  
            market_data_for_maturity = market_data_for_maturities.get_group(maturity)
            strikes = np.array(market_data_for_maturity.index)
            ivol_vector_for_maturity = np.zeros(n_strikes, dtype=float)
            
            for j, strike in enumerate(strikes):
                ivm_value = market_data_for_maturity.loc[strike, 'IVM']
                
                if isinstance(ivm_value, pd.Series):
                    ivm_value = ivm_value.iloc[0]
                ivol_vector_for_maturity[j] = ivm_value

            ivol_table[i] = ivol_vector_for_maturity
            
        
        return ivol_table
    
