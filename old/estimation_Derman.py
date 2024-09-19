# -*- coding: utf-8 -*-
"""
Created on Mon Sep 16 13:56:27 2024

"""

import os
import sys
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
os.chdir(current_dir)
sys.path.append(parent_dir)


import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression



pd.set_option('display.max_rows',None)
# pd.set_option('display.max_columns',None)
# pd.reset_option('display.max_rows')
pd.reset_option('display.max_columns')


class estimation_Derman():
    def __init__(self):
        from import_files import raw_ts
        from data_query import dirdata
        from settings import model_settings
        
        self.raw_ts = raw_ts
        self.K = np.sort(raw_ts.index.unique()).astype(int)
        self.T = np.sort(raw_ts.columns.unique()).astype(int)
        self.ms = model_settings()
        self.settings = self.ms.import_model_settings()
        self.xlsxs = dirdata()
        self.data_files = self.xlsxs
        self.dividend_rate = self.settings[0]['dividend_rate']
        self.risk_free_rate = self.settings[0]['risk_free_rate']
        self.security_settings = self.settings[0]['security_settings']
        self.ticker = self.security_settings[0]
        self.lower_moneyness = self.security_settings[1]
        self.upper_moneyness = self.security_settings[2]
        self.lower_maturity = self.security_settings[3]
        self.upper_maturity = self.security_settings[4]
        self.s = self.security_settings[5]
        self.day_count = self.settings[0]['day_count']
        self.calendar = self.settings[0]['calendar']
        self.calculation_date = self.settings[0]['calculation_date']
        
        print(f'\nspot price: {self.s}')
        print(f'\nstrikes: \n{self.K}')
        print(f'\nmaturities: \n{self.T}')
        

    def compute_one_derman_coef(self, t):
        
        term_struct = self.raw_ts.loc[:,int(t)].dropna()
        K_reg = term_struct.index
        atm_value = self.raw_ts.loc[self.s]
        
        x = np.array(K_reg  - self.s, dtype=float)
        y = np.array(term_struct  - atm_value[t],dtype=float)
        
        model = LinearRegression()
        x = x.reshape(-1,1)
        model.fit(x,y)
            
        b = model.coef_[0]
        alpha = model.intercept_
    
        return b, alpha
    
    def compute_derman_coefs(self):
        derman_coefs = {}
        atm_vols = self.raw_ts.loc[int(self.s)]
        for i, k in enumerate(self.K):
            for j, t in enumerate(self.T):
                atm_value = atm_vols[t]
                b, alpha = self.compute_one_derman_coef(t)
                derman_coefs[t] = [b, alpha, atm_value]
        derman_coefs = pd.DataFrame(derman_coefs)
        derman_coefs['coef'] = ['b','alpha','atm_value']
        derman_coefs.set_index('coef',inplace = True)
        return derman_coefs

