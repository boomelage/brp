#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Sep 12 20:52:50 2024

"""
import os
pwd = str(os.path.dirname(os.path.abspath(__file__)))
os.chdir(pwd)



"""
"""


import pandas as pd
from data_query import dirdatacsv
import numpy as np
from sklearn.linear_model import LinearRegression
pd.set_option('display.max_columns',None)

class derman():
    def __init__(self, derman_coefs = None, data_files = dirdatacsv()):
        self.data_files = data_files
        self.mats = []
        self.derman_coefs = derman_coefs
        
    def retrieve_ts(self):
        
        file = self.data_files[0]
        try:
           ts = pd.read_csv(file)
           ts = ts.set_index(ts.iloc[:, 0]).drop(columns=ts.columns[0])
           ts = ts.astype(float)
           ts.columns = ts.columns.astype(int)
        except Exception:
            print("check working directory files!")
        ts = ts.loc[
            
            5540:5640,
            :
    
            ]
        ts = ts.loc[:, (ts != 0).any(axis=0)]
    
        ks = ts.index.tolist()
        mats = ts.columns.tolist()
        ts = ts/100
        return ks, mats, ts
        
    def compute_derman_ivols(self,maturity,ts):
        TSatmat = ts.loc[:,maturity]
        strikes = ts.index
        S = int(np.median(strikes))
        x = np.array(strikes - S,dtype=float)
        atmvol = np.median(TSatmat)
        y = np.array(TSatmat - atmvol,dtype=float)
        model = LinearRegression()
        x = x.reshape(-1,1)
        model.fit(x,y)
        b = model.coef_[0]
        alpha = model.intercept_
        derman_ivols = model.predict(x)/100
        derman_ivols = derman_ivols*b + alpha + atmvol
        return b, alpha, atmvol, derman_ivols
        
    def get_derman_coefs(self):
        derman_coefs = {}
        ks, mats, ts = self.retrieve_ts()
        for i, maturity in enumerate(mats):
            for j, k in enumerate(ks):
                b, alpha, atmvol, derman_ivols = self.compute_derman_ivols(maturity,ts)
                derman_coefs[int(f"{maturity}")] = [b, alpha, atmvol]
        derman_coefs = pd.DataFrame(derman_coefs)
        derman_coefs['coef'] = ['b','alpha','atmvol']
        derman_coefs.set_index('coef',inplace = True)
        derman_coefs = derman_coefs.loc[:, derman_coefs.loc['atmvol'] != 0]
        return derman_coefs
    
    def derman_ivols_for_market(self,df,derman_coefs):
        b = derman_coefs.loc['b']
        alpha = derman_coefs.loc['alpha']
        K = df['strike_price']
        S = df['spot_price']
        iv = df['atm_vol']
        df['volatility'] = \
            iv + \
                alpha[df['days_to_maturity']] +\
                    b[df['days_to_maturity']]*(K-S)
        return df
    
            
    def compute_one_derman_vol(self,s,k,t,atm_vol):
        moneyness = k-s
        b = self.derman_coefs.loc['b',t]
        alpha = self.derman_coefs.loc['alpha',t]
        derman_vol =  alpha + atm_vol + b*moneyness
        return derman_vol
        
    
    def make_derman_df(self, s, K, T, atm_vol):
        derman_numpy = np.zeros((len(K), len(T)), dtype=float)
        derman_df = pd.DataFrame(derman_numpy)
        derman_df.index = K
        derman_df.columns = T
        for k in K:
            for t in T:
                try:
                    moneyness = k - s
                    b = self.derman_coefs.loc['b', t]
                    alpha = self.derman_coefs.loc['alpha', int(t)]
                    derman_df.loc[k, t] = atm_vol + alpha + moneyness * b
                except Exception:
                    pass
        return derman_df

    def retrieve_derman_from_csv(self):
        derman_coefs = pd.read_csv(r'derman_coefs.csv')
        derman_coefs = derman_coefs.set_index('coef')
        derman_coefs.columns = derman_coefs.columns.astype(int)
        derman_maturities = derman_coefs.columns
        return derman_coefs, derman_maturities
    
    def make_derman_df_for_S(self, s, K, T, atm_vol, contract_details):
        def make_for_s(s, K, T, atm_vol, self,contract_details):
            derman_coefs, derman_maturities = self.retrieve_derman_from_csv()
            contract_details = contract_details[
                contract_details['days_to_maturity'].isin(derman_maturities)
            ].reset_index(drop=True)
    
            # Create the derman DataFrame
            derman_df_for_s = self.make_derman_df(s, K, T, atm_vol)
            return derman_df_for_s
    
        # Call make_for_s to get derman DataFrame
        derman_df_for_s = make_for_s(s, K, T, atm_vol, contract_details)
    
        # Drop columns that contain any zeros
        derman_df_for_s = derman_df_for_s.loc[:, (derman_df_for_s != 0).all(axis=0)]
        return derman_df_for_s




