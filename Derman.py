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
import numpy as np
from sklearn.linear_model import LinearRegression

pd.set_option('display.max_columns',None)

class derman():
    def __init__(
            self, derman_coefs = None, derman_filename=None,implied_vols=None):
        self.derman_filename = derman_filename
        self.mats = []
        self.derman_coefs = derman_coefs
        self.implied_vols = implied_vols
        
    def retrieve_ts(self):
        
        file = self.derman_filename
        try:
           ts = pd.read_csv(file)
           ts = ts.set_index(ts.iloc[:, 0]).drop(columns=ts.columns[0])
           ts = ts.astype(float)
           ts.columns = ts.columns.astype(int)
        except Exception:
            print("check working directory files!")
        ts = ts.loc[
            
            min(self.implied_vols.index):max(self.implied_vols.index),
            :
    
            ]
        ts = ts.loc[:, (ts != 0).any(axis=0)]
    
        ks = ts.index.tolist()
        mats = ts.columns.tolist()
        return ks, mats, ts


    def compute_derman_ivols(self,s,maturity,ts,atm_value):
        TSatmat = ts.loc[:,maturity]
        strikes = ts.index
        x = np.array(strikes - s,dtype=float)
        y = np.array(TSatmat - atm_value,dtype=float)
        model = LinearRegression()
        x = x.reshape(-1,1)
        model.fit(x,y)
        b = model.coef_[0]
        alpha = model.intercept_
        derman_ivols = model.predict(x)
        derman_ivols = derman_ivols*b + alpha + atm_value
        return b, alpha, derman_ivols
        
    def get_derman_coefs(self):
        derman_coefs = {}
        ks, mats, ts = self.retrieve_ts()
        for i, maturity in enumerate(mats):
            for j, k in enumerate(ks):
                b, alpha, atmvol, derman_ivols = self.compute_derman_ivols(maturity,ts)
                derman_coefs[int(f"{maturity}")] = [b, alpha]
        derman_coefs = pd.DataFrame(derman_coefs)
        derman_coefs['coef'] = ['b','alpha']
        derman_coefs.set_index('coef',inplace = True)
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
        
    
    def make_derman_df(self, s, K, T, atm_vol_df):
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
                    derman_df.loc[k, t] = atm_vol_df.loc[s,t] + alpha + moneyness * b
                except Exception:
                    pass
        return derman_df
    
    def make_derman_df_for_S(self, s, K, T, atm_vol_df, contract_details, derman_coefs, derman_maturities):
        def make_for_s(
                s, K, T, 
                atm_vol, contract_details, derman_coefs, derman_maturities):
            contract_details = contract_details[
                contract_details['days_to_maturity'].isin(derman_maturities)
            ].reset_index(drop=True)
    
            # Create the derman DataFrame
            derman_df_for_s = self.make_derman_df(s, K, T, atm_vol_df)
            return derman_df_for_s
    
        # Call make_for_s to get derman DataFrame
        derman_df_for_s = make_for_s(s, K, T, atm_vol_df, contract_details, derman_coefs, derman_maturities)
    
        # Drop columns that contain any zeros
        derman_df_for_s = derman_df_for_s.loc[:, (derman_df_for_s != 0).all(axis=0)]
        return derman_df_for_s

def retrieve_derman_from_csv(filename):
    derman_coefs = pd.read_csv(filename)
    derman_coefs = derman_coefs.set_index('coef')
    derman_coefs.columns = derman_coefs.columns.astype(int)
    return derman_coefs




