# -*- coding: utf-8 -*-
"""
Created on Mon Sep  9 13:54:57 2024

"""
import QuantLib as ql
import numpy as np
import pandas as pd


class model_settings():
    
    def __init__(self, 
            file               =    None,
            dividend_rate      =    0.015, 
            risk_free_rate     =    0.05, 
            day_count          =    ql.Actual365Fixed(), 
            calendar           =    ql.UnitedStates(m=1),
            calculation_date   =    ql.Date.todaysDate()
            ):
        self.dividend_rate = dividend_rate
        self.risk_free_rate = risk_free_rate
        self.calculation_date = calculation_date
        self.day_count = day_count
        self.calendar = calendar
        self.flat_ts = ql.YieldTermStructureHandle(ql.FlatForward(
            self.calculation_date, self.risk_free_rate, self.day_count))
        self.dividend_ts = ql.YieldTermStructureHandle(ql.FlatForward(
            self.calculation_date, self.dividend_rate, self.day_count))
        self.dividend_rate = ql.QuoteHandle(ql.SimpleQuote(dividend_rate)) 
        self.file = file
        ql.Settings.instance().evaluationDate = calculation_date
       
    def import_model_settings(self):
        dividend_rate = self.dividend_rate
        risk_free_rate = self.risk_free_rate
        calculation_date = self.calculation_date
        day_count = self.day_count
        calendar = self.calendar
        dividend_ts = self.dividend_ts
        flat_ts = self.flat_ts
        return {
            "dividend_rate": dividend_rate, 
            "risk_free_rate": risk_free_rate, 
            "calculation_date": calculation_date, 
            "day_count": day_count, 
            "calendar": calendar,
            "flat_ts": flat_ts,
            "dividend_ts": dividend_ts
            }
            
    def make_ql_array(self,size,nparr):
        qlarr = ql.Array(size,1)
        for i in range(size):
            qlarr[i] = float(nparr[i])
        return qlarr
    
    def compute_ql_maturity_dates(self, maturities):
        expiration_dates = np.empty(len(maturities),dtype=object)
        for i, maturity in enumerate(maturities):
            expiration_dates[i] = self.calculation_date + ql.Period(maturity, ql.Days)
        return expiration_dates

    def extract_ivol_matrix_from_market(self):
        df = pd.read_excel(self.file)
        
        df.columns = df.loc[1]
        df = df.iloc[2:,:].reset_index(drop=True).dropna()
        df = df.set_index('Strike')
        strikes = df.index.tolist()
        maturities = df['DyEx'].loc[strikes[0]].unique().tolist()
        
        calls = pd.concat([df.iloc[:, i:i+2] for i in range(
            0, df.shape[1], 4)], axis=1)
        callvols = calls['IVM']
        callvols.columns = maturities
        
        implied_vols_matrix = ql.Matrix(len(strikes),len(maturities),float(0))
        
        for i, maturity in enumerate(maturities):
            for j, strike in enumerate(strikes):
                implied_vols_matrix[j][i] = callvols.iloc[j,i]
        
        return implied_vols_matrix, strikes, maturities, callvols