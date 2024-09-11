# -*- coding: utf-8 -*-
"""
Created on Mon Sep  9 13:54:57 2024

"""
import QuantLib as ql
import numpy as np


class model_settings():
    
    def __init__(self, dividend_rate=0.00, risk_free_rate=0.00):
        self.dividend_rate = dividend_rate
        self.risk_free_rate = risk_free_rate
        self.calculation_date = calculation_date = ql.Date.todaysDate()
        self.day_count = day_count = ql.Actual365Fixed()
        self.calendar = ql.UnitedStates(m=1)

        self.flat_ts = ql.YieldTermStructureHandle(ql.FlatForward(
            self.calculation_date, self.risk_free_rate, self.day_count))
        self.dividend_ts = ql.YieldTermStructureHandle(ql.FlatForward(
            self.calculation_date, self.dividend_rate, self.day_count))
        self.dividend_rate = ql.QuoteHandle(ql.SimpleQuote(dividend_rate)) 
        
        ql.Settings.instance().evaluationDate = calculation_date
    
    
    
    def import_model_settings(self):
        dividend_rate = self.dividend_rate
        risk_free_rate = self.risk_free_rate
        calculation_date = self.calculation_date
        day_count = self.day_count
        calendar = self.calendar
        return dividend_rate, risk_free_rate, calculation_date, day_count,\
            calendar,
            
    def make_ql_array(size,nparr):
        qlarr = ql.Array(size,1)
        for i in range(size):
            qlarr[i] = float(nparr[i])
        return qlarr
    
    
    def compute_ql_maturity_dates(
            maturities, calculation_date=ql.Date.todaysDate()):
        expiration_dates = np.empty(len(maturities),dtype=object)
        for i, maturity in enumerate(maturities):
            expiration_dates[i] = calculation_date + ql.Period(maturity, ql.Days)
        return expiration_dates
