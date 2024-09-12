# -*- coding: utf-8 -*-
"""
Created on Mon Sep  9 13:54:57 2024

"""
import QuantLib as ql
import numpy as np
import pandas as pd


class model_settings():
    
    def __init__(self,
            dividend_rate      =    0.015,
            risk_free_rate     =    0.05, 
            day_count          =    ql.Actual365Fixed(), 
            calendar           =    ql.UnitedStates(m=1),
            calculation_date   =    ql.Date.todaysDate(),
            file               =    None
            ):
        self.dividend_rate = dividend_rate
        self.risk_free_rate = risk_free_rate
        self.dividend_rate = ql.QuoteHandle(ql.SimpleQuote(dividend_rate))
        self.day_count = day_count
        self.calendar = calendar
        self.file = file
        self.calculation_date = calculation_date
        ql.Settings.instance().evaluationDate = calculation_date
        self.flat_ts = ql.YieldTermStructureHandle(ql.FlatForward(
            self.calculation_date, self.risk_free_rate, self.day_count))
        self.dividend_ts = ql.YieldTermStructureHandle(ql.FlatForward(
            self.calculation_date, self.dividend_rate, self.day_count))

    def import_model_settings(self):
        dividend_rate = self.dividend_rate
        risk_free_rate = self.risk_free_rate
        calculation_date = self.calculation_date
        day_count = self.day_count
        calendar = self.calendar
        dividend_ts = self.dividend_ts
        flat_ts = self.flat_ts
        ezimport = [
            "dividend_rate = settings['dividend_rate']",
            "risk_free_rate = settings['risk_free_rate']",
            "calculation_date = settings['calculation_date']",
            "day_count = settings['day_count']",
            "calendar = settings['calendar']",
            "flat_ts = settings['flat_ts']",
            "dividend_ts = settings['dividend_ts']"]
        print("\n")
        for ez in ezimport:
            print(ez)
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
            expiration_dates[i] = self.calculation_date + ql.Period(int(maturity), ql.Days)
        return expiration_dates

    
    
    