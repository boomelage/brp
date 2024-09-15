# -*- coding: utf-8 -*-
"""
Created on Mon Sep  9 13:54:57 2024

"""
import os
pwd = os.path.dirname(os.path.abspath(__file__))
os.chdir(pwd)
import QuantLib as ql
import numpy as np

class model_settings():
    
    def __init__(self,
                 
            dividend_rate      =    0.015,
            risk_free_rate     =    0.05, 
            day_count          =    ql.Actual365Fixed(), 
            calendar           =    ql.UnitedStates(m=1),
            calculation_date   =    ql.Date.todaysDate(),
            ticker             =    'SPX',
            lower_strike       =    5610,
            upper_strike       =    5660,
            lower_maturity     =    None,
            upper_maturity     =    None,
            s                  =    5630
            ):
        self.dividend_rate = dividend_rate
        self.risk_free_rate = risk_free_rate
        self.dividend_rate = ql.QuoteHandle(ql.SimpleQuote(dividend_rate))
        self.day_count = day_count
        self.calendar = calendar
        self.calculation_date = calculation_date
        ql.Settings.instance().evaluationDate = calculation_date
        self.flat_ts = ql.YieldTermStructureHandle(ql.FlatForward(
            self.calculation_date, self.risk_free_rate, self.day_count))
        self.dividend_ts = ql.YieldTermStructureHandle(ql.FlatForward(
            self.calculation_date, self.dividend_rate, self.day_count))
        from data_query import dirdatacsv
        self.csvs = dirdatacsv()
        self.ticker             =    ticker
        self.lower_strike       =    lower_strike
        self.upper_strike       =    upper_strike
        self.lower_maturity     =    lower_maturity
        self.upper_maturity     =    upper_maturity
        self.s                  =    s
        self.security_settings = (
            self.ticker, self.lower_strike, self.upper_strike, 
            self.lower_maturity, self.upper_maturity, self.s)
        
    def import_model_settings(self):
        dividend_rate = self.dividend_rate
        risk_free_rate = self.risk_free_rate
        calculation_date = self.calculation_date
        day_count = self.day_count
        calendar = self.calendar
        dividend_ts = self.dividend_ts
        flat_ts = self.flat_ts
        security_settings = self.security_settings
        ezimport = [
            "",
            "dividend_rate = settings['dividend_rate']",
            "risk_free_rate = settings['risk_free_rate']",
            "calculation_date = settings['calculation_date']",
            "day_count = settings['day_count']",
            "calendar = settings['calendar']",
            "flat_ts = settings['flat_ts']",
            "dividend_ts = settings['dividend_ts']",
            "security_settings = settings['security_settings']",
            "ticker = security_settings[0]",
            "lower_strike = security_settings[1]",
            "upper_strike = security_settings[2]",
            "lower_maturity = security_settings[3]",
            "upper_maturity = security_settings[4]",
            "s = security_settings[5]"
            ]
        def ezprint():
            for ez in ezimport:
                print(ez)
        ezprint()
        return {
            "dividend_rate": dividend_rate, 
            "risk_free_rate": risk_free_rate, 
            "calculation_date": calculation_date, 
            "day_count": day_count, 
            "calendar": calendar,
            "flat_ts": flat_ts,
            "dividend_ts": dividend_ts,
            "security_settings": security_settings
            }
            
    def make_ql_array(self,nparr):
        qlarr = ql.Array(len(nparr),1)
        for i in range(len(nparr)):
            qlarr[i] = float(nparr[i])
        return qlarr
    
    def compute_ql_maturity_dates(self, maturities):
        expiration_dates = np.empty(len(maturities),dtype=object)
        for i, maturity in enumerate(maturities):
            expiration_dates[i] = self.calculation_date + ql.Period(
                int(maturity), ql.Days)
        return expiration_dates
    
    def make_implied_vols_matrix(self, strikes, maturities, term_strucutre):
        implied_vols_matrix = ql.Matrix(len(strikes),len(maturities))
        for i, strike in enumerate(strikes):
            for j, maturity in enumerate(maturities):
                implied_vols_matrix[i][j] = term_strucutre.loc[strike,maturity]
        return implied_vols_matrix
    
    def make_black_var_surface(
            self, expiration_dates, Ks, implied_vols_matrix):
        black_var_surface = ql.BlackVarianceSurface(
            self.calculation_date, self.calendar,
            expiration_dates, Ks,
            implied_vols_matrix, self.day_count)
        return black_var_surface
