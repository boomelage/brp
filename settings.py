# -*- coding: utf-8 -*-
"""
Created on Mon Sep  9 13:54:57 2024

"""
import os
pwd = os.path.dirname(os.path.abspath(__file__))
os.chdir(pwd)
import QuantLib as ql
import numpy as np
from data_query import dirdatacsv
class model_settings():
    
    def __init__(self,
            day_count          =    ql.Actual365Fixed(), 
            calendar           =    ql.UnitedStates(m=1),
            calculation_date   =    ql.Date.todaysDate(),
            dividend_rate      =    9999999,
            risk_free_rate     =    9999999
            ):
        self.csvs               = dirdatacsv()
        self.day_count          = day_count
        self.calendar           = calendar
        self.calculation_date   = calculation_date
        
        self.ticker             =    'SPX'
        
        self.lower_maturity     =    0
        self.upper_maturity     =    999999
        self.s                  =    None

        self.lower_moneyness    =    self.s * 0
        self.upper_moneyness    =    self.s * 999999
        self.security_settings  = (
            self.ticker, self.lower_moneyness, self.upper_moneyness, 
            self.lower_maturity, self.upper_maturity, self.s
            )
        self.risk_free_rate = ql.QuoteHandle(ql.SimpleQuote(risk_free_rate))
        self.dividend_rate = ql.QuoteHandle(ql.SimpleQuote(dividend_rate))
        ql.Settings.instance().evaluationDate = calculation_date
        
    def import_model_settings(self):
        dividend_rate = self.dividend_rate
        risk_free_rate = self.risk_free_rate
        calculation_date = self.calculation_date
        day_count = self.day_count
        calendar = self.calendar
        security_settings = self.security_settings
        ezimport = [
            "",
            "dividend_rate = settings[0]['dividend_rate']",
            "risk_free_rate = settings[0]['risk_free_rate']",
            "",
            "security_settings = settings[0]['security_settings']",
            "s = security_settings[5]",
            "",
            "ticker = security_settings[0]",
            "lower_moneyness = security_settings[1]",
            "upper_moneyness = security_settings[2]",
            "lower_maturity = security_settings[3]",
            "upper_maturity = security_settings[4]",
            "",
            "day_count = settings[0]['day_count']",
            "calendar = settings[0]['calendar']",
            "calculation_date = settings[0]['calculation_date']",
            "",
            ]
        
        def ezprint():
            for ez in ezimport:
                print(ez)
        return [{
            "dividend_rate": dividend_rate, 
            "risk_free_rate": risk_free_rate, 
            "calculation_date": calculation_date, 
            "day_count": day_count, 
            "calendar": calendar,
            "security_settings": security_settings
            }, ezprint]
            
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
                implied_vols_matrix[i][j] = float(term_strucutre.loc[strike,maturity])
        return implied_vols_matrix
    
    def make_black_var_surface(
            self, expiration_dates, Ks, implied_vols_matrix):
        black_var_surface = ql.BlackVarianceSurface(
            self.calculation_date, self.calendar,
            expiration_dates, Ks,
            implied_vols_matrix, self.day_count)
        return black_var_surface

    def make_ts_object(self,rate):
        ts_object = ql.YieldTermStructureHandle(ql.FlatForward(
            self.calculation_date, rate, self.day_count))
        return ts_object
