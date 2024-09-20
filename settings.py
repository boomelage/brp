# -*- coding: utf-8 -*-
"""
Created on Mon Sep  9 13:54:57 2024

"""
import os
import sys
os.chdir(os.path.dirname(os.path.abspath(__file__)))
sys.path.append('term_structure')
sys.path.append('contract_details')
sys.path.append('misc')
import QuantLib as ql
import numpy as np
import pandas as pd
from data_query import dirdatacsv, dirdata
class model_settings():
    
    def __init__(self):
        
        """
        from settings import model_settings
        ms = model_settings()
        """
        self.ticker             =    'SPX'
        self.s                  =    5625
        self.n_k                =    int(1e3)

        from routine_ivol_collection import raw_calls, raw_puts
        
        self.raw_calls = raw_calls
        self.raw_puts = raw_puts
        self.raw_call_K = raw_calls.index
        self.raw_put_K = raw_puts.index
        
        self.call_atmvols = raw_calls.loc[self.s,:].replace(0,np.nan).dropna()
        self.put_atmvols = raw_puts.loc[self.s,:].replace(0,np.nan).dropna()
        
        self.T = self.call_atmvols.index
        self.T =  [
            
            # 2,   3,   
            
            7,   
            
            
            
            # 8,   9,  10,  
            
            
            
            14,  
            
            
            
            
            # 15,  16,  17,  21,  22,  23, 24,  
            
            
            
            
            28,
            
            29,  
            
            30,  
            31, 
            
            # 37,  39,  46, 60,  74, 95, 
            
            106, 
            158, 
            
            # 165, 186, 196, 242, 277, 287, 305, 
            
            368, 
            
            # 459, 487, 640
            ]
        # sep 16th
        
        self.call_K = raw_calls.index[raw_calls.index>self.s]
        self.put_K = raw_puts.index[raw_puts.index<self.s]
        
        self.calibration_call_K = self.call_K[:3]
        self.calibration_put_K = self.put_K[-3:]
        
        
        self.calibration_K = np.array(
            [self.calibration_put_K,self.calibration_call_K]).flatten().tolist()
        
       
        self.day_count          =    ql.Actual365Fixed()
        self.calendar           =    ql.UnitedStates(m=1)
        self.calculation_date   =    ql.Date.todaysDate()
        self.csvs               =    dirdatacsv()
        self.xlsxs              =    dirdata()
        
        ql.Settings.instance().evaluationDate = self.calculation_date
        
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
            self.calculation_date, ql.QuoteHandle(ql.SimpleQuote(rate)), self.day_count))
        return ts_object

    def compute_maturity_date(self,row):
        row['calculation_date'] = self.calculation_date
        row['maturity_date'] = self.calculation_date + ql.Period(
            int(row['days_to_maturity']),ql.Days)
        return row
    
    def heston_price_one_vanilla(
            self,s,k,t,r,g,v0,kappa,theta,sigma,rho,w):
        
        call, put = ql.Option.Call, ql.Option.Put
        option_type = call if w == 'call' else put
        expiration_date = self.calculation_date + ql.Period(t,ql.Days)
        payoff = ql.PlainVanillaPayoff(option_type, k)
        exercise = ql.EuropeanExercise(expiration_date)
        european_option = ql.VanillaOption(payoff, exercise)
        flat_ts = self.make_ts_object(r)
        dividend_ts = self.make_ts_object(g)
        heston_process = ql.HestonProcess(
            flat_ts,dividend_ts, 
            ql.QuoteHandle(ql.SimpleQuote(s)), 
            v0, kappa, theta, sigma, rho)
        engine = ql.AnalyticHestonEngine(
            ql.HestonModel(heston_process), 0.01, 1000)
        european_option.setPricingEngine(engine)
        h_price = european_option.NPV()
        return h_price
    
    def heston_price_vanilla_row(self,row):
        s = row['spot_price']
        k = row['strike_price']
        t = row['days_to_maturity']
        r = row['risk_free_rate']
        g = row['dividend_rate']
        v0 = row['v0']
        kappa = row['kappa']
        theta = row['theta']
        sigma = row['sigma']
        rho = row['rho']
        w = row['w']
        
        date = self.calculation_date + ql.Period(t,ql.Days)
        option_type = ql.Option.Call if w == 'call' else ql.Option.Put
        
        payoff = ql.PlainVanillaPayoff(option_type, k)
        exercise = ql.EuropeanExercise(date)
        european_option = ql.VanillaOption(payoff, exercise)
        flat_ts = self.make_ts_object(r)
        dividend_ts = self.make_ts_object(g)
        heston_process = ql.HestonProcess(
            flat_ts,dividend_ts, 
            ql.QuoteHandle(ql.SimpleQuote(s)), 
            v0, kappa, theta, sigma, rho)
        engine = ql.AnalyticHestonEngine(
            ql.HestonModel(heston_process), 0.01, 1000)
        european_option.setPricingEngine(engine)
        h_price = european_option.NPV()
        row['heston_price'] = h_price
        return row
        
        