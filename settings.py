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
from data_query import dirdatacsv, dirdata
class model_settings():
    
    def __init__(self):
        self.day_count          =    ql.Actual365Fixed()
        self.calendar           =    ql.UnitedStates(m=1)
        self.calculation_date   =    ql.Date.todaysDate()
        self.csvs               =    dirdatacsv()
        self.xlsxs              =    dirdata()
        self.ticker             =    'SPX'
        self.s                  =    5630
        self.n_k                =    int(1e3)

        from routine_ivol_collection import raw_ts
        
        self.raw_ts = raw_ts
        
        self.atm_volvec = raw_ts.loc[self.s,:].replace(0,np.nan).dropna()
        self.raw_T = self.atm_volvec.index.astype(int)
        
        self.raw_T              = [
                                    3, 7, 14, 28, 42, 63, 109, 168
                                    ]
        
        self.raw_K              = [
            5615, 5620, 5625, 5630, 5635, 5640, 5645, 5650
            ]
        
        self.model_vol_ts = raw_ts.loc[self.raw_K,self.raw_T]
        
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
        
        payoff = ql.PlainVanillaPayoff(option_type, k)
        exercise = ql.EuropeanExercise(t)
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
        call, put = ql.Option.Call, ql.Option.Put
        option_type = call if w == 'call' else put
        
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
        row['heston'] = h_price
        return row
        
        