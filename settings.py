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
from bicubic_interpolation import make_bicubic_functional
from derman_test import derman_coefs
from data_query import dirdatacsv, dirdata


class model_settings():
    
    def __init__(self):
        
        """
        from settings import model_settings
        ms = model_settings()
        ms.
        """
        
        self.day_count          =    ql.Actual365Fixed()
        self.calendar           =    ql.UnitedStates(m=1)
        self.calculation_date   =    ql.Date.todaysDate()
        self.csvs               =    dirdatacsv()
        self.xlsxs              =    dirdata()
        ql.Settings.instance().evaluationDate = self.calculation_date
        self.ticker             =    'SPX'
        self.s                  =    1277.92
        self.n_k                =    int(1e3)


        
        self.step = self.s*0.005
        
        self.calibration_call_K = np.arange(
            self.s,
            self.s+5*self.step,
            self.step)
        
        self.calibration_put_K = np.arange(
            self.s-5*self.step,
            self.s,
            self.step)
        
        self.calibration_K = np.unique(np.array(
            [
                self.calibration_put_K,self.calibration_call_K
             ]
            ).flatten().astype(int).tolist())
        
        self.surf_K = np.linspace(self.s*0.5,self.s*1.5,1000).astype(int)
        
        self.atm_vols = [
            
            19.7389,
            21.2123, 
            21.9319,	
            23.0063, 
            23.6643, 
            # 24.1647,  
            # 24.4341
            ]
    
        
        self.ql_T = [
            
            ql.Period(30,ql.Days), 
            ql.Period(60,ql.Days), 
            ql.Period(3,ql.Months), 
            ql.Period(6,ql.Months), 
            ql.Period(12,ql.Months), 
            # ql.Period(18,ql.Months), 
            # ql.Period(24,ql.Months)
            
            ]
        
        self.expiration_dates = np.empty(len(self.ql_T),dtype=object)
        for i, p in enumerate(self.ql_T):
            self.expiration_dates[i] = self.calculation_date + p
        
        self.T = np.zeros(len(self.ql_T),dtype=int)
        for i, date in enumerate(self.expiration_dates):
            self.T[i] = date - self.calculation_date
        
        self.atm_vols = pd.DataFrame(self.atm_vols)/100
        
        self.atm_vols.index = self.T
        
        self.derman_coefs = derman_coefs.loc[[30,60,95,186,368]]
        self.derman_coefs.index = self.T
        
        
        self.derman_ts = pd.DataFrame(
            np.zeros((len(self.surf_K),len(self.T)),dtype=float))
        
        self.derman_ts.index = self.surf_K.astype(int)
        
        self.derman_ts.columns = self.T.astype(int)

        for i, k in enumerate(self.surf_K):
            moneyness = k-self.s
            for j, t in enumerate(self.T):
                self.derman_ts.loc[k,t] = (
                    self.atm_vols.loc[t,0] + \
                    self.derman_coefs.loc[t,0] * moneyness
                )
        self.derman_ts = self.derman_ts.dropna(how="any",axis=0)
        self.derman_ts = self.derman_ts.dropna(how="any",axis=1)

        
        self.bicubic_vol = make_bicubic_functional(
            self.derman_ts, self.surf_K.tolist(), self.T.tolist())
        

        
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

