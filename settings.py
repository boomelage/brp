# -*- coding: utf-8 -*-
"""

general settings

"""
import os
os.chdir(os.path.dirname(os.path.abspath(__file__)))
import sys
import QuantLib as ql
import numpy as np
import pandas as pd
from scipy.stats import norm

class model_settings():
    
    def __init__(self):
        
        sys.path.append('contract_details')
        sys.path.append('train_data')
        sys.path.append('historical_data')
        sys.path.append('misc')
        
        self.day_count          =    ql.Actual365Fixed()
        self.calendar           =    ql.UnitedStates(m=1)
        self.calculation_date   =    ql.Date.todaysDate()
        
        self.default_bar = str("{percentage:3.0f}% | {n_fmt}/{total_fmt} {unit} | "
        "{rate_fmt} | Elapsed: {elapsed} | Remaining: {remaining} | ")
    
    """
    QuantLib time tools
    """    
    def expiration_datef(self,t,calculation_date=None):
        if calculation_date is None:
            calculation_date = self.calculation_date
        expiration_date = calculation_date + ql.Period(int(t),ql.Days)
        return expiration_date
    
    def vexpiration_datef(self,T,calculation_date=None):
        if calculation_date is None:
            calculation_date = self.calculation_date
        vdates = np.vectorize(self.expiration_datef)
        expiration_dates = vdates(T,calculation_date)
        return expiration_dates
    
    
    """
    QuanLib object makers
    """
    def make_implied_vols_matrix(self, strikes, maturities, term_strucutre):
        implied_vols_matrix = ql.Matrix(len(strikes),len(maturities))
        for i, strike in enumerate(strikes):
            for j, maturity in enumerate(maturities):
                implied_vols_matrix[i][j] = float(
                    term_strucutre.loc[strike,maturity])
        return implied_vols_matrix
    
    def make_black_var_surface(
            self, expiration_dates, Ks, implied_vols_matrix, 
            calculation_date=None):
        
        if calculation_date is None:
            calculation_date = self.calculation_date
        ql.Settings.instance().evaluationDate = calculation_date
        
        black_var_surface = ql.BlackVarianceSurface(
            calculation_date, self.calendar,
            expiration_dates, Ks,
            implied_vols_matrix, self.day_count)
        return black_var_surface

    def make_ts_object(self, rate):
        yield_object = ql.FlatForward(
            0, ql.NullCalendar(), 
            ql.QuoteHandle(ql.SimpleQuote(rate)), 
            self.day_count)
        ts_object = ql.YieldTermStructureHandle(yield_object)
        return ts_object
    
    
    """
    miscellaneous convenience
    """
    
    def vmoneyness(self, 
                   S, K, W
                   ):
        
        def compute_moneyness(s,k,w):
            if w == 'call':
                call_moneyness = s/k-1
                return call_moneyness
            elif w == 'put':
                put_moneyness = k/s-1
                return put_moneyness
            else:
                raise ValueError(f'{w} is not a valid put/call flag')

        vrel_moneyness = np.vectorize(compute_moneyness)
        
        moneyness = vrel_moneyness(S,K,W)
        
        return moneyness
    
    def encode_moneyness(self, array):
        moneyness_tags = np.asarray(array)
        moneyness_tags = np.empty_like(array, dtype=object)
        moneyness_tags[array == 0] = 'atm'
        moneyness_tags[array < 0] = 'otm'
        moneyness_tags[array > 0] = 'itm'
        return moneyness_tags
    
    def apply_heston_parameters(self, df, heston_parameters):
        paras = ['v0','theta','eta','kappa','rho']
        for para in paras:
            df[para] = heston_parameters[para]
        return df
    
    
    """
    pricing functions 
    """
    
    def noisyfier(self,prices):
        price = prices.columns[-1]
        
        prices['observed_price'] = prices[price]\
                                .apply(lambda x: x + np.random.normal(
                                    scale=0.15))
        prices['observed_price'] = prices['observed_price']\
                                .apply(lambda x: max(x, 0))
        return prices

    def black_scholes_price(self,s,k,t,r,volatility,w): 
        if w == 'call':
            w = 1
        elif w == 'put':
            w = -1
        else:
            raise KeyError('simple black scholes put/call flag error')
        d1 = (
            np.log(s/k)+(r+volatility**2/2)*t/365
            )/(
                volatility*np.sqrt(t/365)
                )
        d2 = d1-volatility*np.sqrt(t/365)
        
        price = w*(s*norm.cdf(w*d1)-k*np.exp(-r*t/365)*norm.cdf(w*d2))
        
        return price
    
    def ql_black_scholes(self,
            s,k,r,g,
            volatility,w,
            calculation_date, 
            expiration_date
            ):
        ql.Settings.instance().evaluationDate = calculation_date
        if w == 'call':
            option_type = ql.Option.Call
        elif w == 'put':
            option_type = ql.Option.Put
        else:
            raise KeyError("quantlib black scholes flag error")
        
        flat_ts = self.make_ts_object(r)
        divident_ts = self.make_ts_object(g)
        
        initialValue = ql.QuoteHandle(ql.SimpleQuote(s))
        
        volTS = ql.BlackVolTermStructureHandle(
            ql.BlackConstantVol(
                calculation_date, 
                ql.NullCalendar(), 
                ql.QuoteHandle(ql.SimpleQuote(volatility)), 
                self.day_count
                )
            )
        
        process = ql.GeneralizedBlackScholesProcess(
            initialValue, divident_ts, flat_ts, volTS)
        
        engine = ql.AnalyticEuropeanEngine(process)
        
        payoff = ql.PlainVanillaPayoff(option_type, k)
        europeanExercise = ql.EuropeanExercise(expiration_date)
        european_option = ql.VanillaOption(payoff, europeanExercise)
        european_option.setPricingEngine(engine)
        
        ql_black_scholes_price = european_option.NPV()
        return ql_black_scholes_price
    
    
    def ql_heston_price(self,
            s,k,r,g,w,
            v0,kappa,theta,eta,rho,
            calculation_date,
            expiration_date
            ):
        ql.Settings.instance().evaluationDate = calculation_date
        if w == 'call':
            option_type = ql.Option.Call
        elif w == 'put':
            option_type = ql.Option.Put
        else:
            raise KeyError('quantlib heston put/call error')

        payoff = ql.PlainVanillaPayoff(option_type, k)
        exercise = ql.EuropeanExercise(expiration_date)
        european_option = ql.VanillaOption(payoff, exercise)
        
        flat_ts = self.make_ts_object(float(r))
        dividend_ts = self.make_ts_object(float(g))
        heston_process = ql.HestonProcess(
            
            flat_ts,dividend_ts, 
            
            ql.QuoteHandle(ql.SimpleQuote(s)), 
            
            v0, kappa, theta, eta, rho)
        
        heston_model = ql.HestonModel(heston_process)
        
        engine = ql.AnalyticHestonEngine(heston_model)
        
        european_option.setPricingEngine(engine)
        
        h_price = european_option.NPV()
        return h_price
    
    
    def ql_barrier_price(self,
            s,k,t,r,g,calculation_date, w,
            barrier_type_name,barrier,rebate,
            v0, kappa, theta, eta, rho
            ):
        ql.Settings.instance().evaluationDate = calculation_date
        flat_ts = self.make_ts_object(r)
        dividend_ts = self.make_ts_object(g)
        
        spotHandle = ql.QuoteHandle(ql.SimpleQuote(s))
        
        hestonProcess = ql.HestonProcess(
            flat_ts, dividend_ts, spotHandle, 
            v0, kappa, theta, eta, rho)
        
        hestonModel = ql.HestonModel(hestonProcess)
        engine = ql.FdHestonBarrierEngine(hestonModel)
        
        if w == 'call':
            option_type = ql.Option.Call
        elif w == 'put':
            option_type = ql.Option.Put
        else:
            raise KeyError("quantlib black scholes flag error")
        
        if barrier_type_name == 'UpOut':
            barrierType = ql.Barrier.UpOut
        elif barrier_type_name == 'DownOut':
            barrierType = ql.Barrier.DownOut
        elif barrier_type_name == 'UpIn':
            barrierType = ql.Barrier.UpIn
        elif barrier_type_name == 'DownIn':
            barrierType = ql.Barrier.DownIn
        else:
            raise KeyError('barrier flag error')
            
        expiration_date = calculation_date + ql.Period(int(t), ql.Days)
        
        exercise = ql.EuropeanExercise(expiration_date)
        payoff = ql.PlainVanillaPayoff(option_type, k)
        
        barrierOption = ql.BarrierOption(
            barrierType, barrier, rebate, payoff, exercise)
        
        barrierOption.setPricingEngine(engine)
                
        barrier_price = barrierOption.NPV()
        
        return barrier_price
    
    """
    vectorized pricing functions
    """
    
    def vector_black_scholes(self,
            s,k,t,r,volatility,w
            ):
        vblack_scholes_price = np.vectorize(self.black_scholes_price)
        
        black_scholes_prices = vblack_scholes_price(
            s,k,t,r,volatility,w
            )
        return black_scholes_prices
    
    def vector_qlbs(self,
            s,k,r,g,
            volatility,w,
            calculation_date, 
            expiration_date
            ):
        vqlbs = np.vectorize(self.ql_black_scholes)
        
        ql_bsps = vqlbs(
                s,k,r,g,
                volatility,w,
                calculation_date, 
                expiration_date
                )
        return ql_bsps
        
    
    def vector_heston_price(self,
            s,k,r,g,w,
            v0,kappa,theta,eta,rho,
            calculation_date,
            expiration_date
            ):
        
        vql_heston_price = np.vectorize(self.ql_heston_price)
        heston_prices = vql_heston_price(
            s,k,r,g,w,
            v0,kappa,theta,eta,rho,
            calculation_date,
            expiration_date
            )
        return heston_prices
    
    def vector_barrier_price(self,
            s,k,t,r,g,calculation_date, w,
            barrier_type_name,barrier,rebate,
            v0, kappa, theta, eta, rho
            ):
        vql_barrier_price = np.vectorize(self.ql_barrier_price)
        
        barrier_prices = vql_barrier_price(
            s,k,t,r,g,calculation_date, w,
            barrier_type_name,barrier,rebate,
            v0, kappa, theta, eta, rho
            )
        
        return barrier_prices
    
    
    """
    approximations
    """

    def make_bicubic_functional(self,
            s,K,T,atm_volvec,volatility_coefs
            ):
        T = T.astype(int).tolist()
        K = K.astype(int).tolist()
        ql_T = ql.Array(T)
        ql_K = ql.Array(K)
        ql_vols = ql.Matrix(len(K),len(T),0.00)
        
        for i, k in enumerate(ql_K):
            for j, t in enumerate(ql_T):
                ql_vols[i][j] = atm_volvec[t] + volatility_coefs[t]*(k-s)
        
        bicubic_vol = ql.BicubicSpline(ql_T, ql_K, ql_vols)
        return bicubic_vol
    
    
    def make_derman_surface(self, 
            s,K,T,volatility_coefs,atm_volvec
            ):
        ts_df = pd.DataFrame(np.zeros((len(K),len(T)),dtype=float))
        ts_df.index = K
        ts_df.columns = T
        for k in K:
            for t in T:
                moneyness = k-s
                ts_df.loc[k,t] = atm_volvec[t] + volatility_coefs[t]*moneyness
        return ts_df

