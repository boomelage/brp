# -*- coding: utf-8 -*-
"""

general settings

"""
import os
os.chdir(os.path.dirname(os.path.abspath(__file__)))
import sys
import QuantLib as ql
import numpy as np
from scipy.stats import norm

class model_settings():
    

    
    def __init__(self):
        sys.path.append('contract_details')
        sys.path.append('train_data')
        sys.path.append('historical_data')
        sys.path.append('misc')
        self.today              =    ql.Date.todaysDate()
        self.day_count          =    ql.Actual365Fixed()
        self.calendar           =    ql.UnitedStates(m=1)
        self.compounding        =    ql.Continuous
        self.frequency          =    ql.Annual
        
    """
    QuantLib time tools
    """    
    def expiration_datef(self,t,calculation_date=None):
        expiration_date = calculation_date + ql.Period(int(t),ql.Days)
        return expiration_date
    
    def vexpiration_datef(self,T,calculation_date=None):
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
            calculation_date):
        ql.Settings.instance().evaluationDate = calculation_date
        
        black_var_surface = ql.BlackVarianceSurface(
            calculation_date, self.calendar,
            expiration_dates, Ks,
            implied_vols_matrix, self.day_count)
        return black_var_surface
    
    def ql_ts_rg(self,
            r,g,calculation_date
            ):
        
        riskFreeTS = ql.YieldTermStructureHandle(
            ql.FlatForward(
                calculation_date, r, 
                self.day_count, self.compounding, self.frequency
                )
            )
        
        dividendTS = ql.YieldTermStructureHandle(
            ql.FlatForward(
                calculation_date, g, 
                self.day_count, self.compounding, self.frequency
                )
            )
        
        return riskFreeTS, dividendTS
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
        moneyness_tags = np.full_like(
            array, fill_value='not_encoded', dtype=object)
        moneyness_tags[array == 0] = 'atm'
        moneyness_tags[array < 0] = 'otm'
        moneyness_tags[array > 0] = 'itm'
        return moneyness_tags
    
    def apply_heston_parameters(self, df, heston_parameters):
        paras = ['v0','theta','eta','kappa','rho']
        for para in paras:
            df[para] = heston_parameters[para]
        return df
    
    def make_K(self,
            s,spread,atm_spread,step
            ):
        
        K = np.sort(np.unique(
            np.concatenate(
                [
                    np.arange(
                        int(s - spread), int(s - atm_spread), step),
                    np.arange(
                        int(s + atm_spread), int(s + spread) + step, step)
                ]
            )
        ).astype(float)).tolist()
        
        return K
    
    """
    pricing functions 
    """
    
    def noisy_prices(self, prices):
        def noisify_price(price):
            noisy_price = max(price + np.random.normal(scale=0.15),0)
            return noisy_price
        prices_noisyfier = np.vectorize(noisify_price)
        noisy_prices = prices_noisyfier(prices)
        return noisy_prices
    
    def black_scholes_price(self,
            s,k,t,r,volatility,w
            ): 
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
            s, k, t, r,
            volatility,w,
            calculation_date, 
            ):

        if w == 'call':
            option_type = ql.Option.Call
        elif w == 'put':
            option_type = ql.Option.Put
        else:
            raise KeyError("quantlib black scholes flag error")
        
        ql.Settings.instance().evaluationDate = calculation_date
        expiration_date = calculation_date + ql.Period(int(t),ql.Days)
        
        ts_r = ql.YieldTermStructureHandle(
            ql.FlatForward(
                calculation_date, r, 
                self.day_count, self.compounding, self.frequency
                )
            )
        
        initialValue = ql.QuoteHandle(ql.SimpleQuote(s))
        
        volTS = ql.BlackVolTermStructureHandle(
            ql.BlackConstantVol(
                calculation_date, 
                self.calendar, 
                ql.QuoteHandle(ql.SimpleQuote(volatility)), 
                self.day_count
                )
            )

        process = ql.BlackScholesProcess(
            initialValue, ts_r, volTS)
        
        engine = ql.AnalyticEuropeanEngine(process)
        
        payoff = ql.PlainVanillaPayoff(option_type, k)
        europeanExercise = ql.EuropeanExercise(expiration_date)
        european_option = ql.VanillaOption(payoff, europeanExercise)
        european_option.setPricingEngine(engine)
        
        ql_black_scholes_price = european_option.NPV()
        return ql_black_scholes_price
    
    def ql_heston_price(self,
            s,k,t,r,g,w,
            kappa,theta,rho,eta,v0,
            calculation_date
            ):
        ql.Settings.instance().evaluationDate = calculation_date
        expiration_date = calculation_date + ql.Period(int(t),ql.Days)
        ts_r, ts_g = self.ql_ts_rg(r, g, calculation_date)
        
        if w == 'call':
            option_type = ql.Option.Call
        elif w == 'put':
            option_type = ql.Option.Put
        else:
            raise KeyError('quantlib heston put/call error')

        payoff = ql.PlainVanillaPayoff(option_type, k)
        exercise = ql.EuropeanExercise(expiration_date)
        european_option = ql.VanillaOption(payoff, exercise)
        
        heston_process = ql.HestonProcess(
            
            ts_r, ts_g, 
            
            ql.QuoteHandle(ql.SimpleQuote(s)), 
            
            v0, kappa, theta, eta, rho
            
            )
        
        heston_model = ql.HestonModel(heston_process)
        
        engine = ql.AnalyticHestonEngine(heston_model)
        
        european_option.setPricingEngine(engine)
        
        h_price = european_option.NPV()
        return h_price
    
    def ql_barrier_price(self,
            s,k,t,r,g,calculation_date, w,
            barrier_type_name,barrier,rebate,
            kappa,theta,rho,eta,v0
            ):
        ts_r, ts_g = self.ql_ts_rg(r, g, calculation_date)
        ql.Settings.instance().evaluationDate = calculation_date
        heston_process = ql.HestonProcess(
            
            ts_r,ts_g, 
            
            ql.QuoteHandle(ql.SimpleQuote(s)), 
            
            v0, kappa, theta, eta, rho
            
            )
        
        heston_model = ql.HestonModel(heston_process)
        engine = ql.FdHestonBarrierEngine(heston_model)
        
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
    
    def ql_bates_vanilla_price(self,
            s,k,t,r,g,
            kappa,theta,rho,eta,v0,
            lambda_, nu, delta, 
            w, calculation_date
            ):
        
        if w == 'call':
            option_type = ql.Option.Call
        elif w == 'put':
            option_type = ql.Option.Put
        else:
            raise ValueError("call/put flag w sould be either 'call' or 'put'")
        
        ql.Settings.instance().evaluationDate = calculation_date   
        expiration_date = calculation_date + ql.Period(int(t),ql.Days)
        
        ts_r, ts_g = self.ql_ts_rg(r, g, calculation_date)
        
        bates_process = ql.BatesProcess(
            ts_r, 
            ts_g,
            ql.QuoteHandle(ql.SimpleQuote(s)),
            v0, kappa, theta, eta, rho,
            lambda_, nu, delta,
        )
        engine = ql.BatesEngine(ql.BatesModel(bates_process))
        
        payoff = ql.PlainVanillaPayoff(option_type, float(k))
        europeanExercise = ql.EuropeanExercise(expiration_date)
        european_option = ql.VanillaOption(payoff, europeanExercise)
        european_option.setPricingEngine(engine)
        
        bates = european_option.NPV()
        return bates
    
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
        
    def vector_heston_price(self,
            s,k,t,r,g,w,
            kappa,theta,rho,eta,v0,
            calculation_date
            ):
        vql_heston_price = np.vectorize(self.ql_heston_price)
        heston_prices = vql_heston_price(
            s,k,t,r,g,w,
            kappa, theta, rho, eta, v0,
            calculation_date
            )
        return heston_prices
    
    def vector_barrier_price(self,df
            ):
        vql_barrier_price = np.vectorize(self.ql_barrier_price)
        barrier_prices = vql_barrier_price(
            df['spot_price'],
            df['strike_price'],
            df['days_to_maturity'],
            df['risk_free_rate'],
            df['dividend_rate'],
            df['calculation_date'], 
            df['w'],
            df['barrier_type_name'],
            df['barrier'],
            df['rebate'],
            df['kappa'],
            df['theta'],
            df['rho'],
            df['eta'],
            df['v0']
            )
        return barrier_prices
    
    def vector_bates_vanillas(self,
            s,k,t,r,g,
            kappa,theta,rho,eta,v0,
            lambda_, nu, delta, calculation_date
            ):
        v_bates = np.vectorize(self.ql_bates_vanilla_price)
        
        bates_vanillas = v_bates(
            s,k,t,r,g,
            kappa,theta,rho,eta,v0,
            lambda_, nu, delta, calculation_date
            )
        
        return bates_vanillas
        
    """
    approximations
    """

    def derman_volatility(self,s,k,t,coef,atm_vol):
        volatility = atm_vol + (k-s)*coef
        return volatility
    
    def derman_volatilities(self,s,k,t,coef,vol):
        vols = np.vectorize(self.derman_volatility)
        volatilities =  vols(s,k,t,coef,vol)
        return volatilities
            
    def make_bicubic_functional(self,
            s,K,atm_volvec,volatility_coefs
            ):
        T = atm_volvec.index.tolist()
        ql_T = ql.Array(T)
        ql_K = ql.Array(K)
        ql_vols = ql.Matrix(len(K),len(T),0.00)
        
        for i, k in enumerate(ql_K):
            for j, t in enumerate(ql_T):
                ql_vols[i][j] = atm_volvec[t] + volatility_coefs[t]*(k-s)
        
        bicubic_vol = ql.BicubicSpline(ql_T, ql_K, ql_vols)
        return bicubic_vol

        
        
        
        
        
        
        
        
        
        