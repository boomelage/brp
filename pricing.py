
# =============================================================================
# Libraries
# =============================================================================

import numpy as np
import os
import QuantLib as ql
from scipy.stats import norm
from settings import model_settings
ms = model_settings()
settings = ms.import_model_settings()

dividend_rate = settings[0]['dividend_rate']
risk_free_rate = settings[0]['risk_free_rate']

security_settings = settings[0]['security_settings']
s = security_settings[5]

ticker = security_settings[0]
lower_moneyness = security_settings[1]
upper_moneyness = security_settings[2]
lower_maturity = security_settings[3]
upper_maturity = security_settings[4]

day_count = settings[0]['day_count']
calendar = settings[0]['calendar']
calculation_date = settings[0]['calculation_date']

script_dir = os.path.dirname(os.path.abspath(__file__))
os.chdir(script_dir)

def black_scholes_price(row): 
        S =  row['spot_price']
        K =  row['strike_price']
        r =  row['risk_free_rate']
        T =  row['days_to_maturity'] 
        sigma =  row['volatility'] 
        w =  row['w']
    
    
        d1 = (np.log(S/K)+(r+sigma**2/2)*T/365)/(sigma*np.sqrt(T/365))
        d2 = d1-sigma*np.sqrt(T/365)
        price = w*(S*norm.cdf(w*d1)-K*np.exp(-r*T/365)*norm.cdf(w*d2))
        
        
        return price
    
def heston_price_vanilla_row(row):
    try:
        call, put = ql.Option.Call, ql.Option.Put
        option_type = call if row['w'] == 'call' else put
        
        flat_ts = ms.make_ts_object(ms.risk_free_rate)
        dividend_ts = ms.make_ts_object(ms.dividend_rate)
    
        s = row['spot_price']
        k = row['strike_price']
        t = row['days_to_maturity']
        v0 = row['v0']
        theta = row['theta']
        sigma = row['volatility']
        kappa = row['kappa']
        rho = row['rho']
        
        spot_handle = ql.QuoteHandle(ql.SimpleQuote(float(s)))
        
        maturity_date = calculation_date + ql.Period(int(t),ql.Days)
        
        payoff = ql.PlainVanillaPayoff(option_type, k)
        
        exercise = ql.EuropeanExercise(maturity_date)
        
        european_option = ql.VanillaOption(payoff, exercise)
        
        heston_process = ql.HestonProcess(
            flat_ts,                
            dividend_ts,            
            spot_handle,               
            v0,
            kappa,
            theta,        
            sigma,            
            rho                
        )
        
        engine = ql.AnalyticHestonEngine(ql.HestonModel(heston_process), 
                                         0.01, 
                                         1000)
        european_option.setPricingEngine(engine)
        
        h_price_vanilla = european_option.NPV()
        
        row['heston'] = h_price_vanilla
        return row
    
    except Exception:
        row['heston'] = np.nan
        return row


def heston_price_one_vanilla(w,calculation_date,strike_price,maturity_date,
                             spot_price,risk_free_rate,dividend_rate,v0,kappa,
                             theta,sigma,rho):
    call, put = ql.Option.Call, ql.Option.Put
    option_type = call if w == 1 else put
    
    day_count = ql.Actual365Fixed()
    # calendar = ql.UnitedStates(ql.UnitedStates.GovernmentBond)
    ql.Settings.instance().evaluationDate = ql.Date.todaysDate()
    
    price_date = calculation_date
    payoff = ql.PlainVanillaPayoff(option_type, strike_price)
    exercise = ql.EuropeanExercise(maturity_date)
    european_option = ql.VanillaOption(payoff, exercise)
    
    spot_handle = ql.QuoteHandle(ql.SimpleQuote(spot_price))
    
    flat_ts = ql.YieldTermStructureHandle(
        ql.FlatForward(price_date, 
                       float(risk_free_rate), 
                       day_count))
    dividend_yield = ql.YieldTermStructureHandle(
        ql.FlatForward(price_date, 
                       float(dividend_rate), 
                       day_count))
    
    heston_process = ql.HestonProcess(flat_ts, 
                                      dividend_yield, 
                                      spot_handle, 
                                      v0, 
                                      kappa, 
                                      theta, 
                                      sigma, 
                                      rho)
    
    engine = ql.AnalyticHestonEngine(ql.HestonModel(heston_process), 
                                     0.01, 
                                     1000)
    european_option.setPricingEngine(engine)
    
    h_price_vanilla = european_option.NPV()
    return h_price_vanilla


# Binary
# Barrier
# Asian arith/geo 

def noisyfier(prices):
    price = prices.columns[-1]
    
    prices['observed_price'] = prices[price]\
                            .apply(lambda x: x + np.random.normal(scale=0.15))
    prices['observed_price'] = prices['observed_price']\
                            .apply(lambda x: max(x, 0))
    
    return prices




