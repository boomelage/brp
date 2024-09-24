
# =============================================================================
# Libraries
# =============================================================================

import numpy as np
import os
import QuantLib as ql
from scipy.stats import norm
from settings import model_settings
ms = model_settings()

s = ms.s

day_count = ms.day_count
calendar = ms.calendar
calculation_date = ms.calculation_date

script_dir = os.path.dirname(os.path.abspath(__file__))
os.chdir(script_dir)

def black_scholes_price(row): 
        S =  row['spot_price']
        K =  row['strike_price']
        r =  row['risk_free_rate']
        T =  row['days_to_maturity'] 
        sigma =  row['volatility'] 
        w =  row['w']
        if w == 'call':
            w = 1
        else:
            w = -1
    
        d1 = (np.log(S/K)+(r+sigma**2/2)*T/365)/(sigma*np.sqrt(T/365))
        d2 = d1-sigma*np.sqrt(T/365)
        price = w*(S*norm.cdf(w*d1)-K*np.exp(-r*T/365)*norm.cdf(w*d2))
        row['black_scholes'] = price
        return row
    
def heston_price_vanilla_row(row):
    
    if row['w'] == 'call':
        option_type = ql.Option.Call
    elif row['w'] == 'put':
        option_type = ql.Option.Put
    else:
        raise ValueError("flag error")
        
    dividend_rate = row['dividend_rate']
    risk_free_rate = row['risk_free_rate']
    flat_ts = ms.make_ts_object(risk_free_rate)
    dividend_ts = ms.make_ts_object(dividend_rate)
    
    s = row['spot_price']
    k = row['strike_price']
    t = row['days_to_maturity']
    v0 = row['v0']
    theta = row['theta']
    sigma = row['sigma']
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
    
    engine = ql.AnalyticHestonEngine(ql.HestonModel(heston_process),0.01,1000)
    european_option.setPricingEngine(engine)
    h_price_vanilla = european_option.NPV()
    row['heston_price'] = h_price_vanilla
    
    return row

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




