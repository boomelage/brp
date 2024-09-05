
# =============================================================================
# Libraries
# =============================================================================

import numpy as np
import os
import QuantLib as ql
from scipy.stats import norm
script_dir = os.path.dirname(os.path.abspath(__file__))
os.chdir(script_dir)

def black_scholes_price(row): 
    S =  row['spot_price']
    K =  row['strike_price']
    r =  row['risk_free_rate']
    T =  row['years_to_maturity'] 
    sigma =  row['volatility'] 
    w =  row['w']


    d1 = (np.log(S/K)+(r+sigma**2/2)*T)/(sigma*np.sqrt(T))
    d2 = d1-sigma*np.sqrt(T)
    price = w*(S*norm.cdf(w*d1)-K*np.exp(-r*T)*norm.cdf(w*d2))
    return price

def heston_price_vanilla_row(row):
    call, put = ql.Option.Call, ql.Option.Put
    option_type = call if row['w'] == 1 else put
    
    day_count = ql.Actual365Fixed()
    # calendar = ql.UnitedStates(ql.UnitedStates.GovernmentBond)
    ql.Settings.instance().evaluationDate = ql.Date.todaysDate()
    
    price_date = row['calculation_date']
    payoff = ql.PlainVanillaPayoff(option_type, row['strike_price'])
    exercise = ql.EuropeanExercise(row['maturity_date'])
    european_option = ql.VanillaOption(payoff, exercise)
    
    spot_handle = ql.QuoteHandle(ql.SimpleQuote(row['spot_price']))
    
    flat_ts = ql.YieldTermStructureHandle(
        ql.FlatForward(price_date, 
                       float(row['risk_free_rate']), 
                       day_count))
    dividend_yield = ql.YieldTermStructureHandle(
        ql.FlatForward(price_date, 
                       float(row['dividend_rate']), 
                       day_count))
    
    heston_process = ql.HestonProcess(flat_ts, 
                                      dividend_yield, 
                                      spot_handle, 
                                      row['v0'], 
                                      row['kappa'], 
                                      row['theta'], 
                                      row['sigma'], 
                                      row['rho'])
    
    engine = ql.AnalyticHestonEngine(ql.HestonModel(heston_process), 
                                     0.01, 
                                     1000)
    european_option.setPricingEngine(engine)
    
    h_price_vanilla = european_option.NPV()
    return h_price_vanilla

def noisyfier(prices):
    price = prices.columns[-1]
    
    prices['observed_price'] = prices[price]\
                            .apply(lambda x: x + np.random.normal(scale=0.15))
    prices['observed_price'] = prices['observed_price']\
                            .apply(lambda x: max(x, 0))
    
    prices

# =============================================================================
# Independent pricing functions                   Independent pricing functions

def BS_price_vanillas(features):
    
    BS_features = features.copy()

    BS_features['black_scholes'] = BS_features.apply(
        lambda row: black_scholes_price(row), axis = 1)
    BS_vanillas = BS_features.copy()
    return BS_vanillas

def heston_price_vanillas(features):
    
    heston_params = features.copy()
    heston_params['heston_price'] = heston_params.apply(
        heston_price_vanilla_row, 
        axis=1)
    heston_vanillas = heston_params.copy()
    heston_vanillas = heston_vanillas.drop(
        columns = ['calculation_date','maturity_date'])
    return heston_vanillas

# Independent pricing functions                   Independent pricing functions
# =============================================================================

