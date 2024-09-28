# -*- coding: utf-8 -*-
"""
Created on Sat Sep 28 15:01:37 2024

@author: boomelage
"""
import numpy as np
from scipy.stats import norm
import QuantLib as ql
import pandas as pd
from settings import model_settings
ms = model_settings()

"""
# =============================================================================
# # ===========================================================================
# # # def black_scholes_price(
# # #         s,k,t,r,volatility,w
# # #         ): 
# # #     if w == 'call':
# # #         w = 1
# # #     elif w == 'put':
# # #         w = -1
# # #     else:
# # #         raise KeyError('simple black scholes put/call flag error')
# # #     d1 = (
# # #         np.log(s/k)+(r+volatility**2/2)*t/365
# # #         )/(
# # #             volatility*np.sqrt(t/365)
# # #             )
# # #     d2 = d1-volatility*np.sqrt(t/365)
# #     
# # #     price = w*(s*norm.cdf(w*d1)-k*np.exp(-r*t/365)*norm.cdf(w*d2))
# #     
# # #     return price
# # 
# # 
# # 
# # # def ql_heston_price(
# # #         s,k,r,g,w,
# # #         v0,kappa,theta,eta,rho,
# # #         calculation_date,
# # #         expiration_date
# # #         ):
# # #     ql.Settings.instance().evaluationDate = calculation_date
# # #     if w == 'call':
# # #         option_type = ql.Option.Call
# # #     elif w == 'put':
# # #         option_type = ql.Option.Put
# # #     else:
# # #         raise KeyError('quantlib heston put/call error')
# # 
# # #     payoff = ql.PlainVanillaPayoff(option_type, k)
# # #     exercise = ql.EuropeanExercise(expiration_date)
# # #     european_option = ql.VanillaOption(payoff, exercise)
# #     
# # #     flat_ts = ms.make_ts_object(float(r))
# # #     dividend_ts = ms.make_ts_object(float(g))
# # #     heston_process = ql.HestonProcess(
# #         
# # #         flat_ts,dividend_ts, 
# #         
# # #         ql.QuoteHandle(ql.SimpleQuote(s)), 
# #         
# # #         v0, kappa, theta, eta, rho)
# #     
# # #     heston_model = ql.HestonModel(heston_process)
# #     
# # #     engine = ql.AnalyticHestonEngine(heston_model)
# #     
# # #     european_option.setPricingEngine(engine)
# #     
# # #     h_price = european_option.NPV()
# # #     return h_price
# # ===========================================================================
# 
# # ===========================================================================
# # # def vector_black_scholes(
# # #         s,k,t,r,volatility,w
# # #         ):
# # #     vblack_scholes_price = np.vectorize(black_scholes_price)
# #     
# # #     black_scholes_prices = vblack_scholes_price(
# # #         s,k,t,r,volatility,w
# # #         )
# # #     return black_scholes_prices
# #     
# # # def vector_heston_price(
# # #         s,k,r,g,w,
# # #         v0,kappa,theta,eta,rho,
# # #         calculation_date,
# # #         expiration_date
# # #         ):
# #     
# # #     vql_heston_price = np.vectorize(ql_heston_price)
# # #     heston_prices = vql_heston_price(
# # #         s,k,r,g,w,
# # #         v0,kappa,theta,eta,rho,
# # #         calculation_date,
# # #         expiration_date
# # #         )
# # #     return heston_prices
# # ===========================================================================
# =============================================================================
"""


"""
example usage

"""
from routine_calibration_generation import calibration_dataset
from routine_calibration_testing import heston_parameters
test_dataset = ms.apply_heston_parameters(calibration_dataset,heston_parameters)

r = 0.04
g = 0.00
volatility = 0.2
w = ['put']
calculation_date = ms.calculation_date


s = test_dataset['spot_price']
k = test_dataset['strike_price']
t = test_dataset['days_to_maturity']

test_dataset['black_scholes'] = ms.vector_black_scholes(
        s,k,t,r,volatility,w
        )


v0 = heston_parameters['v0']
kappa = heston_parameters['v0']
theta = heston_parameters['v0']
eta = heston_parameters['v0']
rho = heston_parameters['v0']
expiration_date = ms.compute_ql_maturity_dates(t)
test_dataset['heston_price'] = ms.vector_heston_price(
        s,k,r,g,w,
        v0,kappa,theta,eta,rho,
        calculation_date,
        expiration_date
    )



pd.set_option("display.max_columns",None)

test_dataset

# pd.reset_option("display.max_columns")
