# -*- coding: utf-8 -*-
"""
Created on Sat Sep 28 13:15:03 2024

@author: boomelage
"""
import os
os.chdir(os.path.dirname(os.path.abspath(__file__)))
import QuantLib as ql
from settings import model_settings
ms = model_settings()
from routine_calibration_testing import heston_parameters



s = 1277.92
k = 1150.128
r = 0.04
g = 0.00
volatility = 0.21
w = 'call'
t = 10
calculation_date = ms.calculation_date
expiration_date = calculation_date + ql.Period(int(t),ql.Days)



bs = ms.ql_black_scholes(
    s = s, k = k, r = r, g = g,
    volatility = volatility ,w = w,
    calculation_date = calculation_date,
    expiration_date = expiration_date
    )

print(f"\nquantlib black scholes: {bs}\n")


heston = ms.ql_heston_price(
        s=s,k=k,r=r,g=g,w=w,
        v0=heston_parameters.loc['v0'],
        # kappa=heston_parameters['kappa'].loc[0],
        # theta=heston_parameters['theta'].loc[0],
        # eta=heston_parameters['eta'].loc[0],
        # rho=heston_parameters['rho'].loc[0],
        # calculation_date=calculation_date,
        # expiration_date=expiration_date
        )
print(f"\nquantlib heston: {heston}\n")