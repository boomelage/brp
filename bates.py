# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

import os
current_dir = os.path.abspath(os.path.dirname(os.path.abspath(__file__)))
os.chdir(current_dir)
import QuantLib as ql
from settings import model_settings
ms = model_settings()


s = 100
k = 90
r = 0.04
g = 0.0
kappa = 2.0
theta = 0.1
rho = -0.75
eta = 0.3
v0 = 0.02

lambda_ = 0.1  # Jump intensity
nu = 0.05  # Mean jump size
delta = 0.02  # Jump size standard deviation

calculation_date = ms.today
t = 1


bates = ms.ql_bates_vanilla_price(
    s, k, t, r, g, kappa, theta, rho, eta, v0, lambda_, nu, delta, calculation_date)

black_scholes = ms.black_scholes_price(s, k, t, r, 0.18, 'call')

heston = ms.ql_heston_price(
    s, k, t, r, g, 'call', kappa, theta, rho, eta, v0, 
    calculation_date)


print(f"\nBates: {bates}\nHeston: {heston}\nBlack Scholes: {black_scholes}\n")


