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
k = 110
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

w = 'put'


bates = ms.ql_bates_vanilla_price(
    s, k, t, r, g, 
    kappa, theta, rho, eta, v0, lambda_, nu, delta, 
    w, calculation_date
    )

volatility = 0.18

black_scholes = ms.black_scholes_price(
    s, k, t, r, volatility, w
    )


heston = ms.ql_heston_price(
    s, k, t, r, g, w, kappa, theta, rho, eta, v0, 
    calculation_date
    )


barrier_type_name= 'UpIn'
b = 100
rebate = 0.0 



barrier = ms.ql_barrier_price(
    s, k, t, r, g, 
    calculation_date, w, 
    barrier_type_name, b, rebate, 
    kappa, theta, rho, eta, v0)


print(f"\nBates: {bates}\nHeston: {heston}"
      f"\nBlack Scholes: {black_scholes}\nfd barrier B=S: {barrier}\n")


