# -*- coding: utf-8 -*-
"""
Created on Sat Sep 28 15:01:37 2024

@author: boomelage
"""
import numpy as np
from scipy.stats import norm

"""
numpy black scholes
"""
def black_scholes_price(s,k,t,r,volatility,w): 
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

bs = black_scholes_price(s=100,k=90,t=14,r=0.04,volatility=0.2,w='call')

vbs = np.vectorize(black_scholes_price)

nbs = vbs(
    [100,100,100],
    [90,100,110],
    [1,7,14],
    0.04,
    0.2,
    'call'
    )



