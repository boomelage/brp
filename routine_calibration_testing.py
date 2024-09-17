# -*- coding: utf-8 -*-
"""
Created on Tue Sep 17 12:28:08 2024

@author: boomelage
"""

import os
import sys
os.chdir(os.path.dirname(os.path.abspath(__file__)))
sys.path.append('term_structure')
sys.path.append('contract_details')
sys.path.append('misc')


import numpy as np
from routine_calibration import heston_parameters

T = heston_parameters.index

def compute_bs_ivols(T):
    ivol_col = np.empty(len(T),dtype=float)
    for i, t in enumerate(T):
        t = t
        S = heston_parameters.loc[t,'spot_price']
        sigma = heston_parameters.loc[t,'volatility']
        r = 0.05
        K = 5615
        print(S)
        print(sigma)
        print(t)
        
        d1 = (np.log(S/K)+(r+sigma**2/2)*t/365)/(sigma*np.sqrt(t/365))
        print(d1)

        a = t/2*d1
        b = -np.sqrt(t)
        c = np.log(S/K)+r*t/d1
        print(a)
        print(b)
        print(c)
        print("")
        coefficients = [a,b,c]
       
        ivols = np.roots(coefficients)
        ivol_col[i] = ivols[1]
        print(ivol_col)
    heston_parameters['ivol'] = ivol_col
    return heston_parameters


heston_parameters = compute_bs_ivols(T)
heston_parameters['vol_errror'] = heston_parameters['ivol']/heston_parameters['volatility']-1

print(heston_parameters)